#!/usr/bin/env python3
"""
Fine-tuning script for Text-to-SQL models
Trains the model for 1-2 epochs using generated training data
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import os
import logging
from typing import Dict, List
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextToSQLFineTuner:
    def __init__(self, base_model_path: str = "models/model-m3"):
        self.base_model_path = base_model_path
        self.tokenizer = None
        self.model = None
        self.training_data = []
        
    def load_model_and_tokenizer(self):
        """Load the pre-trained model and tokenizer"""
        logger.info(f"Loading model from {self.base_model_path}")
        
        try:
            # Load tokenizer with slow mode to avoid SentencePiece issues
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_path, 
                use_fast=False
            )
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model_path)
            
            # Set to training mode
            self.model.train()
            
            logger.info("✅ Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def load_training_data(self, filename: str = "training_data.json"):
        """Load training data from JSON file"""
        logger.info(f"Loading training data from {filename}")
        
        try:
            with open(filename, 'r') as f:
                self.training_data = json.load(f)
            
            logger.info(f"✅ Loaded {len(self.training_data)} training examples")
            
        except FileNotFoundError:
            logger.error(f"❌ Training data file {filename} not found")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading training data: {e}")
            raise
    
    def prepare_input_text(self, example: Dict) -> str:
        """Prepare input text with schema context"""
        schema = example.get("schema", "")
        question = example.get("question", "")
        
        # Format input text similar to the original model
        input_text = f"""
Database Schema:
{schema}

Question: {question}
"""
        return input_text.strip()
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """Tokenize the training examples"""
        # Prepare input texts
        input_texts = []
        target_texts = []
        
        for i in range(len(examples["question"])):
            # Create example dict for prepare_input_text
            example = {
                "question": examples["question"][i],
                "sql": examples["sql"][i],
                "schema": examples["schema"][i]
            }
            input_texts.append(self.prepare_input_text(example))
            target_texts.append(examples["sql"][i])
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            input_texts,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_texts,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    def create_dataset(self) -> Dataset:
        """Create HuggingFace dataset from training data"""
        logger.info("Creating dataset...")
        
        # Convert to dataset format
        dataset_dict = {
            "question": [ex["question"] for ex in self.training_data],
            "sql": [ex["sql"] for ex in self.training_data],
            "schema": [ex["schema"] for ex in self.training_data],
            "category": [ex["category"] for ex in self.training_data]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Tokenize the dataset
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logger.info(f"✅ Created dataset with {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def setup_training_args(self, output_dir: str = "./fine_tuned_model") -> TrainingArguments:
        """Setup training arguments for fine-tuning"""
        return TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,  # Train for 2 epochs as requested
            per_device_train_batch_size=2,  # Small batch size for memory efficiency
            per_device_eval_batch_size=2,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            eval_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=5e-5,  # Conservative learning rate
            fp16=torch.cuda.is_available(),  # Use mixed precision if available
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
        )
    
    def fine_tune(self, output_dir: str = "./fine_tuned_model"):
        """Run the fine-tuning process"""
        logger.info("🚀 Starting fine-tuning process...")
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Load training data
        self.load_training_data()
        
        # Create dataset
        dataset = self.create_dataset()
        
        # Split dataset (80% train, 20% eval)
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]
        
        logger.info(f"📊 Training examples: {len(train_dataset)}")
        logger.info(f"📊 Evaluation examples: {len(eval_dataset)}")
        
        # Setup training arguments
        training_args = self.setup_training_args(output_dir)
        
        # Create data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Start training
        logger.info("🎯 Starting training...")
        trainer.train()
        
        # Save the fine-tuned model
        logger.info(f"💾 Saving fine-tuned model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Evaluate the model
        logger.info("📊 Evaluating fine-tuned model...")
        eval_results = trainer.evaluate()
        
        logger.info("✅ Fine-tuning completed!")
        logger.info(f"📈 Final evaluation loss: {eval_results['eval_loss']:.4f}")
        
        return eval_results
    
    def test_fine_tuned_model(self, model_path: str = "./fine_tuned_model"):
        """Test the fine-tuned model with sample queries"""
        logger.info("🧪 Testing fine-tuned model...")
        
        # Load fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.eval()
        
        # Test queries
        test_queries = [
            "Show all actors",
            "How many musicals are there?",
            "Find actors older than 20",
            "Show musicals that won awards"
        ]
        
        # Sample schema (simplified)
        schema = """Table 'actors':
  - Actor_ID (INTEGER)
  - Name (TEXT)
  - Musical_ID (INTEGER)
  - Character (TEXT)
  - Duration (TEXT)
  - age (INTEGER)

Table 'musicals':
  - Musical_ID (INTEGER)
  - Name (TEXT)
  - Year (INTEGER)
  - Award (TEXT)
  - Category (TEXT)
  - Nominee (TEXT)
  - Result (TEXT)"""
        
        results = []
        
        for query in test_queries:
            # Prepare input
            input_text = f"""
Database Schema:
{schema}

Question: {query}
"""
            
            # Tokenize
            inputs = tokenizer(
                input_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            # Decode
            generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results.append({
                "question": query,
                "sql": generated_sql
            })
            
            logger.info(f"Q: {query}")
            logger.info(f"A: {generated_sql}")
            logger.info("-" * 50)
        
        return results

def main():
    """Main function to run fine-tuning"""
    print("Text-to-SQL Model Fine-Tuning")
    print("="*50)
    
    # Initialize fine-tuner
    fine_tuner = TextToSQLFineTuner()
    
    try:
        # Run fine-tuning
        eval_results = fine_tuner.fine_tune()
        
        # Test the fine-tuned model
        test_results = fine_tuner.test_fine_tuned_model()
        
        print("\n" + "="*50)
        print("✅ FINE-TUNING COMPLETED!")
        print("="*50)
        print(f"📈 Evaluation Loss: {eval_results['eval_loss']:.4f}")
        print(f"📊 Training Loss: {eval_results.get('train_loss', 'N/A')}")
        print(f"🎯 Test Queries: {len(test_results)}")
        
        print("\n📋 Next Steps:")
        print("1. Test the fine-tuned model in your app")
        print("2. Compare performance with original models")
        print("3. Iterate with more training data if needed")
        
    except Exception as e:
        logger.error(f"❌ Fine-tuning failed: {e}")
        raise

if __name__ == "__main__":
    main() 