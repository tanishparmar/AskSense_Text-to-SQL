import os
import json
import torch
import pandas as pd
import numpy as np
import re
import sqlite3
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    BartTokenizer,
    BartForConditionalGeneration,
    RobertaTokenizer,
    T5ForConditionalGeneration as T5EncoderDecoderModel
)
from tqdm import tqdm
import random
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

class SQLComponentAnalyzer:
    """Analyze SQL components for accuracy evaluation"""
    
    def __init__(self):
        self.components = {
            'select': ['select', 'distinct'],
            'where': ['where', 'and', 'or', 'not', 'in', 'like', 'between'],
            'group_by': ['group by', 'having'],
            'order_by': ['order by', 'asc', 'desc'],
            'join': ['join', 'inner join', 'left join', 'right join', 'outer join'],
            'aggregate': ['count', 'sum', 'avg', 'max', 'min', 'group_concat']
        }
    
    def extract_sql_components(self, sql: str) -> Dict[str, bool]:
        """Extract SQL components from a query"""
        sql_lower = sql.lower()
        components_found = {}
        
        for component_name, keywords in self.components.items():
            found = any(keyword in sql_lower for keyword in keywords)
            components_found[component_name] = found
        
        return components_found
    
    def calculate_component_accuracy(self, predictions: List[str], targets: List[str]) -> Dict[str, float]:
        """Calculate accuracy for each SQL component"""
        component_accuracies = {}
        
        for component_name in self.components.keys():
            correct = 0
            total = 0
            
            for pred, target in zip(predictions, targets):
                pred_components = self.extract_sql_components(pred)
                target_components = self.extract_sql_components(target)
                
                if component_name in pred_components and component_name in target_components:
                    if pred_components[component_name] == target_components[component_name]:
                        correct += 1
                    total += 1
            
            accuracy = (correct / total * 100) if total > 0 else 0.0
            component_accuracies[f"{component_name}_accuracy"] = accuracy
        
        return component_accuracies

class ModelEvaluator:
    """Evaluate multiple pre-trained models on SQL generation"""
    
    def __init__(self, data_path: str = "./spider", db_path: str = "./spider/database"):
        self.data_path = data_path
        self.db_path = db_path
        self.component_analyzer = SQLComponentAnalyzer()
        
        # Set device
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using Apple M3 GPU (MPS)")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")
    
    def load_spider_data(self, split: str = "dev", subset_ratio: float = 0.1) -> List[Dict]:
        """Load Spider dataset with subset sampling"""
        file_path = os.path.join(self.data_path, f"{split}.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Sample 10% of the data
        sample_size = int(len(data) * subset_ratio)
        data = random.sample(data, sample_size)
        print(f"Using {len(data)} samples ({subset_ratio*100:.0f}% of {split} set)")
        
        return data
    
    def load_tables(self) -> Dict:
        """Load table schemas"""
        tables_path = os.path.join(self.data_path, "tables.json")
        with open(tables_path, 'r', encoding='utf-8') as f:
            tables = json.load(f)
        
        tables_dict = {}
        for table in tables:
            tables_dict[table['db_id']] = table
        return tables_dict
    
    def create_input_text(self, question: str, db_id: str, tables_dict: Dict) -> str:
        """Create input text with schema information"""
        if db_id not in tables_dict:
            return f"Question: {question}"
        
        table_info = tables_dict[db_id]
        table_names = table_info.get('table_names_original', [])
        column_names = table_info.get('column_names', [])
        
        # Create schema description
        schema_parts = []
        schema_parts.append(f"Database: {db_id}")
        
        # Group columns by table
        table_columns = {}
        for i, (tab_id, col_name) in enumerate(column_names):
            if tab_id == -1:  # Skip special columns like *
                continue
            
            table_name = table_names[tab_id] if tab_id < len(table_names) else f"table_{tab_id}"
            
            if table_name not in table_columns:
                table_columns[table_name] = []
            table_columns[table_name].append(col_name)
        
        # Create table descriptions
        for table_name, columns in table_columns.items():
            table_desc = f"Table '{table_name}' with columns: {', '.join(columns)}"
            schema_parts.append(table_desc)
        
        schema_text = " | ".join(schema_parts)
        
        prompt = f"""Schema: {schema_text}

IMPORTANT: Only use the tables and columns listed above. Do not create or reference tables that don't exist.

Question: {question}

Generate SQL:"""
        
        return prompt
    
    def load_model(self, model_name: str):
        """Load a pre-trained model"""
        print(f"Loading model: {model_name}")
        
        if model_name.startswith("t5"):
            tokenizer = T5Tokenizer.from_pretrained(model_name)
            model = T5ForConditionalGeneration.from_pretrained(model_name)
        elif model_name.startswith("facebook/bart"):
            tokenizer = BartTokenizer.from_pretrained(model_name)
            model = BartForConditionalGeneration.from_pretrained(model_name)
        elif model_name.startswith("Salesforce/codet5"):
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = T5EncoderDecoderModel.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model.to(self.device)
        model.eval()
        
        return tokenizer, model
    
    def generate_sql(self, tokenizer, model, input_text: str) -> str:
        """Generate SQL using the model"""
        try:
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=256,
                    do_sample=False,
                    num_beams=3,
                    early_stopping=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=False
                )
            
            outputs = outputs.cpu()
            sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return sql
        
        except Exception as e:
            print(f"Error generating SQL: {e}")
            return "SELECT * FROM table"
    
    def evaluate_model(self, model_name: str, data: List[Dict], tables_dict: Dict) -> Dict:
        """Evaluate a single model"""
        print(f"\nEvaluating {model_name}...")
        
        tokenizer, model = self.load_model(model_name)
        
        predictions = []
        targets = []
        questions = []
        
        for item in tqdm(data, desc=f"Generating SQL for {model_name}"):
            # Create input text
            input_text = self.create_input_text(
                item['question'], 
                item['db_id'], 
                tables_dict
            )
            
            # Generate SQL
            pred_sql = self.generate_sql(tokenizer, model, input_text)
            
            predictions.append(pred_sql)
            targets.append(item['query'])
            questions.append(item['question'])
        
        # Calculate component accuracies
        component_accuracies = self.component_analyzer.calculate_component_accuracy(predictions, targets)
        
        # Calculate overall metrics
        exact_matches = sum(1 for p, t in zip(predictions, targets) 
                          if p.strip().lower() == t.strip().lower())
        exact_match_accuracy = (exact_matches / len(predictions)) * 100
        
        # Test execution accuracy
        execution_correct = 0
        for pred, target, db_id in zip(predictions, targets, [item['db_id'] for item in data]):
            try:
                db_file = os.path.join(self.db_path, f"{db_id}.sqlite")
                if os.path.exists(db_file):
                    conn = sqlite3.connect(db_file)
                    pred_result = pd.read_sql_query(pred, conn)
                    target_result = pd.read_sql_query(target, conn)
                    conn.close()
                    
                    if pred_result.equals(target_result):
                        execution_correct += 1
            except:
                pass
        
        execution_accuracy = (execution_correct / len(predictions)) * 100
        
        results = {
            'model_name': model_name,
            'exact_match_accuracy': exact_match_accuracy,
            'execution_accuracy': execution_accuracy,
            'num_samples': len(predictions),
            **component_accuracies
        }
        
        # Clean up
        del tokenizer, model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        return results
    
    def run_comparison(self):
        """Run comparison across all models"""
        print("🚀 Starting Model Comparison Analysis")
        print("=" * 60)
        
        # Load data
        print("Loading Spider dataset...")
        data = self.load_spider_data("dev", subset_ratio=0.1)
        tables_dict = self.load_tables()
        
        # Define models to evaluate
        models = [
            "t5-small",
            "t5-base", 
            "facebook/bart-base",
            "Salesforce/codet5-small",
            "Salesforce/codet5-base"
        ]
        
        results = []
        
        # Evaluate each model
        for model_name in models:
            try:
                result = self.evaluate_model(model_name, data, tables_dict)
                results.append(result)
                print(f"✅ Completed evaluation for {model_name}")
            except Exception as e:
                print(f"❌ Error evaluating {model_name}: {e}")
                continue
        
        # Create comparison table
        if results:
            df = pd.DataFrame(results)
            
            # Reorder columns for better display
            column_order = [
                'model_name', 'exact_match_accuracy', 'execution_accuracy',
                'select_accuracy', 'where_accuracy', 'group_by_accuracy',
                'order_by_accuracy', 'join_accuracy', 'aggregate_accuracy',
                'num_samples'
            ]
            
            df = df[column_order]
            
            # Save results
            os.makedirs('./outputs', exist_ok=True)
            output_file = './outputs/model_comparison_results.json'
            df.to_json(output_file, orient='records', indent=2)
            
            print("\n" + "=" * 60)
            print("📊 MODEL COMPARISON RESULTS")
            print("=" * 60)
            
            # Display results
            for _, row in df.iterrows():
                print(f"\n🔍 {row['model_name']}")
                print(f"   Exact Match Accuracy: {row['exact_match_accuracy']:.2f}%")
                print(f"   Execution Accuracy: {row['execution_accuracy']:.2f}%")
                print(f"   Select Clause Accuracy: {row['select_accuracy']:.2f}%")
                print(f"   Where Clause Accuracy: {row['where_accuracy']:.2f}%")
                print(f"   Group By Accuracy: {row['group_by_accuracy']:.2f}%")
                print(f"   Order By Accuracy: {row['order_by_accuracy']:.2f}%")
                print(f"   Join Accuracy: {row['join_accuracy']:.2f}%")
                print(f"   Aggregate Accuracy: {row['aggregate_accuracy']:.2f}%")
                print(f"   Samples Tested: {row['num_samples']}")
            
            print(f"\n📁 Results saved to: {output_file}")
            
            # Find best model for each component
            print("\n🏆 BEST PERFORMING MODELS BY COMPONENT:")
            print("-" * 40)
            
            components = ['exact_match_accuracy', 'execution_accuracy', 'select_accuracy', 
                        'where_accuracy', 'group_by_accuracy', 'order_by_accuracy', 
                        'join_accuracy', 'aggregate_accuracy']
            
            for component in components:
                best_idx = df[component].idxmax()
                best_model = df.loc[best_idx, 'model_name']
                best_score = df.loc[best_idx, component]
                print(f"{component.replace('_', ' ').title()}: {best_model} ({best_score:.2f}%)")
        
        else:
            print("❌ No results generated. Check for errors above.")

def main():
    """Main function"""
    evaluator = ModelEvaluator()
    evaluator.run_comparison()

if __name__ == "__main__":
    main() 