"""
Predict Module

Handles SQL generation from natural language questions using AI models.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SQLPredictor:
    """Handles SQL generation from natural language questions."""
    
    def __init__(self, model_path: str):
        """
        Initialize SQL predictor with a pre-trained model.
        
        Args:
            model_path: Path to the pre-trained model directory
        """
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load tokenizer - use slow tokenizer to avoid SentencePiece conversion issues
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info(f"Successfully loaded model from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {str(e)}")
            raise
    
    def predict_sql(self, question: str, db_info: str, db_id: str, 
                   method: str = "beam", verbose: bool = False) -> str:
        """
        Generate SQL from natural language question.
        
        Args:
            question: Natural language question
            db_info: Database schema information
            db_id: Database identifier
            method: Generation method ("beam", "greedy", etc.)
            verbose: Whether to print verbose output
            
        Returns:
            Generated SQL query
        """
        try:
            # Create input text with schema context
            input_text = self._create_input_text(question, db_info, db_id)
            
            if verbose:
                logger.info(f"Input text: {input_text}")
            
            # Tokenize input
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate SQL
            if method == "beam":
                outputs = self.model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            elif method == "greedy":
                outputs = self.model.generate(
                    **inputs,
                    max_length=256,
                    do_sample=False
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode output
            generated_sql = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if verbose:
                logger.info(f"Generated SQL: {generated_sql}")
            
            return generated_sql
            
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            return "SELECT * FROM table"
    
    def _create_input_text(self, question: str, db_info: str, db_id: str) -> str:
        """
        Create input text with schema context.
        
        Args:
            question: Natural language question
            db_info: Database schema information
            db_id: Database identifier
            
        Returns:
            Formatted input text for the model
        """
        return f"""
Database Schema:
{db_info}

Available tables: {db_id}

Question: {question}

IMPORTANT: Only use the tables and columns listed above. Do not create or reference tables that don't exist.
"""
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_path': self.model_path,
            'model_type': type(self.model).__name__,
            'tokenizer_type': type(self.tokenizer).__name__,
            'loaded': self.model is not None and self.tokenizer is not None
        } 