#!/usr/bin/env python3
"""
Training Data Generator for Text-to-SQL Fine-Tuning
Generates question-SQL pairs from CSV files
"""

import pandas as pd
import sqlite3
import tempfile
import os
import json
from typing import List, Dict, Tuple
import random

class TrainingDataGenerator:
    def __init__(self):
        self.training_data = []
        
    def create_database_from_csvs(self) -> Tuple[str, str]:
        """Create SQLite database from CSV files and return schema info"""
        # Create temporary database
        db_path = tempfile.mktemp(suffix='.sqlite')
        
        # Read CSV files
        actors_df = pd.read_csv('actor.csv')
        musicals_df = pd.read_csv('musical.csv')
        
        # Convert to SQLite
        conn = sqlite3.connect(db_path)
        actors_df.to_sql('actors', conn, index=False)
        musicals_df.to_sql('musicals', conn, index=False)
        
        # Extract schema information
        schema_info = self._extract_schema(conn)
        
        conn.close()
        return db_path, schema_info
    
    def _extract_schema(self, conn: sqlite3.Connection) -> str:
        """Extract schema information from database"""
        cursor = conn.cursor()
        
        schema_parts = []
        
        # Get table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            table_info = f"Table '{table_name}':\n"
            for col in columns:
                table_info += f"  - {col[1]} ({col[2]})\n"
            schema_parts.append(table_info)
        
        return "\n".join(schema_parts)
    
    def generate_basic_queries(self) -> List[Dict]:
        """Generate basic question-SQL pairs"""
        queries = [
            # Actor queries
            {
                "question": "Show all actors",
                "sql": "SELECT * FROM actors",
                "category": "basic_select"
            },
            {
                "question": "What are all the actor names?",
                "sql": "SELECT Name FROM actors",
                "category": "basic_select"
            },
            {
                "question": "How many actors are there?",
                "sql": "SELECT COUNT(*) FROM actors",
                "category": "aggregation"
            },
            {
                "question": "Find actors older than 20",
                "sql": "SELECT * FROM actors WHERE age > 20",
                "category": "filtering"
            },
            {
                "question": "Show actors with age 20",
                "sql": "SELECT * FROM actors WHERE age = 20",
                "category": "filtering"
            },
            {
                "question": "Find the oldest actor",
                "sql": "SELECT * FROM actors ORDER BY age DESC LIMIT 1",
                "category": "ordering"
            },
            {
                "question": "Show actors sorted by age",
                "sql": "SELECT * FROM actors ORDER BY age",
                "category": "ordering"
            },
            
            # Musical queries
            {
                "question": "Show all musicals",
                "sql": "SELECT * FROM musicals",
                "category": "basic_select"
            },
            {
                "question": "What are all the musical names?",
                "sql": "SELECT Name FROM musicals",
                "category": "basic_select"
            },
            {
                "question": "How many musicals are there?",
                "sql": "SELECT COUNT(*) FROM musicals",
                "category": "aggregation"
            },
            {
                "question": "Find musicals from 1986",
                "sql": "SELECT * FROM musicals WHERE Year = 1986",
                "category": "filtering"
            },
            {
                "question": "Show musicals that won awards",
                "sql": "SELECT * FROM musicals WHERE Result = 'Won'",
                "category": "filtering"
            },
            {
                "question": "Find Tony Award winners",
                "sql": "SELECT * FROM musicals WHERE Award = 'Tony Award' AND Result = 'Won'",
                "category": "filtering"
            },
            
            # JOIN queries
            {
                "question": "Show actors and their musicals",
                "sql": "SELECT actors.Name, musicals.Name FROM actors JOIN musicals ON actors.Musical_ID = musicals.Musical_ID",
                "category": "join"
            },
            {
                "question": "Find actors in The Phantom of the Opera",
                "sql": "SELECT actors.Name FROM actors JOIN musicals ON actors.Musical_ID = musicals.Musical_ID WHERE musicals.Name = 'The Phantom of the Opera'",
                "category": "join"
            },
            {
                "question": "Show musicals with their actors",
                "sql": "SELECT musicals.Name, actors.Name FROM musicals JOIN actors ON musicals.Musical_ID = actors.Musical_ID",
                "category": "join"
            },
            
            # Complex queries
            {
                "question": "How many actors are in each musical?",
                "sql": "SELECT musicals.Name, COUNT(actors.Actor_ID) FROM musicals JOIN actors ON musicals.Musical_ID = actors.Musical_ID GROUP BY musicals.Musical_ID",
                "category": "group_by"
            },
            {
                "question": "Find musicals with more than 1 actor",
                "sql": "SELECT musicals.Name FROM musicals JOIN actors ON musicals.Musical_ID = actors.Musical_ID GROUP BY musicals.Musical_ID HAVING COUNT(actors.Actor_ID) > 1",
                "category": "group_by"
            },
            {
                "question": "What is the average age of actors?",
                "sql": "SELECT AVG(age) FROM actors",
                "category": "aggregation"
            },
            {
                "question": "Show actors by character name",
                "sql": "SELECT Character, Name FROM actors ORDER BY Character",
                "category": "ordering"
            }
        ]
        
        return queries
    
    def generate_variations(self, base_queries: List[Dict]) -> List[Dict]:
        """Generate variations of base queries"""
        variations = []
        
        # Question variations
        question_variations = {
            "Show all actors": [
                "List all actors",
                "Display all actors",
                "Get all actors",
                "Show me all actors"
            ],
            "How many actors are there?": [
                "Count the actors",
                "What is the total number of actors?",
                "How many actors do we have?",
                "Count all actors"
            ],
            "Find actors older than 20": [
                "Show actors over 20 years old",
                "List actors with age greater than 20",
                "Display actors older than 20",
                "Get actors above age 20"
            ],
            "Show all musicals": [
                "List all musicals",
                "Display all musicals",
                "Get all musicals",
                "Show me all musicals"
            ],
            "How many musicals are there?": [
                "Count the musicals",
                "What is the total number of musicals?",
                "How many musicals do we have?",
                "Count all musicals"
            ]
        }
        
        for query in base_queries:
            # Add original query
            variations.append(query)
            
            # Add variations if available
            if query["question"] in question_variations:
                for variation in question_variations[query["question"]]:
                    variations.append({
                        "question": variation,
                        "sql": query["sql"],
                        "category": query["category"]
                    })
        
        return variations
    
    def create_training_data(self) -> List[Dict]:
        """Create complete training dataset"""
        print("🎯 Generating training data from CSV files...")
        
        # Create database and get schema
        db_path, schema_info = self.create_database_from_csvs()
        
        # Generate base queries
        base_queries = self.generate_basic_queries()
        
        # Generate variations
        all_queries = self.generate_variations(base_queries)
        
        # Create training examples
        training_data = []
        
        for query in all_queries:
            training_example = {
                "question": query["question"],
                "sql": query["sql"],
                "schema": schema_info,
                "category": query["category"]
            }
            training_data.append(training_example)
        
        # Clean up
        os.unlink(db_path)
        
        print(f"✅ Generated {len(training_data)} training examples")
        print(f"📊 Categories: {set(q['category'] for q in training_data)}")
        
        return training_data
    
    def save_training_data(self, data: List[Dict], filename: str = "training_data.json"):
        """Save training data to JSON file"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved training data to {filename}")
    
    def validate_sql_queries(self, data: List[Dict]) -> List[Dict]:
        """Validate SQL queries by executing them"""
        print("🔍 Validating SQL queries...")
        
        # Create temporary database
        db_path = tempfile.mktemp(suffix='.sqlite')
        conn = sqlite3.connect(db_path)
        
        # Load CSV data
        actors_df = pd.read_csv('actor.csv')
        musicals_df = pd.read_csv('musical.csv')
        actors_df.to_sql('actors', conn, index=False)
        musicals_df.to_sql('musicals', conn, index=False)
        
        valid_data = []
        cursor = conn.cursor()
        
        for i, example in enumerate(data):
            try:
                cursor.execute(example["sql"])
                result = cursor.fetchall()
                valid_data.append(example)
                print(f"  ✅ Query {i+1}: {example['question']}")
            except Exception as e:
                print(f"  ❌ Query {i+1}: {example['question']} - Error: {e}")
        
        conn.close()
        os.unlink(db_path)
        
        print(f"✅ Validated {len(valid_data)}/{len(data)} queries")
        return valid_data

def main():
    """Main function to generate training data"""
    generator = TrainingDataGenerator()
    
    # Generate training data
    training_data = generator.create_training_data()
    
    # Validate SQL queries
    valid_data = generator.validate_sql_queries(training_data)
    
    # Save training data
    generator.save_training_data(valid_data)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TRAINING DATA SUMMARY")
    print("="*60)
    
    categories = {}
    for example in valid_data:
        cat = example["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    for category, count in categories.items():
        print(f"  {category}: {count} examples")
    
    print(f"\nTotal training examples: {len(valid_data)}")
    print("="*60)

if __name__ == "__main__":
    main() 