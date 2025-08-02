#!/usr/bin/env python3
"""
Test script to evaluate SQL model performance
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from predict import SQLPredictor
import pandas as pd
import sqlite3
import tempfile
import os

def create_test_database():
    """Create a test database with sample data"""
    # Create sample data
    employees_data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'salary': [50000, 60000, 45000, 70000, 55000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'IT']
    }
    
    departments_data = {
        'id': [1, 2, 3],
        'name': ['IT', 'HR', 'Finance'],
        'budget': [100000, 80000, 120000]
    }
    
    # Create temporary database
    db_path = tempfile.mktemp(suffix='.sqlite')
    
    # Convert to SQLite
    conn = sqlite3.connect(db_path)
    pd.DataFrame(employees_data).to_sql('employees', conn, index=False)
    pd.DataFrame(departments_data).to_sql('departments', conn, index=False)
    conn.close()
    
    return db_path

def extract_schema(db_path):
    """Extract schema information"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema_info = []
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        
        table_info = f"Table '{table_name}':\n"
        for col in columns:
            table_info += f"  - {col[1]} ({col[2]})\n"
        schema_info.append(table_info)
    
    conn.close()
    return "\n".join(schema_info)

def test_models():
    """Test all models with sample queries"""
    print("🧪 Testing SQL Models Performance...")
    
    # Create test database
    db_path = create_test_database()
    schema_info = extract_schema(db_path)
    
    # Sample queries to test
    test_queries = [
        "What is the average salary?",
        "Show all employees in the IT department",
        "Find the highest paid employee",
        "Count employees by department",
        "What is the total budget for all departments?"
    ]
    
    # Test each model
    models = {
        'model-m3': 'models/model-m3',
        'model-m3-v2': 'models/model-m3-v2', 
        'model-m3-v4-anti-hallucination': 'models/model-m3-v4-anti-hallucination'
    }
    
    results = {}
    
    for model_name, model_path in models.items():
        print(f"\n📊 Testing {model_name}...")
        try:
            predictor = SQLPredictor(model_path)
            
            model_results = []
            for query in test_queries:
                try:
                    sql = predictor.predict_sql(query, schema_info, "test_db")
                    model_results.append({
                        'query': query,
                        'sql': sql,
                        'status': 'success'
                    })
                    print(f"  ✅ Query: {query}")
                    print(f"     SQL: {sql}")
                except Exception as e:
                    model_results.append({
                        'query': query,
                        'sql': str(e),
                        'status': 'error'
                    })
                    print(f"  ❌ Query: {query} - Error: {e}")
            
            results[model_name] = model_results
            
        except Exception as e:
            print(f"  ❌ Failed to load {model_name}: {e}")
            results[model_name] = []
    
    # Clean up
    os.unlink(db_path)
    
    # Summary
    print("\n" + "="*60)
    print("📈 MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    for model_name, model_results in results.items():
        if model_results:
            success_count = sum(1 for r in model_results if r['status'] == 'success')
            total_count = len(model_results)
            success_rate = (success_count / total_count) * 100
            
            print(f"\n{model_name}:")
            print(f"  Success Rate: {success_rate:.1f}% ({success_count}/{total_count})")
            
            if success_count > 0:
                print("  Sample successful queries:")
                for result in model_results[:2]:  # Show first 2 successful
                    if result['status'] == 'success':
                        print(f"    Q: {result['query']}")
                        print(f"    A: {result['sql']}")
        else:
            print(f"\n{model_name}: Failed to load")
    
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS:")
    print("="*60)
    
    # Analyze results and provide recommendations
    working_models = [name for name, results in results.items() if results and any(r['status'] == 'success' for r in results)]
    
    if working_models:
        print("✅ Your models are working! Consider fine-tuning if:")
        print("  - Success rate is below 80%")
        print("  - SQL queries are not accurate for your specific data")
        print("  - You need support for complex queries")
        print("  - You have domain-specific terminology")
    else:
        print("❌ Models need attention. Consider:")
        print("  - Checking model files are complete")
        print("  - Re-downloading or re-training models")
        print("  - Using different pre-trained models")

if __name__ == "__main__":
    test_models() 