#!/usr/bin/env python3
"""
Test script for the sentence transformer model integration.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from sentence_model import SentenceModelPredictor

def test_sentence_model():
    """Test the sentence transformer model."""
    print("🧪 Testing Sentence Transformer Model...")
    
    try:
        # Initialize the model
        print("📥 Loading all-MiniLM-L6-v2 model...")
        model = SentenceModelPredictor()
        
        # Test basic functionality
        print("✅ Model loaded successfully!")
        
        # Test similarity calculation
        text1 = "What is the average salary?"
        text2 = "Calculate the mean salary"
        text3 = "Show me all users"
        
        similarity = model.get_similarity(text1, text2)
        print(f"📊 Similarity between '{text1}' and '{text2}': {similarity:.3f}")
        
        # Test finding most similar
        query = "What is the average salary?"
        candidates = [
            "Calculate the mean salary",
            "Show me all users", 
            "Find the highest paid employee",
            "Count total employees"
        ]
        
        similar_texts = model.find_most_similar(query, candidates, top_k=3)
        print(f"🔍 Most similar to '{query}':")
        for text, score in similar_texts:
            print(f"   - '{text}' (score: {score:.3f})")
        
        # Test SQL generation enhancement
        question = "What is the average salary by department?"
        schema_info = "Table 'employees' with columns: id (INTEGER), name (TEXT), salary (REAL), department (TEXT)"
        db_id = "test_db"
        
        enhanced_prompt = model.enhance_sql_generation(question, schema_info, db_id)
        print(f"🚀 Enhanced prompt:\n{enhanced_prompt}")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_sentence_model()
    sys.exit(0 if success else 1) 