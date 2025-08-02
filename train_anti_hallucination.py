#!/usr/bin/env python3
"""
Anti-Hallucination Text-to-SQL Training Script for MacBook Air M3
Optimized for 16GB RAM with focus on preventing table/column name hallucination
"""

import os
import json
import torch
import gc
from text2sql_pipeline import TextToSQLTrainer

def main():
    """Main training function optimized for MacBook Air M3"""
    
    print("🚀 Starting Anti-Hallucination Text-to-SQL Training")
    print("=" * 60)
    print("💻 Optimized for MacBook Air M3 (16GB RAM)")
    print("🎯 Focus: Prevent table/column name hallucination")
    print("=" * 60)
    
    # Check available memory
    if torch.backends.mps.is_available():
        print("✅ Apple M3 GPU (MPS) detected")
        device = "mps"
    else:
        print("⚠️  Using CPU (MPS not available)")
        device = "cpu"
    
    # Memory optimization settings
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Initialize trainer with anti-hallucination settings
    trainer = TextToSQLTrainer(
        model_name="google/flan-t5-small",
        data_path="./spider",
        db_path="./spider/database",
        subset_ratio=0.4  # Use 40% of data for better quality
    )
    
    print(f"\n📊 Model Parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")
    print(f"🔧 Device: {trainer.device}")
    print(f"📈 Using {trainer.data_processor.subset_ratio*100:.0f}% of Spider dataset")
    
    # Training configuration optimized for anti-hallucination
    training_config = {
        'output_dir': './text2sql_model_m3_v4_anti_hallucination',
        'num_epochs': 6,  # More epochs for better learning
        'batch_size': 1,  # Small batch size for memory
        'learning_rate': 3e-5,  # Lower learning rate for stability
    }
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Output Directory: {training_config['output_dir']}")
    print(f"   Epochs: {training_config['num_epochs']}")
    print(f"   Batch Size: {training_config['batch_size']}")
    print(f"   Learning Rate: {training_config['learning_rate']}")
    print(f"   Effective Batch Size: {training_config['batch_size'] * 8} (with gradient accumulation)")
    
    # Memory monitoring
    def print_memory_usage():
        if device == "mps":
            allocated = torch.mps.current_allocated_memory() / 1024**3
            reserved = torch.mps.driver_allocated_memory() / 1024**3
            print(f"   GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    print_memory_usage()
    
    try:
        # Start training
        print(f"\n🔥 Starting Training...")
        results = trainer.train_model(**training_config)
        
        print(f"\n✅ Training Completed Successfully!")
        print(f"📁 Model saved to: {training_config['output_dir']}")
        
        # Print results
        print(f"\n📊 Training Results:")
        print(f"   Best Method: {results['best_method'].upper()}")
        print(f"   Execution Accuracy: {results['best_results']['execution_accuracy']:.4f}")
        print(f"   Exact Match Accuracy: {results['best_results']['exact_match_accuracy']:.4f}")
        print(f"   Valid SQL Rate: {results['best_results']['valid_sql_rate']:.4f}")
        
        # Save training summary
        summary = {
            'training_config': training_config,
            'results': results,
            'device': device,
            'model_params': sum(p.numel() for p in trainer.model.parameters()),
            'dataset_ratio': trainer.data_processor.subset_ratio
        }
        
        summary_path = os.path.join(training_config['output_dir'], 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📄 Training summary saved to: {summary_path}")
        
        # Final memory check
        print_memory_usage()
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("💡 Try reducing batch_size or subset_ratio if out of memory")
        raise
    
    finally:
        # Clean up memory
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    
    print(f"\n🎉 Anti-Hallucination Training Complete!")
    print(f"💡 Next steps:")
    print(f"   1. Test the model with: python3 test_model_with_spider_data.py")
    print(f"   2. Use in Streamlit app: streamlit run app.py")
    print(f"   3. Check for hallucination reduction")

if __name__ == "__main__":
    main() 