#!/usr/bin/env python3
"""
Training Data Consolidation Script
Rebuilds the complete training dataset from all available sources
"""

import json
import os
import logging
from typing import List, Dict, Any

def get_base_dir():
    """Get base directory"""
    return os.path.dirname(os.path.abspath(__file__))

def consolidate_all_training_data():
    """Consolidate all training data from all sources"""
    
    base_dir = get_base_dir()
    training_data_dir = os.path.join(base_dir, "training_data")
    
    print("🔄 Consolidating all training data sources...")
    
    all_samples = []
    source_counts = {}
    
    # 1. Load existing complete_kill_analysis.json (if exists)
    complete_analysis_path = os.path.join(training_data_dir, "complete_kill_analysis.json")
    if os.path.exists(complete_analysis_path):
        try:
            with open(complete_analysis_path, 'r') as f:
                complete_data = json.load(f)
                if isinstance(complete_data, list):
                    all_samples.extend(complete_data)
                    source_counts['complete_kill_analysis.json'] = len(complete_data)
                    print(f"✅ Loaded {len(complete_data)} samples from complete_kill_analysis.json")
        except Exception as e:
            print(f"❌ Error loading complete_kill_analysis.json: {e}")
    
    # 2. Load all training batch files
    batch_count = 0
    for file in os.listdir(training_data_dir):
        if file.startswith("training_batch_") and file.endswith(".json"):
            try:
                filepath = os.path.join(training_data_dir, file)
                with open(filepath, 'r') as f:
                    batch_data = json.load(f)
                    if isinstance(batch_data, list):
                        all_samples.extend(batch_data)
                        source_counts[file] = len(batch_data)
                        batch_count += len(batch_data)
                        print(f"✅ Loaded {len(batch_data)} samples from {file}")
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
    
    print(f"✅ Loaded {batch_count} samples from {len(source_counts)-1} training batch files")
    
    # 3. Load large positional dataset
    positional_dataset_path = os.path.join(base_dir, "complete_positional_dataset_all_85_demos.json")
    if os.path.exists(positional_dataset_path):
        try:
            with open(positional_dataset_path, 'r') as f:
                positional_data = json.load(f)
                
            # Extract training samples from structured format
            if isinstance(positional_data, dict) and 'training_samples' in positional_data:
                samples = positional_data['training_samples']
                all_samples.extend(samples)
                source_counts['complete_positional_dataset_all_85_demos.json'] = len(samples)
                print(f"✅ Loaded {len(samples)} samples from large positional dataset")
            elif isinstance(positional_data, list):
                all_samples.extend(positional_data)
                source_counts['complete_positional_dataset_all_85_demos.json'] = len(positional_data)
                print(f"✅ Loaded {len(positional_data)} samples from large positional dataset")
        except Exception as e:
            print(f"❌ Error loading positional dataset: {e}")
    
    # 4. Remove duplicates based on key features (if possible)
    print("🔍 Removing duplicate samples...")
    unique_samples = []
    seen_signatures = set()
    
    for sample in all_samples:
        # Create a signature based on key features to detect duplicates
        try:
            if isinstance(sample, dict):
                # Use killer position, victim position, and timestamp as signature
                signature = (
                    sample.get('killer_x', 0), 
                    sample.get('killer_y', 0),
                    sample.get('victim_x', 0), 
                    sample.get('victim_y', 0),
                    sample.get('timestamp', 0)
                )
                
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    unique_samples.append(sample)
        except:
            # If signature creation fails, keep the sample
            unique_samples.append(sample)
    
    duplicates_removed = len(all_samples) - len(unique_samples)
    print(f"🗑️ Removed {duplicates_removed} duplicate samples")
    
    # 5. Save consolidated dataset
    print("💾 Saving consolidated training dataset...")
    
    # Backup existing file first
    if os.path.exists(complete_analysis_path):
        backup_path = complete_analysis_path + ".backup"
        os.rename(complete_analysis_path, backup_path)
        print(f"📦 Backed up existing file to {backup_path}")
    
    # Save new consolidated dataset
    with open(complete_analysis_path, 'w') as f:
        json.dump(unique_samples, f, indent=2)
    
    print(f"✅ Saved {len(unique_samples)} unique training samples to complete_kill_analysis.json")
    
    # 6. Print summary
    print("\n📊 CONSOLIDATION SUMMARY")
    print("=" * 50)
    for source, count in source_counts.items():
        print(f"  {source}: {count} samples")
    print(f"  Total raw samples: {len(all_samples)}")
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Final unique samples: {len(unique_samples)}")
    print("=" * 50)
    
    return len(unique_samples)

def verify_ai_trainer_can_load():
    """Verify that the AI trainer can now load all the data"""
    print("\n🧪 Testing AI trainer data loading...")
    
    try:
        # Import and test the AI trainer
        import sys
        sys.path.append(get_base_dir())
        from ai_trainer import AITrainer
        
        trainer = AITrainer()
        training_data = trainer._load_all_training_data()
        
        print(f"✅ AI Trainer successfully loaded {len(training_data)} samples")
        return len(training_data)
        
    except Exception as e:
        print(f"❌ AI Trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return 0

def main():
    """Main consolidation process"""
    print("🚀 Training Data Consolidation Script")
    print("=" * 50)
    
    try:
        # Consolidate all data
        total_samples = consolidate_all_training_data()
        
        if total_samples > 0:
            # Test loading
            loaded_samples = verify_ai_trainer_can_load()
            
            if loaded_samples > 0:
                print(f"\n🎉 SUCCESS! Training data consolidated successfully")
                print(f"📈 AI Trainer now has access to {loaded_samples} training samples")
                print("💡 The AI trainer should now show the correct number of samples")
            else:
                print(f"\n⚠️ Data consolidated but AI trainer couldn't load it")
        else:
            print(f"\n❌ No training data found to consolidate")
            
    except Exception as e:
        print(f"\n💥 Consolidation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()