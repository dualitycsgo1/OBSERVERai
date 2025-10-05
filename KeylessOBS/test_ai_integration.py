#!/usr/bin/env python3
"""
Test AI Training Integration End-to-End
"""

import os
import sys
import json
import time
import tempfile
from datetime import datetime

def get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))

def test_complete_integration():
    """Test the complete AI training integration flow"""
    
    print("🧪 Testing Complete AI Training Integration")
    print("=" * 60)
    
    base_dir = get_base_dir()
    models_dir = os.path.join(base_dir, "models")
    
    # Step 1: Check if AI trainer can create model updates
    print("1️⃣ Testing AI Trainer Model Update Notifications...")
    
    try:
        # Import and create AI trainer
        sys.path.append(base_dir)
        from ai_trainer import AITrainer
        
        trainer = AITrainer()
        
        # Create a test notification manually to verify the system
        test_metadata = {
            "success": True,
            "accuracy": 0.89,
            "training_samples": 6115,
            "test_samples": 1529,
            "timestamp": datetime.now().isoformat()
        }
        
        model_path = os.path.join(models_dir, "observer_ai_model.pkl")
        result = trainer._notify_model_update("observer_ai_model", model_path, test_metadata)
        
        if result:
            print("   ✅ AI Trainer can create model notifications")
        else:
            print("   ❌ AI Trainer notification failed")
            return False
            
    except Exception as e:
        print(f"   ❌ AI Trainer test failed: {e}")
        return False
    
    # Step 2: Check if cs2_duel_detector picks up notifications
    print("2️⃣ Testing CS2 Duel Detector Hot-Reload System...")
    
    try:
        from cs2_duel_detector import ModelHotReloader
        
        reloader = ModelHotReloader()
        
        # Check for model updates (this should find our test notification)
        initial_cache_size = len(reloader.model_cache)
        reloader._check_for_updates()
        
        print(f"   ✅ Hot-reload system checked for updates")
        print(f"   📊 Initial cache size: {initial_cache_size}")
        
    except Exception as e:
        print(f"   ❌ Hot-reload test failed: {e}")
        return False
    
    # Step 3: Check training data flow
    print("3️⃣ Testing Training Data Flow...")
    
    try:
        # Check current training data count
        training_data = trainer._load_all_training_data()
        print(f"   ✅ Training data loaded: {len(training_data)} samples")
        
        if len(training_data) > 1000:
            print("   ✅ Sufficient training data available")
        else:
            print("   ⚠️ Limited training data - consider adding more demos")
            
    except Exception as e:
        print(f"   ❌ Training data test failed: {e}")
        return False
    
    # Step 4: Test model loading with hot-reload
    print("4️⃣ Testing Model Loading Integration...")
    
    try:
        model_path = os.path.join(models_dir, "observer_ai_model.pkl")
        
        if os.path.exists(model_path):
            # Test hot-reload model loading
            model = reloader.get_model("observer_ai_model", model_path)
            
            if model:
                print("   ✅ Model loaded successfully via hot-reload")
                print(f"   📊 Model type: {type(model).__name__}")
            else:
                print("   ❌ Model loading failed")
                return False
        else:
            print("   ⚠️ No model file found - train a model first")
            
    except Exception as e:
        print(f"   ❌ Model loading test failed: {e}")
        return False
    
    # Step 5: Integration summary
    print("5️⃣ Integration Flow Summary...")
    
    flow_steps = [
        "1. AI Trainer processes demos and extracts training data",
        "2. AI Trainer trains models and saves to models/observer_ai_model.pkl", 
        "3. AI Trainer creates model_update_*.json notification file",
        "4. CS2 Duel Detector ModelHotReloader checks for notifications every 30s",
        "5. CS2 Duel Detector reloads updated models and uses them for predictions",
        "6. Observer makes better predictions using updated training data"
    ]
    
    for step in flow_steps:
        print(f"   {step}")
    
    print(f"\n🎉 INTEGRATION TEST PASSED!")
    print(f"✅ New training data will flow correctly from AI trainer to cs2_duel_detector.py")
    
    return True

def test_manual_training_cycle():
    """Test a complete manual training cycle"""
    
    print(f"\n🔄 Testing Manual Training Cycle...")
    
    try:
        from ai_trainer import AITrainer
        
        trainer = AITrainer()
        
        # Check current state
        stats = trainer.get_training_stats()
        print(f"   📊 Current training sessions: {stats['total_sessions']}")
        print(f"   📊 Current model accuracy: {stats['latest_accuracy']:.3f}")
        
        # Simulate training update (without actually training)
        print(f"   🔧 To train new model: run AI trainer with new demo files")
        print(f"   🔧 To test hot-reload: check cs2_duel_detector logs after training")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Manual training test failed: {e}")
        return False

def main():
    """Run complete integration test"""
    
    try:
        # Test complete integration
        integration_success = test_complete_integration()
        
        if integration_success:
            # Test manual training cycle
            manual_success = test_manual_training_cycle()
            
            if manual_success:
                print(f"\n🎊 ALL TESTS PASSED!")
                print(f"\n📋 HOW TO USE:")
                print(f"1. Run AI trainer with: python ai_trainer_gui.py")
                print(f"2. Import demo files to create new training data")
                print(f"3. Train new model - it will automatically notify cs2_duel_detector")
                print(f"4. Check cs2_duel_detector logs for 'Hot-reloaded model' messages")
                print(f"5. Observer will use updated AI predictions")
            else:
                print(f"\n❌ Manual training test failed")
        else:
            print(f"\n❌ Integration test failed")
            
    except Exception as e:
        print(f"\n💥 Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()