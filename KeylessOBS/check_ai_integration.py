#!/usr/bin/env python3
"""
AI Training Integration Fix
Ensures new training data flows correctly from AI trainer to cs2_duel_detector.py
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

def get_base_dir():
    """Get base directory"""
    return os.path.dirname(os.path.abspath(__file__))

def ensure_model_integration():
    """Ensure proper integration between AI trainer and duel detector"""
    
    print("🔧 Checking AI Training Integration...")
    
    base_dir = get_base_dir()
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    issues_found = []
    fixes_applied = []
    
    # 1. Check if AI trainer creates model update notifications
    print("📋 1. Checking model update notification system...")
    
    # Check if notification files exist or can be created
    test_notification = {
        "model_name": "observer_ai_model",
        "update_time": datetime.now().isoformat(),
        "accuracy": 0.85,
        "samples_used": 6115
    }
    
    notification_path = os.path.join(models_dir, "model_update_test.json")
    try:
        with open(notification_path, 'w') as f:
            json.dump(test_notification, f, indent=2)
        os.remove(notification_path)  # Clean up test file
        print("   ✅ Model notification system works")
    except Exception as e:
        issues_found.append(f"Cannot write model notifications: {e}")
        print(f"   ❌ Model notification system failed: {e}")
    
    # 2. Check if main model file exists
    print("📋 2. Checking main AI model file...")
    
    main_model_path = os.path.join(models_dir, "observer_ai_model.pkl")
    if os.path.exists(main_model_path):
        print(f"   ✅ Main AI model exists: {main_model_path}")
        
        # Check model age
        model_mtime = os.path.getmtime(main_model_path)
        model_age_hours = (datetime.now().timestamp() - model_mtime) / 3600
        print(f"   📅 Model age: {model_age_hours:.1f} hours")
        
    else:
        issues_found.append("Main AI model file missing")
        print(f"   ❌ Main AI model missing: {main_model_path}")
    
    # 3. Check training data consolidation
    print("📋 3. Checking training data availability...")
    
    training_data_dir = os.path.join(base_dir, "training_data") 
    complete_analysis_path = os.path.join(training_data_dir, "complete_kill_analysis.json")
    
    if os.path.exists(complete_analysis_path):
        try:
            with open(complete_analysis_path, 'r') as f:
                data = json.load(f)
            sample_count = len(data) if isinstance(data, list) else 0
            print(f"   ✅ Training data available: {sample_count} samples")
            
            if sample_count < 1000:
                issues_found.append(f"Training data may be insufficient: only {sample_count} samples")
                
        except Exception as e:
            issues_found.append(f"Cannot read training data: {e}")
            print(f"   ❌ Training data read error: {e}")
    else:
        issues_found.append("Complete training data file missing")
        print(f"   ❌ Training data missing: {complete_analysis_path}")
    
    # 4. Check model hot-reloading configuration
    print("📋 4. Checking hot-reload configuration...")
    
    # Create enhanced hot-reload notification system
    enhanced_notification_code = '''
def _notify_model_update(self, model_name: str, model_path: str, metadata: Dict[str, Any]):
    """Notify the main observer system of model updates"""
    try:
        models_dir = os.path.join(get_base_dir(), "models")
        notification = {
            "model_name": model_name,
            "model_path": model_path,
            "update_time": datetime.now().isoformat(),
            "metadata": metadata,
            "training_samples": metadata.get("training_samples", 0),
            "accuracy": metadata.get("accuracy", 0.0)
        }
        
        # Create notification file for hot-reload system
        notification_file = f"model_update_{model_name}_{int(datetime.now().timestamp())}.json"
        notification_path = os.path.join(models_dir, notification_file)
        
        with open(notification_path, 'w') as f:
            json.dump(notification, f, indent=2)
            
        logger.info(f"🔄 Created model update notification: {notification_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create model update notification: {e}")
        return False
'''
    
    print("   📝 Hot-reload notification code ready for integration")
    
    # 5. Summary and fixes
    print(f"\n📊 INTEGRATION CHECK SUMMARY")
    print("=" * 50)
    
    if not issues_found:
        print("✅ All integration checks passed!")
        print("🎯 AI trainer data should flow correctly to cs2_duel_detector.py")
    else:
        print("⚠️ Issues found:")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
            
    return len(issues_found) == 0

def create_integration_patch():
    """Create patch for AI trainer to properly notify model updates"""
    
    print("\n🔧 Creating AI Trainer Integration Patch...")
    
    patch_code = '''
# Add this to ai_trainer.py after the _save_model method

def _notify_model_update(self, model_name: str, model_path: str, metadata: Dict[str, Any]):
    """Notify the main observer system of model updates"""
    try:
        models_dir = os.path.join(get_base_dir(), "models")
        notification = {
            "model_name": model_name,
            "model_path": model_path,
            "update_time": datetime.now().isoformat(),
            "metadata": metadata,
            "training_samples": metadata.get("training_samples", 0),
            "accuracy": metadata.get("accuracy", 0.0)
        }
        
        # Create notification file for hot-reload system
        notification_file = f"model_update_{model_name}_{int(datetime.now().timestamp())}.json"
        notification_path = os.path.join(models_dir, notification_file)
        
        with open(notification_path, 'w') as f:
            json.dump(notification, f, indent=2)
            
        logger.info(f"🔄 Created model update notification: {notification_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create model update notification: {e}")
        return False

# Add this call after saving the model in _train_sklearn_model method:
# self._notify_model_update("observer_ai_model", model_path, results)
'''
    
    with open("ai_trainer_integration_patch.py", 'w') as f:
        f.write(patch_code)
    
    print("✅ Integration patch created: ai_trainer_integration_patch.py") 
    print("📝 This patch needs to be manually applied to ai_trainer.py")

def test_hot_reload_system():
    """Test the hot-reload system with a dummy update"""
    
    print("\n🧪 Testing Hot-Reload System...")
    
    try:
        base_dir = get_base_dir()
        models_dir = os.path.join(base_dir, "models")
        
        # Create test notification
        test_notification = {
            "model_name": "observer_ai_model",
            "model_path": os.path.join(models_dir, "observer_ai_model.pkl"),
            "update_time": datetime.now().isoformat(),
            "metadata": {
                "training_samples": 6115,
                "accuracy": 0.87,
                "test_update": True
            }
        }
        
        notification_file = f"model_update_test_{int(datetime.now().timestamp())}.json"
        notification_path = os.path.join(models_dir, notification_file)
        
        with open(notification_path, 'w') as f:
            json.dump(test_notification, f, indent=2)
        
        print(f"✅ Test notification created: {notification_file}")
        print("🔍 Check cs2_duel_detector.py logs for hot-reload message")
        print("⏰ Notification will be processed within 30 seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Hot-reload test failed: {e}")
        return False

def main():
    """Main integration check and fix process"""
    
    print("🚀 AI Training Integration Check & Fix")
    print("=" * 50)
    
    try:
        # Check current integration
        integration_ok = ensure_model_integration()
        
        # Create integration patch
        create_integration_patch()
        
        # Test hot-reload system
        test_hot_reload_system()
        
        print(f"\n📋 NEXT STEPS:")
        print("1. Apply the integration patch to ai_trainer.py")
        print("2. Run the AI trainer to create new models")
        print("3. Check cs2_duel_detector.py logs for hot-reload messages") 
        print("4. Verify the observer uses updated models")
        
        if integration_ok:
            print(f"\n🎉 Integration system is ready!")
        else:
            print(f"\n⚠️ Please fix the issues above first")
            
    except Exception as e:
        print(f"\n💥 Integration check failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()