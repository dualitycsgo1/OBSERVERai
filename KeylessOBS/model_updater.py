#!/usr/bin/env python3
"""
Model Updater Integration for CS2 Observer System
Ensures the main observer system automatically uses newly trained models
"""

import os
import pickle
import json
import logging
import time
from datetime import datetime
from typing import Dict, Optional, Any
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelUpdater:
    """
    Manages model updates and integration with the main observer system
    
    Features:
    - Monitors for model updates
    - Validates new models before deployment
    - Backs up previous models
    - Notifies main system of updates
    - Hot-swapping without restart
    """
    
    def __init__(self, 
                 models_dir: str = "models",
                 backup_dir: str = "model_backups",
                 config_file: str = "model_config.json"):
        """
        Initialize Model Updater
        
        Args:
            models_dir: Directory containing trained models
            backup_dir: Directory for model backups
            config_file: Configuration file for model metadata
        """
        self.models_dir = models_dir
        self.backup_dir = backup_dir
        self.config_file = config_file
        
        # Ensure directories exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)
        
        # Model paths
        self.model_files = {
            'observer_ai': os.path.join(models_dir, 'observer_ai_model.pkl'),
            'position_model': os.path.join(models_dir, 'enhanced_position_model.joblib'),
            'kill_analyzer': os.path.join(models_dir, 'kill_pattern_model.pkl')
        }
        
        # Load configuration
        self.config = self._load_config()
        
        # Model metadata
        self.model_metadata = {}
        self._load_model_metadata()
        
        logger.info("Model Updater initialized")
        
    def _load_config(self) -> Dict:
        """Load model configuration"""
        default_config = {
            "auto_update_enabled": True,
            "validation_required": True,
            "backup_on_update": True,
            "max_backups": 10,
            "min_accuracy_threshold": 0.5,
            "update_check_interval": 30,  # seconds
            "supported_models": list(self.model_files.keys())
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    default_config.update(config)
            except Exception as e:
                logger.warning(f"Could not load config: {e}")
                
        return default_config
        
    def _save_config(self):
        """Save model configuration"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save config: {e}")
            
    def _load_model_metadata(self):
        """Load metadata for all models"""
        for model_name, model_path in self.model_files.items():
            if os.path.exists(model_path):
                try:
                    stat = os.stat(model_path)
                    self.model_metadata[model_name] = {
                        'path': model_path,
                        'size': stat.st_size,
                        'modified': stat.st_mtime,
                        'loaded': False,
                        'version': self._get_model_version(model_path)
                    }
                except Exception as e:
                    logger.warning(f"Could not load metadata for {model_name}: {e}")
                    
    def _get_model_version(self, model_path: str) -> str:
        """Get model version from file"""
        try:
            modified_time = os.path.getmtime(model_path)
            return datetime.fromtimestamp(modified_time).strftime("%Y%m%d_%H%M%S")
        except:
            return "unknown"
            
    def check_for_updates(self) -> Dict[str, bool]:
        """
        Check if any models have been updated
        
        Returns:
            Dictionary mapping model names to update status
        """
        updates = {}
        
        for model_name, model_path in self.model_files.items():
            if not os.path.exists(model_path):
                updates[model_name] = False
                continue
                
            try:
                current_metadata = self.model_metadata.get(model_name, {})
                current_modified = current_metadata.get('modified', 0)
                
                stat = os.stat(model_path)
                new_modified = stat.st_mtime
                
                # Check if file has been modified
                if new_modified > current_modified:
                    updates[model_name] = True
                    logger.info(f"Update detected for {model_name}")
                else:
                    updates[model_name] = False
                    
            except Exception as e:
                logger.warning(f"Could not check updates for {model_name}: {e}")
                updates[model_name] = False
                
        return updates
        
    def validate_model(self, model_name: str, model_path: str) -> Dict[str, Any]:
        """
        Validate a model before deployment
        
        Args:
            model_name: Name of the model
            model_path: Path to the model file
            
        Returns:
            Validation results
        """
        validation = {
            'valid': False,
            'error': None,
            'metadata': {},
            'performance': {}
        }
        
        try:
            # Check file exists and is not empty
            if not os.path.exists(model_path):
                validation['error'] = "Model file does not exist"
                return validation
                
            stat = os.stat(model_path)
            if stat.st_size == 0:
                validation['error'] = "Model file is empty"
                return validation
                
            # Try to load the model
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    
                # Basic model validation
                if hasattr(model, 'predict'):
                    validation['metadata']['type'] = type(model).__name__
                    validation['metadata']['has_predict'] = True
                    
                    # If it's a scikit-learn model, get more info
                    if hasattr(model, 'feature_importances_'):
                        validation['metadata']['n_features'] = len(model.feature_importances_)
                        validation['metadata']['feature_importances'] = True
                        
                    if hasattr(model, 'score'):
                        validation['metadata']['has_score'] = True
                        
                    validation['valid'] = True
                    
                else:
                    validation['error'] = "Model does not have predict method"
                    
            elif model_path.endswith('.joblib'):
                import joblib
                model = joblib.load(model_path)
                validation['metadata']['type'] = type(model).__name__
                validation['valid'] = True
                
            else:
                validation['error'] = "Unsupported model format"
                
        except Exception as e:
            validation['error'] = f"Model validation failed: {str(e)}"
            
        return validation
        
    def backup_model(self, model_name: str) -> bool:
        """
        Create backup of existing model
        
        Args:
            model_name: Name of the model to backup
            
        Returns:
            True if backup was successful
        """
        try:
            model_path = self.model_files.get(model_name)
            if not model_path or not os.path.exists(model_path):
                logger.warning(f"No model to backup: {model_name}")
                return False
                
            # Create backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{model_name}_backup_{timestamp}.pkl"
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            # Copy file
            shutil.copy2(model_path, backup_path)
            
            # Clean up old backups
            self._cleanup_old_backups(model_name)
            
            logger.info(f"Model backed up: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup model {model_name}: {e}")
            return False
            
    def _cleanup_old_backups(self, model_name: str):
        """Clean up old model backups"""
        try:
            pattern = f"{model_name}_backup_"
            backups = []
            
            for file in os.listdir(self.backup_dir):
                if file.startswith(pattern):
                    filepath = os.path.join(self.backup_dir, file)
                    backups.append((filepath, os.path.getmtime(filepath)))
                    
            # Sort by modification time (newest first)
            backups.sort(key=lambda x: x[1], reverse=True)
            
            # Remove old backups
            max_backups = self.config.get('max_backups', 10)
            for filepath, _ in backups[max_backups:]:
                try:
                    os.remove(filepath)
                    logger.info(f"Removed old backup: {os.path.basename(filepath)}")
                except Exception as e:
                    logger.warning(f"Could not remove backup {filepath}: {e}")
                    
        except Exception as e:
            logger.warning(f"Backup cleanup failed: {e}")
            
    def update_model(self, model_name: str) -> Dict[str, Any]:
        """
        Update a model and integrate it with the system
        
        Args:
            model_name: Name of the model to update
            
        Returns:
            Update results
        """
        result = {
            'success': False,
            'model_name': model_name,
            'error': None,
            'validation': {},
            'backup_created': False,
            'metadata': {}
        }
        
        try:
            model_path = self.model_files.get(model_name)
            if not model_path:
                result['error'] = f"Unknown model: {model_name}"
                return result
                
            # Validate new model
            if self.config.get('validation_required', True):
                validation = self.validate_model(model_name, model_path)
                result['validation'] = validation
                
                if not validation['valid']:
                    result['error'] = f"Model validation failed: {validation['error']}"
                    return result
                    
            # Backup existing model
            if self.config.get('backup_on_update', True):
                result['backup_created'] = self.backup_model(model_name)
                
            # Update metadata
            stat = os.stat(model_path)
            self.model_metadata[model_name] = {
                'path': model_path,
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'loaded': False,
                'version': self._get_model_version(model_path),
                'updated': datetime.now().isoformat()
            }
            
            result['metadata'] = self.model_metadata[model_name]
            result['success'] = True
            
            logger.info(f"Model {model_name} updated successfully")
            
            # Notify main system
            self._notify_system_update(model_name)
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Model update failed for {model_name}: {e}")
            
        return result
        
    def _notify_system_update(self, model_name: str):
        """Notify the main system that a model has been updated"""
        try:
            # Create update notification file
            notification = {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'action': 'model_updated'
            }
            
            notification_file = f"model_update_{model_name}.json"
            notification_path = os.path.join(self.models_dir, notification_file)
            
            with open(notification_path, 'w') as f:
                json.dump(notification, f, indent=2)
                
            logger.info(f"Update notification created: {notification_file}")
            
        except Exception as e:
            logger.warning(f"Could not create update notification: {e}")
            
    def get_model_status(self) -> Dict[str, Dict]:
        """Get status of all models"""
        status = {}
        
        for model_name, model_path in self.model_files.items():
            model_status = {
                'exists': os.path.exists(model_path),
                'metadata': self.model_metadata.get(model_name, {}),
                'validation': None
            }
            
            if model_status['exists']:
                model_status['validation'] = self.validate_model(model_name, model_path)
                
            status[model_name] = model_status
            
        return status
        
    def auto_update_check(self) -> Dict[str, Any]:
        """
        Perform automatic update check and apply updates if enabled
        
        Returns:
            Results of update check and any updates applied
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'updates_checked': 0,
            'updates_found': 0,
            'updates_applied': 0,
            'updates': {},
            'errors': []
        }
        
        if not self.config.get('auto_update_enabled', True):
            logger.info("Auto-update is disabled")
            return results
            
        try:
            # Check for updates
            updates = self.check_for_updates()
            results['updates_checked'] = len(updates)
            
            for model_name, has_update in updates.items():
                if has_update:
                    results['updates_found'] += 1
                    logger.info(f"Applying auto-update for {model_name}")
                    
                    # Apply update
                    update_result = self.update_model(model_name)
                    results['updates'][model_name] = update_result
                    
                    if update_result['success']:
                        results['updates_applied'] += 1
                    else:
                        results['errors'].append(f"{model_name}: {update_result['error']}")
                        
        except Exception as e:
            results['errors'].append(f"Auto-update check failed: {str(e)}")
            logger.error(f"Auto-update check failed: {e}")
            
        return results

def create_model_integration_patch():
    """
    Create a patch for cs2_duel_detector.py to support hot model reloading
    """
    integration_code = '''
# Model Hot-Reloading Integration
# Add this to your cs2_duel_detector.py

import os
import json
from datetime import datetime

class ModelHotReloader:
    """Hot reload models when they are updated"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.model_cache = {}
        self.last_check = 0
        self.check_interval = 30  # seconds
        
    def get_model(self, model_name, model_path):
        """Get model with hot reloading support"""
        current_time = time.time()
        
        # Check for updates every check_interval seconds
        if current_time - self.last_check > self.check_interval:
            self._check_for_updates()
            self.last_check = current_time
            
        # Return cached model or load new one
        cache_key = f"{model_name}_{model_path}"
        
        if cache_key in self.model_cache:
            return self.model_cache[cache_key]['model']
        else:
            return self._load_model(model_name, model_path)
            
    def _check_for_updates(self):
        """Check for model update notifications"""
        try:
            for file in os.listdir(self.models_dir):
                if file.startswith("model_update_") and file.endswith(".json"):
                    notification_path = os.path.join(self.models_dir, file)
                    
                    with open(notification_path, 'r') as f:
                        notification = json.load(f)
                        
                    model_name = notification.get('model_name')
                    if model_name:
                        self._reload_model(model_name)
                        
                    # Remove notification file
                    os.remove(notification_path)
                    
        except Exception as e:
            logger.warning(f"Update check failed: {e}")
            
    def _reload_model(self, model_name):
        """Reload a specific model"""
        # Clear from cache to force reload
        keys_to_remove = [k for k in self.model_cache.keys() if model_name in k]
        for key in keys_to_remove:
            del self.model_cache[key]
            
        logger.info(f"Model {model_name} cleared from cache for reload")
        
    def _load_model(self, model_name, model_path):
        """Load model and cache it"""
        try:
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            elif model_path.endswith('.joblib'):
                import joblib
                model = joblib.load(model_path)
            else:
                raise ValueError(f"Unsupported model format: {model_path}")
                
            # Cache the model
            cache_key = f"{model_name}_{model_path}"
            self.model_cache[cache_key] = {
                'model': model,
                'loaded_at': datetime.now(),
                'path': model_path
            }
            
            logger.info(f"Model {model_name} loaded and cached")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None

# Usage in cs2_duel_detector.py:
# 
# # Initialize hot reloader (add to __init__)
# self.model_reloader = ModelHotReloader()
# 
# # Replace direct model loading with:
# self.ai_model = self.model_reloader.get_model("observer_ai", "models/observer_ai_model.pkl")
# 
# # In your analysis methods, always get fresh model:
# model = self.model_reloader.get_model("observer_ai", "models/observer_ai_model.pkl")
# if model:
#     prediction = model.predict(features)
'''
    
    patch_file = "model_integration_patch.py"
    with open(patch_file, 'w') as f:
        f.write(integration_code)
        
    print(f"Model integration patch created: {patch_file}")
    print("Add the ModelHotReloader class to your cs2_duel_detector.py")
    
    return patch_file

def test_model_updater():
    """Test the model updater system"""
    print("Testing Model Updater System")
    print("=" * 50)
    
    # Initialize updater
    updater = ModelUpdater()
    
    # Get model status
    status = updater.get_model_status()
    
    for model_name, model_status in status.items():
        print(f"\n{model_name}:")
        print(f"  Exists: {model_status['exists']}")
        if model_status['validation']:
            print(f"  Valid: {model_status['validation']['valid']}")
            if model_status['validation']['error']:
                print(f"  Error: {model_status['validation']['error']}")
                
    # Check for updates
    updates = updater.check_for_updates()
    print(f"\nUpdate Check:")
    for model_name, has_update in updates.items():
        print(f"  {model_name}: {'UPDATE AVAILABLE' if has_update else 'UP TO DATE'}")
        
    return True

if __name__ == "__main__":
    test_model_updater()
    create_model_integration_patch()