#!/usr/bin/env python3
"""
Enhanced CS2 Observer Integration with Positional AI
Integrates the newly trained positional AI models with the observer system
"""

import json
import joblib
import numpy as np
import os
import sys
from typing import Dict, List, Any, Optional
import logging

class PositionalObserverAI:
    """Enhanced observer AI using complete positional and contextual data"""
    
    def __init__(self, model_dir: str = "enhanced_positional_models"):
        self.model_dir = model_dir
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.metadata = {}
        self.feature_columns = []
        
        self.load_models()
    
    def load_models(self):
        """Load all trained models and preprocessing objects"""
        
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        
        try:
            # Load metadata
            metadata_file = os.path.join(self.model_dir, "model_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                    self.feature_columns = self.metadata.get('feature_columns', [])
            
            # Load scalers and encoders with error handling
            scaler_file = os.path.join(self.model_dir, "scalers.joblib")
            if os.path.exists(scaler_file):
                self.scalers = joblib.load(scaler_file)
            
            encoder_file = os.path.join(self.model_dir, "encoders.joblib")
            if os.path.exists(encoder_file):
                self.encoders = joblib.load(encoder_file)
            
            # Load individual models with robust error handling
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('_model.joblib')]
            loaded_models = 0
            
            for model_file in model_files:
                try:
                    task_name = model_file.replace('_model.joblib', '')
                    model_path = os.path.join(self.model_dir, model_file)
                    self.models[task_name] = joblib.load(model_path)
                    loaded_models += 1
                except Exception as model_e:
                    # Skip individual model failures
                    continue
            
            if loaded_models == 0:
                raise RuntimeError("No models could be loaded")
            
        except (Exception, KeyboardInterrupt) as e:
            self.models = {}
            self.scalers = {}  
            self.encoders = {}
            raise RuntimeError(f"Model loading failed: {e}")
    
    def extract_features_from_gamestate(self, gamestate: Dict[str, Any], 
                                      potential_kill: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract features from current gamestate for AI prediction"""
        
        try:
            # Extract player positions and data
            killer_data = potential_kill.get('killer', {})
            victim_data = potential_kill.get('victim', {})
            
            killer_pos = killer_data.get('position', {})
            victim_pos = victim_data.get('position', {})
            killer_angle = killer_data.get('angles', {})
            
            if not killer_pos or not victim_pos:
                return None
            
            # Calculate distance
            distance_3d = np.sqrt(
                (killer_pos.get('x', 0) - victim_pos.get('x', 0))**2 +
                (killer_pos.get('y', 0) - victim_pos.get('y', 0))**2 +
                (killer_pos.get('z', 0) - victim_pos.get('z', 0))**2
            )
            
            # Extract weapon info
            weapon = killer_data.get('weapon', 'unknown').lower()
            
            # Create feature vector matching training format
            features = {
                # Spatial features
                'killer_x': killer_pos.get('x', 0),
                'killer_y': killer_pos.get('y', 0),
                'killer_z': killer_pos.get('z', 0),
                'victim_x': victim_pos.get('x', 0),
                'victim_y': victim_pos.get('y', 0),
                'victim_z': victim_pos.get('z', 0),
                'distance_3d': distance_3d,
                'height_difference': abs(killer_pos.get('z', 0) - victim_pos.get('z', 0)),
                
                # Angular features
                'killer_yaw': killer_angle.get('yaw', 0),
                'killer_pitch': killer_angle.get('pitch', 0),
                
                # Tactical features
                'game_time': gamestate.get('map', {}).get('round_time', 0),
                'game_tick': gamestate.get('provider', {}).get('timestamp', 0),
                
                # Weapon categorization
                'weapon_rifle': 1 if any(w in weapon for w in ['ak47', 'm4a1', 'm4a4']) else 0,
                'weapon_awp': 1 if 'awp' in weapon else 0,
                'weapon_pistol': 1 if any(w in weapon for w in ['glock', 'usp', 'p250', 'deagle']) else 0,
                'weapon_smg': 1 if any(w in weapon for w in ['mp9', 'mac10', 'ump', 'p90']) else 0,
                'weapon_shotgun': 1 if any(w in weapon for w in ['nova', 'xm1014', 'mag7']) else 0,
                'weapon_knife': 1 if 'knife' in weapon else 0,
                
                # Distance categories
                'distance_close': 1 if distance_3d < 500 else 0,
                'distance_medium': 1 if 500 <= distance_3d < 1500 else 0,
                'distance_long': 1 if distance_3d >= 1500 else 0,
                
                # Angle analysis
                'angle_forward': 1 if abs(killer_angle.get('yaw', 0)) < 45 else 0,
                'angle_side': 1 if 45 <= abs(killer_angle.get('yaw', 0)) < 135 else 0,
                'angle_back': 1 if abs(killer_angle.get('yaw', 0)) >= 135 else 0,
                
                # Height advantage
                'height_advantage': 1 if killer_pos.get('z', 0) > victim_pos.get('z', 0) else 0,
                'height_disadvantage': 1 if killer_pos.get('z', 0) < victim_pos.get('z', 0) else 0,
                
                # Map encoding (simplified)
                'map_encoded': 0  # Default value, could be enhanced with map detection
            }
            
            # Create feature vector in correct order
            feature_vector = np.array([features.get(col, 0) for col in self.feature_columns])
            
            # Scale features
            if 'features' in self.scalers:
                feature_vector = self.scalers['features'].transform(feature_vector.reshape(1, -1))
                return feature_vector[0]
            
            return feature_vector
            
        except Exception as e:
            logging.error(f"Feature extraction error: {e}")
            return None
    
    def predict_kill_outcome(self, gamestate: Dict[str, Any], 
                           potential_kill: Dict[str, Any]) -> Dict[str, Any]:
        """Predict various aspects of a potential kill using positional AI"""
        
        features = self.extract_features_from_gamestate(gamestate, potential_kill)
        
        if features is None:
            return {
                'headshot_probability': 0.5,
                'wallbang_probability': 0.1,
                'weapon_effectiveness': 0.5,
                'engagement_type': 'unknown',
                'confidence': 0.0,
                'prediction_source': 'fallback'
            }
        
        predictions = {}
        
        try:
            # Headshot prediction
            if 'headshot_prediction' in self.models:
                headshot_prob = self.models['headshot_prediction'].predict_proba(features.reshape(1, -1))[0]
                predictions['headshot_probability'] = float(headshot_prob[1]) if len(headshot_prob) > 1 else 0.5
            
            # Wallbang prediction
            if 'wallbang_prediction' in self.models:
                wallbang_prob = self.models['wallbang_prediction'].predict_proba(features.reshape(1, -1))[0]
                predictions['wallbang_probability'] = float(wallbang_prob[1]) if len(wallbang_prob) > 1 else 0.1
            
            # Weapon effectiveness
            if 'weapon_effectiveness' in self.models:
                weapon_prob = self.models['weapon_effectiveness'].predict_proba(features.reshape(1, -1))[0]
                predictions['weapon_effectiveness'] = float(weapon_prob[1]) if len(weapon_prob) > 1 else 0.5
            
            # Distance category
            if 'distance_category' in self.models:
                distance_prob = self.models['distance_category'].predict_proba(features.reshape(1, -1))[0]
                predictions['engagement_type'] = 'long_range' if distance_prob[1] > 0.5 else 'close_range'
            
            # Calculate overall confidence based on feature quality
            distance = np.sqrt(np.sum((features[:3] - features[3:6])**2))  # 3D distance from positions
            confidence = min(1.0, max(0.1, 1.0 - (distance / 3000.0)))  # Higher confidence for closer engagements
            
            predictions.update({
                'confidence': float(confidence),
                'prediction_source': 'positional_ai',
                'feature_count': len(features),
                'models_used': list(self.models.keys())
            })
            
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            predictions.update({
                'headshot_probability': 0.5,
                'wallbang_probability': 0.1,
                'weapon_effectiveness': 0.5,
                'engagement_type': 'unknown',
                'confidence': 0.0,
                'prediction_source': 'error_fallback'
            })
        
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        
        return {
            'models_loaded': list(self.models.keys()),
            'feature_count': len(self.feature_columns),
            'model_directory': self.model_dir,
            'metadata': self.metadata,
            'training_samples': self.metadata.get('dataset_info', {}).get('total_samples', 0),
            'capabilities': self.metadata.get('dataset_info', {}).get('capabilities', {}),
            'ready': len(self.models) > 0
        }


def test_positional_ai():
    """Test the positional AI integration"""
    
    print("🎯 Testing Positional Observer AI Integration")
    print("=" * 60)
    
    try:
        # Initialize AI
        ai = PositionalObserverAI()
        
        # Get model info
        info = ai.get_model_info()
        
        print(f"✅ Positional AI loaded:")
        print(f"   Models: {info['models_loaded']}")
        print(f"   Features: {info['feature_count']}")
        print(f"   Training samples: {info['training_samples']}")
        print(f"   Ready: {info['ready']}")
        
        # Test prediction with sample data
        sample_gamestate = {
            'map': {'round_time': 75.5},
            'provider': {'timestamp': 123456}
        }
        
        sample_kill = {
            'killer': {
                'position': {'x': -500, 'y': 1200, 'z': 64},
                'angles': {'yaw': -45, 'pitch': 5},
                'weapon': 'ak47'
            },
            'victim': {
                'position': {'x': -200, 'y': 1500, 'z': 64}
            }
        }
        
        print(f"\n🔍 Testing prediction...")
        predictions = ai.predict_kill_outcome(sample_gamestate, sample_kill)
        
        print(f"✅ Predictions:")
        for key, value in predictions.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")
        
        return ai
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    result = test_positional_ai()
    
    if result:
        print(f"\n✅ Positional AI integration ready!")
        print(f"   Ready for integration with cs2_duel_detector.py")
    else:
        print(f"\n❌ Integration test failed")