#!/usr/bin/env python3
"""
AI Trainer for CS2 Observer System
Allows users to continuously improve the AI model by feeding new demos
Automatically updates the trained model and integrates with the main system
"""

import os
import sys
import json
import pickle
import logging

def get_base_dir():
    """Get base directory for dependencies - works for both script and executable"""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable
        return os.path.dirname(sys.executable)
    else:
        # Running as script
        return os.path.dirname(os.path.abspath(__file__))

# Configure logging with UTF-8 encoding for Windows
# Use user AppData directory for logs to avoid permission issues
import tempfile
log_dir = os.path.join(os.environ.get('APPDATA', tempfile.gettempdir()), 'OBSERVERai')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'ai_trainer.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from pathlib import Path
import shutil

# Import our existing components
from demo_analyzer import DemoAnalyzer
from kill_pattern_analyzer import KillPatternAnalyzer
from comprehensive_demo_processor import ComprehensiveDemoProcessor

logger = logging.getLogger(__name__)

class AITrainer:
    """
    AI Trainer for continuous model improvement
    
    Features:
    - Process new demo files automatically
    - Extract tactical features and patterns
    - Incrementally update existing models
    - Maintain training history and metrics
    - Auto-integration with main observer system
    """
    
    def __init__(self, 
                 models_dir: str = "models",
                 training_data_dir: str = "training_data", 
                 demos_dir: str = "demos",
                 backup_dir: str = "model_backups"):
        """
        Initialize AI Trainer
        
        Args:
            models_dir: Directory containing trained models
            training_data_dir: Directory for training datasets
            demos_dir: Directory to monitor for new demos
            backup_dir: Directory for model backups
        """
        base_dir = get_base_dir()
        self.models_dir = os.path.join(base_dir, models_dir)
        self.training_data_dir = os.path.join(base_dir, training_data_dir)
        self.demos_dir = os.path.join(base_dir, demos_dir)
        self.backup_dir = os.path.join(base_dir, backup_dir)
        
        # Ensure directories exist
        for directory in [models_dir, training_data_dir, demos_dir, backup_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Initialize components
        self.demo_analyzer = DemoAnalyzer()
        self.kill_analyzer = KillPatternAnalyzer()
        self.demo_processor = ComprehensiveDemoProcessor()
        
        # Training configuration
        self.config = {
            "batch_size": 1000,
            "max_demos_per_session": 10,
            "validation_split": 0.2,
            "backup_frequency": 5,  # Backup every N training sessions
            "feature_columns": [
                'distance', 'hp_attacker', 'hp_victim', 'armor_attacker', 'armor_victim',
                'weapon_damage_potential', 'attacker_x', 'attacker_y', 'attacker_z',
                'victim_x', 'victim_y', 'victim_z', 'angle_difference', 'elevation_difference',
                'attacker_velocity', 'victim_velocity', 'crosshair_placement_score',
                'positioning_advantage', 'tactical_situation', 'round_economy_factor',
                'team_advantage', 'map_control_score', 'timing_advantage', 'surprise_factor',
                'experience_rating', 'recent_performance', 'pressure_rating'
            ]
        }
        
        # Set up logger
        self.logger = logging.getLogger('ai_trainer')
        
        # Load training history
        self.history_file = os.path.join(self.training_data_dir, "training_history.json")
        self.training_history = self._load_training_history()
        
        self.logger.info(f"AI Trainer initialized")
        self.logger.info(f"Models: {models_dir}, Training Data: {training_data_dir}")
        self.logger.info(f"Demos: {demos_dir}, Backups: {backup_dir}")
        
    def _load_training_history(self) -> Dict:
        """Load training history from disk"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.getLogger('ai_trainer').warning(f"Could not load training history: {e}")
                
        return {
            "sessions": [],
            "total_demos_processed": 0,
            "total_kills_analyzed": 0,
            "model_versions": [],
            "performance_metrics": []
        }
        
    def _save_training_history(self):
        """Save training history to disk"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.training_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save training history: {e}")
            
    def process_new_demos(self, demo_files: List[str] = None) -> Dict[str, Any]:
        """
        Process new demo files and extract training data
        
        Args:
            demo_files: List of demo file paths. If None, scan demos directory
            
        Returns:
            Dictionary with processing results and statistics
        """
        if demo_files is None:
            demo_files = self._find_new_demos()
            
        if not demo_files:
            logger.info("No new demos to process")
            return {"processed": 0, "new_data": {}}
            
        logger.info(f"Processing {len(demo_files)} new demos...")
        
        results = {
            "processed": 0,
            "failed": 0,
            "new_kills": 0,
            "new_rounds": 0,
            "new_data": {},
            "errors": []
        }
        
        all_new_data = []
        
        for demo_file in demo_files[:self.config["max_demos_per_session"]]:
            try:
                logger.info(f"Processing demo: {os.path.basename(demo_file)}")
                logger.info(f"Demo file exists: {os.path.exists(demo_file)}")
                print(f"🔍 Processing demo: {os.path.basename(demo_file)}")
                print(f"📁 Demo file exists: {os.path.exists(demo_file)}")
                
                # Process demo comprehensively (header + kills with positions)
                demo_result = self.demo_processor.process_demo_comprehensive(demo_file)
                logger.info(f"Demo processing result: success={demo_result.get('success', False)}")
                print(f"📊 Demo processing result: success={demo_result.get('success', False)}")
                
                if not demo_result.get('success') or not demo_result.get('kills'):
                    logger.warning(f"Demo processing failed or no kills found: {demo_file}")
                    logger.warning(f"Map name: {demo_result.get('map_name', 'N/A')}")
                    logger.warning(f"Kill count: {demo_result.get('kill_count', 0)}")
                    print(f"❌ Demo processing failed or no kills found: {demo_file}")
                    print(f"�️ Map name: {demo_result.get('map_name', 'N/A')}")
                    print(f"💀 Kill count: {demo_result.get('kill_count', 0)}")
                    results["failed"] += 1
                    continue
                
                # Get kills in the format expected by the AI trainer
                kill_events = demo_result['kills']
                logger.info(f"Found {len(kill_events)} kill events with positions")
                logger.info(f"Map name: {demo_result.get('map_name', 'unknown')}")
                print(f"💀 Found {len(kill_events)} kill events with positions")
                print(f"🗺️ Map name: {demo_result.get('map_name', 'unknown')}")
                
                # Create demo data structure for feature extraction
                demo_data = {
                    'kills': kill_events,
                    'map_name': demo_result.get('map_name', 'unknown'),
                    'server_name': demo_result.get('server_name', ''),
                    'demo_type': demo_result.get('demo_type', 'Unknown'),
                    'demo_file': os.path.basename(demo_file)
                }
                
                # Extract features using our analyzers (now with proper positions)
                kill_features = self._extract_kill_features_from_comprehensive(kill_events, demo_result, demo_file)
                
                if kill_features:
                    all_new_data.extend(kill_features)
                    results["new_kills"] += len(kill_features)
                    results["processed"] += 1
                    
                    logger.info(f"Extracted {len(kill_features)} kills from {os.path.basename(demo_file)}")
                else:
                    results["failed"] += 1
                    
            except Exception as e:
                logger.error(f"Failed to process demo {demo_file}: {e}")
                results["errors"].append(f"{demo_file}: {str(e)}")
                results["failed"] += 1
                
        # Save new data
        if all_new_data:
            results["new_data"] = self._save_new_training_data(all_new_data)
            logger.info(f"Saved {len(all_new_data)} new training samples")
            
        # Update training history with the comprehensive processing results
        self._update_training_history({
            "demos_processed": [os.path.basename(f) for f in demo_files if os.path.exists(f)],
            "kills_processed": results["new_kills"],
            "processing_results": results,
            "processor_type": "comprehensive",  # Indicate we used comprehensive processor
            "features_extracted": len(all_new_data) if all_new_data else 0
        })
        
        return results
        
    def _find_new_demos(self) -> List[str]:
        """Find demo files that haven't been processed yet"""
        processed_demos = set()
        
        # Get list of already processed demos
        for session in self.training_history.get("sessions", []):
            processed_demos.update(session.get("demos_processed", []))
            
        # Find new demos
        new_demos = []
        if os.path.exists(self.demos_dir):
            for file in os.listdir(self.demos_dir):
                if file.endswith('.dem') and file not in processed_demos:
                    new_demos.append(os.path.join(self.demos_dir, file))
                    
        logger.info(f"Found {len(new_demos)} new demo files")
        return new_demos
        
    def _extract_kill_features(self, demo_data: Dict, demo_file: str) -> List[Dict]:
        """Extract tactical features from kill events"""
        try:
            # Extract kill events from the event stream
            kill_events = [event for event in demo_data.get('events', []) if event.get('type') == 'kill']
            if not kill_events:
                return []
                
            # Use our existing analyzers to extract features
            features = []
            
            for kill_event in kill_events:
                kill = kill_event.get('data')
                
                # Skip if no kill data
                if kill is None:
                    logger.debug(f"Skipping kill event with no data: {kill_event}")
                    continue
                    
                try:
                    # Extract basic kill info
                    kill_features = {
                        'demo_file': os.path.basename(demo_file),
                        'timestamp': datetime.now().isoformat(),
                        'round_num': kill_event.get('round', 0),
                        'tick': kill_event.get('tick', 0),
                        'event_time': kill_event.get('time', 0),
                        
                        # Player info
                        'attacker_name': kill.get('killer', ''),
                        'victim_name': kill.get('victim', ''),
                        'weapon': kill.get('weapon', ''),
                        'is_headshot': kill.get('headshot', False),
                        'wallbang': kill.get('wallbang', False),
                        'distance': kill.get('distance', 0),
                        
                        # Positions from your parser structure
                        'killer_position': kill.get('killer_pos', {}),
                        'victim_position': kill.get('victim_pos', {}),
                        'killer_angle': kill.get('killer_angle', {}),
                        
                        # Legacy position fields for compatibility
                        'attacker_x': kill.get('killer_pos', {}).get('x', 0) if kill.get('killer_pos') else 0,
                        'attacker_y': kill.get('killer_pos', {}).get('y', 0) if kill.get('killer_pos') else 0,
                        'attacker_z': kill.get('killer_pos', {}).get('z', 0) if kill.get('killer_pos') else 0,
                        'victim_x': kill.get('victim_pos', {}).get('x', 0) if kill.get('victim_pos') else 0,
                        'victim_y': kill.get('victim_pos', {}).get('y', 0) if kill.get('victim_pos') else 0,
                        'victim_z': kill.get('victim_pos', {}).get('z', 0) if kill.get('victim_pos') else 0,
                    }
                    
                    # Calculate advanced features using our analyzers
                    if kill:  # Ensure kill data exists
                        advanced_features = self._calculate_advanced_features(kill, demo_data)
                        kill_features.update(advanced_features)
                    else:
                        logger.warning(f"Kill data is None, using defaults")
                        # Add default advanced features
                        default_features = {
                            'distance': 0, 'hp_attacker': 100, 'hp_victim': 100,
                            'armor_attacker': 0, 'armor_victim': 0, 'weapon_damage_potential': 0.5,
                            'elevation_difference': 0, 'angle_difference': 0, 'attacker_velocity': 0,
                            'victim_velocity': 0, 'crosshair_placement_score': 0.5, 'positioning_advantage': 0.5,
                            'tactical_situation': 0.5, 'round_economy_factor': 0.5, 'team_advantage': 0.5,
                            'map_control_score': 0.5, 'timing_advantage': 0.5, 'surprise_factor': 0.5,
                            'experience_rating': 0.5, 'recent_performance': 0.5, 'pressure_rating': 0.5
                        }
                        kill_features.update(default_features)
                    
                    features.append(kill_features)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract features from kill: {e}")
                    continue
                    
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract kill features: {e}")
            return []
    
    def _extract_kill_features_from_comprehensive(self, kill_events: List[Dict], demo_result: Dict, demo_file: str) -> List[Dict]:
        """Extract tactical features from kill events processed by comprehensive demo processor"""
        try:
            if not kill_events:
                return []
                
            features = []
            
            for kill in kill_events:
                # Skip if no kill data
                if kill is None:
                    logger.debug(f"Skipping empty kill event")
                    continue
                    
                try:
                    # The comprehensive processor already converted to the expected format
                    kill_features = {
                        'demo_file': os.path.basename(demo_file),
                        'timestamp': datetime.now().isoformat(),
                        'round_num': kill.get('round_number', 0),
                        'tick': kill.get('tick', 0),
                        'event_time': 0,  # Not available from comprehensive processor
                        
                        # Player info - using comprehensive processor format
                        'attacker_name': kill.get('attacker_name', ''),
                        'victim_name': kill.get('victim_name', ''),
                        'weapon': kill.get('weapon', ''),
                        'is_headshot': kill.get('is_headshot', False),
                        'wallbang': False,  # Not available in current format
                        'distance': kill.get('distance', 0),
                        
                        # Positions - comprehensive processor format
                        'killer_position': kill.get('attacker_pos', {}),
                        'victim_position': kill.get('victim_pos', {}),
                        'killer_angle': {'yaw': kill.get('attacker_view_x', 0), 'pitch': kill.get('attacker_view_y', 0)},
                        
                        # Legacy position fields for compatibility with existing analyzers
                        'attacker_x': kill.get('attacker_pos', {}).get('x', 0),
                        'attacker_y': kill.get('attacker_pos', {}).get('y', 0),
                        'attacker_z': kill.get('attacker_pos', {}).get('z', 0),
                        'victim_x': kill.get('victim_pos', {}).get('x', 0),
                        'victim_y': kill.get('victim_pos', {}).get('y', 0),
                        'victim_z': kill.get('victim_pos', {}).get('z', 0),
                        
                        # Map info from comprehensive processor
                        'map_name': demo_result.get('map_name', 'unknown'),
                        'server_name': demo_result.get('server_name', ''),
                        'demo_type': demo_result.get('demo_type', 'Unknown'),
                    }
                    
                    # Calculate advanced features using existing analyzers
                    # Create a compatible kill dict for the analyzer
                    compatible_kill = {
                        'killer_pos': kill.get('attacker_pos', {}),
                        'victim_pos': kill.get('victim_pos', {}),
                        'distance': kill.get('distance', 0),
                        'weapon': kill.get('weapon', ''),
                        'headshot': kill.get('is_headshot', False),
                        'killer_angle': {'yaw': kill.get('attacker_view_x', 0), 'pitch': kill.get('attacker_view_y', 0)}
                    }
                    
                    advanced_features = self._calculate_advanced_features(compatible_kill, {'map_name': demo_result.get('map_name', 'unknown')})
                    kill_features.update(advanced_features)
                    
                    features.append(kill_features)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract features from kill: {e}")
                    continue
                    
            logger.info(f"Extracted {len(features)} kill features from {len(kill_events)} kill events")
            return features
            
        except Exception as e:
            logger.error(f"Failed to extract kill features from comprehensive data: {e}")
            return []
            
    def _calculate_advanced_features(self, kill: Dict, demo_data: Dict) -> Dict:
        """Calculate advanced tactical features for a kill"""
        features = {}
        
        # Guard against None kill data
        if kill is None:
            logger.warning("Kill data is None in _calculate_advanced_features")
            return self._get_default_features()
        
        try:
            # Distance calculation (use provided distance or calculate if positions available)
            if 'distance' in kill and kill['distance'] > 0:
                distance = kill['distance']
            else:
                # Calculate from positions if available
                killer_pos = kill.get('killer_pos', {})
                victim_pos = kill.get('victim_pos', {})
                
                if killer_pos and victim_pos:
                    kx, ky, kz = killer_pos.get('x', 0), killer_pos.get('y', 0), killer_pos.get('z', 0)
                    vx, vy, vz = victim_pos.get('x', 0), victim_pos.get('y', 0), victim_pos.get('z', 0)
                    distance = math.sqrt((kx - vx)**2 + (ky - vy)**2 + (kz - vz)**2)
                else:
                    distance = 0
                    
            features['distance'] = distance
            
            # Health and armor (these might not be available in kill events, set defaults)
            features['hp_attacker'] = kill.get('killer_health', 100)
            features['hp_victim'] = kill.get('victim_health', 100) 
            features['armor_attacker'] = kill.get('killer_armor', 0)
            features['armor_victim'] = kill.get('victim_armor', 0)
            
            # Weapon analysis
            weapon = kill.get('weapon', '')
            features['weapon_damage_potential'] = self._get_weapon_damage_potential(weapon)
            
            # Positional analysis
            killer_pos = kill.get('killer_pos', {})
            victim_pos = kill.get('victim_pos', {})
            if killer_pos and victim_pos:
                kz = killer_pos.get('z', 0)
                vz = victim_pos.get('z', 0)
                features['elevation_difference'] = abs(kz - vz)
            else:
                features['elevation_difference'] = 0
            
            # Use demo analyzer for more complex features
            if hasattr(self.demo_analyzer, '_calculate_tactical_features'):
                tactical_features = self.demo_analyzer._calculate_tactical_features(kill, demo_data)
                features.update(tactical_features)
            else:
                # Fallback calculations
                features.update({
                    'angle_difference': 0,
                    'attacker_velocity': 0,
                    'victim_velocity': 0,
                    'crosshair_placement_score': 0.5,
                    'positioning_advantage': 0.5,
                    'tactical_situation': 0.5,
                    'round_economy_factor': 0.5,
                    'team_advantage': 0.5,
                    'map_control_score': 0.5,
                    'timing_advantage': 0.5,
                    'surprise_factor': 0.5,
                    'experience_rating': 0.5,
                    'recent_performance': 0.5,
                    'pressure_rating': 0.5
                })
                
        except Exception as e:
            logger.warning(f"Failed to calculate advanced features: {e}")
            # Set default values
            for col in self.config["feature_columns"]:
                if col not in features:
                    features[col] = 0.5 if 'score' in col or 'rating' in col or 'factor' in col else 0
                    
        return features
        
    def _get_default_features(self) -> Dict:
        """Get default feature values when kill data is invalid"""
        return {
            'distance': 0, 'hp_attacker': 100, 'hp_victim': 100,
            'armor_attacker': 0, 'armor_victim': 0, 'weapon_damage_potential': 0.5,
            'elevation_difference': 0, 'angle_difference': 0, 'attacker_velocity': 0,
            'victim_velocity': 0, 'crosshair_placement_score': 0.5, 'positioning_advantage': 0.5,
            'tactical_situation': 0.5, 'round_economy_factor': 0.5, 'team_advantage': 0.5,
            'map_control_score': 0.5, 'timing_advantage': 0.5, 'surprise_factor': 0.5,
            'experience_rating': 0.5, 'recent_performance': 0.5, 'pressure_rating': 0.5
        }
        
    def _get_weapon_damage_potential(self, weapon: str) -> float:
        """Get damage potential score for weapon"""
        weapon_damage = {
            'ak47': 1.0, 'awp': 1.0, 'm4a4': 0.9, 'm4a1_s': 0.9,
            'krieg': 0.85, 'aug': 0.8, 'galil': 0.7, 'famas': 0.7,
            'scout': 0.85, 'deagle': 0.8, 'usp_s': 0.6, 'glock': 0.5,
            'p250': 0.6, 'fiveseven': 0.6, 'tec9': 0.6, 'cz75a': 0.6
        }
        
        weapon_clean = weapon.lower().replace('weapon_', '')
        return weapon_damage.get(weapon_clean, 0.5)
        
    def _save_new_training_data(self, new_data: List[Dict]) -> Dict:
        """Save new training data to disk"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(new_data)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_batch_{timestamp}.json"
            filepath = os.path.join(self.training_data_dir, filename)
            
            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump(new_data, f, indent=2, default=str)
                
            # Also append to master dataset if it exists
            master_file = os.path.join(self.training_data_dir, "complete_kill_analysis.json")
            if os.path.exists(master_file):
                try:
                    with open(master_file, 'r') as f:
                        existing_data = json.load(f)
                    
                    # Append new data
                    existing_data.extend(new_data)
                    
                    # Save updated master file
                    with open(master_file, 'w') as f:
                        json.dump(existing_data, f, indent=2, default=str)
                        
                    logger.info(f"Appended {len(new_data)} samples to master dataset")
                    
                except Exception as e:
                    logger.warning(f"Could not update master dataset: {e}")
                    
            return {
                "filename": filename,
                "samples": len(new_data),
                "filepath": filepath
            }
            
        except Exception as e:
            logger.error(f"Failed to save training data: {e}")
            return {}
            
    def train_model(self, incremental: bool = True) -> Dict[str, Any]:
        """
        Train or update the AI model with new data
        
        Args:
            incremental: If True, update existing model. If False, retrain from scratch
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Starting {'incremental' if incremental else 'full'} model training...")
        
        try:
            # Load training data
            training_data = self._load_all_training_data()
            
            if not training_data or len(training_data) < 100:
                logger.warning("Insufficient training data for model training")
                return {"success": False, "error": "Insufficient training data"}
                
            # Prepare data
            df = pd.DataFrame(training_data)
            
            # Ensure all feature columns exist
            for col in self.config["feature_columns"]:
                if col not in df.columns:
                    df[col] = 0.5 if 'score' in col or 'rating' in col or 'factor' in col else 0
                    
            X = df[self.config["feature_columns"]].fillna(0.5)
            
            # Create target variable (1 for headshots, 0 for body shots as example)
            y = df.get('is_headshot', pd.Series([0] * len(df))).astype(int)
            
            # Backup existing model if it exists
            if incremental:
                self._backup_existing_model()
                
            # Train model
            results = self._train_sklearn_model(X, y, incremental)
            
            # Update training history
            self._update_training_history(results)
            
            logger.info(f"Model training completed: {results.get('accuracy', 'N/A')} accuracy")
            return results
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"success": False, "error": str(e)}
            
    def _load_all_training_data(self) -> List[Dict]:
        """Load all available training data"""
        all_data = []
        
        try:
            # Load master dataset
            master_file = os.path.join(self.training_data_dir, "complete_kill_analysis.json")
            if os.path.exists(master_file):
                with open(master_file, 'r') as f:
                    master_data = json.load(f)
                    all_data.extend(master_data)
                    logger.info(f"Loaded {len(master_data)} samples from master dataset")
                    
            # Load additional training batches
            for file in os.listdir(self.training_data_dir):
                if file.startswith("training_batch_") and file.endswith(".json"):
                    filepath = os.path.join(self.training_data_dir, file)
                    try:
                        with open(filepath, 'r') as f:
                            batch_data = json.load(f)
                            all_data.extend(batch_data)
                    except Exception as e:
                        logger.warning(f"Could not load training batch {file}: {e}")
                        
            logger.info(f"Total training data loaded: {len(all_data)} samples")
            return all_data
            
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return []
            
    def _train_sklearn_model(self, X: pd.DataFrame, y: pd.Series, incremental: bool) -> Dict:
        """Train scikit-learn model"""  
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config["validation_split"], random_state=42
            )
            
            model_path = os.path.join(self.models_dir, "observer_ai_model.pkl")
            
            if incremental and os.path.exists(model_path):
                # Load existing model
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                logger.info("Loaded existing model for incremental training")
            else:
                # Create new model
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
                logger.info("Created new Random Forest model")
                
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
            results = {
                "success": True,
                "accuracy": accuracy,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "model_path": model_path,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Model saved to {model_path}")
            
            # Notify the main observer system of model updates
            self._notify_model_update("observer_ai_model", model_path, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"success": False, "error": str(e)}
            
    def _backup_existing_model(self):
        """Backup the existing model before updating"""
        model_path = os.path.join(self.models_dir, "observer_ai_model.pkl")
        
        if os.path.exists(model_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"observer_ai_model_backup_{timestamp}.pkl"
            backup_path = os.path.join(self.backup_dir, backup_name)
            
            try:
                shutil.copy2(model_path, backup_path)
                logger.info(f"Model backed up to {backup_path}")
            except Exception as e:
                logger.warning(f"Could not backup model: {e}")
                
    def _update_training_history(self, results: Dict):
        """Update training history with new session"""
        session = {
            "timestamp": datetime.now().isoformat(),
            "training_results": results,
            "samples_used": results.get("training_samples", 0),
            "demos_processed": results.get("demos_processed", []),
            "kills_processed": results.get("kills_processed", 0)
        }
        
        self.training_history["sessions"].append(session)
        
        # Update cumulative counters
        if "demos_processed" in results:
            self.training_history["total_demos_processed"] += len(results["demos_processed"])
        
        if "kills_processed" in results:
            self.training_history["total_kills_analyzed"] += results["kills_processed"]
        
        # Add model version if this was a training session
        if "accuracy" in results:
            self.training_history["model_versions"].append({
                "timestamp": session["timestamp"],
                "accuracy": results.get("accuracy", 0),
                "samples": results.get("training_samples", 0)
            })
        
        self._save_training_history()
        
    def auto_integrate_model(self) -> bool:
        """
        Ensure the newly trained model is automatically used by the main system
        """
        try:
            model_path = os.path.join(self.models_dir, "observer_ai_model.pkl")
            
            if not os.path.exists(model_path):
                logger.warning("No trained model found to integrate")
                return False
                
            # The model is already in the correct location for the main system to use
            # Just verify it's accessible
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                
            logger.info("Model integration verified - main system will use updated model")
            return True
            
        except Exception as e:
            logger.error(f"Model integration failed: {e}")
            return False
            
    def get_training_stats(self) -> Dict:
        """Get comprehensive training statistics including historical dataset"""
        # Calculate current training samples from sessions
        current_training_samples = 0
        sessions = self.training_history.get("sessions", [])
        for session in sessions:
            training_results = session.get("training_results", {})
            current_training_samples += training_results.get("training_samples", 0)
        
        # Get current training batch statistics
        current_stats = {
            "total_sessions": len(sessions),
            "current_demos_processed": self.training_history.get("total_demos_processed", 0),
            "current_kills_analyzed": self.training_history.get("total_kills_analyzed", 0),
            "current_training_samples": current_training_samples,
            "model_versions": len(self.training_history.get("model_versions", [])),
            "latest_accuracy": 0,
            "training_data_files": [],
            "model_files": []
        }
        
        # Get historical dataset statistics
        historical_stats = self._get_complete_dataset_stats()
        kill_analysis_stats = self._get_kill_analysis_stats()
        
        # Use the larger dataset (usually kill analysis has more comprehensive data)
        historical_demos = max(historical_stats["demos"], kill_analysis_stats["demos_analyzed"])
        historical_kills = max(historical_stats["kills"], kill_analysis_stats["total_kills"])
        historical_training_samples = historical_stats["training_samples"]
        
        # Combine statistics
        stats = {
            **current_stats,
            "historical_demos": historical_demos,
            "historical_kills": historical_kills,
            "historical_training_samples": historical_training_samples,
            "total_demos_processed": current_stats["current_demos_processed"] + historical_demos,
            "total_kills_analyzed": current_stats["current_kills_analyzed"] + historical_kills,
            "total_training_samples": historical_training_samples + current_stats["current_training_samples"]
        }
        
        # Get latest accuracy
        if self.training_history.get("model_versions"):
            stats["latest_accuracy"] = self.training_history["model_versions"][-1].get("accuracy", 0)
            
        # Count training data files
        if os.path.exists(self.training_data_dir):
            stats["training_data_files"] = [f for f in os.listdir(self.training_data_dir) if f.endswith('.json')]
            
        # Count model files
        if os.path.exists(self.models_dir):
            stats["model_files"] = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl')]
            
        return stats

    def _get_complete_dataset_stats(self) -> Dict:
        """Get statistics from the complete positional dataset"""
        base_dir = get_base_dir()
        dataset_file = os.path.join(base_dir, 'complete_positional_dataset_all_85_demos.json')
        
        self.logger.info(f"Looking for dataset file at: {dataset_file}")
        self.logger.info(f"Base directory: {base_dir}")
        
        if not os.path.exists(dataset_file):
            self.logger.warning(f"Complete dataset file not found: {dataset_file}")
            # List files in base directory for debugging
            try:
                files = os.listdir(base_dir)
                json_files = [f for f in files if f.endswith('.json')]
                self.logger.info(f"JSON files in base directory: {json_files}")
            except Exception as e:
                self.logger.error(f"Error listing base directory: {e}")
            return {"demos": 0, "kills": 0, "training_samples": 0}
            
        try:
            with open(dataset_file, 'r') as f:
                dataset = json.load(f)
                
            training_samples = dataset.get('training_samples', [])
            demo_count = len(training_samples)
            total_training_samples = dataset.get('total_training_samples', 0)
            
            # Calculate total kills from all demos
            total_kills = 0
            for demo in training_samples:
                total_kills += demo.get('kills', 0)
                
            self.logger.info(f"Historical dataset: {demo_count} demos, {total_kills} kills, {total_training_samples} training samples")
            
            return {
                "demos": demo_count,
                "kills": total_kills,
                "training_samples": total_training_samples
            }
            
        except Exception as e:
            self.logger.error(f"Error reading complete dataset: {e}")
            return {"demos": 0, "kills": 0, "training_samples": 0}
            
    def _get_kill_analysis_stats(self) -> Dict:
        """Get statistics from the complete kill analysis"""
        base_dir = get_base_dir()
        kill_analysis_file = os.path.join(base_dir, 'complete_kill_analysis.json')
        
        self.logger.info(f"Looking for kill analysis file at: {kill_analysis_file}")
        
        if not os.path.exists(kill_analysis_file):
            self.logger.warning(f"Kill analysis file not found: {kill_analysis_file}")
            return {"total_kills": 0, "demos_analyzed": 0}
            
        try:
            with open(kill_analysis_file, 'r') as f:
                data = json.load(f)
                
            metadata = data.get('metadata', {})
            total_kills = metadata.get('total_kills', 0)
            demos_analyzed = metadata.get('demos_analyzed', 0)
            
            self.logger.info(f"Kill analysis data: {demos_analyzed} demos, {total_kills} kills analyzed")
            
            return {
                "total_kills": total_kills,
                "demos_analyzed": demos_analyzed
            }
            
        except Exception as e:
            self.logger.error(f"Error reading kill analysis: {e}")
            return {"total_kills": 0, "demos_analyzed": 0}
    
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
                
            logger.info(f"Model update notification created: {notification_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create model update notification: {e}")
            return False

def test_ai_trainer():
    """Test the AI trainer system"""
    print("Testing AI Trainer System")
    print("=" * 50)
    
    # Initialize trainer
    trainer = AITrainer()
    
    # Get stats
    stats = trainer.get_training_stats()
    print(f"Training Sessions: {stats['total_sessions']}")
    print(f"Demos Processed: {stats['total_demos_processed']}")
    print(f"Kills Analyzed: {stats['total_kills_analyzed']}")
    print(f"Model Versions: {stats['model_versions']}")
    print(f"Latest Accuracy: {stats['latest_accuracy']:.3f}")
    print(f"Data Files: {len(stats['training_data_files'])}")
    print(f"Model Files: {len(stats['model_files'])}")
    
    return True

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Train with specific demo file
        demo_file = sys.argv[1]
        if os.path.exists(demo_file):
            print(f"Training AI with demo: {os.path.basename(demo_file)}")
            trainer = AITrainer()
            result = trainer.process_new_demos([demo_file])
            print(f"✓ Processed {result['processed']} demos")
            print(f"✓ Extracted {result['new_kills']} kills")
            if result['failed'] > 0:
                print(f"! {result['failed']} demos failed to process")
            if result.get('errors'):
                for error in result['errors']:
                    print(f"Error: {error}")
        else:
            print(f"Error: Demo file not found: {demo_file}")
    else:
        test_ai_trainer()