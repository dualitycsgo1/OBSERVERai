#!/usr/bin/env python3
"""
CS2 Demo Analyzer - Full AI System with 27 Factors
Advanced positional and contextual analysis for CS2 observer decisions
Based on complete training dataset with 250,230+ samples
"""

import json
import numpy as np
import os
import pickle
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict
import math

class DemoAnalyzer:
    """
    Advanced CS2 Demo Analyzer with full 27-factor analysis
    
    Features:
    - Positional analysis (killer/victim positions, angles)
    - Weapon-specific predictions 
    - Distance categorization (close/medium/long)
    - Height advantage analysis
    - Timing and game state factors
    - Map-specific intelligence
    """
    
    def __init__(self):
        self.models_loaded = False
        self.model = None
        self.feature_columns = [
            "killer_x", "killer_y", "killer_z",
            "victim_x", "victim_y", "victim_z", 
            "distance_3d", "height_difference",
            "killer_yaw", "killer_pitch",
            "game_time", "game_tick",
            "weapon_rifle", "weapon_awp", "weapon_pistol", "weapon_smg", "weapon_shotgun", "weapon_knife",
            "distance_close", "distance_medium", "distance_long",
            "angle_forward", "angle_side", "angle_back",
            "height_advantage", "height_disadvantage",
            "map_encoded"
        ]
        
        # Training data and patterns
        self.training_data = None
        self.weapon_patterns = {}
        self.map_patterns = {}
        self.distance_patterns = {}
        self.angle_patterns = {}
        
        # Load training data if available
        self.load_training_data()
        
    def load_training_data(self):
        """Load the complete training dataset for pattern analysis"""
        try:
            # Try consolidated training data first
            dataset_path = os.path.join("training_data", "complete_kill_analysis.json")
            if not os.path.exists(dataset_path):
                # Fallback to old dataset
                dataset_path = "complete_positional_dataset_all_85_demos.json"
            
            if os.path.exists(dataset_path):
                logging.info("📊 Loading complete training dataset...")
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different data formats
                if isinstance(data, list):
                    # New consolidated format - direct list of training samples
                    self.training_data = {"training_samples": data}
                elif isinstance(data, dict) and "training_samples" in data:
                    # Standard format
                    self.training_data = data
                else:
                    # Old format - wrap in standard structure
                    self.training_data = {"training_samples": data}
                
                # Analyze patterns from training data
                self._analyze_training_patterns()
                logging.info(f"✅ Training data loaded: {len(self.training_data.get('training_samples', []))} samples")
                
        except Exception as e:
            logging.warning(f"Could not load training data: {e}")
            
    def _analyze_training_patterns(self):
        """Analyze patterns from the training data for intelligent predictions"""
        if not self.training_data:
            return
            
        training_samples = self.training_data.get('training_samples', [])
        
        # Weapon effectiveness patterns
        weapon_stats = defaultdict(lambda: {'kills': 0, 'total_distance': 0, 'headshots': 0})
        
        # Map-specific patterns  
        map_stats = defaultdict(lambda: {'kills': 0, 'avg_distance': 0, 'hotspots': []})
        
        # Distance effectiveness
        distance_stats = {'close': 0, 'medium': 0, 'long': 0}
        
        for sample in training_samples:
            kill_samples = sample.get('kill_samples', [])
            
            for kill in kill_samples:
                weapon = kill.get('weapon', 'unknown').lower()
                distance = kill.get('distance', 0)
                headshot = kill.get('headshot', False)
                map_name = sample.get('map_name', 'unknown')
                
                # Weapon analysis
                weapon_stats[weapon]['kills'] += 1
                weapon_stats[weapon]['total_distance'] += distance
                if headshot:
                    weapon_stats[weapon]['headshots'] += 1
                    
                # Map analysis
                map_stats[map_name]['kills'] += 1
                map_stats[map_name]['avg_distance'] += distance
                
                # Distance categorization
                if distance < 500:
                    distance_stats['close'] += 1
                elif distance < 1500:
                    distance_stats['medium'] += 1
                else:
                    distance_stats['long'] += 1
        
        # Store analyzed patterns
        self.weapon_patterns = dict(weapon_stats)
        self.map_patterns = dict(map_stats)
        self.distance_patterns = distance_stats
        
        # Calculate weapon effectiveness scores
        for weapon, stats in self.weapon_patterns.items():
            if stats['kills'] > 0:
                stats['avg_distance'] = stats['total_distance'] / stats['kills']
                stats['headshot_rate'] = stats['headshots'] / stats['kills']
                stats['effectiveness'] = min(1.0, stats['kills'] / 100.0)  # Normalize to 0-1
    
    def load_model(self, model_path: str) -> bool:
        """Load the trained AI model"""
        try:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.models_loaded = True
                logging.info(f"✅ AI model loaded: {model_path}")
                return True
            else:
                # Use pattern-based analysis if no model file
                self.models_loaded = bool(self.training_data)
                logging.info("🧠 Using pattern-based AI analysis")
                return self.models_loaded
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return False
    
    def extract_features(self, gamestate: Dict[str, Any], player_candidates: List[Tuple[str, float]]) -> Dict[str, np.ndarray]:
        """Extract 27 features for each player candidate"""
        
        features = {}
        current_time = gamestate.get('map', {}).get('round_time', 0)
        current_tick = gamestate.get('provider', {}).get('timestamp', 0)
        
        allplayers = gamestate.get('allplayers', {})
        
        for player_id, base_score in player_candidates:
            player_data = allplayers.get(player_id, {})
            if not player_data:
                continue
                
            # Extract positional data
            pos_str = player_data.get('position', '')
            if pos_str:
                try:
                    pos_parts = pos_str.split(', ')
                    killer_x, killer_y, killer_z = map(float, pos_parts)
                except:
                    continue
            else:
                continue
                
            # Extract angle data
            forward_str = player_data.get('forward', '')
            if forward_str:
                try:
                    fwd_parts = forward_str.split(', ')
                    fwd_x, fwd_y, fwd_z = map(float, fwd_parts)
                    killer_yaw = math.atan2(fwd_y, fwd_x)
                    killer_pitch = math.atan2(fwd_z, math.sqrt(fwd_x*fwd_x + fwd_y*fwd_y))
                except:
                    killer_yaw = killer_pitch = 0
            else:
                killer_yaw = killer_pitch = 0
            
            # Analyze potential targets
            best_target_features = None
            best_target_score = 0
            
            for target_id, target_data in allplayers.items():
                if target_id == player_id or target_data.get('team') == player_data.get('team'):
                    continue
                    
                target_pos_str = target_data.get('position', '')
                if not target_pos_str:
                    continue
                    
                try:
                    target_parts = target_pos_str.split(', ')
                    victim_x, victim_y, victim_z = map(float, target_parts)
                except:
                    continue
                
                # Calculate features for this potential engagement
                feature_vector = self._calculate_feature_vector(
                    killer_x, killer_y, killer_z, killer_yaw, killer_pitch,
                    victim_x, victim_y, victim_z,
                    current_time, current_tick,
                    player_data, gamestate
                )
                
                # Score this potential engagement
                engagement_score = self._score_engagement(feature_vector, player_data, target_data)
                
                if engagement_score > best_target_score:
                    best_target_score = engagement_score
                    best_target_features = feature_vector
            
            if best_target_features is not None:
                features[player_id] = best_target_features
                
        return features
    
    def _calculate_feature_vector(self, killer_x: float, killer_y: float, killer_z: float,
                                killer_yaw: float, killer_pitch: float,
                                victim_x: float, victim_y: float, victim_z: float,
                                game_time: float, game_tick: float,
                                player_data: Dict, gamestate: Dict) -> np.ndarray:
        """Calculate the complete 27-feature vector"""
        
        # Distance calculations
        distance_3d = math.sqrt((victim_x - killer_x)**2 + (victim_y - killer_y)**2 + (victim_z - killer_z)**2)
        height_difference = killer_z - victim_z
        
        # Weapon encoding
        weapons = player_data.get('weapons', {})
        weapon_rifle = weapon_awp = weapon_pistol = weapon_smg = weapon_shotgun = weapon_knife = 0
        
        for weapon in weapons.values():
            weapon_name = weapon.get('name', '').lower()
            if any(rifle in weapon_name for rifle in ['ak47', 'm4a4', 'm4a1', 'aug', 'sg553']):
                weapon_rifle = 1
            elif 'awp' in weapon_name:
                weapon_awp = 1
            elif any(pistol in weapon_name for pistol in ['glock', 'usp', 'p250', 'deagle']):
                weapon_pistol = 1
            elif any(smg in weapon_name for smg in ['mp9', 'mp7', 'ump', 'p90']):
                weapon_smg = 1
            elif any(shotgun in weapon_name for shotgun in ['nova', 'xm1014', 'mag7']):
                weapon_shotgun = 1
            elif 'knife' in weapon_name:
                weapon_knife = 1
        
        # Distance categories
        distance_close = 1 if distance_3d < 500 else 0
        distance_medium = 1 if 500 <= distance_3d < 1500 else 0
        distance_long = 1 if distance_3d >= 1500 else 0
        
        # Angle analysis
        dx, dy = victim_x - killer_x, victim_y - killer_y
        target_angle = math.atan2(dy, dx)
        angle_diff = abs(killer_yaw - target_angle)
        angle_diff = min(angle_diff, 2*math.pi - angle_diff)  # Normalize to [0, pi]
        
        angle_forward = 1 if angle_diff < math.pi/4 else 0
        angle_side = 1 if math.pi/4 <= angle_diff < 3*math.pi/4 else 0
        angle_back = 1 if angle_diff >= 3*math.pi/4 else 0
        
        # Height advantage
        height_advantage = 1 if height_difference > 50 else 0
        height_disadvantage = 1 if height_difference < -50 else 0
        
        # Map encoding (simplified)
        map_name = gamestate.get('map', {}).get('name', 'unknown')
        map_encoded = hash(map_name) % 100  # Simple hash-based encoding
        
        # Create feature vector
        features = np.array([
            killer_x, killer_y, killer_z,
            victim_x, victim_y, victim_z,
            distance_3d, height_difference,
            killer_yaw, killer_pitch,
            game_time, game_tick,
            weapon_rifle, weapon_awp, weapon_pistol, weapon_smg, weapon_shotgun, weapon_knife,
            distance_close, distance_medium, distance_long,
            angle_forward, angle_side, angle_back,
            height_advantage, height_disadvantage,
            map_encoded
        ])
        
        return features
    
    def _score_engagement(self, features: np.ndarray, player_data: Dict, target_data: Dict) -> float:
        """Score a potential engagement based on features and patterns"""
        
        if len(features) != 27:
            return 0.0
            
        score = 0.0
        
        # Distance-based scoring
        distance = features[6]  # distance_3d
        if distance < 500:
            score += 0.3  # Close range bonus
        elif distance < 1500:
            score += 0.2  # Medium range
        else:
            score += 0.1  # Long range
            
        # Weapon suitability
        if features[13]:  # AWP
            if distance > 800:
                score += 0.4  # AWP long range bonus
        elif features[12]:  # Rifle
            if 300 < distance < 2000:
                score += 0.3  # Rifle medium-long range bonus
        elif features[14] or features[15]:  # Pistol or SMG
            if distance < 800:
                score += 0.3  # Close range weapons
                
        # Angle advantage
        if features[21]:  # angle_forward
            score += 0.3  # Looking at target
        elif features[23]:  # angle_back
            score -= 0.2  # Looking away penalty
            
        # Height advantage
        if features[24]:  # height_advantage
            score += 0.15
        elif features[25]:  # height_disadvantage
            score -= 0.1
            
        # Health consideration
        player_hp = player_data.get('state', {}).get('health', 100)
        target_hp = target_data.get('state', {}).get('health', 100)
        
        if player_hp > target_hp:
            score += 0.2 * (player_hp - target_hp) / 100.0
        else:
            score -= 0.1 * (target_hp - player_hp) / 100.0
            
        return max(0.0, min(1.0, score))
    
    def predict_best_player(self, gamestate: Dict[str, Any], candidates: List[Tuple[str, float]]) -> Tuple[Optional[str], float]:
        """Predict the best player to observe using 27-factor analysis"""
        
        if not candidates:
            return None, 0.0
            
        if not self.models_loaded and not self.training_data:
            # Fallback to highest base score
            return max(candidates, key=lambda x: x[1])
        
        # Extract features for all candidates
        features = self.extract_features(gamestate, candidates)
        
        best_player = None
        best_score = 0.0
        
        for player_id, base_score in candidates:
            if player_id not in features:
                continue
                
            player_features = features[player_id]
            
            # AI prediction
            if self.model and self.models_loaded:
                try:
                    # Use trained model if available
                    ai_score = self.model.predict_proba([player_features])[0][1]  # Probability of positive class
                except:
                    ai_score = 0.5  # Fallback
            else:
                # Use pattern-based scoring
                ai_score = self._pattern_based_prediction(player_features)
            
            # Combine AI score with base rule-based score
            combined_score = 0.6 * ai_score + 0.4 * base_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_player = player_id
                
        return best_player, best_score
    
    def _pattern_based_prediction(self, features: np.ndarray) -> float:
        """Pattern-based prediction using training data analysis"""
        
        if not self.weapon_patterns:
            return 0.5  # Neutral score
            
        score = 0.5  # Base score
        
        # Weapon effectiveness
        if features[13]:  # AWP
            awp_stats = self.weapon_patterns.get('awp', {})
            if awp_stats.get('effectiveness', 0) > 0.3:
                score += 0.2
        elif features[12]:  # Rifle
            rifle_effectiveness = max(
                self.weapon_patterns.get('ak47', {}).get('effectiveness', 0),
                self.weapon_patterns.get('m4a4', {}).get('effectiveness', 0),
                self.weapon_patterns.get('m4a1', {}).get('effectiveness', 0)
            )
            score += 0.15 * rifle_effectiveness
            
        # Distance patterns
        distance = features[6]
        if distance < 500 and self.distance_patterns.get('close', 0) > 100:
            score += 0.1
        elif 500 <= distance < 1500 and self.distance_patterns.get('medium', 0) > 100:
            score += 0.15
        elif distance >= 1500 and self.distance_patterns.get('long', 0) > 50:
            score += 0.1
            
        return max(0.0, min(1.0, score))

def test_demo_analyzer():
    """Test the demo analyzer system"""
    print("🧠 Testing Demo Analyzer - 27 Factor AI System")
    print("=" * 60)
    
    analyzer = DemoAnalyzer()
    
    print(f"✅ Training data loaded: {analyzer.training_data is not None}")
    print(f"✅ Weapon patterns: {len(analyzer.weapon_patterns)} weapons analyzed")
    print(f"✅ Map patterns: {len(analyzer.map_patterns)} maps analyzed") 
    print(f"✅ Features: {len(analyzer.feature_columns)} factors")
    
    # Test feature extraction
    sample_gamestate = {
        'map': {'round_time': 75.5, 'name': 'de_dust2'},
        'provider': {'timestamp': 12345},
        'allplayers': {
            'player1': {
                'position': '-500.0, 1200.0, 64.0',
                'forward': '0.8, 0.6, 0.0',
                'team': 'CT',
                'state': {'health': 85},
                'weapons': {'0': {'name': 'AK-47'}}
            },
            'player2': {
                'position': '-300.0, 1400.0, 64.0', 
                'forward': '-0.7, -0.7, 0.0',
                'team': 'TERRORIST',
                'state': {'health': 90},
                'weapons': {'0': {'name': 'M4A4'}}
            }
        }
    }
    
    candidates = [('player1', 0.7), ('player2', 0.6)]
    features = analyzer.extract_features(sample_gamestate, candidates)
    
    print(f"\n🎯 Feature extraction test:")
    print(f"   Extracted features for {len(features)} players")
    
    for player_id, feature_vector in features.items():
        print(f"   {player_id}: {len(feature_vector)} features")
        
    # Test prediction
    best_player, confidence = analyzer.predict_best_player(sample_gamestate, candidates)
    print(f"\n🔮 AI Prediction:")
    print(f"   Best player: {best_player}")
    print(f"   Confidence: {confidence:.3f}")
    
    return True

if __name__ == "__main__":
    test_demo_analyzer()