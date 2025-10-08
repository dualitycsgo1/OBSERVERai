#!/usr/bin/env python3
"""
CS2 Predictive Observer Enhancement
Integrates kill pattern analysis with existing observer system for optimal timing
"""
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import threading
import queue
import json

logger = logging.getLogger(__name__)

@dataclass
class PredictionAlert:
    """Alert for predicted kill event"""
    timestamp: float
    player_name: str
    confidence: float
    nav_area: Optional[int]
    position: Tuple[float, float, float]
    predicted_kill_time: float  # When kill is expected
    context: str  # 'duel_setup', 'pre_aim', 'enemy_spotted', etc.
    priority: int  # 1-10, higher = more important

class PredictiveObserver:
    """Enhanced observer that uses kill pattern analysis for better timing"""
    
    def __init__(self, existing_detector=None, kill_analyzer=None):
        self.existing_detector = existing_detector
        self.kill_analyzer = kill_analyzer
        
        # Prediction settings
        self.prediction_window = 4.0  # Look ahead 4 seconds
        self.confidence_threshold = 0.65  # Minimum confidence for camera switch
        self.high_confidence_threshold = 0.85  # Immediate switch threshold
        
        # Prediction state
        self.active_predictions = {}  # player_name -> PredictionAlert
        self.prediction_history = []
        self.last_prediction_time = 0
        
        # Camera control
        self.camera_lock_duration = 2.0  # How long to lock camera on high-confidence prediction
        self.camera_locked_until = 0
        self.current_focus_player = None
        
        # Performance tracking
        self.prediction_stats = {
            'total_predictions': 0,
            'correct_predictions': 0,
            'false_positives': 0,
            'missed_kills': 0,
            'avg_prediction_time': 0.0
        }
        
        logger.info("🔮 Predictive Observer initialized")
        
    def analyze_current_gamestate(self, allplayers: Dict[str, Any], map_name: str) -> List[PredictionAlert]:
        """Analyze current gamestate for kill predictions"""
        current_time = time.time()
        predictions = []
        
        if not self.kill_analyzer or not allplayers:
            return predictions
            
        # Analyze each player for kill potential
        players = [(sid, data) for sid, data in allplayers.items() if sid != 'total']
        
        for player_id, player_data in players:
            if player_data.get('state', {}).get('health', 0) <= 0:
                continue
                
            # Get enemy players for context
            player_team = player_data.get('team', '')
            enemies = [p for _, p in players if p.get('team', '') != player_team and p.get('state', {}).get('health', 0) > 0]
            
            # Predict kill probability
            try:
                kill_prob = self.kill_analyzer.predict_kill_probability(
                    player_data, enemies, map_name, current_time
                )
                
                if kill_prob >= self.confidence_threshold:
                    position = self.parse_position(player_data.get('position', ''))
                    nav_area = self.kill_analyzer.get_nav_area(map_name, position) if position else None
                    
                    # Determine context and priority
                    context, priority = self.analyze_kill_context(player_data, enemies, kill_prob)
                    
                    prediction = PredictionAlert(
                        timestamp=current_time,
                        player_name=player_data.get('name', 'Unknown'),
                        confidence=kill_prob,
                        nav_area=nav_area,
                        position=position,
                        predicted_kill_time=current_time + 2.5,  # Predict kill in 2.5 seconds
                        context=context,
                        priority=priority
                    )
                    
                    predictions.append(prediction)
                    
            except Exception as e:
                logger.error(f"Error predicting for {player_data.get('name', '?')}: {e}")
                
        return predictions
        
    def analyze_kill_context(self, player_data: Dict, enemies: List[Dict], probability: float) -> Tuple[str, int]:
        """Analyze context around potential kill for priority scoring"""
        context = "unknown"
        priority = 5  # Default medium priority
        
        # Check weapon type for context
        weapon = self.get_primary_weapon(player_data)
        if 'awp' in weapon.lower():
            context = "awp_angle"
            priority = 8  # AWP kills are high priority
        elif 'ak47' in weapon.lower() or 'm4' in weapon.lower():
            context = "rifle_duel"
            priority = 7
        elif 'deagle' in weapon.lower():
            context = "deagle_pick"
            priority = 6
            
        # Check positioning context
        position = self.parse_position(player_data.get('position', ''))
        if position:
            # TODO: Use NAV area analysis to determine if player is in:
            # - Common angle positions (higher priority)
            # - Retake positions (medium priority)  
            # - Rotation positions (lower priority)
            pass
            
        # Adjust priority based on confidence
        if probability >= self.high_confidence_threshold:
            priority += 2
        elif probability >= 0.75:
            priority += 1
            
        return context, min(priority, 10)
        
    def should_switch_camera(self, predictions: List[PredictionAlert]) -> Optional[str]:
        """Determine if camera should switch and to which player"""
        current_time = time.time()
        
        # Respect camera lock
        if current_time < self.camera_locked_until:
            return None
            
        if not predictions:
            return None
            
        # Sort predictions by priority and confidence
        sorted_predictions = sorted(predictions, key=lambda p: (p.priority, p.confidence), reverse=True)
        
        best_prediction = sorted_predictions[0]
        
        # High confidence predictions trigger immediate switch
        if best_prediction.confidence >= self.high_confidence_threshold:
            self.camera_locked_until = current_time + self.camera_lock_duration
            self.current_focus_player = best_prediction.player_name
            
            logger.info(f"🎯 HIGH CONFIDENCE switch to {best_prediction.player_name} "
                       f"(confidence: {best_prediction.confidence:.3f}, context: {best_prediction.context})")
            
            return best_prediction.player_name
            
        # Medium confidence requires no other high priority activity
        elif best_prediction.confidence >= self.confidence_threshold:
            # Check if current player still has potential
            current_player_prediction = None
            if self.current_focus_player:
                current_player_prediction = next(
                    (p for p in predictions if p.player_name == self.current_focus_player), None
                )
                
            # Switch if current player has lower priority/confidence
            if (not current_player_prediction or 
                best_prediction.priority > current_player_prediction.priority or
                best_prediction.confidence > current_player_prediction.confidence + 0.1):
                
                self.current_focus_player = best_prediction.player_name
                
                logger.info(f"🎯 Predictive switch to {best_prediction.player_name} "
                           f"(confidence: {best_prediction.confidence:.3f}, context: {best_prediction.context})")
                
                return best_prediction.player_name
                
        return None
        
    def update_existing_detector(self, allplayers: Dict[str, Any], map_name: str) -> Optional[str]:
        """Main integration point with existing CS2DuelDetector"""
        
        # Run prediction analysis
        predictions = self.analyze_current_gamestate(allplayers, map_name)
        
        # Update prediction tracking
        self.active_predictions = {p.player_name: p for p in predictions}
        self.prediction_history.extend(predictions)
        
        # Keep history manageable
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-500:]
            
        # Determine camera switch
        suggested_player = self.should_switch_camera(predictions)
        
        if suggested_player:
            self.prediction_stats['total_predictions'] += 1
            
        return suggested_player
        
    def parse_position(self, pos_str: str) -> Optional[Tuple[float, float, float]]:
        """Parse position string to coordinates"""
        try:
            coords = pos_str.split()
            return (float(coords[0]), float(coords[1]), float(coords[2]))
        except:
            return None
            
    def get_primary_weapon(self, player_data: Dict) -> str:
        """Get player's primary weapon"""
        weapons = player_data.get('weapons', {})
        for weapon_name, weapon_data in weapons.items():
            if weapon_data.get('state') == 'active':
                return weapon_name
        return 'unknown'
        
    def validate_prediction(self, player_name: str, actual_kill_time: float):
        """Validate a prediction against actual kill event"""
        if player_name not in self.active_predictions:
            self.prediction_stats['missed_kills'] += 1
            return
            
        prediction = self.active_predictions[player_name]
        time_diff = abs(actual_kill_time - prediction.predicted_kill_time)
        
        if time_diff <= 3.0:  # Within 3 seconds = correct
            self.prediction_stats['correct_predictions'] += 1
            logger.info(f"✅ Prediction CORRECT: {player_name} killed within {time_diff:.1f}s of prediction")
        else:
            self.prediction_stats['false_positives'] += 1
            logger.info(f"❌ Prediction WRONG: {player_name} kill timing off by {time_diff:.1f}s")
            
    def get_prediction_stats(self) -> Dict[str, Any]:
        """Get prediction performance statistics"""
        total = self.prediction_stats['total_predictions']
        if total == 0:
            return self.prediction_stats
            
        accuracy = self.prediction_stats['correct_predictions'] / total
        
        return {
            **self.prediction_stats,
            'accuracy': accuracy,
            'precision': accuracy,  # Same for this use case
            'prediction_rate': len(self.active_predictions)
        }
        
    def log_performance_summary(self):
        """Log prediction performance summary"""
        stats = self.get_prediction_stats()
        
        logger.info("🔮 Predictive Observer Performance Summary:")
        logger.info(f"   Total Predictions: {stats['total_predictions']}")
        logger.info(f"   Correct: {stats['correct_predictions']}")
        logger.info(f"   Accuracy: {stats.get('accuracy', 0.0):.1%}")
        logger.info(f"   Active Predictions: {len(self.active_predictions)}")

def integrate_with_existing_detector():
    """Example integration with existing CS2DuelDetector"""
    
    print("🔧 Integrating Predictive Observer with existing CS2DuelDetector")
    print("=" * 70)
    
    try:
        # Initialize components
        from kill_pattern_analyzer import DemoKillAnalyzer
        
        kill_analyzer = DemoKillAnalyzer()
        predictive_observer = PredictiveObserver(kill_analyzer=kill_analyzer)
        
        print("✅ Kill Pattern Analyzer initialized")
        print("✅ Predictive Observer initialized")
        
        # Example integration code for CS2DuelDetector
        integration_code = '''
# In your CS2DuelDetector class, add these methods:

def __init__(self, ...):
    # ... existing initialization ...
    
    # Add predictive observer
    try:
        from kill_pattern_analyzer import DemoKillAnalyzer
        from predictive_observer import PredictiveObserver
        
        self.kill_analyzer = DemoKillAnalyzer()
        self.predictive_observer = PredictiveObserver(
            existing_detector=self, 
            kill_analyzer=self.kill_analyzer
        )
        self.use_predictive_observer = True
        logging.info("🔮 Predictive Observer enabled")
    except Exception as e:
        logging.warning(f"Predictive Observer disabled: {e}")
        self.use_predictive_observer = False

def process_gamestate(self, allplayers):
    # ... existing processing ...
    
    # Add predictive analysis
    if self.use_predictive_observer:
        suggested_player = self.predictive_observer.update_existing_detector(
            allplayers, self.current_map
        )
        
        if suggested_player:
            # Override normal duel detection for high-confidence predictions
            logging.info(f"🔮 Predictive override: switching to {suggested_player}")
            self.switch_to_player(suggested_player)
            return
            
    # ... continue with existing duel detection logic ...
        '''
        
        print("\n📝 Integration Code:")
        print(integration_code)
        
        print("\n🎯 Benefits of Predictive Observer:")
        print("   ✅ 2-4 second advance warning before kills")
        print("   ✅ NAV-area aware kill prediction")
        print("   ✅ Weapon-context optimized switching")
        print("   ✅ Learns from demo patterns")
        print("   ✅ Performance tracking and validation")
        
        return True
        
    except Exception as e:
        logger.error(f"Integration failed: {e}")
        return False

if __name__ == "__main__":
    integrate_with_existing_detector()