#!/usr/bin/env python3
"""
Enhanced Observer AI - Advanced Decision Making System
Combines multiple AI systems for intelligent camera control and player selection
"""

import json
import numpy as np
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import math
from collections import defaultdict, deque

try:
    from demo_analyzer import DemoAnalyzer
    from pattern_memory_ai import PatternMemoryAI
    from kill_pattern_analyzer import KillPatternAnalyzer
except ImportError:
    # Fallback imports
    DemoAnalyzer = None
    PatternMemoryAI = None
    KillPatternAnalyzer = None

class EnhancedObserverAI:
    """
    Enhanced Observer AI - Master AI Controller
    
    Integrates multiple AI systems:
    - Demo Analyzer (27-factor analysis)
    - Pattern Memory AI (behavior learning)
    - Kill Pattern Analyzer (combat prediction)
    - Positional Intelligence
    - Team coordination analysis
    """
    
    def __init__(self):
        self.demo_analyzer = None
        self.pattern_memory = None
        self.kill_analyzer = None
        
        # Initialize AI components
        self._initialize_ai_components()
        
        # Decision history for learning
        self.decision_history = deque(maxlen=100)
        self.observation_metrics = {
            'total_decisions': 0,
            'successful_predictions': 0,
            'kill_predictions': 0,
            'correct_kill_predictions': 0,
            'camera_switches': 0,
            'avg_decision_time': 0.0
        }
        
        # Enhanced decision weights
        self.decision_weights = {
            'demo_analyzer': 0.35,      # 27-factor analysis
            'pattern_memory': 0.25,     # Player behavior patterns  
            'kill_analyzer': 0.25,      # Kill prediction
            'rule_based': 0.15          # Traditional rules
        }
        
        # Load configuration
        self.load_configuration()
        
    def _initialize_ai_components(self):
        """Initialize all AI components"""
        try:
            # Initialize Demo Analyzer
            if DemoAnalyzer:
                self.demo_analyzer = DemoAnalyzer()
                self.demo_analyzer.load_model("observer_ai_model.pkl")
                logging.info("✅ Demo Analyzer initialized")
            else:
                logging.warning("⚠️ Demo Analyzer not available")
                
            # Initialize Pattern Memory AI
            if PatternMemoryAI:
                self.pattern_memory = PatternMemoryAI()
                logging.info("✅ Pattern Memory AI initialized")
            else:
                logging.warning("⚠️ Pattern Memory AI not available")
                
            # Initialize Kill Pattern Analyzer
            if KillPatternAnalyzer:
                self.kill_analyzer = KillPatternAnalyzer()
                logging.info("✅ Kill Pattern Analyzer initialized")
            else:
                logging.warning("⚠️ Kill Pattern Analyzer not available")
                
        except Exception as e:
            logging.error(f"Error initializing AI components: {e}")
            
    def load_configuration(self):
        """Load AI configuration settings"""
        try:
            config_file = "ai_config.json"
            default_config = {
                "decision_weights": self.decision_weights,
                "prediction_threshold": 0.6,
                "switch_cooldown": 2.0,
                "learning_rate": 0.1,
                "confidence_threshold": 0.7
            }
            
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    self.decision_weights.update(config.get('decision_weights', {}))
                    self.prediction_threshold = config.get('prediction_threshold', 0.6)
                    self.switch_cooldown = config.get('switch_cooldown', 2.0)
                    self.learning_rate = config.get('learning_rate', 0.1)
                    self.confidence_threshold = config.get('confidence_threshold', 0.7)
            else:
                # Create default config
                with open(config_file, 'w') as f:
                    json.dump(default_config, f, indent=2)
                    
                self.prediction_threshold = 0.6
                self.switch_cooldown = 2.0
                self.learning_rate = 0.1
                self.confidence_threshold = 0.7
                
        except Exception as e:
            logging.error(f"Error loading AI configuration: {e}")
            
    def analyze_gamestate(self, gamestate: Dict[str, Any], previous_gamestate: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Master analysis function - combines all AI systems
        """
        start_time = time.time()
        
        analysis_result = {
            'timestamp': start_time,
            'ai_predictions': {},
            'combined_scores': {},
            'best_player': None,
            'confidence': 0.0,
            'reasoning': [],
            'kill_prediction': None,
            'team_analysis': {}
        }
        
        if not gamestate or 'allplayers' not in gamestate:
            return analysis_result
            
        # Update pattern memory with current state
        if self.pattern_memory:
            self.pattern_memory.observe_gamestate(gamestate, previous_gamestate)
            
        # Get base player candidates
        candidates = self._get_player_candidates(gamestate)
        if not candidates:
            return analysis_result
            
        # Run AI analyses
        analysis_result['ai_predictions'] = self._run_ai_analyses(gamestate, candidates)
        
        # Combine AI scores
        combined_scores = self._combine_ai_scores(candidates, analysis_result['ai_predictions'])
        analysis_result['combined_scores'] = combined_scores
        
        # Select best player
        if combined_scores:
            best_player, confidence = max(combined_scores.items(), key=lambda x: x[1])
            analysis_result['best_player'] = best_player
            analysis_result['confidence'] = confidence
            
        # Generate reasoning
        analysis_result['reasoning'] = self._generate_reasoning(analysis_result)
        
        # Predict potential kills
        analysis_result['kill_prediction'] = self._predict_kills(gamestate, candidates)
        
        # Team analysis
        analysis_result['team_analysis'] = self._analyze_teams(gamestate)
        
        # Update metrics
        decision_time = time.time() - start_time
        self.observation_metrics['total_decisions'] += 1
        self.observation_metrics['avg_decision_time'] = (
            (self.observation_metrics['avg_decision_time'] * (self.observation_metrics['total_decisions'] - 1) + 
             decision_time) / self.observation_metrics['total_decisions']
        )
        
        # Store decision in history
        self.decision_history.append(analysis_result)
        
        return analysis_result
        
    def _get_player_candidates(self, gamestate: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get initial player candidates with base scores"""
        candidates = []
        
        allplayers = gamestate.get('allplayers', {})
        
        for player_id, player_data in allplayers.items():
            # Basic eligibility check
            health = player_data.get('state', {}).get('health', 0)
            if health <= 0:
                continue  # Dead players
                
            # Calculate base score
            base_score = self._calculate_base_score(player_data, gamestate)
            candidates.append((player_id, base_score))
            
        # Sort by base score
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:8]  # Top 8 candidates for analysis
        
    def _calculate_base_score(self, player_data: Dict, gamestate: Dict) -> float:
        """Calculate base rule-based score for a player"""
        score = 0.0
        
        # Health factor
        health = player_data.get('state', {}).get('health', 0)
        if health <= 0:
            return 0.0
            
        score += min(health / 100.0, 1.0) * 0.3
        
        # Weapon factor
        weapons = player_data.get('weapons', {})
        has_primary = False
        for weapon in weapons.values():
            weapon_name = weapon.get('name', '').lower()
            if any(primary in weapon_name for primary in ['ak47', 'm4', 'awp', 'aug', 'sg553']):
                score += 0.3
                has_primary = True
                break
                
        if not has_primary:
            score += 0.1  # Some points for secondary weapons
            
        # Money factor (economic importance)
        money = player_data.get('state', {}).get('money', 0)
        score += min(money / 10000.0, 1.0) * 0.2
        
        # Position factor (simplified)
        position = player_data.get('position', '')
        if position:
            score += 0.2  # Base positioning bonus
            
        return min(1.0, score)
        
    def _run_ai_analyses(self, gamestate: Dict[str, Any], candidates: List[Tuple[str, float]]) -> Dict[str, Dict]:
        """Run all available AI analyses"""
        predictions = {
            'demo_analyzer': {},
            'pattern_memory': {},
            'kill_analyzer': {},
            'availability': {}
        }
        
        # Demo Analyzer (27-factor analysis)
        if self.demo_analyzer:
            try:
                best_player, confidence = self.demo_analyzer.predict_best_player(gamestate, candidates)
                if best_player:
                    predictions['demo_analyzer'][best_player] = confidence
                predictions['availability']['demo_analyzer'] = True
            except Exception as e:
                logging.warning(f"Demo analyzer prediction failed: {e}")
                predictions['availability']['demo_analyzer'] = False
        else:
            predictions['availability']['demo_analyzer'] = False
            
        # Pattern Memory AI
        if self.pattern_memory:
            try:
                for player_id, base_score in candidates:
                    enhanced_score = self.pattern_memory.enhance_player_score(player_id, base_score, gamestate)
                    predictions['pattern_memory'][player_id] = enhanced_score
                predictions['availability']['pattern_memory'] = True
            except Exception as e:
                logging.warning(f"Pattern memory prediction failed: {e}")
                predictions['availability']['pattern_memory'] = False
        else:
            predictions['availability']['pattern_memory'] = False
            
        # Kill Pattern Analyzer
        if self.kill_analyzer:
            try:
                kill_predictions = self.kill_analyzer.predict_next_kills(gamestate, candidates)
                for player_id, kill_prob in kill_predictions.items():
                    predictions['kill_analyzer'][player_id] = kill_prob
                predictions['availability']['kill_analyzer'] = True
            except Exception as e:
                logging.warning(f"Kill analyzer prediction failed: {e}")
                predictions['availability']['kill_analyzer'] = False
        else:
            predictions['availability']['kill_analyzer'] = False
            
        return predictions
        
    def _combine_ai_scores(self, candidates: List[Tuple[str, float]], predictions: Dict[str, Dict]) -> Dict[str, float]:
        """Combine scores from all AI systems using weighted average"""
        combined_scores = {}
        
        for player_id, base_score in candidates:
            total_score = 0.0
            total_weight = 0.0
            
            # Demo Analyzer contribution
            if predictions['availability'].get('demo_analyzer', False):
                demo_score = predictions['demo_analyzer'].get(player_id, 0.0)
                total_score += demo_score * self.decision_weights['demo_analyzer']
                total_weight += self.decision_weights['demo_analyzer']
                
            # Pattern Memory contribution
            if predictions['availability'].get('pattern_memory', False):
                pattern_score = predictions['pattern_memory'].get(player_id, base_score)
                total_score += pattern_score * self.decision_weights['pattern_memory']
                total_weight += self.decision_weights['pattern_memory']
                
            # Kill Analyzer contribution
            if predictions['availability'].get('kill_analyzer', False):
                kill_score = predictions['kill_analyzer'].get(player_id, 0.0)
                total_score += kill_score * self.decision_weights['kill_analyzer']
                total_weight += self.decision_weights['kill_analyzer']
                
            # Rule-based contribution (always available)
            total_score += base_score * self.decision_weights['rule_based']
            total_weight += self.decision_weights['rule_based']
            
            # Calculate final score
            if total_weight > 0:
                combined_scores[player_id] = total_score / total_weight
            else:
                combined_scores[player_id] = base_score
                
        return combined_scores
        
    def _predict_kills(self, gamestate: Dict[str, Any], candidates: List[Tuple[str, float]]) -> Optional[Dict]:
        """Predict potential kills in the next few seconds"""
        if not self.kill_analyzer:
            return None
            
        try:
            kill_predictions = self.kill_analyzer.predict_next_kills(gamestate, candidates)
            
            if kill_predictions:
                best_killer = max(kill_predictions.items(), key=lambda x: x[1])
                return {
                    'most_likely_killer': best_killer[0],
                    'kill_probability': best_killer[1],
                    'all_predictions': kill_predictions,
                    'prediction_window': 5.0  # 5 second prediction window
                }
        except Exception as e:
            logging.warning(f"Kill prediction failed: {e}")
            
        return None
        
    def _analyze_teams(self, gamestate: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze team status and coordination"""
        team_analysis = {
            'team_scores': {},
            'team_coordination': {},
            'team_economy': {},
            'team_positioning': {}
        }
        
        if not self.pattern_memory:
            return team_analysis
            
        # Group players by team
        teams = defaultdict(list)
        for player_id, player_data in gamestate.get('allplayers', {}).items():
            team = player_data.get('team', '')
            if team:
                teams[team].append((player_id, player_data))
                
        # Analyze each team
        for team, players in teams.items():
            if not players:
                continue
                
            # Team coordination score
            coord_score = self.pattern_memory.get_team_coordination_score(team)
            team_analysis['team_coordination'][team] = coord_score
            
            # Team economy
            total_money = sum(p[1].get('state', {}).get('money', 0) for p in players)
            avg_money = total_money / len(players) if players else 0
            team_analysis['team_economy'][team] = {
                'total_money': total_money,
                'avg_money': avg_money,
                'player_count': len(players)
            }
            
            # Team health/alive count
            alive_players = [p for p in players if p[1].get('state', {}).get('health', 0) > 0]
            team_analysis['team_scores'][team] = {
                'alive_count': len(alive_players),
                'total_players': len(players),
                'avg_health': sum(p[1].get('state', {}).get('health', 0) for p in alive_players) / max(len(alive_players), 1)
            }
            
        return team_analysis
        
    def _generate_reasoning(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate human-readable reasoning for the decision"""
        reasoning = []
        
        best_player = analysis_result.get('best_player')
        confidence = analysis_result.get('confidence', 0)
        predictions = analysis_result.get('ai_predictions', {})
        
        if not best_player:
            reasoning.append("No suitable player found")
            return reasoning
            
        # Confidence explanation
        if confidence > 0.8:
            reasoning.append(f"High confidence selection ({confidence:.2f})")
        elif confidence > 0.6:
            reasoning.append(f"Moderate confidence selection ({confidence:.2f})")
        else:
            reasoning.append(f"Low confidence selection ({confidence:.2f})")
            
        # AI system contributions
        availability = predictions.get('availability', {})
        
        if availability.get('demo_analyzer'):
            demo_score = predictions.get('demo_analyzer', {}).get(best_player, 0)
            if demo_score > 0.7:
                reasoning.append("27-factor analysis strongly favors this player")
            elif demo_score > 0.5:
                reasoning.append("Positional analysis supports this choice")
                
        if availability.get('pattern_memory'):
            reasoning.append("Player behavior patterns analyzed")
            
        if availability.get('kill_analyzer'):
            kill_score = predictions.get('kill_analyzer', {}).get(best_player, 0)
            if kill_score > 0.6:
                reasoning.append("High kill probability predicted")
                
        # Kill prediction info
        kill_pred = analysis_result.get('kill_prediction')
        if kill_pred and kill_pred.get('most_likely_killer') == best_player:
            prob = kill_pred.get('kill_probability', 0)
            reasoning.append(f"Predicted to get kill soon ({prob:.1%} chance)")
            
        return reasoning
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'ai_components': {
                'demo_analyzer': self.demo_analyzer is not None,
                'pattern_memory': self.pattern_memory is not None,
                'kill_analyzer': self.kill_analyzer is not None
            },
            'decision_weights': self.decision_weights,
            'metrics': self.observation_metrics,
            'recent_decisions': len(self.decision_history),
            'system_uptime': time.time()
        }

def test_enhanced_observer():
    """Test the enhanced observer AI system"""
    print("🤖 Testing Enhanced Observer AI System")
    print("=" * 60)
    
    ai = EnhancedObserverAI()
    status = ai.get_system_status()
    
    print("AI Components Status:")
    for component, available in status['ai_components'].items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {component}")
        
    print(f"\nDecision Weights:")
    for system, weight in status['decision_weights'].items():
        print(f"  {system}: {weight:.2f}")
        
    # Test with sample gamestate
    sample_gamestate = {
        'map': {'round_time': 75.5, 'name': 'de_dust2'},
        'provider': {'timestamp': 12345},
        'allplayers': {
            'player1': {
                'position': '-500.0, 1200.0, 64.0',
                'forward': '0.8, 0.6, 0.0',
                'team': 'CT',
                'state': {'health': 85, 'armor': 50, 'money': 3500},
                'weapons': {'0': {'name': 'AK-47'}}
            },
            'player2': {
                'position': '-300.0, 1400.0, 64.0', 
                'forward': '-0.7, -0.7, 0.0',
                'team': 'TERRORIST',
                'state': {'health': 90, 'armor': 100, 'money': 2800},
                'weapons': {'0': {'name': 'AWP'}}
            }
        }
    }
    
    print(f"\n🎯 Running AI Analysis...")
    analysis = ai.analyze_gamestate(sample_gamestate)
    
    print(f"Best Player: {analysis.get('best_player', 'None')}")
    print(f"Confidence: {analysis.get('confidence', 0):.3f}")
    
    print(f"\nReasoning:")
    for reason in analysis.get('reasoning', []):
        print(f"  • {reason}")
        
    kill_pred = analysis.get('kill_prediction')
    if kill_pred:
        print(f"\nKill Prediction:")
        print(f"  Most likely killer: {kill_pred.get('most_likely_killer', 'None')}")
        print(f"  Probability: {kill_pred.get('kill_probability', 0):.1%}")
        
    print(f"\nSystem Performance:")
    metrics = status['metrics']
    print(f"  Total decisions: {metrics['total_decisions']}")
    print(f"  Avg decision time: {metrics['avg_decision_time']:.4f}s")
    
    return True

if __name__ == "__main__":
    test_enhanced_observer()