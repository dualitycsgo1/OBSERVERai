#!/usr/bin/env python3
"""
Kill Pattern Analyzer - Advanced Combat Prediction System
Analyzes kill patterns, combat situations, and predicts upcoming kills
"""

import json
import numpy as np
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict, deque
import time
import math

class KillPatternAnalyzer:
    """
    Advanced Kill Pattern Analysis System
    
    Features:
    - Combat situation recognition
    - Kill probability prediction
    - Weapon effectiveness analysis
    - Engagement outcome prediction
    - Combat timing analysis
    - Multi-kill sequence detection
    """
    
    def __init__(self):
        self.kill_patterns = {}
        self.weapon_effectiveness = {}
        self.engagement_history = deque(maxlen=500)
        self.kill_predictions = deque(maxlen=100)
        
        # Combat analysis parameters
        self.engagement_distance_threshold = 2000  # Units
        self.kill_prediction_window = 5.0  # Seconds
        self.multi_kill_window = 10.0  # Seconds
        
        # Load kill analysis data
        self.load_kill_analysis_data()
        
        # Initialize combat models
        self.combat_models = {
            'close_range': {'distance': (0, 500), 'weapons': ['knife', 'shotgun', 'smg'], 'kill_rate': 0.65},
            'medium_range': {'distance': (500, 1500), 'weapons': ['rifle', 'smg'], 'kill_rate': 0.55},
            'long_range': {'distance': (1500, 3000), 'weapons': ['awp', 'rifle'], 'kill_rate': 0.45},
            'extreme_range': {'distance': (3000, 5000), 'weapons': ['awp'], 'kill_rate': 0.35}
        }
        
    def load_kill_analysis_data(self):
        """Load kill analysis dataset"""
        try:
            # Try to load complete kill analysis
            kill_analysis_file = "complete_kill_analysis.json"
            if os.path.exists(kill_analysis_file):
                with open(kill_analysis_file, 'r', encoding='utf-8') as f:
                    kill_data = json.load(f)
                    self._process_kill_data(kill_data)
                    logging.info(f"✅ Kill analysis data loaded: {len(kill_data.get('kills', []))} kills")
            else:
                # Fallback to generating from positional data
                self._generate_kill_patterns_from_positional_data()
                
        except Exception as e:
            logging.warning(f"Could not load kill analysis data: {e}")
            self._initialize_default_patterns()
            
    def _process_kill_data(self, kill_data: Dict):
        """Process loaded kill data into patterns"""
        kills = kill_data.get('kills', [])
        
        # Weapon effectiveness analysis
        weapon_stats = defaultdict(lambda: {
            'kills': 0, 'attempts': 0, 'headshots': 0,
            'distances': [], 'kill_times': []
        })
        
        # Engagement patterns
        engagement_patterns = defaultdict(lambda: {
            'total': 0, 'successful': 0, 'avg_distance': 0,
            'avg_time_to_kill': 0, 'multi_kills': 0
        })
        
        for kill in kills:
            weapon = kill.get('weapon', 'unknown').lower()
            distance = kill.get('distance', 0)
            headshot = kill.get('headshot', False)
            kill_time = kill.get('time_to_kill', 0)
            
            # Update weapon stats
            weapon_stats[weapon]['kills'] += 1
            weapon_stats[weapon]['distances'].append(distance)
            weapon_stats[weapon]['kill_times'].append(kill_time)
            if headshot:
                weapon_stats[weapon]['headshots'] += 1
                
            # Determine engagement type
            engagement_type = self._classify_engagement(distance, weapon)
            engagement_patterns[engagement_type]['total'] += 1
            engagement_patterns[engagement_type]['successful'] += 1
            
        # Calculate weapon effectiveness
        for weapon, stats in weapon_stats.items():
            if stats['kills'] > 0:
                self.weapon_effectiveness[weapon] = {
                    'kill_rate': min(1.0, stats['kills'] / max(stats['attempts'], stats['kills'])),
                    'avg_distance': sum(stats['distances']) / len(stats['distances']),
                    'headshot_rate': stats['headshots'] / stats['kills'],
                    'avg_kill_time': sum(stats['kill_times']) / len(stats['kill_times']) if stats['kill_times'] else 0,
                    'effectiveness_score': self._calculate_weapon_effectiveness(stats)
                }
                
        # Store engagement patterns
        self.kill_patterns = dict(engagement_patterns)
        
    def _generate_kill_patterns_from_positional_data(self):
        """Generate kill patterns from positional dataset"""
        try:
            positional_file = "complete_positional_dataset_all_85_demos.json"
            if not os.path.exists(positional_file):
                return
                
            with open(positional_file, 'r', encoding='utf-8') as f:
                positional_data = json.load(f)
                
            training_samples = positional_data.get('training_samples', [])
            weapon_stats = defaultdict(lambda: {'kills': 0, 'distances': [], 'effectiveness': 0})
            
            for sample in training_samples:
                kill_samples = sample.get('kill_samples', [])
                
                for kill in kill_samples:
                    weapon = kill.get('weapon', 'unknown').lower()
                    distance = kill.get('distance', 0)
                    
                    weapon_stats[weapon]['kills'] += 1
                    weapon_stats[weapon]['distances'].append(distance)
                    
            # Generate effectiveness scores
            for weapon, stats in weapon_stats.items():
                if stats['kills'] > 0:
                    avg_distance = sum(stats['distances']) / len(stats['distances'])
                    kill_count = stats['kills']
                    
                    # Calculate effectiveness based on kills and optimal range
                    effectiveness = min(1.0, kill_count / 100.0)  # Normalize
                    
                    self.weapon_effectiveness[weapon] = {
                        'kill_rate': effectiveness,
                        'avg_distance': avg_distance,
                        'headshot_rate': 0.25,  # Default estimate
                        'avg_kill_time': 1.5,   # Default estimate
                        'effectiveness_score': effectiveness
                    }
                    
            logging.info(f"✅ Generated kill patterns from positional data: {len(self.weapon_effectiveness)} weapons")
            
        except Exception as e:
            logging.warning(f"Could not generate kill patterns from positional data: {e}")
            
    def _initialize_default_patterns(self):
        """Initialize default kill patterns"""
        default_weapons = {
            'ak47': {'kill_rate': 0.7, 'avg_distance': 1200, 'headshot_rate': 0.3, 'effectiveness_score': 0.8},
            'm4a4': {'kill_rate': 0.65, 'avg_distance': 1100, 'headshot_rate': 0.25, 'effectiveness_score': 0.75},
            'm4a1': {'kill_rate': 0.68, 'avg_distance': 1150, 'headshot_rate': 0.28, 'effectiveness_score': 0.77},
            'awp': {'kill_rate': 0.85, 'avg_distance': 2500, 'headshot_rate': 0.6, 'effectiveness_score': 0.9},
            'deagle': {'kill_rate': 0.45, 'avg_distance': 800, 'headshot_rate': 0.4, 'effectiveness_score': 0.5},
            'glock': {'kill_rate': 0.35, 'avg_distance': 600, 'headshot_rate': 0.15, 'effectiveness_score': 0.3},
            'usp': {'kill_rate': 0.4, 'avg_distance': 700, 'headshot_rate': 0.2, 'effectiveness_score': 0.35},
        }
        
        for weapon, stats in default_weapons.items():
            stats['avg_kill_time'] = 1.5
            
        self.weapon_effectiveness = default_weapons
        
    def _classify_engagement(self, distance: float, weapon: str) -> str:
        """Classify engagement type based on distance and weapon"""
        if distance < 500:
            return 'close_range'
        elif distance < 1500:
            return 'medium_range'
        elif distance < 3000:
            return 'long_range'
        else:
            return 'extreme_range'
            
    def _calculate_weapon_effectiveness(self, stats: Dict) -> float:
        """Calculate overall weapon effectiveness score"""
        kills = stats['kills']
        headshot_rate = stats['headshots'] / max(kills, 1)
        avg_distance = sum(stats['distances']) / max(len(stats['distances']), 1)
        
        # Base effectiveness from kill count
        base_effectiveness = min(1.0, kills / 50.0)
        
        # Headshot bonus
        headshot_bonus = headshot_rate * 0.3
        
        # Distance appropriateness (varies by weapon type)
        distance_score = 0.5  # Default neutral
        
        return min(1.0, base_effectiveness + headshot_bonus + distance_score * 0.2)
        
    def analyze_combat_situation(self, gamestate: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current combat situation"""
        
        combat_analysis = {
            'active_engagements': [],
            'potential_engagements': [], 
            'combat_intensity': 0.0,
            'team_advantages': {},
            'high_risk_players': []
        }
        
        if not gamestate or 'allplayers' not in gamestate:
            return combat_analysis
            
        allplayers = gamestate.get('allplayers', {})
        alive_players = {pid: pdata for pid, pdata in allplayers.items() 
                        if pdata.get('state', {}).get('health', 0) > 0}
        
        # Find potential engagements
        engagements = []
        for p1_id, p1_data in alive_players.items():
            for p2_id, p2_data in alive_players.items():
                if p1_id == p2_id or p1_data.get('team') == p2_data.get('team'):
                    continue
                    
                engagement = self._analyze_engagement(p1_id, p1_data, p2_id, p2_data, gamestate)
                if engagement:
                    engagements.append(engagement)
                    
        # Sort engagements by likelihood
        engagements.sort(key=lambda x: x['engagement_probability'], reverse=True)
        
        # Classify engagements
        for engagement in engagements:
            if engagement['engagement_probability'] > 0.7:
                combat_analysis['active_engagements'].append(engagement)
            elif engagement['engagement_probability'] > 0.4:
                combat_analysis['potential_engagements'].append(engagement)
                
        # Calculate combat intensity
        intensity = len(combat_analysis['active_engagements']) * 0.5 + len(combat_analysis['potential_engagements']) * 0.2
        combat_analysis['combat_intensity'] = min(1.0, intensity)
        
        # Identify high-risk players
        for engagement in combat_analysis['active_engagements']:
            if engagement['win_probability'] < 0.3:  # Likely to lose
                combat_analysis['high_risk_players'].append(engagement['player1'])
            if engagement['win_probability'] > 0.7:  # Likely to win
                combat_analysis['high_risk_players'].append(engagement['player2'])
                
        return combat_analysis
        
    def _analyze_engagement(self, p1_id: str, p1_data: Dict, p2_id: str, p2_data: Dict, gamestate: Dict) -> Optional[Dict]:
        """Analyze a potential engagement between two players"""
        
        # Get positions
        p1_pos = p1_data.get('position', '')
        p2_pos = p2_data.get('position', '')
        
        if not p1_pos or not p2_pos:
            return None
            
        try:
            p1_x, p1_y, p1_z = map(float, p1_pos.split(', '))
            p2_x, p2_y, p2_z = map(float, p2_pos.split(', '))
        except:
            return None
            
        # Calculate distance
        distance = math.sqrt((p2_x - p1_x)**2 + (p2_y - p1_y)**2 + (p2_z - p1_z)**2)
        
        if distance > self.engagement_distance_threshold:
            return None  # Too far for engagement
            
        # Analyze weapons
        p1_weapon = self._get_primary_weapon(p1_data)
        p2_weapon = self._get_primary_weapon(p2_data)
        
        # Calculate engagement probability
        engagement_prob = self._calculate_engagement_probability(distance, p1_weapon, p2_weapon)
        
        if engagement_prob < 0.2:
            return None  # Very unlikely engagement
            
        # Calculate win probability for p1
        win_prob = self._calculate_win_probability(p1_data, p2_data, distance, p1_weapon, p2_weapon)
        
        return {
            'player1': p1_id,
            'player2': p2_id,
            'distance': distance,
            'engagement_probability': engagement_prob,
            'win_probability': win_prob,  # For player1
            'p1_weapon': p1_weapon,
            'p2_weapon': p2_weapon,
            'engagement_type': self._classify_engagement(distance, p1_weapon)
        }
        
    def _get_primary_weapon(self, player_data: Dict) -> str:
        """Get player's primary weapon"""
        weapons = player_data.get('weapons', {})
        
        # Priority order: AWP > Rifles > SMGs > Pistols
        weapon_priority = ['awp', 'ak47', 'm4a4', 'm4a1', 'aug', 'sg553', 'famas', 'galil',
                          'mp9', 'mp7', 'ump', 'p90', 'mac10', 'mp5',
                          'deagle', 'usp', 'glock', 'p250', 'tec9', 'cz75']
        
        for weapon in weapons.values():
            weapon_name = weapon.get('name', '').lower()
            for priority_weapon in weapon_priority:
                if priority_weapon in weapon_name:
                    return priority_weapon
                    
        return 'knife'  # Fallback
        
    def _calculate_engagement_probability(self, distance: float, weapon1: str, weapon2: str) -> float:
        """Calculate probability that these players will engage"""
        
        # Base probability from distance
        if distance < 500:
            base_prob = 0.8
        elif distance < 1000:
            base_prob = 0.6
        elif distance < 1500:
            base_prob = 0.4
        else:
            base_prob = 0.2
            
        # Weapon suitability for distance
        w1_suitable = self._is_weapon_suitable_for_distance(weapon1, distance)
        w2_suitable = self._is_weapon_suitable_for_distance(weapon2, distance)
        
        weapon_modifier = (w1_suitable + w2_suitable) / 2.0
        
        return min(1.0, base_prob * (0.7 + 0.3 * weapon_modifier))
        
    def _is_weapon_suitable_for_distance(self, weapon: str, distance: float) -> float:
        """Check if weapon is suitable for given distance"""
        
        weapon_ranges = {
            'awp': (1000, 4000),
            'ak47': (500, 2500), 'm4a4': (500, 2500), 'm4a1': (500, 2500),
            'aug': (600, 2800), 'sg553': (600, 2800),
            'famas': (300, 1800), 'galil': (300, 1800),
            'mp9': (100, 800), 'mp7': (100, 800), 'ump': (100, 1000), 'p90': (100, 1200),
            'deagle': (200, 1500), 'usp': (100, 800), 'glock': (100, 600),
            'knife': (0, 200)
        }
        
        if weapon not in weapon_ranges:
            return 0.5  # Default
            
        min_range, max_range = weapon_ranges[weapon]
        
        if min_range <= distance <= max_range:
            return 1.0
        elif distance < min_range:
            return max(0.1, (distance / min_range) * 0.8)
        else:  # distance > max_range
            return max(0.1, (max_range / distance) * 0.8)
            
    def _calculate_win_probability(self, p1_data: Dict, p2_data: Dict, distance: float, 
                                 weapon1: str, weapon2: str) -> float:
        """Calculate win probability for player 1"""
        
        # Health factor
        p1_health = p1_data.get('state', {}).get('health', 100)
        p2_health = p2_data.get('state', {}).get('health', 100)
        health_ratio = p1_health / max(p2_health, 1)
        
        # Weapon effectiveness
        w1_eff = self.weapon_effectiveness.get(weapon1, {}).get('effectiveness_score', 0.5)
        w2_eff = self.weapon_effectiveness.get(weapon2, {}).get('effectiveness_score', 0.5)
        
        # Distance suitability
        w1_suitable = self._is_weapon_suitable_for_distance(weapon1, distance)
        w2_suitable = self._is_weapon_suitable_for_distance(weapon2, distance)
        
        # Combine factors
        weapon_advantage = (w1_eff * w1_suitable) / max((w2_eff * w2_suitable), 0.1)
        
        # Base win probability
        base_prob = 0.5
        
        # Apply modifiers
        health_modifier = (health_ratio - 1.0) * 0.2
        weapon_modifier = (weapon_advantage - 1.0) * 0.3
        
        win_prob = base_prob + health_modifier + weapon_modifier
        
        return max(0.1, min(0.9, win_prob))
        
    def predict_next_kills(self, gamestate: Dict[str, Any], candidates: List[Tuple[str, float]]) -> Dict[str, float]:
        """Predict kill probabilities for candidate players"""
        
        kill_predictions = {}
        
        combat_analysis = self.analyze_combat_situation(gamestate)
        active_engagements = combat_analysis.get('active_engagements', [])
        potential_engagements = combat_analysis.get('potential_engagements', [])
        
        # Analyze each candidate
        for player_id, base_score in candidates:
            kill_probability = 0.0
            
            # Check active engagements
            for engagement in active_engagements:
                if engagement['player1'] == player_id:
                    kill_probability += engagement['win_probability'] * 0.8
                elif engagement['player2'] == player_id:
                    kill_probability += (1.0 - engagement['win_probability']) * 0.8
                    
            # Check potential engagements
            for engagement in potential_engagements:
                if engagement['player1'] == player_id:
                    kill_probability += engagement['win_probability'] * engagement['engagement_probability'] * 0.4
                elif engagement['player2'] == player_id:
                    kill_probability += (1.0 - engagement['win_probability']) * engagement['engagement_probability'] * 0.4
                    
            # Apply weapon and positioning bonuses
            player_data = gamestate.get('allplayers', {}).get(player_id, {})
            if player_data:
                weapon = self._get_primary_weapon(player_data)
                weapon_bonus = self.weapon_effectiveness.get(weapon, {}).get('effectiveness_score', 0.5) * 0.2
                kill_probability += weapon_bonus
                
            kill_predictions[player_id] = min(1.0, kill_probability)
            
        return kill_predictions

def test_kill_pattern_analyzer():
    """Test the kill pattern analyzer"""
    print("⚔️ Testing Kill Pattern Analyzer")
    print("=" * 50)
    
    analyzer = KillPatternAnalyzer()
    
    print(f"✅ Weapon effectiveness data: {len(analyzer.weapon_effectiveness)} weapons")
    print(f"✅ Combat models: {len(analyzer.combat_models)} range categories")
    
    # Show top weapons by effectiveness
    if analyzer.weapon_effectiveness:
        print(f"\nTop weapons by effectiveness:")
        sorted_weapons = sorted(analyzer.weapon_effectiveness.items(), 
                              key=lambda x: x[1].get('effectiveness_score', 0), reverse=True)
        for weapon, stats in sorted_weapons[:5]:
            eff = stats.get('effectiveness_score', 0)
            avg_dist = stats.get('avg_distance', 0)
            print(f"  {weapon}: {eff:.3f} effectiveness, {avg_dist:.0f} avg distance")
    
    # Test combat analysis
    sample_gamestate = {
        'map': {'round_time': 75.5, 'name': 'de_dust2'},
        'allplayers': {
            'player1': {
                'position': '-500.0, 1200.0, 64.0',
                'team': 'CT',
                'state': {'health': 85, 'armor': 50},
                'weapons': {'0': {'name': 'AK-47'}}
            },
            'player2': {
                'position': '-300.0, 1400.0, 64.0', 
                'team': 'TERRORIST',
                'state': {'health': 90, 'armor': 100},
                'weapons': {'0': {'name': 'AWP'}}
            },
            'player3': {
                'position': '-800.0, 1500.0, 64.0',
                'team': 'CT', 
                'state': {'health': 70, 'armor': 0},
                'weapons': {'0': {'name': 'M4A4'}}
            }
        }
    }
    
    print(f"\n⚔️ Combat Analysis:")
    combat = analyzer.analyze_combat_situation(sample_gamestate)
    print(f"  Active engagements: {len(combat['active_engagements'])}")
    print(f"  Potential engagements: {len(combat['potential_engagements'])}")
    print(f"  Combat intensity: {combat['combat_intensity']:.3f}")
    
    # Test kill predictions
    candidates = [('player1', 0.7), ('player2', 0.8), ('player3', 0.6)]
    kill_preds = analyzer.predict_next_kills(sample_gamestate, candidates)
    
    print(f"\n🎯 Kill Predictions:")
    for player_id, prob in kill_preds.items():
        print(f"  {player_id}: {prob:.3f} kill probability")
        
    return True

if __name__ == "__main__":
    test_kill_pattern_analyzer()