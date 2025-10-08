"""
CS2 OBSERVERai 2.0 (AI Enhanced Duel Detector)
"""

import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional, List, Tuple
import math
import os
import ctypes
import random
from collections import deque, defaultdict
import keyboard  # pip install keyboard
import sys
def get_base_dir():
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable
        return os.path.dirname(sys.executable)
    else:
        # Running as script
        return os.path.dirname(os.path.abspath(__file__))

# Import AI analyzer
try:
    from demo_analyzer import DemoAnalyzer
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    # Create a dummy DemoAnalyzer class for fallback
    class DemoAnalyzer:
        def __init__(self):
            self.models_loaded = False
            self.model = None
        def load_model(self, model_path):
            return False
        def predict_best_player(self, *args, **kwargs):
            return None, 0.0

# Import Line of Sight system
try:
    from line_of_sight import get_geometry_manager, duel_geometrically_possible
    LOS_AVAILABLE = True
except ImportError:
    LOS_AVAILABLE = False
    logging.warning("Line of Sight system not available - may allow duels through walls")

# Import Enhanced Positional AI
try:
    from positional_observer_ai import PositionalObserverAI
    POSITIONAL_AI_AVAILABLE = True
except ImportError:
    POSITIONAL_AI_AVAILABLE = False

# Import Model Update System for hot reloading
try:
    from model_updater import ModelUpdater
    MODEL_UPDATER_AVAILABLE = True
except ImportError:
    MODEL_UPDATER_AVAILABLE = False

class ModelHotReloader:
    """Hot reload models when they are updated"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = os.path.join(get_base_dir(), models_dir)
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
                    logging.info(f"🔄 Hot-reloaded model: {model_name}")
                    
        except Exception as e:
            logging.debug(f"Update check failed: {e}")
            
    def _reload_model(self, model_name):
        """Reload a specific model"""
        # Clear from cache to force reload
        keys_to_remove = [k for k in self.model_cache.keys() if model_name in k]
        for key in keys_to_remove:
            del self.model_cache[key]
            
    def _load_model(self, model_name, model_path):
        """Load model and cache it"""
        try:
            import pickle
            
            if not os.path.exists(model_path):
                return None
                
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            elif model_path.endswith('.joblib'):
                import joblib
                model = joblib.load(model_path)
            else:
                return None
                
            # Cache the model
            cache_key = f"{model_name}_{model_path}"
            self.model_cache[cache_key] = {
                'model': model,
                'loaded_at': datetime.now(),
                'path': model_path
            }
            
            logging.info(f"🤖 Model loaded: {model_name}")
            return model
            
        except Exception as e:
            logging.warning(f"Failed to load model {model_name}: {e}")
            return None

# Windows API setup for keyboard input
user32 = ctypes.windll.user32

# Virtual key codes for numbers 1..0
VK_CODES = {
    1: 0x31, 2: 0x32, 3: 0x33, 4: 0x34, 5: 0x35,
    6: 0x36, 7: 0x37, 8: 0x38, 9: 0x39, 10: 0x30
}
# Virtual key codes for A,Q,E,W,U,I,O,P
HOTKEY_CODES = [0x41, 0x51, 0x45, 0x57, 0x55, 0x49, 0x4F, 0x50]  # Corrected typo in 0x4O to 0x4F
# Virtual key code for "-" (minus/dash) key for campath toggle
VK_MINUS = 0xBD

# Logging setup - Ultra minimal, user-friendly output
# Use user AppData directory for logs to avoid permission issues
import tempfile
log_dir = os.path.join(os.environ.get('APPDATA', tempfile.gettempdir()), 'OBSERVERai')
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'cs2_observer.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Clean format - just the message
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Silence all third-party spam
logging.getLogger('awpy_nav_integration').setLevel(logging.ERROR)
logging.getLogger('kill_pattern_analyzer').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)

# Enable elevation debugging (only for development)
ELEVATION_DEBUG = False  # Set to True for detailed elevation debugging

# ---------------------
# Hjælpefunktioner
# ---------------------

def send_key(vk_code):
    """Send a key press using Windows API (keybd_event for lav-latency enkelttryk)."""
    try:
        user32.keybd_event(vk_code, 0, 0, 0)
        time.sleep(0.035)
        user32.keybd_event(vk_code, 0, 2, 0)
        return True
    except Exception as e:
        pass  # Silently fail keyboard input
        return False

def parse_vec3(s: str) -> Optional[Tuple[float, float, float]]:
    try:
        x, y, z = map(float, s.split(', '))
        return x, y, z
    except:
        return None

def norm2d(v):
    l = math.hypot(v[0], v[1])
    if l == 0: return (0.0, 0.0)
    return (v[0]/l, v[1]/l)

def dot2d(a, b): return a[0]*b[0] + a[1]*b[1]

def distance2d(p1, p2) -> float:
    try:
        return math.hypot(p2[0]-p1[0], p2[1]-p1[1])
    except:
        return float('inf')

def clamp01(x): return max(0.0, min(1.0, x))

def angle_facing_score(player_pos, player_forward, target_pos):
    """
    0..1: 1.0 hvis man kigger direkte mod target (2D), 0 hvis 90+ grader væk.
    """
    try:
        px, py, _ = player_pos
        fx, fy, _ = player_forward
        tx, ty, _ = target_pos
        to_target = (tx - px, ty - py)
        f_n = norm2d((fx, fy))
        t_n = norm2d(to_target)
        dp = dot2d(f_n, t_n)  # [-1..1]
        return max(0.0, dp)
    except:
        return 0.0

def has_awp(player_data: Dict[str, Any]) -> bool:
    """Check if player has an AWP"""
    try:
        weapons = player_data.get('weapons', {}) or {}
        for weapon in weapons.values():
            weapon_name = (weapon.get('name', '') or '').lower()
            if 'awp' in weapon_name:
                return True
        return False
    except:
        return False

def is_approaching_angle(static_pos, static_forward, moving_pos, moving_prev_pos) -> bool:
    """
    Check if moving player is approaching the angle that static player is holding
    Returns True if moving player is getting closer to static player's crosshair line
    """
    try:
        if not static_pos or not static_forward or not moving_pos or not moving_prev_pos:
            return False
            
        # Calculate if moving player is getting closer to the static player's line of sight
        sx, sy, _ = static_pos
        fx, fy, _ = static_forward
        mx, my, _ = moving_pos
        prev_mx, prev_my, _ = moving_prev_pos
        
        # Vector from static player to moving player (current and previous)
        to_moving_current = (mx - sx, my - sy)
        to_moving_prev = (prev_mx - sx, prev_my - sy)
        
        # Normalize static player's forward vector
        forward_norm = norm2d((fx, fy))
        
        # Calculate distances from moving player to static player's line of sight
        # (perpendicular distance to the line)
        current_cross = abs(to_moving_current[0] * forward_norm[1] - to_moving_current[1] * forward_norm[0])
        prev_cross = abs(to_moving_prev[0] * forward_norm[1] - to_moving_prev[1] * forward_norm[0])
        
        # Check if moving player is also moving in the general direction of static player's aim
        dot_current = dot2d(to_moving_current, forward_norm)
        dot_prev = dot2d(to_moving_prev, forward_norm)
        
        # Player is approaching angle if:
        # 1. They're getting closer to the line of sight (smaller cross product)
        # 2. They're moving in the general direction of the AWPer's aim
        # 3. The distance along the aim line is decreasing (getting closer)
        is_approaching_line = current_cross < prev_cross
        is_in_aim_direction = dot_current > 0 and dot_current > dot_prev
        
        return is_approaching_line and is_in_aim_direction
    except:
        return False

# ---------------------
# Historik / Tracking
# ---------------------

class History:
    """Gemmer kort historik pr. spiller til hastighed/ttc/retning og state-deltaer."""
    def __init__(self, maxlen:int=8):
        self.store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, steamid: str, pos, fwd, health:int, round_kills:int, t:float):
        self.store[steamid].append({
            "t": t, "pos": pos, "fwd": fwd, "hp": health, "rk": round_kills
        })

    def last(self, sid:str):
        dq = self.store.get(sid)
        return dq[-1] if dq else None

    def last2(self, sid:str):
        dq = self.store.get(sid)
        if not dq or len(dq) < 2: return None, None
        return dq[-2], dq[-1]

def relative_approach_speed_2d(p1_prev, p1_curr, p2_prev, p2_curr, dt_min=0.05) -> float:
    """Positiv værdi når spillerne nærmer sig hinanden."""
    if not p1_prev or not p1_curr or not p2_prev or not p2_curr:
        return 0.0
    dt1 = max(p1_curr["t"] - p1_prev["t"], dt_min)
    dt2 = max(p2_curr["t"] - p2_prev["t"], dt_min)
    pos1_prev = p1_prev["pos"]; pos1_curr = p1_curr["pos"]
    pos2_prev = p2_prev["pos"]; pos2_curr = p2_curr["pos"]
    if not pos1_prev or not pos1_curr or not pos2_prev or not pos2_curr:
        return 0.0
    v1 = ((pos1_curr[0]-pos1_prev[0])/dt1, (pos1_curr[1]-pos1_prev[1])/dt1)
    v2 = ((pos2_curr[0]-pos2_prev[0])/dt2, (pos2_curr[1]-pos2_prev[1])/dt2)
    rel_v = (v1[0]-v2[0], v1[1]-v2[1])
    r_now = (pos2_curr[0]-pos1_curr[0], pos2_curr[1]-pos1_curr[1])
    r_dir = norm2d(r_now)
    closing_speed = -dot2d(rel_v, r_dir)  # >0 = closing
    return closing_speed

def time_to_contact(distance: float, closing_speed: float, min_speed=1.0) -> float:
    spd = max(min_speed, closing_speed)
    return distance / spd

# ---------------------
# Hovedklasse
# ---------------------

class CS2DuelDetector:
    def __init__(self, config_path: str = "config.json"):
        self.config = self.load_config(config_path)
        self.active = self.config.get('observer', {}).get('enable_auto_switch', True)
        self.campaths_enabled = self.config.get('observer', {}).get('enable_campaths', True)  # Enable/disable campaths (AQWE/UIOP keys)

        # Cooldown/hold/rotation - ULTRA AGGRESSIVE for 2-3 second early switching
        self.switch_cooldown = max(0.5, self.config.get('observer', {}).get('switch_cooldown', 0.5))  # REDUCED to 0.5s for early switching
        self.min_hold_time = 1.0  # REDUCED to 1.0 second for ultra-fast prediction switching
        self.switch_margin = self.config.get('hysteresis', {}).get('switch_margin', 0.04)  # More sensitive
        self.score_ema_alpha = self.config.get('hysteresis', {}).get('score_ema_alpha', 0.55)
        self.rotation_window = self.config.get('rotation', {}).get('window', 1.2)
        self.rotation_delta = self.config.get('rotation', {}).get('delta', 0.04)

        self.last_switch_time = datetime.min
        self.current_target_sid: Optional[str] = None
        self.current_target_start: Optional[datetime] = None
        self.current_target_score_ema: float = 0.0
        self._last_gamestate: Dict[str, Any] = {}

        # Duel detection settings - More sensitive to active engagements
        dd = self.config.get('duel_detection', {})
        self.max_duel_distance = dd.get('max_distance', 1200.0)  # Standard close-range duels
        self.min_duel_distance = dd.get('min_distance', 50.0)    # Allow closer fights
        self.facing_threshold = dd.get('facing_threshold', 0.25)  # More lenient facing
        self.height_diff_max = dd.get('height_difference_max', 450.0)  # Increased for Dust2-style elevation differences
        self.extreme_height_diff_max = dd.get('extreme_height_difference_max', 800.0)  # For very vertical maps
        self.ttc_max = dd.get('ttc_max', 4.0)  # Allow longer approach times
        
        # AWP-specific detection settings
        awp_settings = dd.get('awp_settings', {})
        self.awp_max_distance = awp_settings.get('max_distance', 2800.0)  # Much longer for AWP angles
        self.awp_facing_threshold = awp_settings.get('facing_threshold', 0.15)  # More lenient for angle-holding
        self.awp_approach_detection = awp_settings.get('enable_approach_detection', True)  # Detect approaching enemies

        # Freezetime lockout / F-key
        self.freezetime_fkey_sent = False
        self.last_freezetime_phase = None
        self.freezetime_start: Optional[datetime] = None  # tidsstempel ved FT-start
        self.numbers_lockout_seconds = 19  # 1-0 må først bruges igen 19s efter FT-start

        # Historik/trigger-trackere
        self.history = History(maxlen=8)
        self.last_rk: Dict[str, int] = {}
        self.kill_focus_until: float = 0.0
        self.kill_focus_sid: Optional[str] = None

        self.last_hp: Dict[str, int] = {}
        self.damage_focus_until: float = 0.0
        self.damage_focus_sid: Optional[str] = None

        # Anti-stall
        self.last_any_switch: float = 0.0
        self.failsafe_interval = 5.0  # sek – tving lettere skift hvis intet sker

        # Death detection and reactive switching
        self.last_player_health: Dict[str, int] = {}  # Track health to detect deaths
        self.death_switch_priority_duration = 3.0  # How long to prioritize killer after death
        self.death_switch_active_until: float = 0.0  # Timestamp until death switch is active
        self.death_switch_target: Optional[str] = None  # Who to prioritize after death

        # Enhanced Positional AI Integration
        self.positional_ai = None
        if POSITIONAL_AI_AVAILABLE:
            try:
                self.positional_ai = PositionalObserverAI()
                ai_info = self.positional_ai.get_model_info()
                if ai_info['ready']:
                    logging.info(f"✅ Positional AI: {ai_info['models_loaded']} models ready")
                else:
                    self.positional_ai = None
            except (Exception, KeyboardInterrupt) as e:
                self.positional_ai = None
                logging.info("⚠️ Positional AI unavailable - using fallback")

        # Advanced Kill Prediction System - ULTRA AGGRESSIVE for 2-3 second early prediction
        self.kill_prediction_active = True  # Enable predictive switching
        self.prediction_window = 5.0  # MASSIVELY INCREASED - predict 5 seconds ahead for approach detection
        self.high_tension_duels: Dict[Tuple[str, str], Dict[str, Any]] = {}  # Track high-tension situations

        # Line of Sight / Map Geometry System
        if LOS_AVAILABLE:
            self.geometry_manager = get_geometry_manager()
            logging.info("🗺️ Line of Sight system ready")
        else:
            self.geometry_manager = None
        self.prediction_confidence_threshold = 0.55  # LOWERED - more willing to predict early approaches
        self.high_confidence_override_threshold = 0.70  # LOWERED - easier high-confidence switching
        self.ultra_high_confidence_threshold = 0.85  # LOWERED - more ultra-high predictions for early switching
        self.last_crosshair_data: Dict[str, Dict[str, Any]] = {}  # Track crosshair positioning
        self.engagement_escalation_tracker: Dict[Tuple[str, str], float] = {}  # Track escalating situations

        # Hardcore Prediction System - ULTRA AGGRESSIVE for 2-3 second early switching
        self.hardcore_prediction_active = True  # Enable even earlier prediction
        self.pre_engagement_tracker: Dict[Tuple[str, str], Dict[str, Any]] = {}  # Track pre-engagement phase
        self.crosshair_precision_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=15))  # INCREASED - more history for better prediction
        self.movement_pattern_tracker: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))  # INCREASED - longer movement pattern tracking
        self.hardcore_confidence_threshold = 0.45  # MASSIVELY LOWERED - predict approaches much earlier
        
        # High confidence switching override
        self._force_high_confidence_switch = False  # Flag for bypassing cooldowns
        self._high_confidence_level = 0.0  # Store confidence level for switching speed
        
        # Viewing quality protection
        self._recent_switches = deque(maxlen=5)  # Track recent switches for quality control
        self._max_switches_per_10s = 3  # Limit switches for viewing comfort

        # Duel persistence tracking - prevents switching away from active duels
        self.active_duel_participants: Optional[Tuple[str, str]] = None  # (player1_sid, player2_sid)
        self.active_duel_start_time: Optional[datetime] = None
        self.active_duel_target: Optional[str] = None  # Which participant we're watching
        self.duel_max_duration = 15.0  # Maximum time to lock onto a duel (safety)
        self.duel_separation_distance = 1500.0  # Distance where duel is considered over
        self.last_duel_scores: Dict[str, float] = {}  # Track scores of duel participants
        self.last_duel_switch_time: Optional[datetime] = None  # Debounce duel switching
        self.duel_switch_cooldown = 0.6  # REDUCED - Faster switching between duel participants
        self.duel_switch_threshold = 1.4  # REDUCED - Easier to switch within duel (was 1.8)

        # Pre-Engagement Duel Lock - LOCK-IN when players are approaching/positioning for duel
        self.locked_duel_participants: Optional[Tuple[str, str]] = None  # Players in locked pre-engagement duel
        self.locked_duel_start_time: Optional[datetime] = None
        self.locked_duel_target: Optional[str] = None  # Which participant we're watching
        self.duel_lock_max_duration = 20.0  # Maximum time to lock onto a duel (safety)
        self.pre_engagement_detection_distance = 1800.0  # Distance for pre-engagement detection
        self.duel_resolution_distance = 2500.0  # Distance where we consider duel resolved by separation
        self.pre_engagement_facing_threshold = 0.15  # Lenient facing requirement for pre-engagement
        self.approach_speed_threshold = 5.0  # Minimum approach speed for pre-engagement detection
        self.positioning_time_threshold = 2.0  # Time players must be positioning for lock activation
        
        # Anti-rapid-switching system
        self.last_duel_switch_time: Optional[datetime] = None
        self.duel_switch_cooldown_seconds = 3.0  # Minimum time between duel switches
        self.duel_lock_persistence_bonus = 0.3  # Score bonus for continuing current duel
        self.minimum_duel_lock_duration = 2.5  # Minimum time to stay locked onto a duel

        # Pattern-Aware AI Integration with Memory System
        self.ai_analyzer = None
        self.use_ai = self.config.get('ai', {}).get('enabled', True)
        self.ai_boost_factor = self.config.get('ai', {}).get('boost_factor', 1.5)
        self.use_pattern_memory = self.config.get('ai', {}).get('use_pattern_memory', True)
        
        # Try to load pattern-aware AI system first
        if AI_AVAILABLE and self.use_ai:
            try:
                # First try pattern-aware AI with memory
                if self.use_pattern_memory:
                    from pattern_memory_ai import EnhancedPatternAwareAI
                    self.ai_analyzer = EnhancedPatternAwareAI()
                    if (self.ai_analyzer.base_ai and self.ai_analyzer.base_ai.models_loaded and 
                        self.ai_analyzer.base_ai.model and self.ai_analyzer.pattern_memory):
                        logging.info("🧠 AI System: Pattern-memory with 95.7% accuracy")
                    else:
                        raise ImportError("Pattern memory system incomplete")
                else:
                    # Use enhanced AI without pattern memory
                    from enhanced_observer_ai import EnhancedObserverAI
                    self.ai_analyzer = EnhancedObserverAI()
                    if self.ai_analyzer.models_loaded and self.ai_analyzer.model:
                        logging.info("🧠 AI System: Enhanced model ready")
                    else:
                        raise ImportError("Enhanced AI model not available")
                        
            except ImportError:
                # Fallback to original analyzer
                try:
                    self.ai_analyzer = DemoAnalyzer()
                    
                    # Try to use hot-reloader if available
                    if self.model_reloader:
                        model_path = os.path.join(get_base_dir(), "models", "observer_ai_model.pkl")
                        model = self.model_reloader.get_model("observer_ai_model", model_path)
                        if model:
                            self.ai_analyzer.model = model
                            self.ai_analyzer.models_loaded = True
                            logging.info("🧠 AI System: Basic model loaded with hot-reload support")
                        else:
                            self.ai_analyzer = None
                    elif self.ai_analyzer.load_model('observer_ai_model.pkl'):
                        logging.info("🧠 AI System: Basic model loaded")
                    else:
                        self.ai_analyzer = None
                except:
                    self.ai_analyzer = None

        # Complete Kill Pattern Analysis Integration - WORLD CLASS DATA (12,092 kills!)
        self.kill_pattern_system = None
        self.use_complete_kill_analysis = self.config.get('kill_analysis', {}).get('enabled', True)
        
        if self.use_complete_kill_analysis:
            try:
                # Load the complete kill analysis results
                import json
                # Try consolidated training data first
                kill_analysis_path = os.path.join(get_base_dir(), 'training_data', 'complete_kill_analysis.json')
                if not os.path.exists(kill_analysis_path):
                    # Fallback to root directory
                    kill_analysis_path = os.path.join(get_base_dir(), 'complete_kill_analysis.json')
                with open(kill_analysis_path, 'r', encoding='utf-8') as f:
                    self.kill_analysis_data = json.load(f)
                    
                # Extract prediction model for fast access
                self.prediction_model = self.kill_analysis_data.get('prediction_model', {})
                self.weapon_effectiveness = self.prediction_model.get('weapon_effectiveness', {})
                self.map_hotspots = self.prediction_model.get('map_hotspots', {})
                
                # Load statistical insights from the analysis
                self.weapon_kill_rates = self.kill_analysis_data['analysis_stats']['weapon_stats']
                self.map_kill_distributions = self.kill_analysis_data['analysis_stats']['maps_analyzed']
                
                total_kills = self.kill_analysis_data['analysis_stats']['total_kills_extracted']
                demos_count = self.kill_analysis_data['analysis_stats']['demos_processed']
                logging.info(f"🎯 Kill Analysis: {total_kills} kills from {demos_count} demos")
                
            except Exception as e:
                self.kill_analysis_data = None
                self.prediction_model = {}
                self.weapon_effectiveness = {}
                self.map_hotspots = {}
                self.weapon_kill_rates = {}
                self.map_kill_distributions = {}

        # Model Hot-Reloading System - Auto-update models from AI trainer
        self.model_updater = None
        self.model_reloader = None
        self.auto_model_updates = self.config.get('ai', {}).get('auto_updates', True)
        
        if MODEL_UPDATER_AVAILABLE and self.auto_model_updates:
            try:
                self.model_updater = ModelUpdater()
                self.model_reloader = ModelHotReloader()
                logging.info("🔄 Model Auto-Update System: Ready for hot-reloading")
            except Exception as e:
                logging.warning(f"Model updater initialization failed: {e}")
                self.model_updater = None
                self.model_reloader = None

        # NAV-Aware Predictive Observer System
        self.predictive_observer = None
        self.use_predictive_observer = self.config.get('predictive', {}).get('enabled', True)
        
        if self.use_predictive_observer:
            try:
                from kill_pattern_analyzer import DemoKillAnalyzer
                from predictive_observer import PredictiveObserver
                
                self.kill_analyzer = DemoKillAnalyzer()
                self.predictive_observer = PredictiveObserver(
                    existing_detector=self, 
                    kill_analyzer=self.kill_analyzer
                )
                
                logging.info("🔮 Predictive Observer: NAV-aware with 18k+ areas")
                
            except Exception as e:
                self.use_predictive_observer = False
                self.predictive_observer = None

        # System Summary
        self._log_system_summary()

    def _log_system_summary(self):
        """Log a clean system status summary"""
        systems = []
        if self.positional_ai: systems.append("Positional AI")
        if self.ai_analyzer: systems.append("Pattern AI")  
        if self.geometry_manager: systems.append("Line of Sight")
        if self.kill_analysis_data: systems.append("Kill Analysis")
        if self.predictive_observer: systems.append("Predictive Observer")
        
        if systems:
            logging.info(f"🚀 Systems ready: {', '.join(systems)}")
        else:
            logging.info("🚀 Basic observer ready")

    # ------------- Config --------------

    def load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            config_path_full = os.path.join(get_base_dir(), config_path)
            if os.path.exists(config_path_full):
                with open(config_path_full, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                return config
            else:
                return self.get_default_config()
        except Exception as e:
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        return {
            "observer": {
                "switch_cooldown": 0.5,  # ULTRA REDUCED for 2-3 second early switching
                "enable_auto_switch": True
            },
            "duel_detection": {
                "max_distance": 1200.0,      # Standard close-range duels
                "min_distance": 50.0,        # Allow close fights
                "facing_threshold": 0.25,    # More lenient facing requirement
                "height_difference_max": 450.0,  # Increased for Dust2-style elevation (upper/lower dark)
                "extreme_height_difference_max": 800.0,  # For very vertical scenarios
                "ttc_max": 4.0,              # Longer approach detection
                "awp_settings": {
                    "max_distance": 2800.0,      # Extended range for AWP angles
                    "facing_threshold": 0.15,    # More lenient for angle-holding
                    "enable_approach_detection": True  # Detect enemies approaching AWP angles
                }
            },
            "scoring": {
                "w_orientation_adv": 0.20,     # Increased - current engagement matters
                "w_distance_suit": 0.12,       # Increased - weapon effectiveness matters  
                "w_health": 0.10,              # Increased - current state matters
                "w_weapon_quality": 0.14,      # Increased - current loadout matters
                "w_duel_proximity": 0.25,      # MASSIVELY increased - active duels are key
                "w_recent_perf": 0.12,         # REDUCED from 0.42 - don't favor inactive killers
                "w_isolation": 0.07            # Increased - tactical positioning matters
            },
            "weapon_scores": {
                "awp": 1.0,
                "ak47": 0.9,
                "m4a4": 0.88,
                "m4a1": 0.88,
                "rifle": 0.7,
                "smg": 0.48,
                "shotgun": 0.5,
                "pistol": 0.42,
                "knife": 0.2
            },
            "weapon_ranges": {
                "awp":   {"close": 380, "far": 1300},
                "rifle": {"close": 240, "far": 950},
                "smg":   {"close": 160, "far": 520},
                "shotgun":{"close": 110, "far": 300},
                "pistol":{"close": 150, "far": 520}
            },
            "hysteresis": {
                "switch_margin": 0.03,       # FURTHER REDUCED - Even more sensitive switching for early prediction
                "min_hold_time": 1.0,        # REDUCED - Ultra-fast switching for early predictions
                "score_ema_alpha": 0.85       # INCREASED - Ultra responsive to current action
            },
            "rotation": {
                "window": 1.2,  # sek
                "delta": 0.04   # maks. scoreafstand for rotation
            },
            "triggers": {
                "kill_focus_time": 4.5,      # Much longer focus after kills (AI insight!)
                "damage_drop_hp": 20,        # Even lower threshold for responsiveness  
                "damage_focus_time": 2.0,    # Reduced - kills are more important than damage
                "multi_kill_bonus": 2.0      # Extra time for multi-kills
            },
            "ai": {
                "enabled": True,              # Use enhanced coordinate-based AI model
                "boost_factor": 1.8,          # INCREASED - AI predictions are more important
                "kill_priority_multiplier": 1.3,  # REDUCED from 1.8 - don't over-boost old kills
                "confidence_threshold": 0.2,  # LOWERED - trust AI more
                "use_pattern_memory": True,   # Use pattern recognition from training data
                "memory_weight": 0.6,         # INCREASED to 60% - pattern memory is key!
                "similarity_threshold": 0.4   # LOWERED - find more pattern matches
            },
            "kill_analysis": {
                "enabled": True,              # Use complete kill analysis data (12,092 kills!)
                "use_weapon_effectiveness": True,  # Use real weapon performance data
                "use_map_intelligence": True,  # Use map-specific kill patterns  
                "confidence_boost": 1.15,    # Boost confidence when real data available
                "weapon_data_weight": 0.30,  # Weight for weapon effectiveness (was 0.25)
                "map_data_weight": 0.05      # Weight for map-specific patterns
            },
            "kill_prediction": {
                "enabled": True,              # Enable predictive kill detection
                "confidence_threshold": 0.55,  # MASSIVELY LOWERED - predict approaches 2-3 seconds earlier
                "high_confidence_override": 0.70,  # LOWERED - much faster early switching  
                "ultra_high_confidence": 0.85,  # LOWERED - more early predictions
                "prediction_window": 5.0,    # INCREASED - longer prediction window for approaches
                "viewing_quality_protection": True,  # Limit switches for spectator comfort
                "max_switches_per_10s": 4,   # INCREASED - allow more switches for early prediction
                "hardcore_prediction": True,  # Enable pre-engagement detection
                "hardcore_confidence_threshold": 0.45  # MASSIVELY LOWERED - catch approaches much earlier
            },
            "weapon_aware_distances": {
                "awp_max_distance": 2800,     # AWP maximum engagement distance
                "rifle_vs_rifle_max": 2200,   # Rifle vs rifle maximum distance
                "rifle_vs_close_max": 1800,   # Rifle vs pistol/SMG maximum distance
                "standard_max_distance": 1200  # Standard close-range maximum distance
            }
        }

    # ------------- Historik/Triggers --------------

    def update_history_and_triggers(self, allplayers: Dict[str, Any], now_ts: float, gamestate: Dict[str, Any] = None):
        trig_cfg = self.config.get('triggers', {})
        kill_focus_time = trig_cfg.get('kill_focus_time', 2.2)
        damage_drop_hp = trig_cfg.get('damage_drop_hp', 35)
        damage_focus_time = trig_cfg.get('damage_focus_time', 1.7)

        # Check for death of currently spectated player FIRST
        self._check_spectated_player_death(allplayers, now_ts, gamestate)

        # opdater historik + tjek kills/damage
        for sid, p in allplayers.items():
            if sid == "total":
                continue
            state = p.get('state', {}) or {}
            hp = int(state.get('health', 0))
            rk = int(state.get('round_kills', 0))
            pos = parse_vec3(p.get('position', '')) if p.get('position') else None
            fwd = parse_vec3(p.get('forward', '')) if p.get('forward') else None

            self.history.update(sid, pos, fwd, hp, rk, now_ts)
            
            # Update death detection tracking
            self.last_player_health[sid] = hp

            # Enhanced Kill trigger (AI learned kills are 96.7% important!)
            prev_rk = self.last_rk.get(sid, rk)
            if rk > prev_rk:
                kill_diff = rk - prev_rk
                base_focus_time = kill_focus_time
                
                # Multi-kill bonus (AI insight: kills are everything!)
                if kill_diff >= 2:  # Double kill or more
                    multi_kill_bonus = trig_cfg.get('multi_kill_bonus', 2.0)
                    base_focus_time *= multi_kill_bonus
                    logging.info(f"🔥 {p.get('name','?')} multi-kill! ({kill_diff}x)")
                else:
                    logging.info(f"💀 {p.get('name','?')} got a kill")
                
                self.kill_focus_sid = sid
                self.kill_focus_until = now_ts + base_focus_time
                
                # Force immediate switch for kills (overrides cooldowns)
                if self._can_switch_now() or kill_diff >= 2:
                    slot = self.get_player_slot(sid, gamestate)
                    if slot:
                        self.switch_to_player(slot)
                        self._update_current_target(sid, 1.0)  # Max score for kill focus
                        
            self.last_rk[sid] = rk

            # Damage trigger (HP faldt markant)
            prev_hp = self.last_hp.get(sid, hp)
            if hp > 0 and prev_hp - hp >= damage_drop_hp:
                attacker_sid = self._guess_attacker(sid, allplayers)
                if attacker_sid:
                    self.damage_focus_sid = attacker_sid
                    self.damage_focus_until = now_ts + damage_focus_time
                    an = allplayers.get(attacker_sid, {}).get('name', '?')
                    vn = p.get('name','?')
                    pass  # Damage trigger
            self.last_hp[sid] = hp

        # udløb fokusvinduer
        if self.kill_focus_sid and now_ts > self.kill_focus_until:
            self.kill_focus_sid = None
        if self.damage_focus_sid and now_ts > self.damage_focus_until:
            self.damage_focus_sid = None
        if self.death_switch_target and now_ts > self.death_switch_active_until:
            self.death_switch_target = None

    def _check_spectated_player_death(self, allplayers: Dict[str, Any], now_ts: float, gamestate: Dict[str, Any]):
        """Check if currently spectated player died and handle immediate switching"""
        if not self.current_target_sid:
            return
            
        current_player = allplayers.get(self.current_target_sid, {})
        current_hp = current_player.get('state', {}).get('health', 0)
        previous_hp = self.last_player_health.get(self.current_target_sid, current_hp)
        
        # Detect death (health went from >0 to 0)
        if previous_hp > 0 and current_hp <= 0:
            victim_name = current_player.get('name', 'Unknown')
            logging.info(f"☠️ {victim_name} died - switching away")
            
            # Priority 1: Switch to other active duel if available
            duels = self.find_potential_duels(allplayers)
            best_duel_target = self._find_best_alternative_duel(duels, allplayers, now_ts)
            
            if best_duel_target:
                slot = self.get_player_slot(best_duel_target, gamestate)
                if slot:
                    self.switch_to_player(slot)
                    self._update_current_target(best_duel_target, 0.85)  # High priority for duel
                    target_name = allplayers.get(best_duel_target, {}).get('name', 'Unknown')
                    logging.info(f"⚔️ Switching to {target_name} (alternative duel)")
                    return
            
            # Priority 2: Switch to the killer if no other duels
            killer_sid = self._find_killer_by_recent_kills(allplayers)
            if killer_sid:
                slot = self.get_player_slot(killer_sid, gamestate)
                if slot:
                    self.switch_to_player(slot)
                    self._update_current_target(killer_sid, 0.75)  # Good priority for killer
                    
                    # Set death switch priority to keep focus on killer briefly
                    self.death_switch_target = killer_sid
                    self.death_switch_active_until = now_ts + self.death_switch_priority_duration
                    
                    killer_name = allplayers.get(killer_sid, {}).get('name', 'Unknown')
                    logging.info(f"🎯 Switching to {killer_name} (killer priority)")
                    return
            
            # Priority 3: End duel tracking if current target died
            if self._is_duel_active() and self.current_target_sid in self.active_duel_participants:
                pass  # Ending duel tracking
                self._end_duel_tracking()
            
            pass  # No immediate alternative found

    def _find_best_alternative_duel(self, duels: List[Tuple[str, str, float]], allplayers: Dict[str, Any], now_ts: float) -> Optional[str]:
        """Find the best alternative duel target when current spectated player dies"""
        if not duels:
            return None
            
        best_target = None
        best_score = 0.0
        
        for sid1, sid2, dist in duels:
            # Score both participants
            score1 = self.score_player_in_duel(sid1, sid2, allplayers, dist, now_ts)
            score2 = self.score_player_in_duel(sid2, sid1, allplayers, dist, now_ts)
            
            # Pick the better one from this duel
            if score1 > score2 and score1 > best_score:
                best_target = sid1
                best_score = score1
            elif score2 > best_score:
                best_target = sid2
                best_score = score2
                
        return best_target

    def _find_killer_by_recent_kills(self, allplayers: Dict[str, Any]) -> Optional[str]:
        """Find player who most likely got the kill by looking for recent kill count increase"""
        best_killer = None
        best_score = 0.0
        
        for sid, player in allplayers.items():
            if sid == 'total':
                continue
                
            current_kills = player.get('state', {}).get('round_kills', 0)
            previous_kills = self.last_rk.get(sid, current_kills)
            
            # Look for recent kill increase
            if current_kills > previous_kills:
                kill_diff = current_kills - previous_kills
                
                # Calculate killer score based on kill recency and other factors
                killer_score = kill_diff * 2.0  # Base score for kills
                
                # Boost for players with better positioning/weapons
                player_hp = player.get('state', {}).get('health', 0)
                if player_hp > 0:  # Must be alive
                    weapon_score, _, _ = self.get_best_weapon_score(player, self.config.get('weapon_scores', {}))
                    killer_score += weapon_score * 0.5
                    
                    # Boost for higher HP (survivor advantage)
                    killer_score += (player_hp / 100.0) * 0.3
                    
                    if killer_score > best_score:
                        best_killer = sid
                        best_score = killer_score
        
        return best_killer

    def _is_rifle_weapon(self, weapon_name: str, weapon_type: str) -> bool:
        """Check if weapon is a rifle (AK, M4, etc.)"""
        rifle_names = ['ak47', 'ak-47', 'm4a4', 'm4a1', 'aug', 'sg553', 'sg556', 'famas', 'galil']
        return any(rifle in weapon_name.lower() for rifle in rifle_names) or 'rifle' in weapon_type.lower()
    
    def _is_close_range_weapon(self, weapon_name: str, weapon_type: str) -> bool:
        """Check if weapon is pistol/SMG (close range)"""
        close_range_names = ['glock', 'usp', 'p250', 'tec-9', 'cz75', 'five-seven', 'dual', 'deagle',
                           'mp9', 'mac-10', 'mp7', 'ump', 'p90', 'bizon', 'mp5']
        return (any(name in weapon_name.lower() for name in close_range_names) or 
                'smg' in weapon_type.lower() or 'pistol' in weapon_type.lower())
    
    def _get_weapon_aware_max_distance(self, p1_awp: bool, p2_awp: bool, p1_rifle: bool, p2_rifle: bool,
                                     p1_close: bool, p2_close: bool, current_dist: float) -> float:
        """Calculate maximum duel distance based on weapons involved"""
        
        # AWP gets maximum distance
        if p1_awp or p2_awp:
            return self.awp_max_distance  # 2800 units
            
        # Rifle vs anything = extended distance (rifles dominate at range)
        if p1_rifle or p2_rifle:
            if p1_rifle and p2_rifle:
                return 2200.0  # Rifle vs rifle - long range duels common
            else:
                return 1800.0  # Rifle vs pistol/SMG - rifle advantage at range
                
        # Both close-range weapons = standard distance
        if p1_close and p2_close:
            return self.max_duel_distance  # 1200 units
            
        # Mixed unknown weapons = standard distance
        return self.max_duel_distance

    def _detect_hardcore_pre_engagement(self, allplayers: Dict[str, Any], now_ts: float) -> Optional[Tuple[str, float]]:
        """
        Hardcore prediction - detect kills even before direct engagement starts.
        Looks for: positioning, crosshair pre-placement, movement patterns, map knowledge
        """
        if not self.hardcore_prediction_active:
            return None
            
        best_prediction = None
        best_confidence = 0.0
        
        # Analyze all players for pre-engagement indicators
        for sid1, p1 in allplayers.items():
            if sid1 == 'total' or p1.get('state', {}).get('health', 0) <= 0:
                continue
                
            # Get player positioning and weapon info
            p1_pos = parse_vec3(p1.get('position', ''))
            p1_fwd = parse_vec3(p1.get('forward', ''))
            p1_weapon_score, p1_weapon_name, p1_weapon_type = self.get_best_weapon_score(p1, self.config.get('weapon_scores', {}))
            
            if not p1_pos or not p1_fwd:
                continue
                
            # Look for potential targets this player might engage
            for sid2, p2 in allplayers.items():
                if sid2 == 'total' or sid2 == sid1 or p2.get('state', {}).get('health', 0) <= 0:
                    continue
                    
                if p1.get('team', '') == p2.get('team', ''):
                    continue  # Same team
                    
                p2_pos = parse_vec3(p2.get('position', ''))
                if not p2_pos:
                    continue
                    
                # Calculate hardcore prediction confidence
                hardcore_confidence = self._calculate_hardcore_prediction_confidence(
                    sid1, sid2, p1, p2, p1_pos, p2_pos, p1_fwd, now_ts
                )
                
                if hardcore_confidence > best_confidence and hardcore_confidence >= self.hardcore_confidence_threshold:
                    best_confidence = hardcore_confidence
                    best_prediction = sid1
                    
        if best_prediction and best_confidence >= self.hardcore_confidence_threshold:
            pred_name = allplayers.get(best_prediction, {}).get('name', 'Unknown')
            logging.info(f"🎯 Pre-engagement: {pred_name} ({best_confidence:.0%} confidence)")
            return best_prediction, best_confidence
            
        return None

    def _calculate_hardcore_prediction_confidence(self, attacker_sid: str, target_sid: str, 
                                                attacker: Dict[str, Any], target: Dict[str, Any],
                                                attacker_pos: Tuple[float, float, float], target_pos: Tuple[float, float, float],
                                                attacker_fwd: Tuple[float, float, float], now_ts: float) -> float:
        """Calculate confidence for hardcore pre-engagement prediction"""
        
        confidence = 0.0
        distance = distance2d(attacker_pos, target_pos)
        
        # 1. Crosshair Pre-placement (30% weight) - MOST IMPORTANT for hardcore prediction
        crosshair_score = angle_facing_score(attacker_pos, attacker_fwd, target_pos)
        self.crosshair_precision_history[attacker_sid].append((now_ts, crosshair_score))
        
        # Check crosshair consistency over time (pre-aiming)
        recent_crosshair_scores = [score for ts, score in self.crosshair_precision_history[attacker_sid] 
                                 if now_ts - ts <= 2.0]  # Last 2 seconds
        
        if len(recent_crosshair_scores) >= 3:  # LOWERED - need less data for faster prediction
            avg_crosshair = sum(recent_crosshair_scores) / len(recent_crosshair_scores)
            consistency = 1.0 - (max(recent_crosshair_scores) - min(recent_crosshair_scores))  # Lower variance = higher consistency
            
            if avg_crosshair > 0.5 and consistency > 0.6:  # LOWERED - Good pre-aiming (was 0.7/0.8)
                confidence += 0.30  # Full 30% for good pre-aim
            elif avg_crosshair > 0.3 and consistency > 0.4:  # LOWERED - Decent pre-aiming (was 0.5/0.6)
                confidence += 0.22  # INCREASED
            elif avg_crosshair > 0.15:  # MASSIVELY LOWERED - Even basic aiming (was 0.3)
                confidence += 0.15  # INCREASED - reward any directional aiming
            elif avg_crosshair > 0.05:  # NEW - Minimal aiming gets some reward
                confidence += 0.08  # NEW - catch very early positioning
        
        # 2. Movement Pattern Analysis (25% weight) - positioning for advantage
        movement_confidence = self._analyze_hardcore_movement_patterns(attacker_sid, target_sid, now_ts)
        confidence += 0.25 * movement_confidence
        
        # 3. Weapon Advantage at Current Distance (20% weight)
        attacker_weapon_score, attacker_weapon_name, attacker_weapon_type = self.get_best_weapon_score(attacker, self.config.get('weapon_scores', {}))
        target_weapon_score, _, _ = self.get_best_weapon_score(target, self.config.get('weapon_scores', {}))
        
        weapon_advantage = attacker_weapon_score / max(target_weapon_score, 0.1)
        distance_suitability = self.weapon_distance_suitability(attacker_weapon_name, attacker_weapon_type, distance)
        weapon_confidence = min(weapon_advantage * distance_suitability, 1.0)
        confidence += 0.20 * weapon_confidence
        
        # 4. Positional Advantage (15% weight) - cover, angles, etc.
        positional_confidence = self._analyze_positional_advantage(attacker_sid, target_sid, attacker_pos, target_pos, now_ts)
        confidence += 0.15 * positional_confidence
        
        # 5. Target Vulnerability (10% weight) - target not aware, low HP, etc.
        target_state = target.get('state', {})
        target_hp = float(target_state.get('health', 100))
        target_fwd = parse_vec3(target.get('forward', ''))
        
        vulnerability = 0.0
        if target_hp < 50:  # Low HP target
            vulnerability += 0.6
        elif target_hp < 80:
            vulnerability += 0.3
            
        # Check if target is looking away
        if target_fwd:
            target_awareness = angle_facing_score(target_pos, target_fwd, attacker_pos)
            if target_awareness < 0.3:  # Target not looking at attacker
                vulnerability += 0.4
                
        confidence += 0.10 * min(vulnerability, 1.0)
        
        return min(confidence, 1.0)

    def _analyze_hardcore_movement_patterns(self, attacker_sid: str, target_sid: str, now_ts: float) -> float:
        """Analyze movement patterns for hardcore prediction"""
        
        attacker_history = self.history.store.get(attacker_sid, deque())
        if len(attacker_history) < 3:
            return 0.0
            
        movement_confidence = 0.0
        
        # Check for tactical positioning (slowing down, taking cover, etc.)
        recent_positions = [(h['t'], h['pos']) for h in list(attacker_history)[-5:] if h.get('pos')]
        
        if len(recent_positions) >= 3:
            # Calculate speed changes
            speeds = []
            for i in range(1, len(recent_positions)):
                dt = max(recent_positions[i][0] - recent_positions[i-1][0], 0.1)
                dist = distance2d(recent_positions[i][1], recent_positions[i-1][1])
                speed = dist / dt
                speeds.append(speed)
                
            if speeds:
                recent_speed = speeds[-1] if speeds else 0
                
                # Slowing down for precise aim = good indicator
                if recent_speed < 10:  # Very slow/stopped
                    movement_confidence += 0.8
                elif recent_speed < 30:  # Slow movement
                    movement_confidence += 0.5
                elif recent_speed < 60:  # Moderate movement
                    movement_confidence += 0.2
                    
        return min(movement_confidence, 1.0)

    def _analyze_positional_advantage(self, attacker_sid: str, target_sid: str, 
                                    attacker_pos: Tuple[float, float, float], target_pos: Tuple[float, float, float], 
                                    now_ts: float) -> float:
        """Analyze positional advantages for hardcore prediction"""
        
        positional_confidence = 0.0
        
        # Height advantage
        height_diff = attacker_pos[2] - target_pos[2]
        if height_diff > 50:  # Attacker above target
            positional_confidence += 0.3
        elif height_diff < -50:  # Attacker below (disadvantage)
            positional_confidence -= 0.2
            
        # Distance analysis based on historical patterns
        distance = distance2d(attacker_pos, target_pos)
        
        # Check if this is a known strong position (simplified heuristic)
        # In a real implementation, you'd have map-specific position data
        if 200 < distance < 800:  # Medium range = often advantageous
            positional_confidence += 0.2
        elif distance < 200:  # Close range = depends on weapons
            positional_confidence += 0.1
            
        return max(0.0, min(positional_confidence, 1.0))

    def _predict_imminent_kills(self, allplayers: Dict[str, Any], duels: List[Tuple[str, str, float]], now_ts: float) -> Optional[Tuple[str, float]]:
        """
        Advanced kill prediction system - predicts which player is most likely to get a kill soon.
        Uses multiple indicators: positioning, crosshair placement, movement patterns, weapon advantages.
        """
        if not self.kill_prediction_active or not duels:
            return None
            
        best_prediction = None
        best_confidence = 0.0
        
        for sid1, sid2, dist in duels:
            p1 = allplayers.get(sid1, {})
            p2 = allplayers.get(sid2, {})
            
            if not p1 or not p2:
                continue
                
            # Analyze both directions of the duel
            confidence1 = self._calculate_kill_probability(sid1, sid2, p1, p2, dist, allplayers, now_ts)
            confidence2 = self._calculate_kill_probability(sid2, sid1, p2, p1, dist, allplayers, now_ts)
            
            # Take the higher confidence prediction
            if confidence1 > best_confidence and confidence1 >= self.prediction_confidence_threshold:
                best_prediction = sid1
                best_confidence = confidence1
                
            if confidence2 > best_confidence and confidence2 >= self.prediction_confidence_threshold:
                best_prediction = sid2
                best_confidence = confidence2
                
        if best_prediction and best_confidence >= self.prediction_confidence_threshold:
            pred_name = allplayers.get(best_prediction, {}).get('name', 'Unknown')
            logging.info(f"🔮 Kill prediction: {pred_name} ({best_confidence:.0%} confidence)")
            return best_prediction, best_confidence
            
        return None

    def _calculate_kill_probability(self, attacker_sid: str, target_sid: str, attacker: Dict[str, Any], 
                                  target: Dict[str, Any], distance: float, allplayers: Dict[str, Any], now_ts: float) -> float:
        """Calculate probability that attacker will kill target based on 12,092 real kills analysis"""
        
        attacker_state = attacker.get('state', {})
        target_state = target.get('state', {})
        
        # Base factors
        attacker_hp = float(attacker_state.get('health', 0))
        target_hp = float(target_state.get('health', 0))
        
        if attacker_hp <= 0 or target_hp <= 0:
            return 0.0
            
        confidence = 0.0
        
        # 1. Health Advantage (18% weight) - slightly reduced for weapon focus
        health_ratio = attacker_hp / max(target_hp, 1.0)
        health_confidence = min(health_ratio / 2.0, 1.0)  # Normalize to 0-1
        confidence += 0.18 * health_confidence
        
        # 2. Real-World Weapon Advantage (30% weight) - INCREASED using our 12,092 kill dataset
        attacker_weapon_score, attacker_weapon, _ = self.get_best_weapon_score(attacker, self.config.get('weapon_scores', {}))
        target_weapon_score, target_weapon, _ = self.get_best_weapon_score(target, self.config.get('weapon_scores', {}))
        
        # Enhanced weapon analysis using real data
        weapon_confidence = 0.0
        if hasattr(self, 'weapon_kill_rates') and self.weapon_kill_rates:
            # Get actual kill counts from our dataset
            attacker_kills = self._get_weapon_kill_count(attacker_weapon)
            target_kills = self._get_weapon_kill_count(target_weapon)
            total_kills = sum(self.weapon_kill_rates.values())
            
            if total_kills > 0:
                attacker_effectiveness = attacker_kills / total_kills
                target_effectiveness = target_kills / total_kills
                
                # Real-world weapon advantage calculation
                if target_effectiveness > 0:
                    weapon_advantage = attacker_effectiveness / target_effectiveness
                    weapon_confidence = min(weapon_advantage / 3.0, 1.0)  # Normalize
                else:
                    weapon_confidence = min(attacker_effectiveness * 5.0, 1.0)
                
                # Special bonuses for dominant weapons from our analysis
                if attacker_kills > 4000:  # AK-47 dominance (4422 kills)
                    weapon_confidence = min(weapon_confidence * 1.2, 1.0)
                elif attacker_kills > 2000:  # M4A1 strength (2097 kills) 
                    weapon_confidence = min(weapon_confidence * 1.15, 1.0)
                elif attacker_kills > 900 and 'awp' in attacker_weapon.lower():  # AWP precision (953 kills)
                    weapon_confidence = min(weapon_confidence * 1.25, 1.0)  # Higher bonus for AWP
        else:
            # Fallback to basic calculation
            weapon_advantage = attacker_weapon_score / max(target_weapon_score, 0.1)
            weapon_confidence = min(weapon_advantage / 2.0, 1.0)
            
        confidence += 0.30 * weapon_confidence
        
        # 2.5. Enhanced Positional AI Predictions (15% weight) - NEW!
        if self.positional_ai:
            try:
                attacker_pos_vec = parse_vec3(attacker.get('position', ''))
                target_pos_vec = parse_vec3(target.get('position', ''))
                
                if attacker_pos_vec and target_pos_vec:
                    potential_kill = {
                        'killer': {
                            'position': {
                                'x': attacker_pos_vec[0],
                                'y': attacker_pos_vec[1],
                                'z': attacker_pos_vec[2]
                            },
                            'angles': {
                                'yaw': float(attacker_state.get('yaw', 0)),
                                'pitch': float(attacker_state.get('pitch', 0))
                            },
                            'weapon': attacker_weapon.lower() if attacker_weapon else 'unknown'
                        },
                        'victim': {
                            'position': {
                                'x': target_pos_vec[0],
                                'y': target_pos_vec[1],
                                'z': target_pos_vec[2]
                            }
                        }
                    }
                    
                    gamestate = self._last_gamestate
                    ai_predictions = self.positional_ai.predict_kill_outcome(gamestate, potential_kill)
                    
                    # Combine AI predictions into confidence
                    ai_confidence = (
                        ai_predictions.get('headshot_probability', 0.5) * 0.4 +  # Headshot likelihood
                        ai_predictions.get('weapon_effectiveness', 0.5) * 0.3 +  # Weapon effectiveness in position
                        (1.0 - ai_predictions.get('wallbang_probability', 0.1)) * 0.2 +  # Lower wallbang = cleaner shot
                        ai_predictions.get('confidence', 0.5) * 0.1  # AI's own confidence
                    )
                    
                    confidence += 0.15 * ai_confidence
                    
                    # Debug log for high-confidence AI predictions
                    if ai_predictions.get('confidence', 0) > 0.7:
                        pass  # Positional AI prediction
                
            except Exception as e:
                pass  # AI prediction error
        
        # 3. Positioning Advantage (17% weight) - Crosshair on target (reduced from 20% to make room for AI)
        attacker_pos = parse_vec3(attacker.get('position', ''))
        target_pos = parse_vec3(target.get('position', ''))
        attacker_fwd = parse_vec3(attacker.get('forward', ''))
        
        crosshair_confidence = 0.0
        if attacker_pos and target_pos and attacker_fwd:
            crosshair_score = angle_facing_score(attacker_pos, attacker_fwd, target_pos)
            # High crosshair precision indicates ready to shoot
            if crosshair_score > 0.8:  # Very precise aim
                crosshair_confidence = 1.0
            elif crosshair_score > 0.6:  # Good aim
                crosshair_confidence = 0.7
            elif crosshair_score > 0.4:  # Decent aim
                crosshair_confidence = 0.4
        confidence += 0.17 * crosshair_confidence
        
        # 4. Movement Pattern Analysis (13% weight) - reduced to make room for AI
        movement_confidence = self._analyze_movement_for_kill_prediction(attacker_sid, target_sid, now_ts)
        confidence += 0.13 * movement_confidence
        
        # 5. Distance Suitability (10% weight)
        distance_confidence = self.weapon_distance_suitability(attacker_weapon, "", distance)
        confidence += 0.10 * distance_confidence
        
        # 6. Recent Performance Boost (10% weight)
        recent_kills = attacker_state.get('round_kills', 0)
        performance_confidence = min(recent_kills * 0.3, 1.0)  # Each kill adds confidence
        confidence += 0.10 * performance_confidence
        
        # 7. Special bonuses for high-tension situations
        duel_key = tuple(sorted([attacker_sid, target_sid]))
        if duel_key in self.high_tension_duels:
            tension_data = self.high_tension_duels[duel_key]
            tension_duration = now_ts - tension_data.get('start_time', now_ts)
            
            # Weapon-aware tension bonuses
            is_rifle_duel = self._is_rifle_weapon(attacker_weapon, "")
            
            if is_rifle_duel:
                # Rifle duels can develop over longer distances/time
                if tension_duration > 2.0:  # Rifles = longer buildup time
                    confidence += 0.18  # Higher bonus for rifle tension
                elif tension_duration > 1.0:
                    confidence += 0.12
            else:
                # Close-range weapons = faster tension resolution
                if tension_duration > 1.0:  # Shorter buildup for close weapons
                    confidence += 0.15
                elif tension_duration > 0.5:
                    confidence += 0.08
        
        # 8. Map-Specific Intelligence from Real Data (5% weight)
        map_confidence = 0.0
        if hasattr(self, 'map_kill_distributions') and self.map_kill_distributions:
            # Our analysis shows: de_ancient: 2836, de_nuke: 2482, de_inferno: 2021, etc.
            # Use last known gamestate for map detection
            current_map = getattr(self, '_current_map', 'unknown')
            if current_map == 'unknown' and hasattr(self, '_last_gamestate'):
                current_map = self._last_gamestate.get('map', {}).get('name', '').lower()
            
            if current_map in self.map_kill_distributions:
                map_kills = self.map_kill_distributions[current_map]
                total_map_kills = sum(self.map_kill_distributions.values())
                
                if total_map_kills > 0:
                    # Higher kill volume maps = more predictable patterns
                    map_activity_factor = min(map_kills / total_map_kills * 8.0, 1.0)
                    map_confidence = map_activity_factor
                    
                    # Special bonuses for high-activity maps from our data
                    if map_kills > 2500:  # de_ancient/de_nuke tier (2800+ kills)
                        map_confidence *= 1.1  # More predictable on active maps
                    elif map_kills > 2000:  # de_inferno tier (2021 kills)
                        map_confidence *= 1.05
        
        confidence += 0.05 * map_confidence
        
        # 9. Escalation bonus - situations getting more intense
        escalation_bonus = self._get_escalation_bonus(attacker_sid, target_sid, now_ts)
        confidence += escalation_bonus
        
        return min(confidence, 1.0)  # Cap at 100%

    def _analyze_movement_for_kill_prediction(self, attacker_sid: str, target_sid: str, now_ts: float) -> float:
        """Analyze movement patterns to predict kill likelihood"""
        
        attacker_prev, attacker_curr = self.history.last2(attacker_sid)
        target_prev, target_curr = self.history.last2(target_sid)
        
        if not all([attacker_prev, attacker_curr, target_prev, target_curr]):
            return 0.0
            
        # Calculate relative movement
        closing_speed = relative_approach_speed_2d(attacker_prev, attacker_curr, target_prev, target_curr)
        
        movement_confidence = 0.0
        
        # ULTRA SENSITIVE approach detection - catch even slow movements 2-3 seconds earlier
        if closing_speed > 30:  # LOWERED - Fast approach (was 50)
            movement_confidence += 0.7  # INCREASED confidence
        elif closing_speed > 10:  # LOWERED - Medium approach (was 20)  
            movement_confidence += 0.5  # INCREASED confidence
        elif closing_speed > 2:  # MASSIVELY LOWERED - Slow approach (was 0, now catch crawling)
            movement_confidence += 0.35  # INCREASED - even slow movement matters
        elif closing_speed > -5:  # NEW - Even stationary positioning gets some confidence
            movement_confidence += 0.15  # NEW - reward good positioning
            
        # Check if attacker stopped moving (aiming for precise shot)
        attacker_pos_prev = attacker_prev.get('pos')
        attacker_pos_curr = attacker_curr.get('pos')
        
        if attacker_pos_prev and attacker_pos_curr:
            attacker_movement = distance2d(attacker_pos_prev, attacker_pos_curr)
            time_diff = attacker_curr.get('t', 0) - attacker_prev.get('t', 0)
            
            if time_diff > 0:
                attacker_speed = attacker_movement / time_diff
                
                # Standing still often indicates preparing to shoot
                if attacker_speed < 5:  # Very slow/still
                    movement_confidence += 0.4
                elif attacker_speed < 15:  # Slow movement
                    movement_confidence += 0.2
        
        return min(movement_confidence, 1.0)

    def _get_escalation_bonus(self, attacker_sid: str, target_sid: str, now_ts: float) -> float:
        """Calculate bonus for escalating situations - FASTER escalation detection"""
        
        duel_key = tuple(sorted([attacker_sid, target_sid]))
        
        if duel_key not in self.engagement_escalation_tracker:
            self.engagement_escalation_tracker[duel_key] = now_ts
            return 0.0
            
        # How long has this engagement been building? - ULTRA AGGRESSIVE for 2-3 second early switching
        escalation_time = now_ts - self.engagement_escalation_tracker[duel_key]
        
        if escalation_time > 1.5:  # MASSIVELY REDUCED - high probability after just 1.5s
            return 0.20  # INCREASED bonus for longer engagement
        elif escalation_time > 0.8:  # MASSIVELY REDUCED - medium probability after 0.8s
            return 0.15  # INCREASED bonus
        elif escalation_time > 0.3:  # MASSIVELY REDUCED - even faster detection (0.3s!)
            return 0.10  # INCREASED bonus
        elif escalation_time > 0.1:  # NEW - Any engagement tracking gets some bonus
            return 0.05  # NEW - immediate small bonus
            
        return 0.0

    def _track_high_tension_situations(self, duels: List[Tuple[str, str, float]], allplayers: Dict[str, Any], now_ts: float):
        """Track situations that are building to kills"""
        
        current_duels = set()
        
        for sid1, sid2, dist in duels:
            duel_key = tuple(sorted([sid1, sid2]))
            current_duels.add(duel_key)
            
            p1 = allplayers.get(sid1, {})
            p2 = allplayers.get(sid2, {})
            
            # Check for tension indicators
            tension_indicators = 0
            
            # 1. Close distance
            if dist < 400:
                tension_indicators += 1
                
            # 2. Both players aiming at each other
            p1_pos = parse_vec3(p1.get('position', ''))
            p2_pos = parse_vec3(p2.get('position', ''))
            p1_fwd = parse_vec3(p1.get('forward', ''))
            p2_fwd = parse_vec3(p2.get('forward', ''))
            
            if all([p1_pos, p2_pos, p1_fwd, p2_fwd]):
                p1_aiming = angle_facing_score(p1_pos, p1_fwd, p2_pos) > 0.5
                p2_aiming = angle_facing_score(p2_pos, p2_fwd, p1_pos) > 0.5
                
                if p1_aiming and p2_aiming:
                    tension_indicators += 2  # Mutual aiming is high tension
                elif p1_aiming or p2_aiming:
                    tension_indicators += 1
                    
            # 3. Recent damage taken
            p1_hp = p1.get('state', {}).get('health', 100)
            p2_hp = p2.get('state', {}).get('health', 100)
            
            if p1_hp < 80 or p2_hp < 80:  # Someone took damage
                tension_indicators += 1
                
            # 4. Good weapons
            p1_weapon_score, _, _ = self.get_best_weapon_score(p1, self.config.get('weapon_scores', {}))
            p2_weapon_score, _, _ = self.get_best_weapon_score(p2, self.config.get('weapon_scores', {}))
            
            if p1_weapon_score > 0.6 or p2_weapon_score > 0.6:
                tension_indicators += 1
                
            # Track high tension situations - LOWERED threshold for faster detection
            if tension_indicators >= 2:  # REDUCED from 3 to 2 - faster tension detection
                if duel_key not in self.high_tension_duels:
                    self.high_tension_duels[duel_key] = {
                        'start_time': now_ts,
                        'max_tension': tension_indicators,
                        'participants': [sid1, sid2]
                    }
                    p1_name = p1.get('name', 'Unknown')
                    p2_name = p2.get('name', 'Unknown')
                    logging.info(f"🔥 High tension: {p1_name} vs {p2_name}")
                else:
                    # Update existing tension
                    self.high_tension_duels[duel_key]['max_tension'] = max(
                        self.high_tension_duels[duel_key]['max_tension'], 
                        tension_indicators
                    )
        
        # Clean up resolved tensions
        resolved_duels = set(self.high_tension_duels.keys()) - current_duels
        for duel_key in resolved_duels:
            del self.high_tension_duels[duel_key]
            if duel_key in self.engagement_escalation_tracker:
                del self.engagement_escalation_tracker[duel_key]

    def _guess_attacker(self, victim_sid: str, allplayers: Dict[str, Any]) -> Optional[str]:
        vic = allplayers.get(victim_sid, {})
        vic_team = vic.get('team','')
        vic_pos = parse_vec3(vic.get('position','')) if vic.get('position') else None
        if not vic_pos: return None
        best_sid = None
        best_score = 0.0
        for sid, p in allplayers.items():
            if sid in ('total', victim_sid):
                continue
            if p.get('team','') == vic_team:
                continue
            pos = parse_vec3(p.get('position','')) if p.get('position') else None
            fwd = parse_vec3(p.get('forward','')) if p.get('forward') else None
            if not pos or not fwd:
                continue
            d = distance2d(pos, vic_pos)
            if d > 1200:
                continue
            fscore = angle_facing_score(pos, fwd, vic_pos)
            s = clamp01(0.6*(1.0 - min(d/1200.0, 1.0)) + 0.4*fscore)
            if s > best_score:
                best_score = s
                best_sid = sid
        return best_sid

    # ------------- Scoring -------------

    def get_best_weapon_score(self, player_data: Dict[str, Any], weapon_scores: Dict[str, float]) -> Tuple[float, str, str]:
        weapons = player_data.get('weapons', {}) or {}
        best = 0.12
        best_name = "pistol"
        best_type = ""
        for w in weapons.values():
            name = (w.get('name', '') or '').lower()
            wtype = (w.get('type', '') or '').lower()
            if 'knife' in name and len(weapons) > 1:
                continue
            sc = 0.12
            
            # Use real-world weapon effectiveness data from 12,092 kills dataset
            if hasattr(self, 'weapon_kill_rates') and self.weapon_kill_rates:
                # Map weapon names to our dataset (AK-47: 4422 kills, M4A1: 2097 kills, etc.)
                weapon_key = name.upper().replace('-', '-')  # Keep original format
                if 'ak47' in name or 'ak-47' in name:
                    weapon_key = 'AK-47'
                elif 'm4a1' in name:
                    weapon_key = 'M4A1'
                elif 'm4a4' in name:
                    weapon_key = 'M4A4'
                elif 'awp' in name:
                    weapon_key = 'AWP'
                elif 'usp' in name:
                    weapon_key = 'USP-S'
                
                weapon_kills = self.weapon_kill_rates.get(weapon_key, 0)
                if weapon_kills > 0:
                    # Calculate real-world effectiveness based on actual kill data
                    total_kills = sum(self.weapon_kill_rates.values())
                    real_world_score = min(weapon_kills / total_kills * 8.0, 1.0)  # Scale to realistic range
                    sc = max(sc, real_world_score)
                    
                    # Special bonuses for top performers from our dataset
                    if weapon_kills > 4000:  # AK-47 tier (4422 kills)
                        sc = max(sc, 0.95)
                    elif weapon_kills > 2000:  # M4A1 tier (2097 kills)
                        sc = max(sc, 0.88)
                    elif weapon_kills > 900:  # AWP tier (953 kills)
                        sc = max(sc, 0.92)  # AWP gets high score despite lower volume
                    elif weapon_kills > 500:  # Strong secondary weapons
                        sc = max(sc, 0.75)
            
            # Use weapon effectiveness model if available
            if hasattr(self, 'weapon_effectiveness') and self.weapon_effectiveness:
                weapon_clean = name.replace('-', '').replace('_', '').lower()
                if weapon_clean in self.weapon_effectiveness:
                    effectiveness_data = self.weapon_effectiveness[weapon_clean]
                    kill_rate = effectiveness_data.get('kill_rate', 0.0)
                    prediction_boost = effectiveness_data.get('prediction_boost', 1.0)
                    model_score = kill_rate * prediction_boost
                    sc = max(sc, model_score)
            
            # Fallback to config-based scoring
            for key, val in weapon_scores.items():
                if key in name:
                    sc = val; break
            if 'rifle' in wtype or 'sniper' in wtype:
                sc += 0.1
            if sc > best:
                best = sc; best_name = name; best_type = wtype
        return best, best_name, best_type

    def _get_weapon_kill_count(self, weapon_name: str) -> int:
        """Get actual kill count for a weapon from our 12,092 kill dataset"""
        if not hasattr(self, 'weapon_kill_rates') or not self.weapon_kill_rates:
            return 0
            
        # Map weapon names to dataset keys
        weapon_key = weapon_name.upper()
        if 'ak47' in weapon_name.lower() or 'ak-47' in weapon_name.lower():
            weapon_key = 'AK-47'
        elif 'm4a1' in weapon_name.lower():
            weapon_key = 'M4A1'  
        elif 'm4a4' in weapon_name.lower():
            weapon_key = 'M4A4'
        elif 'awp' in weapon_name.lower():
            weapon_key = 'AWP'
        elif 'usp' in weapon_name.lower():
            weapon_key = 'USP-S'
        elif 'deagle' in weapon_name.lower() or 'desert' in weapon_name.lower():
            weapon_key = 'Desert Eagle'
        elif 'glock' in weapon_name.lower():
            weapon_key = 'Glock-18'
        elif 'p250' in weapon_name.lower():
            weapon_key = 'P250'
        elif 'famas' in weapon_name.lower():
            weapon_key = 'FAMAS'
        elif 'galil' in weapon_name.lower():
            weapon_key = 'Galil AR'
            
        return self.weapon_kill_rates.get(weapon_key, 0)

    def weapon_distance_suitability(self, weapon_name: str, weapon_type: str, dist: float) -> float:
        wr = self.config.get('weapon_ranges', {})
        name = "awp" if "awp" in weapon_name else \
               "rifle" if ("rifle" in weapon_type or "m4" in weapon_name or "ak" in weapon_name) else \
               "smg" if "smg" in weapon_type else \
               "shotgun" if ("shotgun" in weapon_type or "nova" in weapon_name or "mag-7" in weapon_name) else \
               "pistol"
        r = wr.get(name, {"close": 200, "far": 700})
        if dist <= r["close"]:
            return clamp01(dist / max(1.0, r["close"]))
        elif dist >= r["far"]:
            return clamp01(r["far"] / max(1.0, dist))
        else:
            return 1.0

    def isolation_score(self, sid: str, allplayers: Dict[str, Any], radius_close=340.0) -> float:
        me = allplayers.get(sid, {})
        my_team = me.get('team', '')
        my_pos = parse_vec3(me.get('position', '')) if me.get('position') else None
        if not my_pos or not my_team:
            return 0.0
        allies = 0
        enemies = 0
        for osid, od in allplayers.items():
            if osid in ('total', sid):
                continue
            if od.get('state', {}).get('health', 0) <= 0:
                continue
            opos = parse_vec3(od.get('position', '')) if od.get('position') else None
            if not opos: continue
            if distance2d(my_pos, opos) <= radius_close:
                if od.get('team', '') == my_team:
                    allies += 1
                else:
                    enemies += 1
        if enemies >= 2 and allies <= 1: return 1.0
        if enemies == 1 and allies == 0: return 0.75
        if enemies >= 3: return 0.9
        if allies == 0 and enemies == 0: return 0.25
        return clamp01(0.35 + 0.16*enemies - 0.1*allies)

    def score_player_in_duel(self, sid_me: str, sid_op: str, allplayers: Dict[str, Any], dist: float, t: float) -> float:
        scw = self.config.get('scoring', {})
        wscores = self.config.get('weapon_scores', {})

        me = allplayers.get(sid_me, {})
        op = allplayers.get(sid_op, {})
        me_state = me.get('state', {}) or {}
        op_state = op.get('state', {}) or {}

        if me_state.get('health', 0) <= 0 or op_state.get('health', 0) <= 0:
            return 0.0

        me_pos = parse_vec3(me.get('position','')) if me.get('position') else None
        op_pos = parse_vec3(op.get('position','')) if op.get('position') else None
        me_fwd = parse_vec3(me.get('forward','')) if me.get('forward') else None
        op_fwd = parse_vec3(op.get('forward','')) if op.get('forward') else None

        o1 = angle_facing_score(me_pos, me_fwd, op_pos) if (me_pos and me_fwd and op_pos) else 0.0
        o2 = angle_facing_score(op_pos, op_fwd, me_pos) if (op_pos and op_fwd and me_pos) else 0.0
        orient_me = clamp01(0.65*o1 + 0.35*max(0.0, o1 - o2))

        me_hp = float(me_state.get('health', 0)); op_hp = float(op_state.get('health', 0))
        health_score = clamp01(min(me_hp / max(op_hp, 1.0), 2.0) / 2.0)

        w_me, wname, wtype = self.get_best_weapon_score(me, wscores)
        weapon_quality = clamp01(w_me)
        dist_suit = self.weapon_distance_suitability(wname, wtype, dist)
        duel_prox = 1.0 if dist < 280 else (0.75 if dist < 600 else 0.45)

        rk = int(me_state.get('round_kills', 0))
        recent_perf = clamp01(min(0.22*rk, 0.66))

        iso = self.isolation_score(sid_me, allplayers)

        p1_prev, p1_curr = self.history.last2(sid_me)
        p2_prev, p2_curr = self.history.last2(sid_op)
        closing = relative_approach_speed_2d(p1_prev, p1_curr, p2_prev, p2_curr)
        ttc = time_to_contact(dist, closing, min_speed=1.0)
        ttc_bonus = 1.0 if ttc <= self.ttc_max else max(0.0, (self.ttc_max / ttc))

        flashed = float(me_state.get('flashed', 0))
        blind_pen = 0.0
        if flashed >= 75: blind_pen = 0.45
        elif flashed >= 35: blind_pen = 0.2

        score = (
            scw.get('w_orientation_adv', 0.33) * orient_me +
            scw.get('w_distance_suit', 0.14) * dist_suit +
            scw.get('w_health', 0.12) * health_score +
            scw.get('w_weapon_quality', 0.14) * weapon_quality +
            scw.get('w_duel_proximity', 0.16) * duel_prox +
            scw.get('w_recent_perf', 0.06) * recent_perf +
            scw.get('w_isolation', 0.05) * iso
        )
        score *= clamp01(0.6 + 0.4*ttc_bonus)
        score *= (1.0 - blind_pen)
        
        # AWP-specific scoring bonus for long-range engagements
        if has_awp(me) and dist > 800:  # AWP at long range
            awp_bonus = min(1.3, 1.0 + (dist - 800) / 2000)  # Up to 30% bonus for very long ranges
            score *= awp_bonus
            pass  # AWP bonus applied
        
        return clamp01(score)

    # ------------- Duel-finder -------------

    def find_potential_duels(self, allplayers: Dict[str, Any]) -> List[Tuple[str, str, float]]:
        duels = []
        players = [(sid, data) for sid, data in allplayers.items() if sid != 'total']
        for i, (sid1, p1) in enumerate(players):
            for sid2, p2 in players[i+1:]:
                if p1.get('team','') == p2.get('team','') or not p1.get('team') or not p2.get('team'):
                    continue
                if p1.get('state', {}).get('health', 0) <= 0 or p2.get('state', {}).get('health', 0) <= 0:
                    continue
                pos1 = parse_vec3(p1.get('position','')) if p1.get('position') else None
                pos2 = parse_vec3(p2.get('position','')) if p2.get('position') else None
                if not pos1 or not pos2:
                    continue
                
                dist = distance2d(pos1, pos2)
                
                # Advanced weapon-aware distance detection
                p1_has_awp = has_awp(p1)
                p2_has_awp = has_awp(p2)
                p1_weapon_score, p1_weapon_name, p1_weapon_type = self.get_best_weapon_score(p1, self.config.get('weapon_scores', {}))
                p2_weapon_score, p2_weapon_name, p2_weapon_type = self.get_best_weapon_score(p2, self.config.get('weapon_scores', {}))
                
                # Determine weapon categories for distance logic
                p1_has_rifle = self._is_rifle_weapon(p1_weapon_name, p1_weapon_type)
                p2_has_rifle = self._is_rifle_weapon(p2_weapon_name, p2_weapon_type)
                p1_has_pistol_smg = self._is_close_range_weapon(p1_weapon_name, p1_weapon_type)
                p2_has_pistol_smg = self._is_close_range_weapon(p2_weapon_name, p2_weapon_type)
                
                # Dynamic distance limits based on weapons involved
                max_distance = self._get_weapon_aware_max_distance(
                    p1_has_awp, p2_has_awp, p1_has_rifle, p2_has_rifle, 
                    p1_has_pistol_smg, p2_has_pistol_smg, dist
                )
                
                # Dynamic facing threshold based on weapon types
                facing_threshold = self.awp_facing_threshold if (p1_has_awp or p2_has_awp) else self.facing_threshold
                
                if dist > max_distance or dist < self.min_duel_distance:
                    continue
                    
                # Smart elevation logic - context-aware height difference checking
                try:
                    height_diff = abs(pos1[2] - pos2[2])
                    
                    # Dynamic height threshold based on distance and weapon type
                    effective_height_threshold = self.height_diff_max
                    
                    # For longer distances, allow more height difference (angled shots, ramps, etc.)
                    if dist > 800:
                        effective_height_threshold = self.extreme_height_diff_max
                    elif dist > 400:
                        # Graduated height allowance based on distance
                        height_multiplier = 1.0 + (dist - 400) / 800  # 1.0 to 1.5x
                        effective_height_threshold = self.height_diff_max * height_multiplier
                    
                    # AWP gets even more height tolerance (long-range angle shots)
                    if p1_has_awp or p2_has_awp:
                        effective_height_threshold *= 1.5
                    
                    # Skip only if height difference is extreme
                    if height_diff > effective_height_threshold:
                        if ELEVATION_DEBUG:
                            pass  # Skip elevated duel
                        continue
                    elif height_diff > self.height_diff_max:
                        if ELEVATION_DEBUG:
                            pass  # Allow elevated duel
                        
                except:
                    continue

                # Line of Sight check - prevent duels through walls
                if LOS_AVAILABLE:
                    try:
                        if not duel_geometrically_possible(pos1, pos2, self.geometry_manager):
                            pass  # LOS blocked
                            continue
                        else:
                            pass  # LOS clear
                    except Exception as e:
                        pass  # LOS error
                        # Continue without LOS check on error

                # Weapon-specific approach detection and facing logic
                should_include_duel = False
                awp_involved = p1_has_awp or p2_has_awp
                rifle_involved = p1_has_rifle or p2_has_rifle
                
                if awp_involved:
                    # For AWP scenarios, use more sophisticated detection
                    f1 = parse_vec3(p1.get('forward','')) if p1.get('forward') else None
                    f2 = parse_vec3(p2.get('forward','')) if p2.get('forward') else None
                    
                    if p1_has_awp and f1:
                        # Check if p2 is approaching p1's angle or in p1's line of sight
                        facing_score = angle_facing_score(pos1, f1, pos2)
                        if facing_score >= facing_threshold:
                            should_include_duel = True
                            pass  # AWP duel detected
                        
                        # Check if p2 is approaching p1's angle
                        if self.awp_approach_detection:
                            p2_prev, p2_curr = self.history.last2(sid2)
                            if p2_prev and p2_curr:
                                p2_prev_pos = p2_prev.get("pos")
                                if p2_prev_pos and is_approaching_angle(pos1, f1, pos2, p2_prev_pos):
                                    should_include_duel = True
                                    pass  # AWP approach detected
                    
                    if p2_has_awp and f2:
                        # Check if p1 is approaching p2's angle or in p2's line of sight
                        facing_score = angle_facing_score(pos2, f2, pos1)
                        if facing_score >= facing_threshold:
                            should_include_duel = True
                            pass  # AWP duel detected
                        
                        # Check if p1 is approaching p2's angle
                        if self.awp_approach_detection:
                            p1_prev, p1_curr = self.history.last2(sid1)
                            if p1_prev and p1_curr:
                                p1_prev_pos = p1_prev.get("pos")
                                if p1_prev_pos and is_approaching_angle(pos2, f2, pos1, p1_prev_pos):
                                    should_include_duel = True
                                    pass  # AWP approach detected
                
                else:
                    # Standard duel detection (non-AWP)
                    p1_prev, p1_curr = self.history.last2(sid1)
                    p2_prev, p2_curr = self.history.last2(sid2)
                    closing = relative_approach_speed_2d(p1_prev, p1_curr, p2_prev, p2_curr)
                    
                    if closing >= 0.05 or dist <= 600:
                        should_include_duel = True
                    else:
                        # Check facing for slower/static engagements
                        f1 = parse_vec3(p1.get('forward','')) if p1.get('forward') else None
                        f2 = parse_vec3(p2.get('forward','')) if p2.get('forward') else None
                        fscore = 0.0
                        if pos1 and f1 and pos2: fscore = max(fscore, angle_facing_score(pos1, f1, pos2))
                        if pos2 and f2 and pos1: fscore = max(fscore, angle_facing_score(pos2, f2, pos1))
                        if fscore >= facing_threshold:
                            should_include_duel = True
                
                if should_include_duel:
                    if awp_involved:
                        duel_type = "AWP"
                    elif rifle_involved:
                        duel_type = "RIFLE"
                    else:
                        duel_type = "CLOSE-RANGE"
                    pass  # Duel found
                    duels.append((sid1, sid2, dist))
                    
        return duels

    # ------------- Valg og switching -------------

    def choose_best_player(self, gamestate: Dict[str, Any]) -> Tuple[Optional[str], float, Optional[str], float]:
        allplayers = gamestate.get('allplayers', {}) or {}
        duels = self.find_potential_duels(allplayers)
        now_t = time.time()
        
        # Track high tension situations for kill prediction
        self._track_high_tension_situations(duels, allplayers, now_t)
        
        # HARDCORE Prediction - detect pre-engagement setups (but respect duel locks)
        hardcore_prediction = self._detect_hardcore_pre_engagement(allplayers, now_t)
        if hardcore_prediction:
            hardcore_killer, hardcore_confidence = hardcore_prediction
            
            # Hardcore predictions must respect duel locks
            if (hardcore_killer != self.current_target_sid and 
                hardcore_confidence >= self.hardcore_confidence_threshold):
                
                # DUEL LOCK PROTECTION: Hardcore predictions cannot override locked duels
                should_use_hardcore = True
                
                if self._is_duel_locked():
                    # Only allow hardcore predictions within the locked duel
                    if hardcore_killer in self.locked_duel_participants:
                        hardcore_name = allplayers.get(hardcore_killer, {}).get('name', 'Unknown')
                        logging.info(f"🎯 Duel prediction: {hardcore_name} ({hardcore_confidence:.0%})")
                    else:
                        hardcore_name = allplayers.get(hardcore_killer, {}).get('name', 'Unknown')
                        pass  # Prediction blocked by duel lock
                        should_use_hardcore = False
                
                if self.death_switch_target:
                    should_use_hardcore = False  # Still respect death switches
                
                if should_use_hardcore:
                    hardcore_name = allplayers.get(hardcore_killer, {}).get('name', 'Unknown')
                    logging.info(f"🎯 Early setup: {hardcore_name} ({hardcore_confidence:.0%})")
                    return hardcore_killer, hardcore_confidence, None, 0.0
        
        # Advanced Kill Prediction - predict imminent kills BEFORE they happen (but respect duel locks)
        kill_prediction = self._predict_imminent_kills(allplayers, duels, now_t)
        if kill_prediction:
            predicted_killer, prediction_confidence = kill_prediction
            
            # If prediction is strong and target is different, return prediction as best choice
            if (predicted_killer != self.current_target_sid and 
                prediction_confidence >= self.prediction_confidence_threshold):
                
                # DUEL LOCK PROTECTION: Kill predictions must respect locked duels
                should_use_prediction = True
                
                if self._is_duel_locked():
                    # Only allow predictions within the locked duel
                    if predicted_killer in self.locked_duel_participants:
                        pred_name = allplayers.get(predicted_killer, {}).get('name', 'Unknown')
                        logging.info(f"🔮 Duel prediction: {pred_name} ({prediction_confidence:.0%})")
                    else:
                        pred_name = allplayers.get(predicted_killer, {}).get('name', 'Unknown')
                        pass  # Prediction blocked by duel lock
                        should_use_prediction = False
                
                # HIGH CONFIDENCE PREDICTIONS - quality-focused approach
                if should_use_prediction and prediction_confidence >= self.high_confidence_override_threshold:
                    # Respect existing high-priority situations for viewing continuity
                    if self.death_switch_target or self.kill_focus_sid:
                        should_use_prediction = False  # Maintain narrative flow
                        pass  # Maintaining current focus
                    else:
                        should_use_prediction = True  # Use high-quality predictions
                        if self._is_duel_active():
                            pass  # Upgrading prediction target
                else:
                    # Normal confidence - be more conservative
                    if (self.death_switch_target or self.kill_focus_sid):
                        should_use_prediction = False
                    elif self._is_duel_active() and prediction_confidence < 0.75:
                        should_use_prediction = False
                
                if should_use_prediction:
                    pred_name = allplayers.get(predicted_killer, {}).get('name', 'Unknown')
                    confidence_desc = "HIGH CONFIDENCE" if prediction_confidence >= 0.85 else "MEDIUM CONFIDENCE"
                    logging.info(f"🔮 Prioritizing {pred_name} ({prediction_confidence:.0%})")
                    
                    # Mark this as a high confidence prediction for fast switching
                    if prediction_confidence >= self.high_confidence_override_threshold:
                        self._force_high_confidence_switch = True
                        self._high_confidence_level = prediction_confidence  # Store confidence for switching speed
                    
                    return predicted_killer, prediction_confidence, None, 0.0
        
        # Check if current duel is resolved
        if self._is_duel_active() and self._is_duel_resolved(allplayers):
            self._end_duel_tracking()
        
        # If we're actively tracking a duel and it's not resolved, stick with it
        if self._is_duel_active() and not self._is_duel_resolved(allplayers):
            current_target = self.active_duel_target
            sid1, sid2 = self.active_duel_participants
            
            # Make sure both duel participants are still alive and in a duel
            target_duel = self._find_duel_for_player(current_target, duels)
            if target_duel:
                # Score both participants to see if we should switch within the duel
                p1, p2 = target_duel
                dist = next((d for s1, s2, d in duels if (s1 == p1 and s2 == p2) or (s1 == p2 and s2 == p1)), 0)
                
                score1 = self.score_player_in_duel(p1, p2, allplayers, dist, time.time())
                score2 = self.score_player_in_duel(p2, p1, allplayers, dist, time.time())
                
                # Determine best target with higher threshold and debouncing
                can_switch_within_duel = True
                if self.last_duel_switch_time:
                    time_since_switch = (datetime.now() - self.last_duel_switch_time).total_seconds()
                    can_switch_within_duel = time_since_switch >= self.duel_switch_cooldown
                
                # Only switch if there's a significant advantage and cooldown has passed
                if can_switch_within_duel and score1 > score2 * self.duel_switch_threshold:
                    best_target = p1
                    best_score = score1
                elif can_switch_within_duel and score2 > score1 * self.duel_switch_threshold:
                    best_target = p2
                    best_score = score2
                else:
                    # Keep current target - either scores are close or cooldown active
                    best_target = current_target
                    best_score = score1 if current_target == p1 else score2
                    if not can_switch_within_duel:
                        pass  # Duel switch cooldown
                
                # Update duel target if we're switching within the duel
                if best_target != current_target:
                    self.active_duel_target = best_target
                    self.last_duel_switch_time = datetime.now()
                    target_name = allplayers.get(best_target, {}).get('name', '?')
                    score_diff = max(score1, score2) / min(score1, score2) if min(score1, score2) > 0 else 0
                    logging.info(f"🔄 Duel switch → {target_name}")
                
                return best_target, best_score, None, 0.0  # No second choice during duel lock
            else:
                # Current target is no longer in a duel, end tracking
                logging.info("🔓 Duel ended")
                self._end_duel_tracking()
        
        if not duels:
            return None, 0.0, None, 0.0

        best_sid = None; best_score = 0.0
        second_sid = None; second_score = 0.0
        now_t = time.time()

        # Calculate rule-based scores first to get candidates
        duel_candidates = []
        for sid1, sid2, dist in duels:
            s1 = self.score_player_in_duel(sid1, sid2, allplayers, dist, now_t)
            s2 = self.score_player_in_duel(sid2, sid1, allplayers, dist, now_t)
            duel_candidates.extend([(sid1, s1), (sid2, s2)])
        
        # Sort by rule-based score to get top candidates
        duel_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = duel_candidates[:4]  # Top 4 candidates
        
        # Try AI prediction on top candidates only (not stuck on one player)
        ai_prediction = None
        ai_confidence = 0.0
        if self.ai_analyzer and self.use_ai and top_candidates:
            try:
                # Create mini-gamestate with just top candidates for AI
                ai_candidates = [sid for sid, _ in top_candidates]
                ai_result = self.ai_analyzer.predict_best_observer_target(gamestate, candidates=ai_candidates)
                if isinstance(ai_result, tuple):
                    ai_prediction, ai_confidence = ai_result
                else:
                    ai_prediction = ai_result
                    ai_confidence = 0.8  # Default confidence
                
                # Only use AI prediction if it's among the top rule-based candidates
                if ai_prediction in ai_candidates:
                    # Check if we're using pattern memory
                    if hasattr(self.ai_analyzer, 'pattern_memory') and self.ai_analyzer.pattern_memory:
                        logging.info(f"🧠 AI suggests: {ai_prediction} ({ai_confidence:.0%})")
                    else:
                        logging.info(f"🧠 AI suggests: {ai_prediction} ({ai_confidence:.0%})")
                else:
                    ai_prediction = None  # AI suggestion not in top candidates, ignore it
            except Exception as e:
                pass  # AI prediction failed

        # Enhanced AI boost system based on training insights
        for sid, sc in duel_candidates:
            # Get player for kill analysis (AI learned kills are 96.7% important!)
            player = allplayers.get(sid, {})
            player_state = player.get('state', {})
            recent_kills = int(player_state.get('round_kills', 0))
            
            # AI boost with kill-aware weighting
            if ai_prediction == sid and sid != self.current_target_sid:
                # Base boost factor
                base_boost = 1.1 + (0.4 * ai_confidence)  # 1.1-1.5x
                
                # MASSIVE boost for players with recent kills (AI insight!)
                if recent_kills >= 2:
                    kill_multiplier = 1.8  # 80% extra boost for multi-kills
                elif recent_kills >= 1:
                    kill_multiplier = 1.4  # 40% extra boost for single kills
                else:
                    kill_multiplier = 1.0  # No kill bonus
                
                final_boost = base_boost * kill_multiplier
                sc *= final_boost
                
                if kill_multiplier > 1.0:
                    pass  # AI+Kill boost applied
                else:
                    pass  # AI boost applied
            
            # Additional kill-based scoring boost (rule-based)
            if recent_kills > 0:
                kill_bonus = min(recent_kills * 0.2, 0.6)  # Up to 60% bonus for kills
                sc += kill_bonus
            
            if sc > best_score:
                second_sid, second_score = best_sid, best_score
                best_sid, best_score = sid, sc
            elif sc > second_score and sid != best_sid:
                second_sid, second_score = sid, sc

        # If we're selecting a new best player, check if they're in a duel and start tracking
        if best_sid and not self._is_duel_active():
            duel_participants = self._find_duel_for_player(best_sid, duels)
            if duel_participants:
                p1, p2 = duel_participants
                # Only start duel tracking for high-quality duels (score > 0.4)
                if best_score > 0.4:
                    self._start_duel_tracking(p1, p2, best_sid, allplayers)

        return best_sid, best_score, second_sid, second_score

    def get_player_slot(self, steamid: str, gamestate: Dict[str, Any]) -> Optional[int]:
        try:
            allplayers = gamestate.get('allplayers', {})
            target = allplayers.get(steamid)
            if target:
                observer_slot = target.get('observer_slot')
                if observer_slot is not None:
                    return (observer_slot % 10) + 1 if observer_slot != 9 else 10
            lst = []
            for sid, pdata in allplayers.items():
                if sid == 'total': continue
                lst.append((sid, pdata.get('observer_slot', 0)))
            lst.sort(key=lambda x: x[1])
            for i, (sid, _) in enumerate(lst):
                if sid == steamid: return i+1
            return None
        except Exception as e:
            pass  # Slot conversion error
            return None

    def _numbers_lockout_active(self) -> bool:
        """True hvis 1-0 ikke må trykkes pga freezetime-lås (19s fra FT-start)."""
        if not self.freezetime_start:
            return False
        return (datetime.now() - self.freezetime_start).total_seconds() < self.numbers_lockout_seconds

    def _can_switch_now(self, force_high_confidence: bool = False, confidence_level: float = 0.85) -> bool:
        if self._numbers_lockout_active():
            return False
        
        # Viewing quality protection - prevent too many switches
        now = datetime.now()
        recent_switches = [t for t in self._recent_switches if (now - t).total_seconds() <= 10.0]
        if len(recent_switches) >= self._max_switches_per_10s:
            pass  # Limiting switches for quality
            return False
        
        # HIGH CONFIDENCE PREDICTIONS with quality-focused timing
        if force_high_confidence:
            now = datetime.now()
            
            # Ultra-high confidence (90%+) = moderate speed for quality viewing
            if confidence_level >= self.ultra_high_confidence_threshold:
                if (now - self.last_switch_time).total_seconds() < 0.6:  # Reasonable cooldown for viewing
                    return False
                if self.current_target_start and (now - self.current_target_start).total_seconds() < 1.0:  # Allow time to see the setup
                    return False
                return True
            
            # High confidence (85-90%) = slight speed increase for important moments
            else:
                if (now - self.last_switch_time).total_seconds() < 0.7:  # Slightly faster than normal
                    return False
                if self.current_target_start and (now - self.current_target_start).total_seconds() < 1.2:  # Slightly faster than normal
                    return False
                return True
        
        # Normal cooldown logic
        now = datetime.now()
        if (now - self.last_switch_time).total_seconds() < self.switch_cooldown:
            return False
        if self.current_target_start and (now - self.current_target_start).total_seconds() < self.min_hold_time:
            return False
        return True

    def _should_switch_by_score(self, new_score: float) -> bool:
        cur = self.current_target_score_ema
        return new_score >= cur * (1.0 + self.switch_margin)

    def _update_current_target(self, sid: str, inst_score: float):
        a = self.score_ema_alpha
        if self.current_target_sid != sid:
            self.current_target_sid = sid
            self.current_target_start = datetime.now()
            self.current_target_score_ema = inst_score
        else:
            self.current_target_score_ema = a*inst_score + (1-a)*self.current_target_score_ema

    def switch_to_player(self, slot: int):
        """Tryk på 1..0 for at skifte – respekterer 19s freezetime-lås."""
        try:
            if self._numbers_lockout_active():
                # Log kun på debug-niveau for ikke at støje
                pass  # Numbers lockout active
                return
            if slot in VK_CODES:
                vk_code = VK_CODES[slot]
                if send_key(vk_code):
                    key_name = str(slot) if slot != 10 else '0'
                    logging.info(f"→ Switched to {key_name}")
                    self.last_switch_time = datetime.now()
                    self.last_any_switch = time.time()
        except Exception as e:
            pass  # Switch error

    def toggle_active(self):
        self.active = not self.active
        status = "AKTIVT" if self.active else "INAKTIVT"
        print(f"Duel detection: {status}", flush=True)
        logging.info(f"Observer toggled: {status}")

    def toggle_campaths(self):
        """Toggle campaths (AQWE/UIOP hotkeys) on/off"""
        self.campaths_enabled = not self.campaths_enabled
        status = "enabled" if self.campaths_enabled else "disabled"
        print(f"🎬 Campaths {status}", flush=True)
        logging.info(f"🎬 Campaths {status}")

    # ------------- Main processing -------------

    def process_gamestate(self, gamestate: Dict[str, Any]):
        try:
            if not self.active:
                return

            self._last_gamestate = gamestate
            allplayers = gamestate.get('allplayers', {}) or {}
            now_ts = time.time()
            now = datetime.now()
            
            # PRIORITY 1: Track pre-engagement duels and lock onto them FIRST
            self._update_duel_lock_tracking(allplayers, now)
            
            # Track current map for kill analysis integration
            map_info = gamestate.get('map', {})
            current_map = None
            if map_info:
                current_map = map_info.get('name', '').lower()
                if current_map:
                    # Only set if different from current
                    if current_map != getattr(self, '_current_map', None):
                        self._current_map = current_map
                        
                        # Log map-specific kill analysis data (only once per map)
                        if hasattr(self, 'map_kill_distributions') and self.map_kill_distributions:
                            map_kills = self.map_kill_distributions.get(current_map, 0)
                            if map_kills > 0:
                                logging.info(f"🗺️ Map: {current_map}")
            
                # Update map geometry if available - FIXED: Only when map actually changes
                if self.geometry_manager and LOS_AVAILABLE:
                    # Check if map actually changed (not just missing attribute)
                    last_geometry_map = getattr(self, '_geometry_map', None)
                    if current_map != last_geometry_map:
                        self._geometry_map = current_map  # Store for comparison
                        try:
                            self.geometry_manager.set_current_map(current_map)
                            logging.info(f"🗺️ Map: {current_map}")
                        except Exception as e:
                            pass  # Geometry load failed
            
            self.update_history_and_triggers(allplayers, now_ts, gamestate)

            # fase
            round_phase = (gamestate.get('round') or {}).get('phase')

            # Debugging: Log map_phase and round.phase
            pass  # Round phase debug

            # Handle FreezetimeStarted event using round_phase
            if round_phase == "freezetime" and not self.freezetime_fkey_sent:
                self.freezetime_start = datetime.now()
                self.freezetime_fkey_sent = True

                # Log FreezetimeStart
                logging.info("❄️ Freezetime started")
                print("[Freezetime] Freezetime started.")

                # Simulate pressing a random hotkey (A, Q, W, E, U, I, O, P) only once if campaths are enabled
                if self.campaths_enabled:
                    hotkey_code = random.choice(HOTKEY_CODES)
                    if send_key(hotkey_code):
                        hotkey_char = chr(hotkey_code)
                        pass  # Random hotkey pressed
                        print(f"[Freezetime] Pressed random hotkey: {hotkey_char}")
                else:
                    print(f"[Freezetime] Campaths disabled - no hotkey pressed")

            # Handle FreezetimeEnded event using round_phase
            if round_phase != "freezetime" and self.freezetime_fkey_sent:
                self.freezetime_fkey_sent = False

                # Log FreezetimeStop
                logging.info("🚀 Round started")
                print("[Freezetime] Freezetime ended.")

            # Lockout for player keys (1-0) during freezetime
            if self.freezetime_start and (datetime.now() - self.freezetime_start).total_seconds() < 19:
                pass  # Numbers lockout active
                return

            # DUEL LOCK PROTECTION: If duel is locked, triggers are heavily restricted
            if self._is_duel_locked():
                # ABSOLUTE DUEL LOCK PRIORITY - only allow triggers within the locked duel
                trigger_sid = None
                
                # Check if any trigger is within the locked duel
                potential_triggers = []
                if self.death_switch_target and now_ts <= self.death_switch_active_until:
                    if self.death_switch_target in self.locked_duel_participants:
                        potential_triggers.append(('death', self.death_switch_target))
                
                if self.kill_focus_sid and self.kill_focus_sid in self.locked_duel_participants:
                    potential_triggers.append(('kill', self.kill_focus_sid))
                
                if self.damage_focus_sid and self.damage_focus_sid in self.locked_duel_participants:
                    potential_triggers.append(('damage', self.damage_focus_sid))
                
                # Only honor triggers within the locked duel
                if potential_triggers:
                    # Prioritize death > kill > damage for duel participants
                    trigger_type, trigger_sid = potential_triggers[0]  # Already sorted by priority
                    
                    if trigger_sid != self.current_target_sid:
                        # Switch within locked duel
                        can_switch = self._can_switch_now()
                        if can_switch:
                            slot = self.get_player_slot(trigger_sid, gamestate)
                            if slot:
                                self.switch_to_player(slot)
                                self._update_current_target(trigger_sid, 0.9)
                                self.locked_duel_target = trigger_sid  # Update duel target
                                trigger_name = allplayers.get(trigger_sid, {}).get('name', '?')
                                logging.info(f"🎯 → {trigger_name} ({trigger_type})")
                                return
                        else:
                            pass  # Trigger ignored - cooldown
                
                # If no valid triggers within duel, ignore all external triggers
                if self.death_switch_target and self.death_switch_target not in self.locked_duel_participants:
                    logging.info(f"🎯 [DUEL LOCK PROTECTION] Ignoring death trigger - not in locked duel")
                if self.kill_focus_sid and self.kill_focus_sid not in self.locked_duel_participants:
                    logging.info(f"🎯 [DUEL LOCK PROTECTION] Ignoring kill trigger - not in locked duel")
                if self.damage_focus_sid and self.damage_focus_sid not in self.locked_duel_participants:
                    logging.info(f"🎯 [DUEL LOCK PROTECTION] Ignoring damage trigger - not in locked duel")
                
                # Skip all trigger processing - duel lock is absolute
                return
            
            # NORMAL TRIGGER PRIORITY: Only when no duel is locked
            trigger_sid = None
            if self.death_switch_target and now_ts <= self.death_switch_active_until:
                trigger_sid = self.death_switch_target
                logging.debug(f"[DEATH PRIORITY] Maintaining focus on killer for {self.death_switch_active_until - now_ts:.1f}s more")
            elif self.kill_focus_sid:
                trigger_sid = self.kill_focus_sid
            elif self.damage_focus_sid:
                trigger_sid = self.damage_focus_sid

            if trigger_sid and trigger_sid != self.current_target_sid:
                # Normal trigger processing (when no duel locked)
                should_switch = True
                
                # High-confidence predictions can override cooldowns for trigger switches
                can_switch = self._can_switch_now(
                    force_high_confidence=self._force_high_confidence_switch,
                    confidence_level=self._high_confidence_level
                )
                if not can_switch and trigger_sid == self.death_switch_target:
                    # Death switches always override cooldowns
                    can_switch = True
                elif not can_switch and trigger_sid == self.kill_focus_sid:
                    # Kill triggers can override short cooldowns
                    time_since_switch = (datetime.now() - self.last_switch_time).total_seconds()
                    if time_since_switch >= 0.5:  # Reduced from full cooldown
                        can_switch = True
                        logging.info(f"[KILL TRIGGER OVERRIDE] Overriding cooldown for kill trigger")
                
                if should_switch and can_switch:
                    slot = self.get_player_slot(trigger_sid, gamestate)
                    if slot:
                        self.switch_to_player(slot)
                        self._update_current_target(trigger_sid, 0.9)
                        return  # trigger har forrang i kort tid

            # NAV-AWARE PREDICTIVE ANALYSIS (2-4 seconds ahead prediction)
            predictive_override_sid = None
            if self.use_predictive_observer and self.predictive_observer:
                try:
                    map_name = getattr(self, '_current_map', 'de_dust2')
                    suggested_player = self.predictive_observer.update_existing_detector(allplayers, map_name)
                    
                    if suggested_player:
                        # Find steam ID for suggested player
                        for sid, player_data in allplayers.items():
                            if sid != 'total' and player_data.get('name') == suggested_player:
                                predictive_override_sid = sid
                                break
                                
                        if predictive_override_sid and predictive_override_sid != self.current_target_sid:
                            # Get prediction confidence for logging
                            predictions = self.predictive_observer.active_predictions
                            confidence = predictions.get(suggested_player, type('obj', (object,), {'confidence': 0.0})()).confidence
                            
                            # High confidence predictions override normal selection
                            if confidence >= self.predictive_observer.high_confidence_threshold:
                                if self._can_switch_now():
                                    slot = self.get_player_slot(predictive_override_sid, gamestate)
                                    if slot:
                                        self.switch_to_player(slot)
                                        self._update_current_target(predictive_override_sid, min(0.95, confidence))
                                        logging.info(f"🔮 PREDICTIVE OVERRIDE: {suggested_player} (confidence: {confidence:.3f})")
                                        return  # Skip normal selection for high confidence predictions
                except Exception as e:
                    logging.error(f"Predictive observer error: {e}")

            # NORMAL VALG
            # DUEL LOCK: If pre-engagement duel detected, only allow switching within that duel
            if self._is_duel_locked():
                current_in_duel = (self.current_target_sid and 
                                 self.current_target_sid in self.locked_duel_participants)
                
                if current_in_duel:
                    # We're watching one participant - check if we should switch to the other
                    other_participant = None
                    for participant in self.locked_duel_participants:
                        if participant != self.current_target_sid:
                            other_participant = participant
                            break
                    
                    if other_participant:
                        other_player = allplayers.get(other_participant, {})
                        current_player = allplayers.get(self.current_target_sid, {})
                        
                        # Switch if other participant has higher kill probability
                        try:
                            other_score = self.score_player_in_duel(other_participant, self.current_target_sid, allplayers, 
                                                                   distance2d(parse_vec3(other_player.get('position', '')) or (0,0,0),
                                                                            parse_vec3(current_player.get('position', '')) or (0,0,0)), now_ts)
                            current_score = self.current_target_score_ema
                        except Exception:
                            # Skip switching if position calculation fails
                            return
                        
                        # Easier switching within locked duel (especially as engagement intensifies)
                        if other_score > current_score * 1.1 and self._can_switch_now():  # Lower threshold for duel switching
                            slot = self.get_player_slot(other_participant, gamestate)
                            if slot:
                                self.switch_to_player(slot)
                                self._update_current_target(other_participant, other_score)
                                current_name = allplayers.get(self.current_target_sid, {}).get('name', '?')
                                other_name = allplayers.get(other_participant, {}).get('name', '?')
                                logging.info(f"🎯 [DUEL SWITCH] Switched from {current_name} to {other_name} within locked duel")
                                self.locked_duel_target = other_participant
                    
                    return  # LOCKED - don't consider any other targets during duel
                
                else:
                    # We're not watching the locked duel - switch to it immediately
                    # Use the pre-determined target or pick the participant with higher kill probability
                    target_sid = self.locked_duel_target
                    
                    if not target_sid or target_sid not in self.locked_duel_participants:
                        # Re-calculate best target in locked duel
                        p1, p2 = self.locked_duel_participants
                        p1_data = allplayers.get(p1, {})
                        p2_data = allplayers.get(p2, {})
                        
                        if p1_data and p2_data:
                            try:
                                p1_pos = parse_vec3(p1_data.get('position', '')) or (0,0,0)
                                p2_pos = parse_vec3(p2_data.get('position', '')) or (0,0,0)
                                dist = distance2d(p1_pos, p2_pos)
                                
                                p1_score = self.score_player_in_duel(p1, p2, allplayers, dist, now_ts)
                                p2_score = self.score_player_in_duel(p2, p1, allplayers, dist, now_ts)
                                
                                target_sid = p1 if p1_score >= p2_score else p2
                                self.locked_duel_target = target_sid
                            except Exception:
                                # Fallback to first participant if scoring fails
                                target_sid = p1
                                self.locked_duel_target = target_sid
                    
                    if target_sid:
                        slot = self.get_player_slot(target_sid, gamestate)
                        
                        if slot and self._can_switch_now(force_high_confidence=True, confidence_level=0.9):
                            self.switch_to_player(slot)
                            try:
                                target_score = max(self.score_player_in_duel(target_sid, other_sid, allplayers, 
                                                                           distance2d(parse_vec3(allplayers.get(target_sid, {}).get('position', '')) or (0,0,0),
                                                                                    parse_vec3(allplayers.get(other_sid, {}).get('position', '')) or (0,0,0)), now_ts)
                                                 for other_sid in self.locked_duel_participants if other_sid != target_sid)
                            except Exception:
                                target_score = 0.5  # Default score if calculation fails
                            self._update_current_target(target_sid, target_score)
                            target_name = allplayers.get(target_sid, {}).get('name', '?')
                            logging.info(f"🎯 [DUEL LOCK] Switched to {target_name} - locked pre-engagement duel!")
                    
                    return  # LOCKED - duel takes absolute priority

            best_sid, best_score, second_sid, second_score = self.choose_best_player(gamestate)
            if not best_sid:
                # Anti-stall
                if now_ts - self.last_any_switch > self.failsafe_interval and self._can_switch_now():
                    candidate = self._pick_any_live(allplayers)
                    if candidate:
                        slot = self.get_player_slot(candidate, gamestate)
                        if slot:
                            self.switch_to_player(slot)
                            self._update_current_target(candidate, 0.4)
                return

            if best_score < 0.28:
                if now_ts - self.last_any_switch > self.failsafe_interval and self._can_switch_now():
                    slot = self.get_player_slot(best_sid, gamestate)
                    if slot:
                        self.switch_to_player(slot)
                        self._update_current_target(best_sid, best_score)
                return

            # Rotation
            if second_sid and (best_sid != second_sid) and abs(best_score - second_score) <= self.rotation_delta:
                if self._can_switch_now() and (datetime.now() - (self.current_target_start or datetime.min)).total_seconds() >= self.rotation_window:
                    pick = second_sid if self.current_target_sid == best_sid else best_sid
                    slot = self.get_player_slot(pick, gamestate)
                    if slot:
                        self.switch_to_player(slot)
                        self._update_current_target(pick, second_score if pick == second_sid else best_score)
                else:
                    if self.current_target_sid:
                        self._update_current_target(self.current_target_sid, max(0.0, self.current_target_score_ema*0.95))
                return

            # Hysterese with high confidence override
            can_switch = self._can_switch_now(
                force_high_confidence=self._force_high_confidence_switch,
                confidence_level=self._high_confidence_level
            )
            if (self.current_target_sid != best_sid and can_switch and
                (self.current_target_sid is None or self._should_switch_by_score(best_score))):
                slot = self.get_player_slot(best_sid, gamestate)
                if slot:
                    if self._force_high_confidence_switch:
                        if self._high_confidence_level >= self.ultra_high_confidence_threshold:
                            logging.info(f"🎯 [QUALITY SWITCH] {self._high_confidence_level:.0%} confidence - balanced for viewing experience")
                        else:
                            logging.info(f"📈 [PRIORITY SWITCH] {self._high_confidence_level:.0%} confidence - slightly accelerated")
                    self.switch_to_player(slot)
                    self._update_current_target(best_sid, best_score)
                    # Track switch for viewing quality control
                    self._recent_switches.append(datetime.now())
                    # Reset flags after use
                    self._force_high_confidence_switch = False
                    self._high_confidence_level = 0.0
                return

            # Opdatér EMA for nuværende target
            if self.current_target_sid == best_sid:
                self._update_current_target(best_sid, best_score)
            elif self.current_target_sid:
                self._update_current_target(self.current_target_sid, max(0.0, self.current_target_score_ema*0.95))

        except Exception as e:
            logging.error(f"Fejl i gamestate processing: {e}")

    def _pick_any_live(self, allplayers: Dict[str, Any]) -> Optional[str]:
        for sid, p in allplayers.items():
            if sid == 'total': continue
            if p.get('state',{}).get('health',0) > 0:
                return sid
        return None

    # ------------- Duel Persistence Management -------------

    def _is_duel_active(self) -> bool:
        """Check if we're currently locked onto an active duel"""
        return (self.active_duel_participants is not None and 
                self.active_duel_start_time is not None and
                self.active_duel_target is not None)

    def _is_duel_resolved(self, allplayers: Dict[str, Any]) -> bool:
        """Check if the current active duel has been resolved"""
        if not self._is_duel_active():
            return True
            
        sid1, sid2 = self.active_duel_participants
        p1 = allplayers.get(sid1, {})
        p2 = allplayers.get(sid2, {})
        
        # Check if either player died
        p1_hp = p1.get('state', {}).get('health', 0)
        p2_hp = p2.get('state', {}).get('health', 0)
        if p1_hp <= 0 or p2_hp <= 0:
            logging.info(f"[DUEL RESOLVED] Player death - {p1.get('name', '?')} ({p1_hp}hp) vs {p2.get('name', '?')} ({p2_hp}hp)")
            return True
            
        # Check if players are too far apart (duel broke off)
        try:
            pos1 = parse_vec3(p1.get('position', '')) if p1.get('position') else None
            pos2 = parse_vec3(p2.get('position', '')) if p2.get('position') else None
            if pos1 and pos2:
                dist = distance2d(pos1, pos2)
                if dist > self.duel_separation_distance:
                    logging.info(f"[DUEL RESOLVED] Players separated - distance: {dist:.0f} units")
                    return True
        except Exception:
            # Ignore position parsing errors for duel resolution
            pass
                
        # Safety timeout - don't lock onto duels forever
        if self.active_duel_start_time:
            elapsed = (datetime.now() - self.active_duel_start_time).total_seconds()
            if elapsed > self.duel_max_duration:
                logging.info(f"[DUEL RESOLVED] Safety timeout after {elapsed:.1f}s")
                return True
                
        return False

    def _start_duel_tracking(self, sid1: str, sid2: str, target_sid: str, allplayers: Dict[str, Any]):
        """Start tracking a duel between two players"""
        self.active_duel_participants = (sid1, sid2)
        self.active_duel_start_time = datetime.now()
        self.active_duel_target = target_sid
        self.last_duel_switch_time = datetime.now()  # Initialize switch cooldown
        
        p1_name = allplayers.get(sid1, {}).get('name', '?')
        p2_name = allplayers.get(sid2, {}).get('name', '?')
        target_name = allplayers.get(target_sid, {}).get('name', '?')
        
        logging.info(f"🔒 Duel lock: {p1_name} vs {p2_name} → watching {target_name}")

    def _end_duel_tracking(self):
        """Stop tracking the current duel"""
        if self._is_duel_active():
            logging.info("[DUEL ENDED] Releasing duel lock - can switch to other targets")
        self.active_duel_participants = None
        self.active_duel_start_time = None
        self.active_duel_target = None
        self.last_duel_switch_time = None  # Reset switch cooldown
        self.last_duel_scores.clear()

    # ------------- Pre-Engagement Duel Lock Detection -------------

    def _detect_pre_engagement_duel(self, allplayers: Dict[str, Any], now: datetime) -> Optional[Tuple[str, str]]:
        """
        Detect when two players are in pre-engagement phase (approaching, positioning for duel).
        Uses NAV intelligence and smart prioritization to avoid rapid switching.
        Returns tuple of (player1_sid, player2_sid) if pre-engagement duel detected.
        """
        pre_engagement_pairs = []
        
        # Find potential pre-engagement duels
        for sid1, p1 in allplayers.items():
            if sid1 == 'total' or p1.get('state', {}).get('health', 0) <= 0:
                continue
                
            for sid2, p2 in allplayers.items():
                if sid2 == 'total' or sid2 == sid1 or p2.get('state', {}).get('health', 0) <= 0:
                    continue
                
                # Must be on different teams
                if p1.get('team', '') == p2.get('team', ''):
                    continue
                
                # Check if they're in pre-engagement positioning
                pre_engagement_score = self._calculate_pre_engagement_score(sid1, sid2, p1, p2, allplayers, now)
                
                if pre_engagement_score >= 0.65:  # INCREASED threshold to reduce noise
                    pre_engagement_pairs.append((sid1, sid2, pre_engagement_score))
        
        # SMART PRIORITIZATION - avoid rapid switching between similar duels
        if pre_engagement_pairs:
            # 1. PERSISTENCE BONUS: If we already have a locked duel, heavily favor continuing it
            if self.locked_duel_participants:
                current_pair = tuple(sorted(self.locked_duel_participants))
                for sid1, sid2, score in pre_engagement_pairs:
                    pair = tuple(sorted([sid1, sid2]))
                    if pair == current_pair:
                        # MASSIVE bonus for continuing current duel (reduces switching)
                        enhanced_score = score + 0.3  # Big persistence bonus
                        return current_pair  # Strong preference to continue
            
            # 2. SPECTATOR TARGET PRIORITY: If current target is in a duel, prefer that duel
            if self.current_target_sid:
                for sid1, sid2, score in pre_engagement_pairs:
                    if self.current_target_sid in (sid1, sid2) and score >= 0.7:  # Higher threshold
                        return tuple(sorted([sid1, sid2]))
            
            # 3. QUALITY OVER QUANTITY: Only switch to significantly better duels
            pre_engagement_pairs.sort(key=lambda x: x[2], reverse=True)
            best_score = pre_engagement_pairs[0][2]
            
            # Require high-quality duel for switching (0.75+ score)
            if best_score >= 0.75:
                return tuple(sorted([pre_engagement_pairs[0][0], pre_engagement_pairs[0][1]]))
            
        return None

    def _calculate_pre_engagement_score(self, sid1: str, sid2: str, p1: Dict[str, Any], p2: Dict[str, Any], 
                                       allplayers: Dict[str, Any], now: datetime) -> float:
        """Calculate how likely these two players are to engage in a duel soon - ENHANCED with NAV intelligence"""
        
        # Get positions and orientations
        p1_pos = parse_vec3(p1.get('position', '')) if p1.get('position') else None
        p2_pos = parse_vec3(p2.get('position', '')) if p2.get('position') else None
        p1_fwd = parse_vec3(p1.get('forward', '')) if p1.get('forward') else None
        p2_fwd = parse_vec3(p2.get('forward', '')) if p2.get('forward') else None
        
        if not all([p1_pos, p2_pos, p1_fwd, p2_fwd]):
            return 0.0
        
        score = 0.0
        distance = distance2d(p1_pos, p2_pos)
        
        # 1. Distance suitability (20% weight) - must be in reasonable duel range
        if distance <= self.pre_engagement_detection_distance:
            # Sweet spot for engagement prediction
            if 400 <= distance <= 1200:  # Optimal pre-engagement range
                distance_score = 1.0
            elif 200 <= distance <= 1600:  # Good range
                distance_score = 0.8
            else:  # Acceptable range
                distance_score = 0.6
            score += 0.20 * distance_score
        else:
            return 0.0  # Too far for pre-engagement
        
        # 2. ISOLATION FACTOR (25% weight) - CRITICAL for avoiding multi-engagement chaos
        isolation_score = self._calculate_duel_isolation(sid1, sid2, allplayers)
        score += 0.25 * isolation_score
        
        # If this isn't an isolated duel, heavily penalize it
        if isolation_score < 0.5:
            score *= 0.4  # Major penalty for chaotic multi-player situations
        
        # 3. Mutual awareness/facing (20% weight) - are they aware of each other?
        p1_facing_p2 = angle_facing_score(p1_pos, p1_fwd, p2_pos)
        p2_facing_p1 = angle_facing_score(p2_pos, p2_fwd, p1_pos)
        
        # More strict facing requirements to reduce noise
        if p1_facing_p2 >= 0.4 and p2_facing_p1 >= 0.4:  # INCREASED thresholds
            score += 0.20  # Both facing each other clearly
        elif p1_facing_p2 >= 0.3 or p2_facing_p1 >= 0.3:  # INCREASED threshold
            score += 0.12  # One facing the other clearly
        elif max(p1_facing_p2, p2_facing_p1) >= 0.2:  # INCREASED threshold
            score += 0.06  # Minimal clear awareness
        
        # 4. Approach/positioning movement (20% weight)
        approach_score = self._calculate_approach_score(sid1, sid2, now)
        score += 0.20 * approach_score
        
        # 5. Line of sight with NAV intelligence (10% weight)
        if self.geometry_manager and LOS_AVAILABLE:
            try:
                if duel_geometrically_possible(p1_pos, p2_pos, self.geometry_manager):
                    score += 0.10
                    # NAV BONUS: If this is a known duel spot (common engagement area)
                    if hasattr(self, 'predictive_observer') and self.predictive_observer:
                        try:
                            nav_bonus = self.predictive_observer.get_engagement_area_score(p1_pos, p2_pos)
                            score += 0.05 * nav_bonus  # Bonus for known engagement areas
                        except:
                            pass
            except:
                return 0.0  # No line of sight = no duel
        else:
            score += 0.03  # Minimal credit without geometry
        
        # 6. Weapon suitability and kill prediction (5% weight)
        if self.positional_ai:
            try:
                ai_confidence = self._get_positional_ai_confidence(sid1, sid2, p1, p2, distance)
                score += 0.05 * ai_confidence
            except:
                pass
        
        return min(score, 1.0)

    def _calculate_duel_isolation(self, sid1: str, sid2: str, allplayers: Dict[str, Any]) -> float:
        """
        Calculate how isolated this potential duel is (critical for avoiding rapid switching).
        Returns 1.0 for perfect 1v1 isolation, 0.0 for chaotic multi-player situations.
        """
        p1 = allplayers.get(sid1, {})
        p2 = allplayers.get(sid2, {})
        
        p1_pos = parse_vec3(p1.get('position', '')) if p1.get('position') else None
        p2_pos = parse_vec3(p2.get('position', '')) if p2.get('position') else None
        
        if not p1_pos or not p2_pos:
            return 0.0
        
        # Calculate duel center point
        center_x = (p1_pos[0] + p2_pos[0]) / 2
        center_y = (p1_pos[1] + p2_pos[1]) / 2
        center_pos = (center_x, center_y, 0)
        
        duel_distance = distance2d(p1_pos, p2_pos)
        isolation_radius = max(800, duel_distance * 1.5)  # Dynamic isolation check
        
        # Count other players near this duel
        nearby_enemies = 0
        nearby_allies = 0
        
        p1_team = p1.get('team', '')
        p2_team = p2.get('team', '')
        
        for other_sid, other_player in allplayers.items():
            if other_sid == 'total' or other_sid in (sid1, sid2):
                continue
                
            if other_player.get('state', {}).get('health', 0) <= 0:
                continue
                
            other_pos = parse_vec3(other_player.get('position', '')) if other_player.get('position') else None
            if not other_pos:
                continue
                
            # Distance to duel center
            dist_to_duel = distance2d(other_pos, center_pos)
            
            if dist_to_duel <= isolation_radius:
                other_team = other_player.get('team', '')
                if other_team == p1_team or other_team == p2_team:
                    nearby_allies += 1
                else:
                    nearby_enemies += 1
        
        # Calculate isolation score
        total_nearby = nearby_enemies + nearby_allies
        
        if total_nearby == 0:
            return 1.0  # Perfect isolation - clean 1v1
        elif total_nearby == 1:
            return 0.8  # Very good - one other player nearby
        elif total_nearby == 2:
            return 0.6  # Acceptable - small skirmish
        elif total_nearby == 3:
            return 0.3  # Poor - getting chaotic
        else:
            return 0.1  # Very poor - multi-player chaos
        
    def _get_positional_ai_confidence(self, sid1: str, sid2: str, p1: Dict[str, Any], p2: Dict[str, Any], distance: float) -> float:
        """Get AI confidence for this potential duel"""
        try:
            # Simplified feature extraction for pre-engagement
            features = {
                'distance': distance,
                'p1_health': p1.get('state', {}).get('health', 100),
                'p2_health': p2.get('state', {}).get('health', 100),
                'p1_armor': p1.get('state', {}).get('armor', 0),
                'p2_armor': p2.get('state', {}).get('armor', 0),
            }
            
            # Get AI prediction
            prediction = self.positional_ai.predict_engagement_quality(features)
            return prediction
        except:
            return 0.5  # Neutral if AI fails

    def _calculate_approach_score(self, sid1: str, sid2: str, now: datetime) -> float:
        """Calculate how much the players are approaching each other"""
        
        # Get movement history
        p1_prev, p1_curr = self.history.last2(sid1)
        p2_prev, p2_curr = self.history.last2(sid2)
        
        if not all([p1_prev, p1_curr, p2_prev, p2_curr]):
            return 0.0
        
        # Calculate relative approach speed
        closing_speed = relative_approach_speed_2d(p1_prev, p1_curr, p2_prev, p2_curr)
        
        approach_score = 0.0
        
        # Reward approach movement
        if closing_speed > 20:  # Fast approach
            approach_score += 0.8
        elif closing_speed > 10:  # Medium approach
            approach_score += 0.6
        elif closing_speed > self.approach_speed_threshold:  # Slow approach
            approach_score += 0.4
        elif closing_speed > 0:  # Any approach
            approach_score += 0.2
        
        # Check for tactical positioning (slowing down for engagement)
        p1_recent_positions = list(self.history.store.get(sid1, []))[-4:]
        p2_recent_positions = list(self.history.store.get(sid2, []))[-4:]
        
        # Look for deceleration patterns (common before engagement)
        if len(p1_recent_positions) >= 3:
            p1_speeds = []
            for i in range(1, len(p1_recent_positions)):
                if p1_recent_positions[i].get('pos') and p1_recent_positions[i-1].get('pos'):
                    dt = max(p1_recent_positions[i]['t'] - p1_recent_positions[i-1]['t'], 0.1)
                    movement = distance2d(p1_recent_positions[i]['pos'], p1_recent_positions[i-1]['pos'])
                    speed = movement / dt
                    p1_speeds.append(speed)
            
            # Check for deceleration (slowing down for engagement)
            if len(p1_speeds) >= 2 and p1_speeds[-1] < p1_speeds[0] * 0.7:  # 30% slower
                approach_score += 0.2  # Bonus for tactical slowing
        
        return min(approach_score, 1.0)

    def _update_duel_lock_tracking(self, allplayers: Dict[str, Any], now: datetime):
        """Update pre-engagement duel lock tracking with anti-rapid-switching"""
        
        # ANTI-RAPID-SWITCHING: If we just switched duels, enforce cooldown
        if (self.last_duel_switch_time and 
            (now - self.last_duel_switch_time).total_seconds() < self.duel_switch_cooldown_seconds):
            return  # Still in cooldown - don't switch duels
        
        # PERSISTENCE: If we have a current duel and it hasn't been active long enough, stick with it
        if (self.locked_duel_participants and self.locked_duel_start_time and 
            (now - self.locked_duel_start_time).total_seconds() < self.minimum_duel_lock_duration):
            # Check if current duel is still valid (not resolved)
            if not self._is_locked_duel_resolved(allplayers, now):
                return  # Stay locked to current duel
        
        # Detect new pre-engagement duels
        pre_engagement_duel = self._detect_pre_engagement_duel(allplayers, now)
        
        if pre_engagement_duel:
            if self.locked_duel_participants != pre_engagement_duel:
                # SWITCHING LOGIC: Only switch if significantly better or current is resolved
                should_switch = False
                
                if not self.locked_duel_participants:
                    # No current duel - safe to lock onto new one
                    should_switch = True
                elif self._is_locked_duel_resolved(allplayers, now):
                    # Current duel resolved - safe to switch
                    should_switch = True
                else:
                    # Have current duel - only switch if new one is MUCH better
                    current_score = self._calculate_duel_pair_score(self.locked_duel_participants, allplayers, now)
                    new_score = self._calculate_duel_pair_score(pre_engagement_duel, allplayers, now)
                    
                    # Require 40% better score to switch (high bar)
                    if new_score > current_score * 1.4:
                        should_switch = True
                        logging.info(f"🎯 [DUEL OVERRIDE] Switching to much better duel (score: {new_score:.2f} vs {current_score:.2f})")
                
                if should_switch:
                    # Log the previous duel end if switching
                    if self.locked_duel_participants:
                        prev_p1_name = allplayers.get(self.locked_duel_participants[0], {}).get('name', '?')
                        prev_p2_name = allplayers.get(self.locked_duel_participants[1], {}).get('name', '?')
                        logging.info(f"🎯 [DUEL ENDED] {prev_p1_name} vs {prev_p2_name} - Switching to new duel")
                    
                    # Switch to new duel
                    self.locked_duel_participants = pre_engagement_duel
                    self.locked_duel_start_time = now
                    self.last_duel_switch_time = now  # Record switch time
                    
                    p1_name = allplayers.get(pre_engagement_duel[0], {}).get('name', '?')
                    p2_name = allplayers.get(pre_engagement_duel[1], {}).get('name', '?')
                    logging.info(f"🎯 [DUEL LOCK] {p1_name} vs {p2_name} - Pre-engagement detected, DUEL LOCKED!")
                    
                    # Set initial target to the player with higher kill probability
                    try:
                        p1_data = allplayers.get(pre_engagement_duel[0], {})
                        p2_data = allplayers.get(pre_engagement_duel[1], {})
                        p1_pos = parse_vec3(p1_data.get('position', '')) or (0,0,0)
                        p2_pos = parse_vec3(p2_data.get('position', '')) or (0,0,0)
                        dist = distance2d(p1_pos, p2_pos)
                        
                        p1_score = self.score_player_in_duel(pre_engagement_duel[0], pre_engagement_duel[1], allplayers, dist, now.timestamp())
                        p2_score = self.score_player_in_duel(pre_engagement_duel[1], pre_engagement_duel[0], allplayers, dist, now.timestamp())
                        
                        self.locked_duel_target = pre_engagement_duel[0] if p1_score >= p2_score else pre_engagement_duel[1]
                    except Exception:
                        # Fallback to first participant if scoring fails
                        self.locked_duel_target = pre_engagement_duel[0]
            else:
                # Continue existing duel lock - check if it should continue
                pass
        else:
            # No pre-engagement duel detected
            if self.locked_duel_participants:
                # Check if duel is resolved
                if self._is_locked_duel_resolved(allplayers, now):
                    p1_name = allplayers.get(self.locked_duel_participants[0], {}).get('name', '?')
                    p2_name = allplayers.get(self.locked_duel_participants[1], {}).get('name', '?')
                    logging.info(f"🎯 [DUEL RESOLVED] {p1_name} vs {p2_name} - Duel lock released")
                    self.locked_duel_participants = None
                    self.locked_duel_start_time = None
                    self.locked_duel_target = None

    def _calculate_duel_pair_score(self, duel_pair: Tuple[str, str], allplayers: Dict[str, Any], now: datetime) -> float:
        """Calculate overall quality score for a duel pair"""
        if not duel_pair:
            return 0.0
            
        p1_data = allplayers.get(duel_pair[0], {})
        p2_data = allplayers.get(duel_pair[1], {})
        
        if not p1_data or not p2_data:
            return 0.0
            
        return self._calculate_pre_engagement_score(duel_pair[0], duel_pair[1], p1_data, p2_data, allplayers, now)

    def _is_duel_locked(self) -> bool:
        """Check if there's an active duel lock that should prevent switching away"""
        return (self.locked_duel_participants is not None and 
                self.locked_duel_start_time is not None)

    def _is_locked_duel_resolved(self, allplayers: Dict[str, Any], now: datetime) -> bool:
        """Check if the locked duel has been resolved"""
        if not self._is_duel_locked():
            return True
            
        p1_sid, p2_sid = self.locked_duel_participants
        p1 = allplayers.get(p1_sid, {})
        p2 = allplayers.get(p2_sid, {})
        
        # Check if either player died (duel resolved by elimination)
        p1_hp = p1.get('state', {}).get('health', 0)
        p2_hp = p2.get('state', {}).get('health', 0)
        if p1_hp <= 0 or p2_hp <= 0:
            return True
            
        # Check if players separated too much (duel resolved by disengagement)
        try:
            p1_pos = parse_vec3(p1.get('position', '')) if p1.get('position') else None
            p2_pos = parse_vec3(p2.get('position', '')) if p2.get('position') else None
            if p1_pos and p2_pos:
                distance = distance2d(p1_pos, p2_pos)
                if distance > self.duel_resolution_distance:
                    return True
        except Exception:
            # Ignore position parsing errors for locked duel resolution
            pass
                
        # Safety timeout - don't lock forever
        if self.locked_duel_start_time:
            elapsed = (now - self.locked_duel_start_time).total_seconds()
            if elapsed > self.duel_lock_max_duration:
                return True
                
        return False



    def _find_duel_for_player(self, target_sid: str, duels: List[Tuple[str, str, float]]) -> Optional[Tuple[str, str]]:
        """Find which duel the target player is participating in"""
        for sid1, sid2, _ in duels:
            if target_sid == sid1:
                return (sid1, sid2)
            elif target_sid == sid2:
                return (sid1, sid2)
        return None

# ---------------------
# HTTP server
# ---------------------

class GameStateHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            gamestate = json.loads(post_data.decode('utf-8'))
            duel_detector.process_gamestate(gamestate)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'OK')
        except Exception as e:
            logging.error(f"Fejl i HTTP handler: {e}")
            self.send_response(500)
            self.end_headers()

    def log_message(self, format, *args):
        pass

def start_server(port: int = 8082):
    try:
        server_address = ('', port)
        httpd = HTTPServer(server_address, GameStateHandler)
        logging.info(f"🌐 Server started on port {port}")
        httpd.serve_forever()
    except Exception as e:
        logging.info(f"❌ Server failed to start: {e}")

def control_interface():
    print("\n🎯 CS2 Auto Observer - Enhanced v4.2")
    print("Commands: 'toggle', 'status', 'quit'")
    print("Running...\n")

    while True:
        try:
            cmd = input("> ").strip()
            cmd_lower = cmd.lower()
            
            if cmd_lower in ('quit', 'exit'):
                break
            elif cmd == "":
                # Handle empty input (just Enter key)
                continue
            elif cmd_lower == 'toggle':
                duel_detector.active = not duel_detector.active
                print(f"Observer: {'ON' if duel_detector.active else 'OFF'}")
            elif cmd_lower == 'status':
                print(f"Status: {'ON' if duel_detector.active else 'OFF'}")
                print(f"Cooldown: {duel_detector.switch_cooldown}s")
                print(f"Min holdetid: {duel_detector.min_hold_time}s")
                print(f"Switch margin: {duel_detector.switch_margin}")
                print(f"Aktuel target: {duel_detector.current_target_sid} EMA={duel_detector.current_target_score_ema:.2f}")
                print(f"Kill focus: {duel_detector.kill_focus_sid} (t={max(0.0, duel_detector.kill_focus_until - time.time()):.1f}s)")
                print(f"Damage focus: {duel_detector.damage_focus_sid} (t={max(0.0, duel_detector.damage_focus_until - time.time()):.1f}s)")
                print(f"Death switch: {duel_detector.death_switch_target} (t={max(0.0, duel_detector.death_switch_active_until - time.time()):.1f}s)")
                print(f"Kill prediction: {'ENABLED' if duel_detector.kill_prediction_active else 'DISABLED'} (threshold: {duel_detector.prediction_confidence_threshold})")
                print(f"High tension duels: {len(duel_detector.high_tension_duels)} active")
                
                # Duel tracking status
                if duel_detector._is_duel_active():
                    elapsed = (datetime.now() - duel_detector.active_duel_start_time).total_seconds()
                    p1, p2 = duel_detector.active_duel_participants
                    target = duel_detector.active_duel_target
                    allplayers = duel_detector._last_gamestate.get('allplayers', {})
                    p1_name = allplayers.get(p1, {}).get('name', p1[:8])
                    p2_name = allplayers.get(p2, {}).get('name', p2[:8])
                    target_name = allplayers.get(target, {}).get('name', target[:8])
                    print(f"Duel tracking: AKTIV - {p1_name} vs {p2_name} (watching {target_name}, {elapsed:.1f}s)")
                else:
                    print(f"Duel tracking: INAKTIV")
                
                if duel_detector.freezetime_start:
                    left = duel_detector.numbers_lockout_seconds - (datetime.now() - duel_detector.freezetime_start).total_seconds()
                    left = max(0.0, left)
                    print(f"Numbers-lockout: {'AKTIV' if left>0 else 'INAKTIV'} ({left:.1f}s tilbage)")
            elif cmd_lower.startswith('cooldown '):
                try:
                    duel_detector.switch_cooldown = float(cmd_lower.split()[1])
                    print(f"Cooldown sat til {duel_detector.switch_cooldown}s")
                except:
                    print("Ugyldig værdi")
            elif cmd_lower.startswith('hold '):
                try:
                    duel_detector.min_hold_time = float(cmd_lower.split()[1])
                    print(f"Min holdetid sat til {duel_detector.min_hold_time}s")
                except:
                    print("Ugyldig værdi")
            elif cmd_lower.startswith('margin '):
                try:
                    duel_detector.switch_margin = float(cmd_lower.split()[1])
                    print(f"Switch-margin sat til {duel_detector.switch_margin}")
                except:
                    print("Ugyldig værdi")
            else:
                print("Ukendt kommando")
        except KeyboardInterrupt:
            break

def global_hotkey_listener():
    """Global hotkey listener that works independently of input() command loop"""
    while True:
        try:
            # Listen for either ' (toggle observer) or - (toggle campaths)
            event = keyboard.read_event()
            if event.event_type == keyboard.KEY_DOWN:
                if event.name == "'" or event.name == "apostrophe":
                    print("\n🎯 Hotkey detected: Toggling observer...")
                    duel_detector.toggle_active()
                    print("> ", end="", flush=True)  # Restore prompt
                elif event.name == "-" or event.name == "minus":
                    print("\n🎬 Hotkey detected: Toggling campaths...")
                    duel_detector.toggle_campaths()
                    print("> ", end="", flush=True)  # Restore prompt
        except Exception as e:
            print(f"Keyboard event error: {e}")
            time.sleep(0.1)  # Small delay to prevent busy loop

# ---------------------
# Main
# ---------------------

if __name__ == "__main__":
    print("🎯 CS2 Auto Observer - Enhanced v4.2")
    print("=====================================")

    duel_detector = CS2DuelDetector()

    # Start global hotkey listener thread
    hotkey_thread = threading.Thread(target=global_hotkey_listener, daemon=True)
    hotkey_thread.start()

    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    try:
        control_interface()
    except KeyboardInterrupt:
        logging.info("👋 Observer stopped")

    print("Observer shutdown complete.")
