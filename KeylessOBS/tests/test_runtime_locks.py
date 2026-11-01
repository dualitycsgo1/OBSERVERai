"""Exercise full camera processing so locks/triggers cannot bypass the forecast."""

from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cs2_duel_detector as runtime
from demo_analyzer import DemoAnalyzer
from observer_model import FEATURE_NAMES, HORIZON_SECONDS, SCHEMA_VERSION, TARGET


class LowProbabilityEstimator:
    """Realistic probability scale; fixture scores do not assert real accuracy."""
    classes_ = [0, 1]
    n_features_in_ = len(FEATURE_NAMES)

    def predict_proba(self, values):
        return [[.9, .1] if features[0] < 75 else [.8, .2] for features in values]


def player(team, health, x):
    return {'name': team, 'team': team, 'position': f'{x}, 0, 0',
            'forward': '1, 0, 0', 'state': {'health': health, 'armor': 50},
            'weapons': {'weapon_0': {'name': 'weapon_ak47', 'state': 'active', 'ammo_clip': 30}}}


class RuntimeLockTests(unittest.TestCase):
    def setUp(self):
        self.state = {'round': {'phase': 'live'}, 'allplayers': {
            'a': player('T', 50, 0), 'b': player('CT', 100, 100),
        }}

    def detector(self, locked=True, model=True):
        detector = runtime.CS2DuelDetector.__new__(runtime.CS2DuelDetector)
        detector.config = {'geometry': {'require_verified_visibility': False}}
        detector.active = True
        detector.geometry_manager = None
        detector._refresh_observer_model = lambda: None
        detector._update_duel_lock_tracking = lambda *args: None
        detector.update_history_and_triggers = lambda *args: None
        detector.freezetime_fkey_sent = False
        detector.freezetime_start = None
        detector.campaths_enabled = False
        detector.death_switch_target = None
        detector.death_switch_active_until = 0
        detector.kill_focus_sid = None
        detector.kill_focus_until = 0
        detector.damage_focus_sid = None
        detector.damage_focus_until = 0
        detector.locked_duel_participants = ('a', 'b') if locked else None
        detector.locked_duel_start_time = datetime.now() - timedelta(seconds=1) if locked else None
        detector.locked_duel_target = 'a' if locked else None
        detector.use_predictive_observer = True
        detector.predictive_observer = Mock()
        detector.find_potential_duels = lambda _: [('a', 'b', 100)]
        detector.current_target_sid = 'a'
        detector.current_target_start = datetime.now() - timedelta(seconds=10)
        detector.current_target_score_ema = 1.0  # Previous event trigger score.
        detector.last_switch_time = datetime.now() - timedelta(seconds=10)
        detector.last_any_switch = time.time()
        detector._can_switch_now = lambda **kwargs: True
        detector.get_player_slot = lambda sid, _: {'a': 1, 'b': 2}.get(sid)
        detector.switch_to_player = Mock()
        detector.score_ema_alpha = .55
        detector.rotation_delta = .04
        detector.rotation_window = 1.2
        detector.switch_margin = .08
        detector.failsafe_interval = 10
        detector._force_high_confidence_switch = False
        detector._high_confidence_level = 0
        detector._recent_switches = deque(maxlen=5)
        detector.ai_analyzer = DemoAnalyzer() if model else None
        if model:
            detector.ai_analyzer.set_model_bundle(dict(
                schema_version=SCHEMA_VERSION, target=TARGET,
                horizon_seconds=HORIZON_SECONDS, feature_names=FEATURE_NAMES,
                model=LowProbabilityEstimator(), train_demo_ids=['train'],
                validation_demo_ids=['heldout'], evaluation={'promotion_passed': True}))
        return detector

    def test_full_locked_processing_uses_model_after_legacy_trigger_score(self):
        detector = self.detector()
        detector.process_gamestate(self.state)
        detector.switch_to_player.assert_called_once_with(2)
        self.assertEqual(detector.current_target_sid, 'b')
        self.assertEqual(detector.current_target_score_ema, .2)
        detector.predictive_observer.update_existing_detector.assert_not_called()

    def test_unlocked_model_does_not_apply_legacy_point_28_threshold(self):
        detector = self.detector(locked=False)
        detector.process_gamestate(self.state)
        detector.switch_to_player.assert_called_once_with(2)

    def test_model_keeps_valid_locked_pair_outside_legacy_candidate_gates(self):
        detector = self.detector()
        detector.find_potential_duels = lambda _: []
        self.assertEqual(detector.choose_best_player(self.state), ('b', .2, 'a', .1))

    def test_current_kill_event_is_held_until_expiry_then_forecast_resumes(self):
        detector = self.detector()
        detector.kill_focus_sid = 'a'
        detector.kill_focus_until = time.time() + 1
        detector.process_gamestate(self.state)
        detector.switch_to_player.assert_not_called()
        detector.kill_focus_until = time.time() - 1
        detector.process_gamestate(self.state)
        detector.switch_to_player.assert_called_once_with(2)

    def test_external_event_does_not_freeze_a_valid_locked_model_selection(self):
        detector = self.detector()
        self.state['allplayers']['c'] = player('T', 60, 200)
        detector.kill_focus_sid = 'c'
        detector.kill_focus_until = time.time() + 1
        detector.process_gamestate(self.state)
        detector.switch_to_player.assert_called_once_with(2)

    def test_model_selection_still_respects_camera_cooldown(self):
        detector = self.detector()
        detector._can_switch_now = lambda **kwargs: False
        detector.process_gamestate(self.state)
        detector.switch_to_player.assert_not_called()

    def test_blocked_lock_releases_before_cooldown_or_minimum_hold(self):
        detector = self.detector()
        detector._direct_duel_allowed = lambda *args: False
        detector.last_duel_switch_time = datetime.now()
        detector.duel_switch_cooldown_seconds = 3
        detector.minimum_duel_lock_duration = 2.5
        detector._detect_pre_engagement_duel = lambda *args: None
        runtime.CS2DuelDetector._update_duel_lock_tracking(detector, self.state['allplayers'], datetime.now())
        self.assertIsNone(detector.locked_duel_participants)
        self.assertIsNone(detector.locked_duel_target)

    def test_new_lock_uses_forecast_instead_of_heuristic_initial_target(self):
        detector = self.detector(locked=False)
        detector._last_gamestate = self.state
        detector.last_duel_switch_time = None
        detector._detect_pre_engagement_duel = lambda *args: ('a', 'b')
        detector.score_player_in_duel = Mock(side_effect=AssertionError('Legacy scoring must not run'))
        runtime.CS2DuelDetector._update_duel_lock_tracking(detector, self.state['allplayers'], datetime.now())
        self.assertEqual(detector.locked_duel_target, 'b')
        detector.score_player_in_duel.assert_not_called()

    def test_legacy_locked_branch_is_reachable_without_a_model(self):
        detector = self.detector(model=False)
        detector.use_predictive_observer = False
        detector.current_target_score_ema = .1
        detector.score_player_in_duel = lambda *args: .9
        detector.process_gamestate(self.state)
        detector.switch_to_player.assert_called_once_with(2)

    def test_central_switch_guard_does_not_send_same_slot_or_restart_cooldown(self):
        detector = self.detector(locked=False)
        detector._last_gamestate = self.state
        previous_switch = detector.last_switch_time
        previous_any_switch = detector.last_any_switch
        with patch.object(runtime, 'send_key', return_value=True) as send:
            switched = runtime.CS2DuelDetector.switch_to_player(detector, 1)
        self.assertFalse(switched)
        send.assert_not_called()
        self.assertEqual(detector.last_switch_time, previous_switch)
        self.assertEqual(detector.last_any_switch, previous_any_switch)

    def test_idle_same_target_cannot_delay_later_better_model_challenger(self):
        detector = self.detector(locked=False)
        detector.switch_to_player = runtime.CS2DuelDetector.switch_to_player.__get__(detector)
        detector._can_switch_now = runtime.CS2DuelDetector._can_switch_now.__get__(detector)
        detector.switch_cooldown = 4.2
        detector.min_hold_time = 1.1
        detector._max_switches_per_10s = 3
        detector.last_switch_time = datetime.now() - timedelta(seconds=5)
        detector.last_any_switch = time.time() - 6
        detector.failsafe_interval = 5
        previous_switch = detector.last_switch_time
        previous_any_switch = detector.last_any_switch
        detector.choose_best_player = lambda _: (None, 0, None, 0)
        detector._pick_any_live = lambda _: 'a'
        with patch.object(runtime, 'send_key', return_value=True) as send:
            detector.process_gamestate(self.state)
            send.assert_not_called()
            self.assertEqual(detector.last_switch_time, previous_switch)
            self.assertEqual(detector.last_any_switch, previous_any_switch)
            self.assertTrue(detector._can_switch_now())
            # A real .2 challenger now beats the held player's .1, while the
            # earlier no-op cannot manufacture another 4.2-second wait.
            detector.choose_best_player = runtime.CS2DuelDetector.choose_best_player.__get__(detector)
            detector.process_gamestate(self.state)
            send.assert_called_once_with(runtime.VK_CODES[2])
        self.assertEqual(detector.current_target_sid, 'b')

    def test_low_score_failsafe_does_not_request_the_already_held_target(self):
        detector = self.detector(locked=False, model=False)
        detector.use_predictive_observer = False
        detector.last_any_switch = time.time() - 20
        detector.choose_best_player = lambda _: ('a', .1, None, 0)
        detector.process_gamestate(self.state)
        detector.switch_to_player.assert_not_called()


if __name__ == '__main__':
    unittest.main()
