"""Regression coverage for the live observer, with no keyboard/server startup."""
import copy
import os
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cs2_duel_detector as runtime
from demo_analyzer import DemoAnalyzer
from model_updater import ModelUpdater
from observer_model import FEATURE_NAMES, HORIZON_SECONDS, SCHEMA_VERSION, TARGET


class FixtureEstimator:
    """Synthetic estimator fixture; these scores are not empirical accuracy."""
    classes_ = [0, 1]
    n_features_in_ = len(FEATURE_NAMES)

    def __init__(self, probability=.7):
        self.probability = probability

    def predict_proba(self, values):
        return [[1-self.probability, self.probability] for _ in values]


def bundle(probability=.7):
    return dict(schema_version=SCHEMA_VERSION, target=TARGET,
                horizon_seconds=HORIZON_SECONDS, feature_names=FEATURE_NAMES,
                model=FixtureEstimator(probability), train_demo_ids=['train-a', 'train-b'],
                validation_demo_ids=['validation-a', 'validation-b'],
                evaluation={'promotion_passed': True})


def player(team='T', pos='0, 0, 0'):
    return {'team': team, 'position': pos, 'forward': '1, 0, 0',
            'state': {'health': 100, 'armor': 50},
            'weapons': {'weapon_0': {'name': 'weapon_ak47', 'state': 'active', 'ammo_clip': 30}}}


class RuntimeTests(unittest.TestCase):
    def detector(self, strict=False):
        detector = runtime.CS2DuelDetector.__new__(runtime.CS2DuelDetector)
        detector.config = {'geometry': {'require_verified_visibility': strict}}
        detector.geometry_manager = SimpleNamespace(current_map='de_test')
        detector.pre_engagement_detection_distance = 2000
        detector._geometry_cache = {}
        return detector

    def test_parse_vectors_and_reject_nonfinite(self):
        self.assertEqual(runtime.parse_vec3('1,2, 3'), (1, 2, 3))
        self.assertEqual(runtime.parse_vec3([1, 2, 3]), (1, 2, 3))
        self.assertIsNone(runtime.parse_vec3('nan, 0, 0'))
        self.assertIsNone(runtime.parse_vec3('1, 2'))

    def test_aim_uses_pitch_instead_of_flat_heading(self):
        self.assertAlmostEqual(runtime.angle_facing_score((0, 0, 0), (1, 0, 0), (100, 0, 100)), 2**-.5)
        self.assertEqual(runtime.angle_facing_score((0, 0, 0), (0, 0, 1), (0, 0, 100)), 1)
        self.assertEqual(runtime.angle_facing_score((0, 0, 0), (1, 0, 0), (-100, 0, 0)), 0)

    def test_geometry_unknown_is_distinct_from_blocked(self):
        a, b = player(), player('CT', '100, 0, 0')
        for strict, result, expected in ((False, False, False), (False, None, True),
                                         (True, None, False), (True, True, True)):
            with self.subTest(strict=strict, result=result), patch.object(runtime, 'duel_geometrically_possible', return_value=result):
                self.assertEqual(self.detector(strict)._direct_duel_allowed(a, b), expected)

    def test_invalid_position_never_becomes_a_duel(self):
        self.assertFalse(self.detector()._direct_duel_allowed(player(pos='nan,0,0'), player('CT')))

    def test_repeated_paths_reuse_geometry(self):
        detector = self.detector()
        with patch.object(runtime, 'duel_geometrically_possible', return_value=True) as query:
            detector._direct_duel_allowed(player(), player('CT', '100,0,0'))
            detector._direct_duel_allowed(player(), player('CT', '100,0,0'))
            detector._direct_duel_allowed(player('CT', '100,0,0'), player())
            self.assertIs(detector._duel_visibility(player(), player('CT', '100,0,0')), True)
        self.assertEqual(query.call_count, 1)

    def test_blocked_pair_has_zero_score_in_all_prediction_paths(self):
        detector = self.detector()
        a, b = player(), player('CT', '100,0,0')
        with patch.object(runtime, 'duel_geometrically_possible', return_value=False):
            self.assertEqual(detector.score_player_in_duel('a', 'b', {'a': a, 'b': b}, 100, 0), 0)
            self.assertEqual(detector._calculate_kill_probability('a', 'b', a, b, 100, {}, 0), 0)
            self.assertEqual(detector._calculate_hardcore_prediction_confidence('a', 'b', a, b, (0,0,0), (100,0,0), (1,0,0), 0), 0)
            self.assertEqual(detector._calculate_pre_engagement_score('a', 'b', a, b, {}, None), 0)

    def test_validated_forecast_ranks_candidates_without_heuristic_overrides(self):
        detector = self.detector()
        detector.ai_analyzer = DemoAnalyzer()
        detector.ai_analyzer.set_model_bundle(bundle())
        detector.locked_duel_participants = None
        detector.locked_duel_start_time = None
        detector.find_potential_duels = lambda players: [('a', 'b', 100)]
        state = {'allplayers': {'a': player(), 'b': player('CT', '100,0,0'), 'total': 2}}
        self.assertEqual(detector.choose_best_player(state), ('a', .7, 'b', .7))

    def test_first_model_can_arrive_after_observer_start(self):
        detector = self.detector()
        detector.ai_analyzer = DemoAnalyzer()
        detector.auto_model_updates = True
        detector.model_reloader = SimpleNamespace(get_model=lambda *args: bundle(.8))
        detector._refresh_observer_model()
        self.assertTrue(detector.ai_analyzer.models_loaded)

    def test_hot_reload_replaces_changed_bundle_and_preserves_last_good(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, 'observer_ai_model.pkl')
            reloader = runtime.ModelHotReloader(directory)
            reloader.check_interval = 0
            self.assertIsNone(reloader.get_model('observer_ai', model_path))
            with open(model_path, 'wb') as stream:
                pickle.dump(bundle(.7), stream)
            old = reloader.get_model('observer_ai', model_path)
            self.assertEqual(old['model'].probability, .7)
            with open(model_path, 'wb') as stream:
                stream.write(b'partial invalid file')
            self.assertIs(reloader.get_model('observer_ai', model_path), old)
            with open(model_path, 'wb') as stream:
                pickle.dump(bundle(.9), stream)
            stat = os.stat(model_path)
            os.utime(model_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
            self.assertEqual(reloader.get_model('observer_ai', model_path)['model'].probability, .9)

    def test_legacy_headshot_model_rejected_by_live_loader_and_updater(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, 'observer_ai_model.pkl')
            with open(model_path, 'wb') as stream:
                pickle.dump(FixtureEstimator(), stream)
            self.assertIsNone(runtime.ModelHotReloader(directory).get_model('observer_ai', model_path))
            updater = ModelUpdater(models_dir=directory, backup_dir=os.path.join(directory, 'backups'),
                                   config_file=os.path.join(directory, 'config.json'))
            self.assertFalse(updater.validate_model('observer_ai', model_path)['valid'])
            with open(model_path, 'wb') as stream:
                pickle.dump(bundle(), stream)
            self.assertTrue(updater.validate_model('observer_ai', model_path)['valid'])


if __name__ == '__main__':
    unittest.main()
