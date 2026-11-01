"""Live inference for the same future-kill feature contract used by training."""
import logging
import os
import pickle
import sys

from observer_model import FEATURE_NAMES, feature_vector, predict_probabilities, validate_model_bundle


class DemoAnalyzer:
    def __init__(self):
        self.models_loaded = False
        self.model = None
        self.feature_columns = list(FEATURE_NAMES)
        # Historical kill counts are descriptive data, not a trained kill model.
        self.training_data = None
        self.weapon_patterns = {}
        self.map_patterns = {}
        self.distance_patterns = {}
        self.angle_patterns = {}

    def set_model_bundle(self, bundle):
        """Validate before swapping, preserving the last usable model on failure."""
        try:
            validate_model_bundle(bundle)
        except (ValueError, TypeError) as error:
            logging.warning("Rejected observer model: %s", error)
            return False
        self.model = bundle
        self.models_loaded = True
        return True

    def load_model(self, model_path):
        try:
            if not os.path.isabs(model_path) and not os.path.exists(model_path):
                base = os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else os.path.abspath(__file__))
                model_path = os.path.join(base, "models", os.path.basename(model_path))
            with open(model_path, "rb") as handle:
                bundle = pickle.load(handle)
            return self.set_model_bundle(bundle)
        except (OSError, ValueError, TypeError, EOFError, pickle.UnpicklingError) as error:
            logging.warning("Observer model unavailable: %s", error)
            return False

    def extract_features(self, gamestate, player_candidates):
        features = {}
        for player_id, _ in player_candidates:
            values = feature_vector(gamestate, player_id)
            if values is not None:
                features[player_id] = values
        return features

    def predict_best_player(self, gamestate, candidates):
        """Return the highest two-second kill probability among valid candidates.

        With no validated model, no AI probability is available. The observer can
        explicitly choose its rule-based fallback instead of treating 0.5 as AI.
        """
        if not candidates or not self.models_loaded or self.model is None:
            return None, 0.0
        try:
            scores = predict_probabilities(self.model, gamestate, [player_id for player_id, _ in candidates])
            return min(scores.items(), key=lambda item: (-item[1], item[0])) if scores else (None, 0.0)
        except (ValueError, TypeError, IndexError) as error:
            logging.warning("Observer prediction failed: %s", error)
            return None, 0.0

    def predict_best_observer_target(self, gamestate, candidates):
        return self.predict_best_player(gamestate, candidates)
