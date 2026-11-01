#!/usr/bin/env python3
"""
AI Trainer for CS2 Observer System
Train and evaluate future-kill models on pre-event demo snapshots.
Only candidates that pass held-out comparison are published.
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

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Scikit-learn imports
from sklearn.ensemble import RandomForestClassifier

import shutil

# Import our existing components
from comprehensive_demo_processor import ComprehensiveDemoProcessor
from observer_model import FEATURE_NAMES, HORIZON_SECONDS, SCHEMA_VERSION, TARGET, validate_model_bundle
from observer_training import prepare_prediction_samples, demo_split, evaluate_candidate

logger = logging.getLogger(__name__)

class AITrainer:
    """
    AI Trainer with measured model promotion

    Features:
    - Process new demo files automatically
    - Extract tactical features and patterns
    - Refit on cumulative validated snapshots
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
        for directory in [self.models_dir, self.training_data_dir, self.demos_dir, self.backup_dir]:
            os.makedirs(directory, exist_ok=True)

        # Initialize components
        self.demo_processor = ComprehensiveDemoProcessor()

        # Training configuration
        self.config = {
            "batch_size": 1000,
            "max_demos_per_session": 10,
            "validation_split": 0.2,
            "backup_frequency": 5,  # Backup every N training sessions
            "feature_columns": list(FEATURE_NAMES)
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
        """Import pre-event snapshots with observed future labels, never kill-only rows."""
        if demo_files is None:
            demo_files = self._find_new_demos()
        results = {"processed": 0, "failed": 0, "new_kills": 0, "new_rounds": 0,
                   "new_samples": 0, "new_data": {}, "errors": [], "demos_processed": []}
        new_samples = []
        for demo_file in demo_files[:self.config["max_demos_per_session"]]:
            try:
                parsed = self.demo_processor.process_demo_comprehensive(demo_file)
                if not parsed.get("success"):
                    raise ValueError(parsed.get("error", "Demo parser failed"))
                samples = parsed.get("prediction_samples", parsed.get("training_samples", []))
                prepared, rejected = prepare_prediction_samples(samples)
                if not prepared:
                    raise ValueError("No valid pre-event prediction samples. Kill-event/headshot data cannot train a future-kill model.")
                if len(prepared) != len(samples):
                    raise ValueError(f"Parser returned invalid/duplicate prediction samples: {rejected}")
                new_samples.extend(samples)
                results["processed"] += 1
                results["new_kills"] += len(parsed.get("kills", []))
                results["new_samples"] += len(samples)
                results["demos_processed"].append(os.path.basename(demo_file))
            except Exception as error:
                results["failed"] += 1
                results["errors"].append(f"{demo_file}: {error}")
                logger.warning("Demo import failed: %s", error)
        if new_samples:
            results["new_data"] = self._save_new_training_data(new_samples)
            if not results["new_data"]:
                # Do not mark files processed when their extracted data was not saved.
                results["failed"] += results["processed"]
                results["processed"] = 0
                results["demos_processed"] = []
                results["errors"].append("Failed to save prediction samples")
        self._update_training_history({"demos_processed": results["demos_processed"],
                                       "kills_processed": results["new_kills"],
                                       "processing_results": results})
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
                if file.lower().endswith(('.dem', '.dem.zst')) and file not in processed_demos:
                    new_demos.append(os.path.join(self.demos_dir, file))

        logger.info(f"Found {len(new_demos)} new demo files")
        return new_demos

    def _save_new_training_data(self, new_data: List[Dict]) -> Dict:
        """Save one source of truth per batch; do not append the same rows to master."""
        try:
            filename = f"prediction_batch_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
            filepath = os.path.join(self.training_data_dir, filename)
            temporary = filepath + ".tmp"
            with open(temporary, 'w', encoding='utf-8') as handle:
                json.dump(new_data, handle, indent=2, allow_nan=False)
            os.replace(temporary, filepath)
            return {"filename": filename, "samples": len(new_data), "filepath": filepath}
        except Exception as error:
            logger.error("Failed to save training data: %s", error)
            return {}

    def train_model(self, incremental: bool = True) -> Dict[str, Any]:
        """Retrain on cumulative snapshots, evaluate by demo, then optionally publish.

        RandomForest.fit is a full refit, not incremental learning. `incremental`
        is retained for GUI/API compatibility; all valid accumulated rows are used.
        """
        try:
            rows, rejected = prepare_prediction_samples(self._load_all_training_data())
            if len(rows) < 100:
                raise ValueError(f"Insufficient pre-event prediction data ({len(rows)} valid samples). "
                                 "Need at least 100 examples, both outcomes, and five independent demos. "
                                 "Historical kill/headshot rows are not valid future-kill training data.")
            train, validation = demo_split(rows, os.path.join(self.training_data_dir, "validation_manifest.json"),
                                           self.config["validation_split"])
            model_path = os.path.join(self.models_dir, "observer_ai_model.pkl")
            incumbent = None
            legacy_model_ignored = False
            if os.path.exists(model_path):
                with open(model_path, 'rb') as handle:
                    existing = pickle.load(handle)
                try:
                    incumbent = validate_model_bundle(existing)
                except ValueError:
                    legacy_model_ignored = True
            model = RandomForestClassifier(n_estimators=150, min_samples_leaf=3, random_state=42, n_jobs=-1)
            model.fit([row["features"] for row in train], [row["label"] for row in train])
            # Thread-pool startup costs dominate tiny live inference batches.
            model.n_jobs = 1
            evaluation = evaluate_candidate(model, train, validation, incumbent)
            results = {"success": True, "published": evaluation["promotion_passed"],
                       "target": TARGET, "horizon_seconds": HORIZON_SECONDS,
                       "accuracy": evaluation["candidate"]["accuracy_at_0_5"],
                       "training_samples": len(train), "test_samples": len(validation),
                       "training_demos": sorted({row["demo_id"] for row in train}),
                       "validation_demos": sorted({row["demo_id"] for row in validation}),
                       "rejected_samples": rejected, "legacy_model_ignored": legacy_model_ignored,
                       "evaluation": evaluation, "training_mode": "cumulative_full_refit",
                       "timestamp": datetime.now().isoformat()}
            if evaluation["promotion_passed"]:
                bundle = {"schema_version": SCHEMA_VERSION, "target": TARGET,
                          "horizon_seconds": HORIZON_SECONDS, "feature_names": list(FEATURE_NAMES),
                          "model": model, "train_demo_ids": results["training_demos"],
                          "validation_demo_ids": results["validation_demos"], "evaluation": evaluation}
                validate_model_bundle(bundle)
                self._backup_existing_model()
                temporary = model_path + ".tmp"
                with open(temporary, 'wb') as handle:
                    pickle.dump(bundle, handle)
                os.replace(temporary, model_path)
                results["model_path"] = model_path
                self._notify_model_update("observer_ai_model", model_path, results)
            else:
                results["message"] = "Candidate evaluated but not deployed: " + "; ".join(evaluation["rejection_reasons"])
            report_path = os.path.join(self.training_data_dir, "latest_evaluation.json")
            with open(report_path + ".tmp", 'w', encoding='utf-8') as handle:
                json.dump(results, handle, indent=2, allow_nan=False)
            os.replace(report_path + ".tmp", report_path)
            self._update_training_history(results)
            return results
        except Exception as error:
            logger.error("Model training failed: %s", error)
            results = {"success": False, "published": False, "error": str(error),
                       "timestamp": datetime.now().isoformat()}
            try:
                report_path = os.path.join(self.training_data_dir, "latest_evaluation.json")
                with open(report_path + ".tmp", 'w', encoding='utf-8') as handle:
                    json.dump(results, handle, indent=2, allow_nan=False)
                os.replace(report_path + ".tmp", report_path)
            except OSError:
                logger.warning("Could not save failed-training report")
            self._update_training_history(results)
            return results

    def _load_all_training_data(self) -> List[Dict]:
        """Read batches plus legacy sources; preparation deduplicates and rejects legacy rows."""
        all_data = []
        for filename in sorted(os.listdir(self.training_data_dir)):
            if not (filename == "complete_kill_analysis.json" or
                    filename.startswith(("prediction_batch_", "training_batch_")) and filename.endswith('.json')):
                continue
            try:
                with open(os.path.join(self.training_data_dir, filename), encoding='utf-8') as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    data = data.get("prediction_samples", data.get("training_samples", []))
                if isinstance(data, list):
                    all_data.extend(data)
            except (OSError, ValueError) as error:
                logger.warning("Could not read training source %s: %s", filename, error)
        return all_data

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
        if results.get("published"):
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
                validate_model_bundle(pickle.load(f))

            logger.info("Model integration verified - main system will use updated model")
            return True

        except Exception as e:
            logger.error(f"Model integration failed: {e}")
            return False

    def get_training_stats(self) -> Dict:
        """Count unique eligible examples, not accumulated refits or legacy kill rows."""
        rows, rejected = prepare_prediction_samples(self._load_all_training_data())
        sessions = self.training_history.get("sessions", [])
        versions = [session["training_results"] for session in sessions
                    if session.get("training_results", {}).get("published")]
        groups = {row["demo_id"] for row in rows}
        return {
            "total_sessions": len(sessions), "current_demos_processed": len(groups),
            "current_kills_analyzed": self.training_history.get("total_kills_analyzed", 0),
            "current_training_samples": len(rows), "model_versions": len(versions),
            "latest_accuracy": versions[-1].get("accuracy", 0) if versions else 0,
            "historical_demos": 0, "historical_kills": 0, "historical_training_samples": 0,
            "total_demos_processed": len(groups),
            "total_kills_analyzed": self.training_history.get("total_kills_analyzed", 0),
            "total_training_samples": len(rows), "rejected_samples": rejected,
            "training_data_files": sorted(file for file in os.listdir(self.training_data_dir)
                                          if file.startswith(("prediction_batch_", "training_batch_")) and file.endswith('.json')),
            "model_files": sorted(file for file in os.listdir(self.models_dir) if file.endswith('.pkl')),
        }

    def _notify_model_update(self, model_name: str, model_path: str, metadata: Dict[str, Any]):
        """Notify the main observer system of model updates"""
        try:
            models_dir = self.models_dir
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
