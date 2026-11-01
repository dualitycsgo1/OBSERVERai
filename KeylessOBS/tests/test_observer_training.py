"""Synthetic regression tests establish correctness, not real-demo accuracy."""
import copy
import json
import random
import sys
from pathlib import Path

import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_trainer import AITrainer
from demo_analyzer import DemoAnalyzer
from observer_model import (FEATURE_NAMES, HORIZON_SECONDS, TARGET, feature_vector,
                            validate_model_bundle, predict_probability, predict_probabilities)
from observer_training import demo_split, evaluate_candidate, prepare_prediction_samples


def snapshot(health=100):
    return {"map": {"name": "de_test"}, "allplayers": {
        "a": {"team": "T", "position": "0,0,0", "forward": "1,0,0",
              "state": {"health": health, "armor": 50},
              "weapons": {"held": {"name": "weapon_ak47", "state": "active", "ammo_clip": 20},
                          "bag": {"name": "weapon_awp", "state": "holstered", "ammo_clip": 10}}},
        "b": {"team": "CT", "position": "100,0,0", "forward": "-1,0,0",
              "state": {"health": 50 if health == 100 else 100, "armor": 50},
              "weapons": {"held": {"name": "weapon_ak47", "state": "active", "ammo_clip": 20}}},
    }}


def sample(demo="demo-a", tick=100, player="a", label=1, state=None):
    return {"demo_id": demo, "sample_tick": tick, "horizon_seconds": HORIZON_SECONDS,
            "gamestate": state or snapshot(), "player_id": player, "label": label,
            "target": TARGET, "snapshot_phase": "pre_event"}


def synthetic_samples():
    records = []
    for demo in range(8):
        for tick in range(20):
            # A deliberately simple learnable future outcome in independent groups.
            # Only a fixture; this says nothing about real CS2 forecasting quality.
            health = 100 if tick % 2 else 50
            state = snapshot(health)
            records.extend([sample(f"demo-{demo}", tick, "a", int(health == 100), state),
                            sample(f"demo-{demo}", tick, "b", int(health != 100), state)])
    return records


def make_trainer(tmp_path):
    return AITrainer(*(str(tmp_path / name) for name in ("models", "data", "demos", "backup")))


def test_shared_features_use_active_weapon_and_three_dimensional_angles():
    state = snapshot()
    state["allplayers"]["b"]["position"] = "0,0,100"
    features = feature_vector(state, "a")
    assert len(features) == len(FEATURE_NAMES)
    assert features[FEATURE_NAMES.index("weapon_rifle")] == 1
    assert features[FEATURE_NAMES.index("weapon_sniper")] == 0
    assert features[FEATURE_NAMES.index("nearest_enemy_angle")] == pytest.approx(90)
    assert features[FEATURE_NAMES.index("nearest_enemy_distance")] == pytest.approx(100)
    rows, rejected = prepare_prediction_samples([sample(state=state)])
    assert not rejected
    assert rows[0]["features"] == DemoAnalyzer().extract_features(state, [("a", 1.0)])["a"]


def test_dead_targets_and_missing_direction_are_not_fabricated():
    state = snapshot()
    state["allplayers"]["b"]["state"]["health"] = 0
    assert feature_vector(state, "a") is None
    state = snapshot()
    del state["allplayers"]["a"]["forward"]
    assert feature_vector(state, "a") is None
    state["allplayers"]["a"]["view_angles"] = {"yaw": 0, "pitch": 0}
    assert feature_vector(state, "a")[FEATURE_NAMES.index("best_aim_angle")] == 0


def test_kill_events_and_post_event_rows_cannot_become_prediction_data():
    wrong_phase = sample()
    wrong_phase["snapshot_phase"] = "post_event"
    rows, rejected = prepare_prediction_samples([{"is_headshot": True}, wrong_phase])
    assert rows == []
    assert rejected["legacy_or_wrong_target"] == 1


def test_duplicate_sources_are_deduplicated_and_conflicts_fail():
    row = sample()
    rows, rejected = prepare_prediction_samples([row, copy.deepcopy(row)])
    assert len(rows) == 1
    assert rejected["duplicate"] == 1
    conflicting = copy.deepcopy(row)
    conflicting["label"] = 0
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        prepare_prediction_samples([row, conflicting])


def test_source_row_permutations_produce_identical_training_and_predictions():
    records = synthetic_samples()
    expected, _ = prepare_prediction_samples(records)
    random.Random(123).shuffle(records)
    reordered, _ = prepare_prediction_samples(records)
    assert reordered == expected
    first = RandomForestClassifier(n_estimators=20, min_samples_leaf=3, random_state=42, n_jobs=1)
    second = RandomForestClassifier(n_estimators=20, min_samples_leaf=3, random_state=42, n_jobs=1)
    first.fit([row["features"] for row in expected], [row["label"] for row in expected])
    second.fit([row["features"] for row in reordered], [row["label"] for row in reordered])
    matrix = [row["features"] for row in expected]
    assert first.predict_proba(matrix).tolist() == second.predict_proba(matrix).tolist()
    assert first.estimators_[0].tree_.__getstate__()["nodes"].tolist() == second.estimators_[0].tree_.__getstate__()["nodes"].tolist()


def test_validation_membership_stays_fixed_when_new_demos_arrive(tmp_path):
    rows, _ = prepare_prediction_samples(synthetic_samples())
    manifest = str(tmp_path / "manifest.json")
    train, validation = demo_split(rows, manifest)
    original = json.loads(Path(manifest).read_text())["assignments"]
    new_records = [sample("new-demo", tick, label=tick % 2) for tick in range(10)]
    extended, _ = prepare_prediction_samples(synthetic_samples() + new_records)
    new_train, new_validation = demo_split(extended, manifest)
    updated = json.loads(Path(manifest).read_text())["assignments"]
    assert all(updated[group] == membership for group, membership in original.items())
    assert not {row["demo_id"] for row in train} & {row["demo_id"] for row in validation}
    assert not {row["demo_id"] for row in new_train} & {row["demo_id"] for row in new_validation}


def test_constant_model_does_not_pass_promotion(tmp_path):
    rows, _ = prepare_prediction_samples(synthetic_samples())
    train, validation = demo_split(rows, str(tmp_path / "manifest.json"))
    model = DummyClassifier(strategy="prior").fit([row["features"] for row in train], [row["label"] for row in train])
    evaluation = evaluate_candidate(model, train, validation)
    assert not evaluation["promotion_passed"]
    assert evaluation["brier_improvement_over_baseline"]["mean"] == 0


def test_legacy_headshot_training_is_rejected_without_publishing(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer._save_new_training_data([{"is_headshot": index % 2} for index in range(200)])
    result = trainer.train_model()
    assert not result["success"] and not result["published"]
    assert "pre-event" in result["error"]
    assert not (tmp_path / "models" / "observer_ai_model.pkl").exists()


def test_training_to_inference_and_identical_retrain_preserves_deployed_model(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer._save_new_training_data(synthetic_samples())
    result = trainer.train_model()
    assert result["success"] and result["published"]
    assert not set(result["training_demos"]) & set(result["validation_demos"])
    model_path = Path(result["model_path"])
    original = model_path.read_bytes()
    analyzer = DemoAnalyzer()
    assert analyzer.load_model(str(model_path))
    assert analyzer.model["model"].n_jobs == 1
    assert analyzer.predict_best_player(snapshot(100), [("a", 0), ("b", 1)])[0] == "a"
    assert analyzer.predict_best_player(snapshot(50), [("a", 1), ("b", 0)])[0] == "b"
    assert trainer.auto_integrate_model()
    notifications = list((tmp_path / "models").glob("model_update_*.json"))
    second = trainer.train_model()
    assert second["success"] and not second["published"]
    assert second["evaluation"]["brier_improvement_over_incumbent"]["mean"] == 0
    assert model_path.read_bytes() == original
    assert list((tmp_path / "models").glob("model_update_*.json")) == notifications
    assert trainer.get_training_stats()["total_training_samples"] == 320


def test_batched_inference_matches_individual_predictions_and_calls_estimator_once(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer._save_new_training_data(synthetic_samples())
    result = trainer.train_model()
    analyzer = DemoAnalyzer()
    assert analyzer.load_model(result["model_path"])
    bundle = analyzer.model
    individual = {player: predict_probability(bundle, snapshot(), player) for player in ("a", "b")}
    original = bundle["model"].predict_proba
    calls = []
    def record_batch(matrix):
        calls.append(len(matrix))
        return original(matrix)
    bundle["model"].predict_proba = record_batch
    assert predict_probabilities(bundle, snapshot(), ["a", "b", "a", "missing"]) == individual
    assert calls == [2]
    calls.clear()
    assert analyzer.predict_best_player(snapshot(), [("a", 1), ("b", 1)])[0] == "a"
    assert calls == [2]


def test_model_contract_rejects_legacy_and_retains_last_good(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer._save_new_training_data(synthetic_samples())
    result = trainer.train_model()
    analyzer = DemoAnalyzer()
    assert analyzer.load_model(result["model_path"])
    original = analyzer.model
    assert not analyzer.set_model_bundle(original["model"])
    assert analyzer.model is original
    invalid = dict(original, target="is_headshot")
    with pytest.raises(ValueError, match="target"):
        validate_model_bundle(invalid)
    leaked = dict(original, train_demo_ids=original["validation_demo_ids"])
    with pytest.raises(ValueError, match="overlap"):
        validate_model_bundle(leaked)
    for changed in ({"train_demo_ids": []}, {"validation_demo_ids": None}, {"evaluation": []}):
        with pytest.raises(ValueError):
            validate_model_bundle(dict(original, **changed))


def test_contaminated_incumbent_cannot_be_compared_on_its_training_demos(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer._save_new_training_data(synthetic_samples())
    result = trainer.train_model()
    analyzer = DemoAnalyzer()
    assert analyzer.load_model(result["model_path"])
    rows, _ = prepare_prediction_samples(synthetic_samples())
    train, validation = demo_split(rows, str(tmp_path / "data" / "validation_manifest.json"))
    contaminated = dict(analyzer.model, train_demo_ids=result["validation_demos"])
    with pytest.raises(ValueError, match="fresh external test"):
        evaluate_candidate(analyzer.model["model"], train, validation, contaminated)


def test_failed_and_unattempted_demos_remain_retryable(tmp_path):
    trainer = make_trainer(tmp_path)
    demos = [str(tmp_path / "demos" / name) for name in ("good.dem", "bad.dem", "later.dem")]
    for filename in demos:
        Path(filename).touch()
    trainer.config["max_demos_per_session"] = 2
    def parse(filename):
        if filename.endswith("bad.dem"):
            return {"success": False, "error": "corrupt demo"}
        return {"success": True, "prediction_samples": [sample()], "kills": [{}]}
    trainer.demo_processor.process_demo_comprehensive = parse
    result = trainer.process_new_demos(demos)
    assert result["processed"] == 1 and result["failed"] == 1
    assert result["demos_processed"] == ["good.dem"]
    assert set(map(lambda value: Path(value).name, trainer._find_new_demos())) == {"bad.dem", "later.dem"}


def test_compressed_demos_are_discovered(tmp_path):
    trainer = make_trainer(tmp_path)
    (tmp_path / "demos" / "match.dem.zst").touch()
    assert [Path(value).name for value in trainer._find_new_demos()] == ["match.dem.zst"]


def test_gui_does_not_integrate_a_rejected_candidate(tmp_path):
    from ai_trainer_gui import AITrainerGUI
    from types import SimpleNamespace
    gui = AITrainerGUI.__new__(AITrainerGUI)
    gui.root = SimpleNamespace(after=lambda delay, callback: callback())
    gui.status_var = SimpleNamespace(set=lambda value: None)
    gui.progress_var = SimpleNamespace(set=lambda value: None)
    gui.incremental_var = SimpleNamespace(get=lambda: True)
    gui.log = lambda message: None
    gui.training_finished = lambda: None
    integrations = []
    gui.trainer = SimpleNamespace(
        process_new_demos=lambda filenames: {"processed": 1, "failed": 0, "new_kills": 1},
        train_model=lambda **kwargs: {"success": True, "published": False},
        auto_integrate_model=lambda: integrations.append(True),
    )
    gui.training_worker([str(tmp_path / "demo.dem")])
    assert integrations == []
