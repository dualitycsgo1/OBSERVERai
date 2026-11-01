"""Reproduce an isolated evaluation of parser JSON, without deploying live models.

Example, from the repository root:
    python KeylessOBS/evaluate_demos.py --results-dir work/parser-results \
        --output-dir work/evaluation-new --learning-curve --strict-next-killer

The output directory must be new and outside KeylessOBS. Imports do not train,
read demos, create directories, or import the optional scientific dependencies.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def _payload(path):
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("success") is not True:
        raise ValueError(f"{path.name}: parser did not report success")
    samples = data.get("prediction_samples", data.get("training_samples"))
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path.name}: no prediction samples")
    if not isinstance(data.get("demo_id"), str) or not data["demo_id"]:
        raise ValueError(f"{path.name}: missing content-based demo identity")
    if any(sample.get("demo_id") != data["demo_id"] for sample in samples):
        raise ValueError(f"{path.name}: inconsistent demo identity")
    return data, samples


def _records(files, summary):
    """Hold at most one source JSON document in memory at a time."""
    summary.clear()
    for path in files:
        data, samples = _payload(path)
        summary.append({"file": path.name, "demo_id": data["demo_id"],
                        "map_name": data.get("map_name", "unknown"),
                        "kills": len(data.get("kills", [])), "samples": len(samples),
                        "positive_samples": sum(sample.get("label") == 1 for sample in samples),
                        "data_quality": data.get("data_quality", {})})
        yield from samples


def _learning_curve(rows, manifest_path):
    from sklearn.ensemble import RandomForestClassifier
    from observer_training import demo_split, evaluate_candidate
    train, validation = demo_split(rows, str(manifest_path))
    groups = sorted({row["demo_id"] for row in train})
    curve, previous = [], None
    for count in range(1, len(groups) + 1):
        subset = [row for row in train if row["demo_id"] in groups[:count]]
        point = {"training_demo_count": count, "training_demo_ids": groups[:count],
                 "training_samples": len(subset)}
        if {row["label"] for row in subset} != {0, 1}:
            point["error"] = "This subset does not contain both outcomes"
        else:
            model = RandomForestClassifier(n_estimators=150, min_samples_leaf=3,
                                           random_state=42, n_jobs=-1)
            model.fit([row["features"] for row in subset], [row["label"] for row in subset])
            model.n_jobs = 1
            point["evaluation"] = evaluate_candidate(model, subset, validation, previous)
            previous = {"model": model, "train_demo_ids": groups[:count]}
        curve.append(point)
    return curve


def _strict_next_killer(files, bundle):
    from observer_model import FEATURE_NAMES, feature_vector, validate_model_bundle
    validate_model_bundle(bundle)
    held_out = set(bundle["validation_demo_ids"])
    per_demo = []
    for path in files:
        data, samples = _payload(path)
        if data["demo_id"] not in held_out:
            continue
        snapshots, kills = {}, defaultdict(list)
        for sample in samples:
            snapshots.setdefault(sample["sample_tick"], sample)
        for order, kill in enumerate(data.get("kills", [])):
            if (kill.get("attacker_team") in ("T", "CT") and
                    kill.get("victim_team") in ("T", "CT") and
                    kill["attacker_team"] != kill["victim_team"]):
                kills[kill["round_number"]].append((kill["time"], kill["tick"], order, kill["attacker_steam_id"]))
        for values in kills.values():
            values.sort()
        matrix, windows = [], []
        for tick, sample in sorted(snapshots.items()):
            if "sample_time" not in sample:
                raise ValueError("Strict next-killer evaluation requires parser JSON with sample_time")
            elapsed = sample["sample_time"]
            upcoming = [kill for kill in kills[sample["gamestate"]["map"]["round"] + 1]
                        if kill[0] > elapsed and kill[1] > tick and kill[0] <= elapsed + 2.0]
            if not upcoming:
                continue
            players, start = [], len(matrix)
            for player_id in sorted(sample["gamestate"]["allplayers"]):
                features = feature_vector(sample["gamestate"], player_id)
                if features is not None:
                    players.append(player_id)
                    matrix.append(features)
            if not players:
                raise ValueError("An action snapshot has no eligible prediction candidates")
            tied = len({kill[3] for kill in upcoming if kill[1] == upcoming[0][1]}) > 1
            windows.append((players, start, len(matrix), upcoming[0][3], tied))
        probabilities = bundle["model"].predict_proba(matrix)[:, 1] if matrix else []
        correct = nearest_correct = missing = ties = 0
        for players, start, end, killer, tied in windows:
            chosen = min(zip(players, probabilities[start:end]), key=lambda item: (-item[1], item[0]))[0]
            nearest = min(zip(players, matrix[start:end]),
                          key=lambda item: (item[1][FEATURE_NAMES.index("nearest_enemy_distance")], item[0]))[0]
            correct += chosen == killer
            nearest_correct += nearest == killer
            missing += killer not in players
            ties += tied
        per_demo.append({"demo_id": data["demo_id"], "map_name": data.get("map_name", "unknown"),
                         "action_snapshots": len(windows), "model_correct": correct,
                         "nearest_enemy_correct": nearest_correct,
                         "next_killer_missing_from_candidates": missing,
                         "simultaneous_killer_snapshots": ties})
    totals = {field: sum(demo[field] for demo in per_demo) for field in
              ("action_snapshots", "model_correct", "nearest_enemy_correct",
               "next_killer_missing_from_candidates", "simultaneous_killer_snapshots")}
    count = totals["action_snapshots"]
    return {**totals, "model_coverage": totals["model_correct"] / count if count else None,
            "nearest_enemy_coverage": totals["nearest_enemy_correct"] / count if count else None,
            "validation_demos": len(per_demo), "per_demo": per_demo,
            "definition": "First enemy kill in (sample_time, sample_time+2s], later tick, same live round; same-tick ties use demo event order. Static ranking without mesh filtering or camera cooldown."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory of successful demo-parser JSON files")
    parser.add_argument("--output-dir", type=Path, required=True, help="New isolated directory outside KeylessOBS; existing paths are refused")
    parser.add_argument("--learning-curve", action="store_true", help="Compare growing training-demo subsets on the same fixed holdout")
    parser.add_argument("--strict-next-killer", action="store_true", help="Also evaluate the first upcoming enemy killer; requires sample_time")
    args = parser.parse_args(argv)
    source, output = Path(__file__).resolve().parent, args.output_dir.resolve()
    if output == source or source in output.parents:
        parser.error("--output-dir must be outside KeylessOBS and its active model/data directories")
    if output.exists():
        parser.error("--output-dir already exists; choose a fresh directory to preserve previous evaluations")
    if not args.results_dir.is_dir():
        parser.error("--results-dir must be an existing directory")
    files = sorted(path for path in args.results_dir.resolve().glob("*.json") if path.name != "parse_summary.json")
    if len(files) < 5:
        parser.error("Need parser JSON from at least five independent demos")

    # Scientific imports and filesystem writes happen only during an explicit run.
    from ai_trainer import AITrainer
    from observer_training import prepare_prediction_samples
    summary = []

    class EvaluationTrainer(AITrainer):
        def _load_all_training_data(self):
            return _records(files, summary)

    output.mkdir(parents=True, exist_ok=False)
    trainer = EvaluationTrainer(*(str(output / name) for name in ("models", "data", "demos", "backups")))
    result = trainer.train_model(incremental=False)
    report = {"dataset": list(summary), "training_result": result,
              "deployment_scope": "Isolated evaluation directory only; live observer models are not updated",
              "limitations": "Demo-grouped validation is not an independent final test; repeated selection can overfit the holdout. Overlapping snapshots are not independent trials."}
    report_path = output / "evaluation.json"
    try:
        if result.get("success") and args.learning_curve:
            rows, _ = prepare_prediction_samples(_records(files, summary))
            report["learning_curve"] = _learning_curve(rows, output / "data" / "validation_manifest.json")
        if result.get("success") and args.strict_next_killer:
            if result.get("published"):
                import pickle
                with Path(result["model_path"]).open("rb") as handle:
                    bundle = pickle.load(handle)
                report["strict_next_killer_evaluation"] = _strict_next_killer(files, bundle)
            else:
                report["strict_next_killer_evaluation"] = {"available": False, "reason": "Candidate failed the promotion gate; no validated model was saved"}
    except (OSError, ValueError, KeyError, TypeError) as error:
        report["optional_evaluation_error"] = str(error)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(json.dumps({"success": result.get("success"), "isolated_candidate_passed": result.get("published"),
                      "error": result.get("error"), "report": str(report_path)}, indent=2))
    return 0 if result.get("success") and "optional_evaluation_error" not in report else 1


if __name__ == "__main__":
    raise SystemExit(main())
