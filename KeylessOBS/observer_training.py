"""Leakage-resistant evaluation for the observer's two-second kill forecast."""
import hashlib
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, log_loss

from observer_model import FEATURE_NAMES, HORIZON_SECONDS, TARGET, feature_vector

# Parallel forest probability sums can differ in the last floating-point bit.
# A numerical no-op is not evidence that another model should be deployed.
MIN_BRIER_IMPROVEMENT = 1e-6


def prepare_prediction_samples(records):
    """Keep genuine pre-event examples and deduplicate across batch/master files."""
    rows, seen, rejected = [], {}, Counter()
    for record in records:
        try:
            if not isinstance(record, dict) or record.get("target") != TARGET:
                rejected["legacy_or_wrong_target"] += 1
                continue
            if record.get("snapshot_phase") != "pre_event":
                raise ValueError("snapshot_phase must be pre_event")
            if float(record["horizon_seconds"]) != HORIZON_SECONDS:
                raise ValueError("unsupported forecast horizon")
            demo_id, tick, player_id = record["demo_id"], record["sample_tick"], record["player_id"]
            if not isinstance(demo_id, str) or not demo_id or not isinstance(player_id, str):
                raise ValueError("missing demo/player identity")
            if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
                raise ValueError("invalid sample tick")
            label = record["label"]
            if label not in (0, 1) or not isinstance(label, (int, bool)):
                raise ValueError("label must be an observed future-kill outcome, 0 or 1")
            features = feature_vector(record["gamestate"], player_id)
            if features is None:
                raise ValueError("incomplete or non-alive prediction snapshot")
            identity = (demo_id, tick, player_id)
            canonical = (int(label), tuple(features))
            if identity in seen:
                if seen[identity] != canonical:
                    raise ValueError("Conflicting duplicate prediction sample")
                rejected["duplicate"] += 1
                continue
            seen[identity] = canonical
            rows.append({"demo_id": demo_id, "sample_tick": tick, "player_id": player_id,
                         "map_name": record["gamestate"].get("map", {}).get("name", "unknown"),
                         "label": int(label), "features": features})
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            if "Conflicting duplicate" in str(error):
                raise
            rejected[str(error)] += 1
    # Parser maps and source files need not preserve row order. A fixed RNG seed
    # alone does not make bootstrap/tree training reproducible when row order
    # changes, so use the content-based identity as the canonical training order.
    rows.sort(key=lambda row: (row["demo_id"], row["sample_tick"], row["player_id"]))
    return rows, dict(rejected)


def demo_split(rows, manifest_path, validation_fraction=0.2):
    """Persist demo membership, so a previous training demo never becomes a test."""
    groups = sorted({row["demo_id"] for row in rows})
    if len(groups) < 5:
        raise ValueError("Need at least five independent demos (including two held-out demos)")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_split must be between 0 and 1")
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("target") != TARGET or not isinstance(manifest.get("assignments"), dict):
            raise ValueError("Invalid validation manifest")
        assignments = manifest["assignments"]
        if any(value not in ("train", "validation") for value in assignments.values()):
            raise ValueError("Invalid validation membership")
    else:
        ordered = sorted(groups, key=lambda group: hashlib.sha256(group.encode()).hexdigest())
        held_out = set(ordered[:max(2, math.ceil(len(groups) * validation_fraction))])
        assignments = {group: "validation" if group in held_out else "train" for group in groups}
    for group in groups:
        if group not in assignments:
            bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) / 0xffffffff
            assignments[group] = "validation" if bucket < validation_fraction else "train"
    train = [row for row in rows if assignments[row["demo_id"]] == "train"]
    validation = [row for row in rows if assignments[row["demo_id"]] == "validation"]
    if len({row["demo_id"] for row in train}) < 2 or len({row["demo_id"] for row in validation}) < 2:
        raise ValueError("Need at least two training and two held-out demos in the fixed split")
    for name, subset in (("training", train), ("validation", validation)):
        if {row["label"] for row in subset} != {0, 1}:
            raise ValueError(f"The {name} demos must contain both kill and no-kill samples")
    manifest = {"target": TARGET, "horizon_seconds": HORIZON_SECONDS, "assignments": assignments}
    temporary = manifest_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    os.replace(temporary, manifest_path)
    return train, validation


def top1_kill_coverage(rows, probabilities):
    """Among snapshots with an upcoming kill, did the top-ranked player get one?"""
    snapshots = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        snapshots[(row["demo_id"], row["sample_tick"])].append((row["player_id"], row["label"], float(probability)))
    hits, opportunities = 0, 0
    for candidates in snapshots.values():
        if any(candidate[1] for candidate in candidates):
            opportunities += 1
            # Stable tie break makes comparisons reproducible.
            winner = min(candidates, key=lambda candidate: (-candidate[2], candidate[0]))
            hits += winner[1]
    return hits / opportunities if opportunities else None, opportunities


def metrics(rows, probabilities):
    labels = [row["label"] for row in rows]
    coverage, opportunities = top1_kill_coverage(rows, probabilities)
    return {
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "accuracy_at_0_5": float(accuracy_score(labels, np.asarray(probabilities) >= 0.5)),
        "top1_kill_coverage": coverage, "action_snapshots": opportunities,
        "samples": len(rows), "positive_samples": sum(labels),
    }


def grouped_brier_improvement(rows, reference, candidate):
    """Paired bootstrap of per-demo mean error; correlated ticks stay together."""
    improvements = defaultdict(list)
    for row, old, new in zip(rows, reference, candidate):
        label = row["label"]
        improvements[row["demo_id"]].append((label - old) ** 2 - (label - new) ** 2)
    means = np.asarray([np.mean(values) for _, values in sorted(improvements.items())])
    generator = np.random.default_rng(42)
    bootstrap = generator.choice(means, size=(2000, len(means)), replace=True).mean(axis=1)
    return {"mean": float(means.mean()), "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)), "independent_demos": len(means)}


def evaluate_candidate(model, train_rows, validation_rows, incumbent=None):
    features = [row["features"] for row in validation_rows]
    candidate_probabilities = model.predict_proba(features)[:, 1]
    prevalence = float(np.mean([row["label"] for row in train_rows]))
    baseline_probabilities = np.full(len(validation_rows), prevalence)
    baseline = metrics(validation_rows, baseline_probabilities)
    candidate = metrics(validation_rows, candidate_probabilities)
    # A simple camera-ranking benchmark: observe the player closest to an enemy.
    distance_probabilities = [1.0 / (1.0 + row["features"][FEATURE_NAMES.index("nearest_enemy_distance")]) for row in validation_rows]
    distance_coverage, _ = top1_kill_coverage(validation_rows, distance_probabilities)
    improvement = grouped_brier_improvement(validation_rows, baseline_probabilities, candidate_probabilities)
    reasons = []
    if improvement["ci95_low"] <= MIN_BRIER_IMPROVEMENT or candidate["log_loss"] >= baseline["log_loss"]:
        reasons.append("No demonstrated probability improvement over the training-prevalence baseline")
    if candidate["top1_kill_coverage"] is None or candidate["top1_kill_coverage"] < distance_coverage:
        reasons.append("Camera ranking is worse than the nearest-enemy baseline")
    evaluation = {"candidate": candidate, "prevalence_baseline": baseline,
                  "nearest_enemy_top1_coverage": distance_coverage,
                  "brier_improvement_over_baseline": improvement,
                  "minimum_brier_improvement": MIN_BRIER_IMPROVEMENT,
                  "scope": "fixed held-out demos; repeated model selection can overfit this validation set"}
    evaluation["per_demo"] = {}
    for demo_id in sorted({row["demo_id"] for row in validation_rows}):
        indexes = [index for index, row in enumerate(validation_rows) if row["demo_id"] == demo_id]
        subset = [validation_rows[index] for index in indexes]
        probabilities = candidate_probabilities[indexes]
        evaluation["per_demo"][demo_id] = {
            "map_name": subset[0].get("map_name", "unknown"),
            "candidate": metrics(subset, probabilities),
            "prevalence_baseline": metrics(subset, baseline_probabilities[indexes]),
            "nearest_enemy_top1_coverage": top1_kill_coverage(subset, [distance_probabilities[index] for index in indexes])[0],
        }
    if incumbent is not None:
        test_groups = {row["demo_id"] for row in validation_rows}
        if test_groups & set(incumbent.get("train_demo_ids", ())):
            raise ValueError("Incumbent has trained on current validation demos; use a fresh external test set")
        old_probabilities = incumbent["model"].predict_proba(features)[:, 1]
        old = metrics(validation_rows, old_probabilities)
        delta = grouped_brier_improvement(validation_rows, old_probabilities, candidate_probabilities)
        evaluation["incumbent"] = old
        evaluation["brier_improvement_over_incumbent"] = delta
        if delta["ci95_low"] <= MIN_BRIER_IMPROVEMENT or candidate["log_loss"] > old["log_loss"] or candidate["top1_kill_coverage"] < old["top1_kill_coverage"]:
            reasons.append("No demonstrated improvement over the deployed model on unseen demos")
    evaluation["promotion_passed"] = not reasons
    evaluation["rejection_reasons"] = reasons
    return evaluation
