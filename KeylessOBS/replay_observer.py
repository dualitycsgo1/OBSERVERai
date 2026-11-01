"""Replay sampled demo snapshots through the real observer without input effects.

This is an offline state-machine check, not rendered POV or broadcast validation.
Only load model bundles you trust: the observer's model format uses pickle.
"""
import argparse
import json
import logging
import os
from pathlib import Path
import pickle
import sys
import time as real_time
from datetime import datetime as RealDateTime, timedelta
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext


def replay(parsed_path, model_path, meshes_dir, output_dir, limit=None, mesh_enabled=True,
           mesh_game_patch=None, model_enabled=True):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    appdata = output_dir / "appdata"
    appdata.mkdir(exist_ok=True)
    os.environ["APPDATA"] = str(appdata)

    # No keyboard module hooks or native input calls are allowed in this process.
    keyboard_stub = ModuleType("keyboard")
    blocked_native_attempts = []
    def forbidden(*args, **kwargs):
        blocked_native_attempts.append("Native input, registration, or real sleep attempted")
        raise AssertionError("Native input or hotkey registration attempted during offline replay")
    keyboard_stub.add_hotkey = keyboard_stub.hook = keyboard_stub.wait = forbidden
    sys.modules["keyboard"] = keyboard_stub

    import cs2_duel_detector as observer
    from line_of_sight import MapGeometryManager
    from observer_model import validate_model_bundle, predict_probabilities
    observer_source_sha256 = __import__("hashlib").sha256(Path(observer.__file__).read_bytes()).hexdigest()

    with open(model_path, "rb") as stream:
        bundle = validate_model_bundle(pickle.load(stream))
    model_sha256 = __import__("hashlib").sha256(Path(model_path).read_bytes()).hexdigest()
    bundle["model"].n_jobs = 1
    with open(parsed_path, encoding="utf-8") as stream:
        parsed = json.load(stream)
    if parsed["demo_id"] not in bundle["validation_demo_ids"]:
        raise ValueError("Replay demo must be held out from training")
    groups = {}
    for sample in parsed["prediction_samples"]:
        if "sample_time" not in sample:
            raise ValueError("Reparse the demo with the snapshot-time exporter")
        key = (sample["gamestate"]["map"]["round"], sample["sample_tick"])
        group = groups.setdefault(key, {"time": sample["sample_time"], "tick": sample["sample_tick"],
                                        "state": sample["gamestate"], "labels": {}})
        group["labels"][sample["player_id"]] = sample["label"]
    ordered = sorted(groups.values(), key=lambda group: group["time"])
    if limit is not None:
        ordered = ordered[:limit]
    if not ordered:
        raise ValueError("No replay snapshots")
    # The exporter uses Source 2 live rounds and never emits the round's last
    # incomplete 2s horizon. Do not invent frozen/dead-player game states here.
    ids = sorted({sid for group in ordered for sid in group["state"]["allplayers"]})
    if len(ids) > 10:
        raise ValueError("Stable spectator slots require at most ten player identities")
    slots = {sid: index + 1 for index, sid in enumerate(ids)}
    slot_ids = {slot: sid for sid, slot in slots.items()}

    class Clock:
        seconds = ordered[0]["time"]
        def timestamp(self):
            return 1767225600.0 + self.seconds
    clock = Clock()
    class ReplayDateTime(RealDateTime):
        @classmethod
        def now(cls, tz=None):
            result = RealDateTime(2026, 1, 1) + timedelta(seconds=clock.seconds)
            return result if tz is None else result.replace(tzinfo=tz)

    class Errors(logging.Handler):
        def __init__(self):
            super().__init__(logging.ERROR)
            self.messages = []
        def emit(self, record):
            self.messages.append(record.getMessage())
    errors = Errors()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)
    root_logger.addHandler(errors)
    geometry = MapGeometryManager(str(meshes_dir)) if mesh_enabled else None
    clock_module = SimpleNamespace(time=clock.timestamp, monotonic=lambda: clock.seconds, sleep=forbidden)
    input_requests, switches, trace = [], [], []
    camera = {"sid": None}
    processing_seconds = []
    started = real_time.perf_counter()

    def safe_key(code):
        input_requests.append({"time": clock.seconds, "code": code})
        return True

    with patch.object(observer, "get_base_dir", return_value=str(Path(model_path).resolve().parent.parent)), \
            patch.object(observer, "get_geometry_manager", return_value=geometry), \
            patch.object(observer, "time", clock_module), \
            patch.object(observer, "datetime", ReplayDateTime), \
            patch.object(observer, "send_key", side_effect=safe_key), \
            patch.object(observer, "user32", SimpleNamespace(keybd_event=forbidden)), \
            (nullcontext() if model_enabled else patch.object(observer.ModelHotReloader, "get_model", return_value=None)):
        config_path = Path(observer.__file__).resolve().parent / "config.json"
        detector = observer.CS2DuelDetector(str(config_path))
        detector.campaths_enabled = False
        if model_enabled and not detector._has_validated_forecast():
            raise RuntimeError("Isolated trained model was not loaded")
        # This changes only inference execution parallelism, never fitted trees
        # or scores. Small per-player requests do not benefit from thread pools.
        if model_enabled:
            detector.ai_analyzer.model["model"].n_jobs = 1
        else:
            detector.ai_analyzer = None
            detector.use_ai = False
        detector.auto_model_updates = False
        original_switch = detector.switch_to_player
        def safe_switch(slot):
            previous_inputs = len(input_requests)
            original_switch(slot)  # Exercises the real cooldown/bookkeeping with send_key stubbed.
            if len(input_requests) > previous_inputs:
                previous = camera["sid"]
                camera["sid"] = slot_ids[slot]
                switches.append({"time": clock.seconds, "target": camera["sid"], "changed": previous != camera["sid"]})
        detector.switch_to_player = safe_switch
        original_slot = detector.get_player_slot
        def safe_slot(sid, state):
            result = original_slot(sid, state)
            if result is not None and result != slots.get(sid):
                raise AssertionError("Observer slot mapping differs from the simulated spectator slots")
            return result
        detector.get_player_slot = safe_slot
        for index, group in enumerate(ordered):
            clock.seconds = group["time"]
            state, labels = group["state"], group["labels"]
            for sid, player in state["allplayers"].items():
                player["observer_slot"] = slots[sid] - 1
            errors_before = len(errors.messages)
            process_started = real_time.perf_counter()
            detector.process_gamestate(state)
            processing_seconds.append(real_time.perf_counter() - process_started)
            positives = [sid for sid, label in labels.items() if label == 1]
            held = camera["sid"]
            future_kills = [kill for kill in parsed["kills"]
                            if kill["round_number"] == state["map"]["round"] + 1
                            and kill["tick"] > group["tick"]
                            and group["time"] < kill["time"] <= group["time"] + 2
                            and kill["attacker_team"] != kill["victim_team"]]
            next_kill = min(future_kills, key=lambda kill: (kill["time"], kill["tick"])) if future_kills else None
            next_killer = next_kill["attacker_steam_id"] if next_kill else None
            # Diagnostics run after the state-machine step and never alter its
            # selected target. They distinguish eligibility from camera holds.
            pool = {sid for first, second, _ in detector.find_potential_duels(state["allplayers"])
                    for sid in (first, second)}
            raw_scores = predict_probabilities(bundle, state, sorted(state["allplayers"]))
            raw_top = min(raw_scores, key=lambda sid: (-raw_scores[sid], sid)) if raw_scores else None
            eligible_scores = {sid: score for sid, score in raw_scores.items() if sid in pool}
            eligible_top = min(eligible_scores, key=lambda sid: (-eligible_scores[sid], sid)) if eligible_scores else None
            trace.append({
                "time": clock.seconds, "tick": group["tick"], "round": state["map"]["round"],
                "held_target": held, "internal_target": detector.current_target_sid,
                "target_present": held in state["allplayers"] if held else False,
                "action_snapshot": bool(positives),
                "held_target_kills_within_2s": labels.get(held) == 1,
                "next_killer_within_2s": next_killer,
                "held_target_is_next_killer": bool(next_kill and held == next_killer),
                "duel_pool_size": len(pool), "next_killer_in_duel_pool": bool(next_kill and next_killer in pool),
                "raw_model_top1": raw_top, "raw_model_top1_is_next_killer": bool(next_kill and raw_top == next_killer),
                "duel_pool_model_top1": eligible_top,
                "duel_pool_model_top1_is_next_killer": bool(next_kill and eligible_top == next_killer),
                "processing_errors": len(errors.messages) - errors_before,
            })
            if (index + 1) % 500 == 0:
                print(json.dumps({"processed": index + 1, "total": len(ordered)}), flush=True)
    root_logger.removeHandler(errors)
    actions = [row for row in trace if row["action_snapshot"]]
    covered = sum(row["held_target_kills_within_2s"] for row in actions)
    duration = ordered[-1]["time"] - ordered[0]["time"]
    strict_actions = [row for row in trace if row["next_killer_within_2s"] is not None]
    strict_covered = sum(row["held_target_is_next_killer"] for row in strict_actions)
    patch_matches = bool(mesh_game_patch and
                         str(parsed.get("patch_version")) == mesh_game_patch.replace(".", ""))
    result = {
        "demo_file": Path(parsed_path).name, "demo_id": parsed["demo_id"],
        "map_name": parsed["map_name"], "model_path": str(Path(model_path).resolve()),
        "model_sha256": model_sha256,
        "held_out": True, "mesh_enabled": mesh_enabled,
        "runtime_model_enabled": model_enabled,
        "geometry_loaded": bool(geometry and geometry.use_real_geometry),
        "geometry_sources": geometry.sources if geometry else {},
        "geometry_errors": geometry.errors if geometry else {},
        "snapshots": len(trace), "action_snapshots": len(actions),
        "held_target_positive_action_snapshots": covered,
        "held_target_future_kill_coverage": covered / len(actions) if actions else None,
        "strict_next_killer_action_snapshots": len(strict_actions),
        "held_target_next_killer_hits": strict_covered,
        "held_target_next_killer_coverage": strict_covered / len(strict_actions) if strict_actions else None,
        "next_killer_in_duel_pool": sum(row["next_killer_in_duel_pool"] for row in strict_actions),
        "next_killer_duel_pool_coverage": sum(row["next_killer_in_duel_pool"] for row in strict_actions) / len(strict_actions) if strict_actions else None,
        "raw_model_top1_next_killer_hits": sum(row["raw_model_top1_is_next_killer"] for row in strict_actions),
        "raw_model_top1_next_killer_coverage": sum(row["raw_model_top1_is_next_killer"] for row in strict_actions) / len(strict_actions) if strict_actions else None,
        "duel_pool_model_top1_next_killer_hits": sum(row["duel_pool_model_top1_is_next_killer"] for row in strict_actions),
        "duel_pool_model_top1_next_killer_coverage": sum(row["duel_pool_model_top1_is_next_killer"] for row in strict_actions) / len(strict_actions) if strict_actions else None,
        "switch_requests": len(switches),
        "camera_target_changes": sum(item["changed"] for item in switches),
        "camera_target_changes_per_demo_minute": sum(item["changed"] for item in switches) / (duration / 60) if duration else None,
        "sampled_timeline_seconds": duration,
        "snapshots_with_target": sum(row["held_target"] is not None for row in trace),
        "snapshots_target_absent": sum(bool(row["held_target"]) and not row["target_present"] for row in trace),
        "internal_target_camera_mismatches": sum(row["held_target"] != row["internal_target"] for row in trace),
        "processing_errors": len(errors.messages), "error_messages": errors.messages[:30],
        "real_input_events": 0, "native_input_blocked": True,
        "blocked_native_input_attempts": len(blocked_native_attempts),
        "source_sha256": observer_source_sha256,
        "estimator_inference_jobs": 1,
        "processing_seconds_total": sum(processing_seconds),
        "processing_seconds_mean": sum(processing_seconds) / len(processing_seconds),
        "processing_seconds_max": max(processing_seconds),
        "demo_patch_version": parsed.get("patch_version"),
        "mesh_game_patch": mesh_game_patch if mesh_enabled else None,
        "geometry_patch_version_matches": patch_matches if mesh_enabled else None,
        "wall_seconds": real_time.perf_counter() - started,
        "scope": "process_gamestate replay of sampled live snapshots; held target evaluated against overlapping next-2s kill labels",
        "limitations": [
            "This is not actual POV kill capture, rendered video, input delivery, or measured GSI/network latency.",
            "Only alive-player snapshots are exported; round_kills and freeze/death transition states are absent.",
            "The last incomplete two-second horizon of each round is excluded.",
            "Repeated overlapping action windows are correlated and are not independent kills.",
            "This is a system smoke check; verify mesh patch metadata against the demo before interpreting geometry-dependent accuracy.",
            "Spectator slot mappings are simulated; real key delivery is blocked.",
            "The replay clock follows demo timestamps and excludes processing and command delivery delay.",
            "Doors, smoke and bullet penetration are not simulated.",
        ],
    }
    name = parsed["map_name"] + ("_mesh" if mesh_enabled else "_unknown")
    if not model_enabled:
        name += "_fallback"
    (output_dir / (name + "_summary.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output_dir / (name + "_trace.jsonl")).open("w", encoding="utf-8") as stream:
        for row in trace:
            stream.write(json.dumps(row) + "\n")
    (output_dir / (name + "_switches.json")).write_text(json.dumps(switches, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parsed_demo")
    parser.add_argument("--model", required=True)
    parser.add_argument("--meshes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--without-mesh", action="store_true")
    parser.add_argument("--without-model", action="store_true", help="Disable the runtime model; retain it only for separate raw-score diagnostics")
    parser.add_argument("--mesh-game-patch", help="Official mesh asset game patch, e.g. 1.41.9.4")
    args = parser.parse_args()
    replay(args.parsed_demo, args.model, args.meshes, args.output, args.limit, not args.without_mesh,
           args.mesh_game_patch, not args.without_model)


if __name__ == "__main__":
    main()
