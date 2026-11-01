"""Replay sampled demo snapshots through the real observer without input effects.

This is an offline state-machine check, not rendered POV or broadcast validation.
Only load model bundles you trust: the observer's model format uses pickle.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json
import logging
import math
import os
from pathlib import Path
import pickle
import sys
import time as real_time
from datetime import datetime as RealDateTime, timedelta
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from contextlib import nullcontext


class CausalReplayReconstructor:
    """Restore confirmed deaths/round kills using only information already seen.

    Positions, weapons and ammo come from the current or previous snapshots.
    Kill-event positions are deliberately not used. Missing players without a
    confirmed death retain unknown health rather than being declared dead.
    """

    def __init__(self, kills, maximum_gap_seconds=2.0):
        if not math.isfinite(maximum_gap_seconds) or maximum_gap_seconds <= 0:
            raise ValueError("maximum_gap_seconds must be positive and finite")
        self.maximum_gap_seconds = maximum_gap_seconds
        self.kills = []
        self.ignored_events = 0
        for event in kills:
            try:
                copied = deepcopy(event)
                copied["time"] = float(copied["time"])
                copied["tick"] = int(copied["tick"])
                copied["round_number"] = int(copied["round_number"])
                if not math.isfinite(copied["time"]):
                    raise ValueError("Invalid event time")
                self.kills.append(copied)
            except (KeyError, TypeError, ValueError, OverflowError):
                self.ignored_events += 1
        self.round_key = None
        self.previous = {}
        self.previous_time = None
        self.previous_tick = None
        self.dead = set()
        self.interval_start_time = self.interval_start_tick = None
        self.excluded_before_interval = set()
        self.interval_resets = []

    @staticmethod
    def _identity(value):
        return str(value) if value is not None and str(value) not in ("", "0") else None

    def apply(self, gamestate, sample_time, sample_tick):
        sample_time, sample_tick = float(sample_time), int(sample_tick)
        if not math.isfinite(sample_time):
            raise ValueError("Snapshot time must be finite")
        state = deepcopy(gamestate)
        map_info = state.get("map") or {}
        round_key = (map_info.get("name"), int(map_info["round"]))
        gap_reset = self.previous_time is not None and sample_time - self.previous_time > self.maximum_gap_seconds
        if round_key != self.round_key or gap_reset:
            self.interval_resets.append({"time": sample_time, "round": round_key[1],
                                         "reason": "round changed" if round_key != self.round_key else "observed telemetry gap"})
            self.previous, self.dead = {}, set()
            self.previous_time = self.previous_tick = None
            self.round_key = round_key
            self.interval_start_time, self.interval_start_tick = sample_time, sample_tick
        elif ((self.previous_time is not None and sample_time < self.previous_time)
              or (self.previous_tick is not None and sample_tick < self.previous_tick)):
            raise ValueError("Causal replay snapshots must arrive in chronological order")

        raw_players = state.get("allplayers") or {}
        current = {str(sid): deepcopy(player) for sid, player in raw_players.items()
                   if sid != "total" and isinstance(player, dict)}
        players = deepcopy(self.previous)
        players.update(current)
        round_events = [event for event in self.kills if event["round_number"] == round_key[1] + 1]
        # Warmup/knife rounds may reuse round_number=1 before the actual live
        # samples. Without an event phase marker, earlier events cannot safely
        # establish deaths or counters in this observed live interval.
        before_interval = [event for event in round_events
                           if event["time"] < self.interval_start_time or event["tick"] < self.interval_start_tick]
        self.excluded_before_interval.update((event["round_number"], event["tick"],
                                             event.get("attacker_steam_id"), event.get("victim_steam_id"))
                                            for event in before_interval)
        past_events = [event for event in round_events
                       if self.interval_start_time <= event["time"] <= sample_time
                       and self.interval_start_tick <= event["tick"] <= sample_tick]
        confirmed_dead = set()
        enemy_kills = Counter()
        for event in past_events:
            attacker = self._identity(event.get("attacker_steam_id"))
            victim = self._identity(event.get("victim_steam_id"))
            for identity, prefix in ((attacker, "attacker"), (victim, "victim")):
                if identity and identity not in players:
                    # Names/teams are facts in this already-observed event;
                    # unavailable positions, view and weapon states stay absent.
                    players[identity] = {"team": event.get(prefix + "_team"),
                                         "name": event.get(prefix + "_name"), "state": {"health": None}}
            if victim:
                confirmed_dead.add(victim)
            if (attacker and victim and attacker != victim
                    and event.get("attacker_team") in ("T", "CT")
                    and event.get("victim_team") in ("T", "CT")
                    and event["attacker_team"] != event["victim_team"]):
                enemy_kills[attacker] += 1

        unknown_missing = []
        for sid, player in players.items():
            player_state = player.setdefault("state", {})
            if sid in confirmed_dead:
                current_health = (current.get(sid, {}).get("state") or {}).get("health")
                if isinstance(current_health, (int, float)) and current_health > 0:
                    raise ValueError("Snapshot says player is alive after a confirmed same-round death")
                player_state["health"] = 0
            elif sid not in current:
                # GSI expresses unavailable health by omission. Do not send a
                # fabricated numeric zero (death) or a nonnumeric null to the
                # legacy numeric consumer.
                player_state.pop("health", None)
                unknown_missing.append(sid)
            player_state["round_kills"] = enemy_kills[sid]
        # Preserve the original reported roster size where supplied. Otherwise
        # this is only the roster already observed, not a future identity list.
        reported_total = raw_players.get("total")
        state["allplayers"] = players
        if reported_total is not None:
            state["allplayers"]["total"] = reported_total
        reconstruction = {
            "enabled": True, "past_kill_events": len(past_events),
            "confirmed_dead": sorted(confirmed_dead),
            "newly_confirmed_dead": sorted(confirmed_dead - self.dead),
            "unknown_missing": sorted(unknown_missing), "ignored_events": self.ignored_events,
            "interval_start_time": self.interval_start_time, "interval_start_tick": self.interval_start_tick,
            "excluded_pre_interval_events_this_round": len(before_interval),
            "positions_and_ammo": "current or past snapshots only; no event-position or ammo imputation",
            "kill_counter": "cumulative enemy kills since the first observed live state of this interval",
            "left_censored_start": "events before the first sampled live state are excluded because warmup can reuse round numbers",
        }
        state["_replay_reconstruction"] = reconstruction
        self.previous, self.dead = deepcopy(players), confirmed_dead
        self.previous_time, self.previous_tick = sample_time, sample_tick
        return state


def replay(parsed_path, model_path, meshes_dir, output_dir, limit=None, mesh_enabled=True,
           mesh_game_patch=None, model_enabled=True, camera_policy="broadcast", reconstruct_events=False):
    if camera_policy not in ("broadcast", "legacy"):
        raise ValueError("camera_policy must be broadcast or legacy")
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
    source_fingerprints = {}
    for source_name in ("cs2_duel_detector.py", "observer_runtime_policy.py", "observer_policy.py",
                        "observer_model.py", "replay_observer.py"):
        source_path = Path(observer.__file__).resolve().parent / source_name
        if source_path.is_file():
            source_fingerprints[source_name] = __import__("hashlib").sha256(source_path.read_bytes()).hexdigest()

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
    first_interval_end = ordered[0]["time"]
    for previous, current in zip(ordered, ordered[1:]):
        if (current["state"]["map"]["round"] != previous["state"]["map"]["round"]
                or current["time"] - previous["time"] > 2.0):
            break
        first_interval_end = current["time"]
    # The exporter never emits the round's last incomplete 2s horizon.
    # Slot identities are a fixed actuator mapping only, never injected as
    # future roster members or features by the causal reconstruction.
    ids = sorted({sid for group in ordered for sid in group["state"]["allplayers"]})
    if len(ids) > 10:
        raise ValueError("Stable spectator slots require at most ten player identities")
    slots = {sid: index + 1 for index, sid in enumerate(ids)}
    slot_ids = {slot: sid for sid, slot in slots.items()}
    reconstructor = CausalReplayReconstructor(parsed["kills"]) if reconstruct_events else None

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
        detector.config.setdefault("observer", {})["camera_policy"] = camera_policy
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
            success = original_switch(slot)  # Preserve the real actuator acknowledgement.
            if len(input_requests) > previous_inputs:
                previous = camera["sid"]
                camera["sid"] = slot_ids[slot]
                switches.append({"time": clock.seconds, "target": camera["sid"], "changed": previous != camera["sid"]})
            return success
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
            state = (reconstructor.apply(group["state"], group["time"], group["tick"])
                     if reconstructor else deepcopy(group["state"]))
            labels = group["labels"]
            for sid, player in state["allplayers"].items():
                if sid != "total" and sid in slots:
                    player["observer_slot"] = slots[sid] - 1
            errors_before = len(errors.messages)
            detector.last_observer_decision = None
            process_started = real_time.perf_counter()
            detector.process_gamestate(state)
            processing_seconds.append(real_time.perf_counter() - process_started)
            positives = [sid for sid, label in labels.items() if label == 1]
            held = camera["sid"]
            held_health = (state["allplayers"].get(held, {}).get("state") or {}).get("health")
            future_kills = [kill for kill in parsed["kills"]
                            if kill["round_number"] == state["map"]["round"] + 1
                            and kill["tick"] > group["tick"]
                            and group["time"] < kill["time"] <= group["time"] + 2
                            and kill["attacker_team"] != kill["victim_team"]]
            next_kill = min(future_kills, key=lambda kill: (kill["time"], kill["tick"])) if future_kills else None
            next_killer = next_kill["attacker_steam_id"] if next_kill else None
            next_victim = next_kill["victim_steam_id"] if next_kill else None
            # Diagnostics run after the state-machine step and never alter its
            # selected target. They distinguish eligibility from camera holds.
            pool = {sid for first, second, _ in detector.find_potential_duels(state["allplayers"])
                    for sid in (first, second)}
            raw_scores = predict_probabilities(bundle, state, sorted(state["allplayers"]))
            raw_top = min(raw_scores, key=lambda sid: (-raw_scores[sid], sid)) if raw_scores else None
            eligible_scores = {sid: score for sid, score in raw_scores.items() if sid in pool}
            eligible_top = min(eligible_scores, key=lambda sid: (-eligible_scores[sid], sid)) if eligible_scores else None
            adapter = getattr(detector, "_broadcast_runtime", None)
            policy_state = getattr(getattr(adapter, "policy", None), "state", None)
            policy_context = getattr(adapter, "last_context", None)
            actual_policy_ids = (sorted(candidate.player_id for candidate in policy_context.candidates if candidate.alive)
                                 if camera_policy == "broadcast" and policy_context is not None
                                 and policy_context.now == clock.timestamp() else None)
            actual_policy_scored_count = (sum(candidate.kill_probability is not None for candidate in policy_context.candidates)
                                          if actual_policy_ids is not None else None)
            trace.append({
                "time": clock.seconds, "tick": group["tick"], "round": state["map"]["round"],
                "held_target": held, "internal_target": detector.current_target_sid,
                "target_present": held in state["allplayers"] if held else False,
                "held_target_health": held_health,
                "action_snapshot": bool(positives),
                "held_target_kills_within_2s": labels.get(held) == 1,
                "next_killer_within_2s": next_killer,
                "next_victim_within_2s": next_victim,
                "next_kill_time": next_kill["time"] if next_kill else None,
                "next_kill_tick": next_kill["tick"] if next_kill else None,
                "next_kill_weapon": next_kill["weapon"] if next_kill else None,
                "held_target_is_next_killer": bool(next_kill and held == next_killer),
                "held_target_is_next_victim": bool(next_kill and held == next_victim),
                "held_target_in_next_kill_duel": bool(next_kill and held in (next_killer, next_victim)),
                "duel_pool_size": len(pool), "next_killer_in_duel_pool": bool(next_kill and next_killer in pool),
                "raw_model_top1": raw_top, "raw_model_top1_is_next_killer": bool(next_kill and raw_top == next_killer),
                "raw_model_scored_candidate_count": len(raw_scores),
                "duel_pool_model_top1": eligible_top,
                "duel_pool_model_top1_is_next_killer": bool(next_kill and eligible_top == next_killer),
                "direct_duel_pool_metric_scope": "legacy direct-duel diagnostic; not the broadcast candidate pool",
                "actual_policy_candidate_ids": actual_policy_ids,
                "actual_policy_candidate_count": len(actual_policy_ids) if actual_policy_ids is not None else None,
                "actual_policy_model_scored_candidate_count": actual_policy_scored_count,
                "next_killer_in_actual_policy_pool": (bool(next_kill and next_killer in actual_policy_ids)
                                                       if actual_policy_ids is not None else None),
                "processing_errors": len(errors.messages) - errors_before,
                "camera_policy": camera_policy,
                "policy_decision": deepcopy(getattr(detector, "last_observer_decision", None)),
                "policy_state": asdict(policy_state) if is_dataclass(policy_state) else None,
                "reconstruction": deepcopy(state.get("_replay_reconstruction")),
            })
            if (index + 1) % 500 == 0:
                print(json.dumps({"processed": index + 1, "total": len(ordered)}), flush=True)
    root_logger.removeHandler(errors)
    actions = [row for row in trace if row["action_snapshot"]]
    covered = sum(row["held_target_kills_within_2s"] for row in actions)
    duration = ordered[-1]["time"] - ordered[0]["time"]
    strict_actions = [row for row in trace if row["next_killer_within_2s"] is not None]
    strict_covered = sum(row["held_target_is_next_killer"] for row in strict_actions)
    after_first_interval = [row for row in strict_actions if row["time"] > first_interval_end]
    first_interval_actions = [row for row in strict_actions if row["time"] <= first_interval_end]
    first_interval_exclusion = {
        "scope": "Diagnostic excluding the first observed interval; no model retraining or claim that all remaining intervals are verified normal competitive rounds",
        "excluded_time_range": [ordered[0]["time"], first_interval_end],
        "first_interval_next_kill_weapons": sorted({row["next_kill_weapon"] for row in first_interval_actions}),
        "excluded_strict_opportunities": len(first_interval_actions),
        "remaining_strict_opportunities": len(after_first_interval),
        "held_next_killer_hits": sum(row["held_target_is_next_killer"] for row in after_first_interval),
        "held_next_killer_coverage": (sum(row["held_target_is_next_killer"] for row in after_first_interval) / len(after_first_interval)
                                      if after_first_interval else None),
        "held_next_kill_duel_hits": sum(row["held_target_in_next_kill_duel"] for row in after_first_interval),
        "held_next_kill_duel_coverage": (sum(row["held_target_in_next_kill_duel"] for row in after_first_interval) / len(after_first_interval)
                                         if after_first_interval else None),
        "raw_model_next_killer_hits": sum(row["raw_model_top1_is_next_killer"] for row in after_first_interval),
    }
    patch_matches = bool(mesh_game_patch and
                         str(parsed.get("patch_version")) == mesh_game_patch.replace(".", ""))
    result = {
        "demo_file": Path(parsed_path).name, "demo_id": parsed["demo_id"],
        "map_name": parsed["map_name"], "model_path": str(Path(model_path).resolve()),
        "model_sha256": model_sha256,
        "held_out": True, "mesh_enabled": mesh_enabled,
        "runtime_model_enabled": model_enabled,
        "camera_policy": camera_policy, "event_reconstruction_enabled": reconstruct_events,
        "geometry_loaded": bool(geometry and geometry.use_real_geometry),
        "geometry_sources": geometry.sources if geometry else {},
        "geometry_errors": geometry.errors if geometry else {},
        "snapshots": len(trace), "action_snapshots": len(actions),
        "held_target_positive_action_snapshots": covered,
        "held_target_future_kill_coverage": covered / len(actions) if actions else None,
        "strict_next_killer_action_snapshots": len(strict_actions),
        "held_target_next_killer_hits": strict_covered,
        "held_target_next_killer_coverage": strict_covered / len(strict_actions) if strict_actions else None,
        "held_target_next_victim_hits": sum(row["held_target_is_next_victim"] for row in strict_actions),
        "held_target_next_kill_duel_hits": sum(row["held_target_in_next_kill_duel"] for row in strict_actions),
        "held_target_next_kill_duel_coverage": sum(row["held_target_in_next_kill_duel"] for row in strict_actions) / len(strict_actions) if strict_actions else None,
        "first_observed_interval_exclusion_diagnostic": first_interval_exclusion,
        "next_killer_in_duel_pool": sum(row["next_killer_in_duel_pool"] for row in strict_actions),
        "next_killer_duel_pool_coverage": sum(row["next_killer_in_duel_pool"] for row in strict_actions) / len(strict_actions) if strict_actions else None,
        "raw_model_top1_next_killer_hits": sum(row["raw_model_top1_is_next_killer"] for row in strict_actions),
        "raw_model_top1_next_killer_coverage": sum(row["raw_model_top1_is_next_killer"] for row in strict_actions) / len(strict_actions) if strict_actions else None,
        "raw_model_scored_snapshots": sum(row["raw_model_scored_candidate_count"] > 0 for row in trace),
        "duel_pool_model_top1_next_killer_hits": sum(row["duel_pool_model_top1_is_next_killer"] for row in strict_actions),
        "duel_pool_model_top1_next_killer_coverage": sum(row["duel_pool_model_top1_is_next_killer"] for row in strict_actions) / len(strict_actions) if strict_actions else None,
        "direct_duel_pool_metric_scope": "find_potential_duels legacy direct-duel diagnostic, including when replaying broadcast policy",
        "actual_policy_pool_observed_snapshots": sum(row["actual_policy_candidate_ids"] is not None for row in trace),
        "actual_policy_model_scored_snapshots": sum((row["actual_policy_model_scored_candidate_count"] or 0) > 0 for row in trace),
        "next_killer_in_actual_policy_pool": sum(row["next_killer_in_actual_policy_pool"] is True for row in strict_actions),
        "actual_policy_pool_next_killer_coverage": (sum(row["next_killer_in_actual_policy_pool"] is True for row in strict_actions) / len(strict_actions)
                                                    if camera_policy == "broadcast" and strict_actions else None),
        "switch_requests": len(switches),
        "camera_target_changes": sum(item["changed"] for item in switches),
        "camera_target_changes_per_demo_minute": sum(item["changed"] for item in switches) / (duration / 60) if duration else None,
        "sampled_timeline_seconds": duration,
        "snapshots_with_target": sum(row["held_target"] is not None for row in trace),
        "snapshots_target_absent": sum(bool(row["held_target"]) and not row["target_present"] for row in trace),
        "snapshots_target_confirmed_dead": sum(bool(row["held_target"]) and row["held_target_health"] == 0 for row in trace),
        "snapshots_target_health_unknown": sum(bool(row["held_target"]) and row["held_target_health"] is None for row in trace),
        "policy_modes": dict(Counter((row["policy_decision"] or {}).get("mode", camera_policy) for row in trace)),
        "policy_reasons": dict(Counter((row["policy_decision"] or {}).get("reason", "legacy path") for row in trace)),
        "reconstruction_unknown_snapshots": sum(bool((row["reconstruction"] or {}).get("unknown_missing")) for row in trace),
        "reconstruction_excluded_pre_interval_events": len(reconstructor.excluded_before_interval) if reconstructor else None,
        "reconstruction_intervals": reconstructor.interval_resets if reconstructor else None,
        "internal_target_camera_mismatches": sum(row["held_target"] != row["internal_target"] for row in trace),
        "processing_errors": len(errors.messages), "error_messages": errors.messages[:30],
        "real_input_events": 0, "native_input_blocked": True,
        "blocked_native_input_attempts": len(blocked_native_attempts),
        "source_sha256": observer_source_sha256,
        "source_fingerprints": source_fingerprints,
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
            ("Confirmed same-round deaths and enemy round-kill counters were causally reconstructed; ammo is the actual sampled value."
             if reconstruct_events else "Only alive-player snapshots are exported; round_kills and freeze/death transition states are absent."),
            "Freeze states, bomb objectives, world/suicide deaths omitted by the exporter, and events after the final sampled horizon are not reconstructed.",
            "Events before the first sampled state in each live round are excluded to avoid warmup/knife-round deaths with reused round numbers.",
            "Causal reconstruction resets after observed telemetry gaps over two seconds; events in the gaps are not assigned to the next observed interval.",
            "The last incomplete two-second horizon of each round is excluded.",
            "Repeated overlapping action windows are correlated and are not independent kills.",
            "This is a system smoke check; verify mesh patch metadata against the demo before interpreting geometry-dependent accuracy.",
            "Spectator slot mappings are simulated; real key delivery is blocked.",
            "The replay clock follows demo timestamps and excludes processing and command delivery delay.",
            "Doors, smoke and bullet penetration are not simulated.",
        ],
    }
    name = parsed["map_name"] + ("_mesh" if mesh_enabled else "_unknown")
    name += "_" + camera_policy
    if reconstruct_events:
        name += "_events"
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
    parser.add_argument("--camera-policy", choices=("broadcast", "legacy"), default="broadcast")
    parser.add_argument("--reconstruct-events", action="store_true", help="Restore only already-observed same-round deaths and enemy kill counters")
    args = parser.parse_args()
    replay(args.parsed_demo, args.model, args.meshes, args.output, args.limit, not args.without_mesh,
           args.mesh_game_patch, not args.without_model, args.camera_policy, args.reconstruct_events)


if __name__ == "__main__":
    main()
