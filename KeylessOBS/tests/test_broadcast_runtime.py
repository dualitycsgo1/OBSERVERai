"""Full GSI-to-policy-to-actuator tests; Windows input is always mocked."""
import copy
from datetime import datetime, timedelta
from pathlib import Path
import sys
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cs2_duel_detector as runtime


def player(team, x, slot, health=100):
    return {"team": team, "position": f"{x},0,0", "forward": "1,0,0", "observer_slot": slot,
            "state": {"health": health, "round_kills": 0},
            "weapons": {"active": {"name": "weapon_ak47", "state": "active", "ammo_clip": 30}}}


def state():
    return {"map": {"name": "de_test", "round": 1}, "round": {"phase": "live"}, "allplayers": {
        "a": player("T", 0, 0), "b": player("T", 10, 1),
        "c": player("CT", 100, 2), "d": player("CT", 120, 3)}}


def detector(held="a", visibility=False, settings=None):
    instance = runtime.CS2DuelDetector.__new__(runtime.CS2DuelDetector)
    instance.config = {"observer": {"camera_policy": "broadcast"}, "broadcast_policy": settings or {}}
    instance.active, instance.campaths_enabled = True, False
    instance.geometry_manager = None
    instance.current_target_sid = held
    instance.current_target_start = datetime.now() - timedelta(seconds=5) if held else None
    instance.current_target_score_ema = 0.01
    instance.score_ema_alpha = 0.55
    instance.last_switch_time = datetime.now() - timedelta(seconds=5)
    instance.last_any_switch = 0
    instance.freezetime_start = None
    instance._refresh_observer_model = Mock()
    instance._duel_visibility = Mock(return_value=visibility)
    instance._rank_model_candidates = Mock(return_value=[("a", .01), ("b", .02), ("c", .03), ("d", .04)])
    instance._update_duel_lock_tracking = Mock(side_effect=AssertionError("Legacy lock path must not run"))
    instance.update_history_and_triggers = Mock(side_effect=AssertionError("Legacy triggers can send duplicate keys"))
    instance._check_spectated_player_death = Mock(side_effect=AssertionError("Legacy death actuator must not run"))
    return instance


def process(instance, payload, now, success=True):
    with patch.object(runtime.time, "time", return_value=now), patch.object(runtime, "send_key", return_value=success) as send:
        instance.process_gamestate(payload)
    return send


def test_failed_send_does_not_change_held_target_or_spend_cut_budget():
    instance, payload = detector(), state()
    process(instance, payload, 100)
    payload["allplayers"]["a"]["state"]["health"] = 0
    payload["allplayers"]["c"]["state"]["round_kills"] = 1
    send = process(instance, payload, 100.5, success=False)
    send.assert_called_once_with(runtime.VK_CODES[2])
    assert instance.current_target_sid == "a"
    assert instance._broadcast_runtime.policy.state.successful_cuts == 0
    assert instance.last_observer_decision["acknowledged"] is False
    send = process(instance, payload, 101, success=True)
    send.assert_called_once_with(runtime.VK_CODES[2])
    assert instance.current_target_sid == "b"
    assert instance._broadcast_runtime.policy.state.successful_cuts == 1


def test_confirmed_death_makes_one_cut_and_cannot_run_legacy_actuators():
    instance, payload = detector(), state()
    process(instance, payload, 100)
    payload["allplayers"]["a"]["state"]["health"] = 0
    payload["allplayers"]["c"]["state"]["round_kills"] = 1
    send = process(instance, payload, 100.1)
    assert send.call_count == 1
    assert instance.last_observer_decision["reason"] == "confirmed_death"
    assert instance.last_observer_decision["urgent"]
    instance.update_history_and_triggers.assert_not_called()
    instance._check_spectated_player_death.assert_not_called()
    instance._update_duel_lock_tracking.assert_not_called()


def test_lone_low_health_clutch_player_is_held_without_direct_visibility():
    instance, payload = detector(), state()
    del payload["allplayers"]["b"]
    payload["allplayers"]["a"]["state"]["health"] = 5
    for now in (100, 110, 120):
        send = process(instance, payload, now)
        send.assert_not_called()
        assert instance.current_target_sid == "a"
        assert instance.last_observer_decision["reason"] == "preserve_clutch"
    assert instance._broadcast_runtime.last_context.alive_team_counts == {"T": 1, "CT": 2}


def test_missing_health_or_missing_held_player_is_not_a_death_or_clutch():
    instance, payload = detector(), state()
    del payload["allplayers"]["a"]["state"]["health"]
    send = process(instance, payload, 100)
    send.assert_not_called()
    assert instance.last_observer_decision["reason"] == "telemetry_gap"
    context = instance._broadcast_runtime.last_context
    assert "a" not in context.dead_player_ids
    assert context.alive_team_counts is None
    del payload["allplayers"]["a"]
    payload["allplayers"]["total"] = 4
    send = process(instance, payload, 101)
    send.assert_not_called()
    assert instance.last_observer_decision["reason"] == "telemetry_gap"
    assert instance._broadcast_runtime.last_context.alive_team_counts is None


def test_quiet_low_probabilities_do_not_force_top_two_rotation():
    instance, payload = detector(), state()
    payload["allplayers"]["c"]["position"] = "6000,0,0"
    payload["allplayers"]["d"]["position"] = "7000,0,0"
    for now in (100, 102, 110):
        send = process(instance, payload, now)
        send.assert_not_called()
        assert instance.current_target_sid == "a"
    assert instance._broadcast_runtime.policy.state.successful_cuts == 0


def test_blocked_players_remain_setup_candidates_and_all_alive_are_scored_once():
    instance, payload = detector(held=None), state()
    send = process(instance, payload, 100)
    assert send.call_count == 1
    assert instance.current_target_sid in {"a", "b"}
    context = instance._broadcast_runtime.last_context
    assert {candidate.player_id for candidate in context.candidates} == {"a", "b", "c", "d"}
    assert all(candidate.direct_visibility == "blocked" for candidate in context.candidates)
    instance._rank_model_candidates.assert_called_once()
    assert set(instance._rank_model_candidates.call_args.args[1]) == {"a", "b", "c", "d"}


def test_event_supported_local_trade_can_override_minimum_shot():
    instance, payload = detector(visibility=True, settings={"minimum_shot_seconds": 5}), state()
    payload["allplayers"]["e"] = player("CT", 150, 4)
    process(instance, payload, 100)
    payload["allplayers"]["a"]["state"]["round_kills"] = 1
    payload["allplayers"]["c"]["state"]["health"] = 0
    process(instance, payload, 100.1)
    payload["allplayers"]["b"]["weapons"]["active"]["ammo_clip"] = 29
    send = process(instance, payload, 101)
    send.assert_called_once_with(runtime.VK_CODES[2])
    assert instance.last_observer_decision["reason"] == "follow_trade"
    assert instance.last_observer_decision["urgent"]
    assert instance.current_target_sid == "b"


def test_overview_respects_campath_toggle_and_counts_only_successful_input():
    instance, payload = detector(), state()
    payload["round"]["phase"] = "freezetime"
    process(instance, payload, 100).assert_not_called()
    instance.campaths_enabled = True
    process(instance, payload, 101, success=False).assert_called_once()
    assert instance.current_target_sid == "a"
    assert instance._broadcast_runtime.policy.state.successful_cuts == 0
    process(instance, payload, 102).assert_called_once()
    assert instance.current_target_sid is None
    assert instance._broadcast_runtime.policy.state.successful_cuts == 1
    process(instance, payload, 103).assert_not_called()


def test_live_phase_can_leave_overview_without_legacy_nineteen_second_lock():
    instance, payload = detector(), state()
    instance.freezetime_start = datetime.now()
    payload["bomb"] = {"state": "defusing", "player": "c"}
    send = process(instance, payload, 100)
    send.assert_called_once_with(runtime.VK_CODES[3])
    assert instance.last_observer_decision["reason"] == "confirmed_objective"
    assert instance.current_target_sid == "c"


def test_disabled_observer_cannot_actuate_or_update_policy_history():
    instance, payload = detector(), state()
    instance.active = False
    send = process(instance, payload, 100)
    send.assert_not_called()
    assert not hasattr(instance, "_broadcast_runtime")


def test_cold_start_counts_and_long_gap_deltas_are_not_fresh_kills():
    instance, payload = detector(), state()
    payload["allplayers"]["a"]["state"]["round_kills"] = 2
    process(instance, payload, 100)
    assert instance._broadcast_runtime.kill_times == {}
    payload["allplayers"]["a"]["state"]["round_kills"] = 3
    process(instance, payload, 110)
    assert instance._broadcast_runtime.kill_times == {}


def test_unknown_round_phase_does_not_send_an_overview_or_speculative_player_key():
    instance, payload = detector(held=None), state()
    instance.campaths_enabled = True
    del payload["round"]
    send = process(instance, payload, 100)
    send.assert_not_called()
    assert instance._broadcast_runtime.policy.state.successful_cuts == 0


def test_anchor_pressure_requires_opponents_own_incoming_movement():
    instance, payload = detector(), state()
    payload["allplayers"]["c"]["position"] = "1000,0,0"
    payload["allplayers"]["d"]["position"] = "3000,0,0"
    process(instance, payload, 100)
    payload["allplayers"]["c"]["position"] = "900,0,0"
    process(instance, payload, 100.5)
    defender = next(candidate for candidate in instance._broadcast_runtime.last_context.candidates if candidate.player_id == "c")
    assert not defender.is_anchor and defender.approaching_enemy_count == 0
    payload["allplayers"]["a"]["position"] = "100,0,0"
    payload["allplayers"]["b"]["position"] = "110,0,0"
    process(instance, payload, 101)
    defender = next(candidate for candidate in instance._broadcast_runtime.last_context.candidates if candidate.player_id == "c")
    assert defender.is_anchor and defender.approaching_enemy_count == 2


def test_missing_previously_known_teammate_without_total_is_not_a_clutch():
    instance, payload = detector(), state()
    process(instance, payload, 100)
    del payload["allplayers"]["b"]
    process(instance, payload, 100.1).assert_not_called()
    context = instance._broadcast_runtime.last_context
    assert context.alive_team_counts is None
    assert "b" not in context.dead_player_ids
    assert instance.last_observer_decision["reason"] != "preserve_clutch"
    assert instance._rank_model_candidates.call_count == 1
    assert all(candidate.kill_probability is None for candidate in context.candidates)


def test_unknown_health_of_a_nearby_teammate_does_not_prove_unsupported_anchor():
    instance, payload = detector(), state()
    payload["allplayers"]["c"]["position"] = "1000,0,0"
    payload["allplayers"]["d"]["position"] = "1100,0,0"
    process(instance, payload, 100)
    del payload["allplayers"]["b"]["state"]["health"]
    payload["allplayers"]["c"]["position"] = "900,0,0"
    payload["allplayers"]["d"]["position"] = "1000,0,0"
    process(instance, payload, 100.5)
    context = instance._broadcast_runtime.last_context
    held = next(candidate for candidate in context.candidates if candidate.player_id == "a")
    assert held.approaching_enemy_count == 2
    assert not held.is_anchor
    assert context.alive_team_counts is None
    assert instance._rank_model_candidates.call_count == 1
    assert all(candidate.kill_probability is None for candidate in context.candidates)


def test_confirmed_round_boundary_clears_story_but_preserves_view_and_cut_budget():
    instance, payload = detector(held=None), state()
    process(instance, payload, 200)
    held_id = instance.current_target_sid
    payload["allplayers"][held_id]["state"]["health"] = 50
    process(instance, payload, 200.1)
    adapter = instance._broadcast_runtime
    assert adapter.policy.state.last_conflict_at is not None
    assert adapter.policy.state.mode == "engagement"
    previous_cuts = adapter.policy.state.cut_times
    previous_count = adapter.policy.state.successful_cuts
    payload["map"]["round"] = 2
    payload["allplayers"][held_id]["state"]["health"] = 100
    process(instance, payload, 200.2).assert_not_called()
    assert instance.current_target_sid == held_id
    assert adapter.damage_times == {}
    assert adapter.policy.state.last_conflict_at is None
    assert adapter.policy.state.last_kill_at is None
    assert adapter.policy.state.trade_group_id is None
    assert adapter.policy.state.cut_times == previous_cuts
    assert adapter.policy.state.successful_cuts == previous_count == 1
    assert instance.last_observer_decision["reason"] != "engagement_grace"


def test_partial_map_identity_does_not_clear_encounter_or_known_roster():
    instance, payload = detector(), state()
    process(instance, payload, 200)
    payload["allplayers"]["a"]["state"]["health"] = 50
    process(instance, payload, 200.1)
    adapter = instance._broadcast_runtime
    identity, roster = adapter.round_key, dict(adapter.known_roster)
    for offset, map_payload in enumerate(({}, {"name": "de_test"}, {"round": 1})):
        partial = copy.deepcopy(payload)
        partial["map"] = map_payload
        process(instance, partial, 200.2 + offset * .1)
        assert adapter.round_key == identity
        assert adapter.known_roster == roster
        assert adapter.damage_times["a"] == 200.1
        assert adapter.policy.state.last_conflict_at is not None


def test_round_boundary_clears_old_death_confirmation_and_retains_known_roster():
    instance, payload = detector(), state()
    process(instance, payload, 100)
    payload["allplayers"]["b"]["state"]["health"] = 0
    process(instance, payload, 100.1)
    assert instance._broadcast_runtime.last_context.alive_team_counts == {"T": 1, "CT": 2}
    # Missing the previously dead player is a new-round telemetry gap, not
    # evidence that the next round starts with that teammate still dead.
    payload["map"]["round"] = 2
    del payload["allplayers"]["b"]
    process(instance, payload, 100.2)
    context = instance._broadcast_runtime.last_context
    assert context.alive_team_counts is None
    assert "b" not in context.dead_player_ids
    assert instance.last_observer_decision["reason"] != "preserve_clutch"
