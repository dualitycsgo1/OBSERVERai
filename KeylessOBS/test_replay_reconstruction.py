"""Causal boundaries for offline GSI reconstruction; no observer/native input."""
from copy import deepcopy
import unittest

from replay_observer import CausalReplayReconstructor


def player(team, health=100, x=0, ammo=20):
    return {"team": team, "position": f"{x},0,0", "forward": "1,0,0",
            "state": {"health": health, "armor": 0},
            "weapons": {"0": {"name": "weapon_glock", "state": "active", "ammo_clip": ammo}}}


def snapshot(players, round_number=0):
    return {"map": {"name": "de_dust2", "round": round_number},
            "round": {"phase": "live"}, "allplayers": players}


def kill(time=2, tick=20, round_number=1, attacker="a", victim="b", victim_team="CT"):
    return {"time": time, "tick": tick, "round_number": round_number,
            "attacker_steam_id": attacker, "victim_steam_id": victim,
            "attacker_team": "T", "victim_team": victim_team,
            "attacker_name": "attacker", "victim_name": "victim",
            "victim_pos": {"x": 999, "y": 999, "z": 999}}


class ReconstructionTests(unittest.TestCase):
    def test_future_kill_never_changes_current_health_or_counter(self):
        reconstructor = CausalReplayReconstructor([kill()])
        result = reconstructor.apply(snapshot({"a": player("T"), "b": player("CT")}), 1, 10)
        self.assertEqual(result["allplayers"]["b"]["state"]["health"], 100)
        self.assertEqual(result["allplayers"]["a"]["state"]["round_kills"], 0)
        self.assertEqual(result["_replay_reconstruction"]["past_kill_events"], 0)

    def test_confirmed_death_uses_only_previous_position_and_ammo(self):
        reconstructor = CausalReplayReconstructor([kill()])
        reconstructor.apply(snapshot({"a": player("T"), "b": player("CT", x=10, ammo=7)}), 1, 10)
        result = reconstructor.apply(snapshot({"a": player("T", ammo=5)}), 2, 20)
        dead = result["allplayers"]["b"]
        self.assertEqual(dead["state"]["health"], 0)
        self.assertEqual(dead["position"], "10,0,0")
        self.assertEqual(dead["weapons"]["0"]["ammo_clip"], 7)
        self.assertEqual(result["allplayers"]["a"]["weapons"]["0"]["ammo_clip"], 5)
        self.assertEqual(result["allplayers"]["a"]["state"]["round_kills"], 1)

    def test_missing_player_without_kill_is_unknown_not_dead(self):
        reconstructor = CausalReplayReconstructor([])
        reconstructor.apply(snapshot({"a": player("T"), "b": player("CT")}), 1, 10)
        result = reconstructor.apply(snapshot({"a": player("T")}), 2, 20)
        self.assertNotIn("health", result["allplayers"]["b"]["state"])
        self.assertEqual(result["_replay_reconstruction"]["confirmed_dead"], [])
        self.assertEqual(result["_replay_reconstruction"]["unknown_missing"], ["b"])

    def test_both_time_and_tick_must_have_been_observed(self):
        reconstructor = CausalReplayReconstructor([kill(time=1, tick=20)])
        result = reconstructor.apply(snapshot({"a": player("T"), "b": player("CT")}), 1, 10)
        self.assertEqual(result["allplayers"]["b"]["state"]["health"], 100)
        self.assertEqual(result["_replay_reconstruction"]["past_kill_events"], 0)

    def test_equal_snapshot_tick_and_time_are_causal_known_events(self):
        result = CausalReplayReconstructor([kill()]).apply(snapshot({"a": player("T")}), 2, 20)
        self.assertEqual(result["allplayers"]["b"]["state"]["health"], 0)

    def test_unseen_victim_has_no_invented_position_or_weapon(self):
        result = CausalReplayReconstructor([kill()]).apply(snapshot({"a": player("T")}), 2, 20)
        dead = result["allplayers"]["b"]
        self.assertEqual(dead["state"]["health"], 0)
        self.assertNotIn("position", dead)
        self.assertNotIn("weapons", dead)

    def test_future_identity_is_not_injected_into_roster(self):
        result = CausalReplayReconstructor([kill(victim="future-player")]).apply(snapshot({"a": player("T")}), 1, 10)
        self.assertEqual(set(result["allplayers"]), {"a"})

    def test_warmup_death_with_reused_round_number_is_not_live_death(self):
        reconstructor = CausalReplayReconstructor([kill(time=0.5, tick=5)])
        result = reconstructor.apply(snapshot({"a": player("T"), "b": player("CT")}), 1, 10)
        self.assertEqual(result["allplayers"]["b"]["state"]["health"], 100)
        self.assertEqual(result["allplayers"]["a"]["state"]["round_kills"], 0)
        self.assertEqual(result["_replay_reconstruction"]["excluded_pre_interval_events_this_round"], 1)

    def test_round_boundary_resets_deaths_and_kill_counts(self):
        reconstructor = CausalReplayReconstructor([kill()])
        reconstructor.apply(snapshot({"a": player("T")}), 2, 20)
        result = reconstructor.apply(snapshot({"a": player("T"), "b": player("CT")}, 1), 5, 50)
        self.assertEqual(result["allplayers"]["b"]["state"]["health"], 100)
        self.assertEqual(result["allplayers"]["a"]["state"]["round_kills"], 0)
        self.assertEqual(result["_replay_reconstruction"]["confirmed_dead"], [])

    def test_same_round_number_after_knife_round_gap_resets_causal_interval(self):
        reconstructor = CausalReplayReconstructor([kill()])
        reconstructor.apply(snapshot({"a": player("T")}), 2, 20)
        result = reconstructor.apply(snapshot({"a": player("T"), "b": player("CT")}), 6, 60)
        self.assertEqual(result["allplayers"]["b"]["state"]["health"], 100)
        self.assertEqual(result["allplayers"]["a"]["state"]["round_kills"], 0)
        self.assertEqual(reconstructor.interval_resets[-1]["reason"], "observed telemetry gap")

    def test_teamkill_death_confirmed_without_enemy_kill_credit(self):
        result = CausalReplayReconstructor([kill(victim_team="T")]).apply(snapshot({"a": player("T")}), 2, 20)
        self.assertEqual(result["allplayers"]["b"]["state"]["health"], 0)
        self.assertEqual(result["allplayers"]["a"]["state"]["round_kills"], 0)

    def test_repeated_snapshot_does_not_double_count_kill(self):
        reconstructor = CausalReplayReconstructor([kill()])
        first = reconstructor.apply(snapshot({"a": player("T")}), 2, 20)
        second = reconstructor.apply(snapshot({"a": player("T")}), 3, 30)
        self.assertEqual(first["allplayers"]["a"]["state"]["round_kills"], 1)
        self.assertEqual(second["allplayers"]["a"]["state"]["round_kills"], 1)
        self.assertEqual(second["_replay_reconstruction"]["newly_confirmed_dead"], [])

    def test_input_and_cached_snapshots_are_not_mutated(self):
        state = snapshot({"a": player("T"), "b": player("CT", x=10)})
        original = deepcopy(state)
        reconstructor = CausalReplayReconstructor([kill()])
        returned = reconstructor.apply(state, 1, 10)
        returned["allplayers"]["b"]["position"] = "future-mutation"
        self.assertEqual(state, original)
        result = reconstructor.apply(snapshot({"a": player("T")}), 2, 20)
        self.assertEqual(result["allplayers"]["b"]["position"], "10,0,0")

    def test_contradictory_positive_health_after_death_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "alive after"):
            CausalReplayReconstructor([kill()]).apply(snapshot({"a": player("T"), "b": player("CT")}), 2, 20)

    def test_out_of_order_snapshots_cannot_keep_future_state(self):
        reconstructor = CausalReplayReconstructor([])
        reconstructor.apply(snapshot({"a": player("T")}), 2, 20)
        with self.assertRaisesRegex(ValueError, "chronological"):
            reconstructor.apply(snapshot({"a": player("T")}), 1, 10)

    def test_invalid_event_is_ignored_without_fabricating_data(self):
        result = CausalReplayReconstructor([{"round_number": 1}]).apply(snapshot({"a": player("T")}), 1, 10)
        self.assertEqual(result["_replay_reconstruction"]["ignored_events"], 1)
        self.assertEqual(result["_replay_reconstruction"]["confirmed_dead"], [])


if __name__ == "__main__":
    unittest.main()
