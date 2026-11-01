"""Causal editorial rules and real-view acknowledgment, without model/OS imports."""

from dataclasses import replace
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from observer_policy import Candidate, Context, ObserverPolicy, PolicyConfig


class ObserverPolicyTests(unittest.TestCase):
    def setUp(self):
        self.policy = ObserverPolicy()

    def decide(self, now, held, candidates, acknowledge=True, **kwargs):
        context = Context(now, held, candidates, **kwargs)
        decision = self.policy.decide(context)
        if acknowledge:
            self.assertTrue(self.policy.ack(decision, True, now, decision.target_id))
        return decision

    def test_blocked_current_sightline_does_not_exclude_an_interesting_setup(self):
        candidate = Candidate('a', 'T', direct_visibility='blocked', setup_interest=.8)
        decision = self.decide(0, None, [candidate])
        self.assertEqual((decision.action, decision.target_id, decision.mode), ('switch', 'a', 'setup'))

    def test_kill_probability_alone_cannot_create_interest_or_empty_cycling(self):
        candidates = [Candidate('a', 'T', kill_probability=.99), Candidate('b', 'CT', kill_probability=.98)]
        for now in (0, 10, 40):
            decision = self.decide(now, 'a', candidates)
            self.assertEqual(decision.action, 'hold')
            self.assertEqual(decision.target_id, 'a')
        self.assertEqual(self.policy.state.successful_cuts, 0)

    def test_low_probability_near_ties_do_not_force_top_two_rotation(self):
        candidates = [Candidate('a', 'T', setup_interest=.5, kill_probability=.01),
                      Candidate('b', 'CT', setup_interest=.5, kill_probability=.02)]
        for now in (0, 5, 10, 19):
            decision = self.decide(now, 'a', candidates)
            self.assertEqual((decision.action, decision.target_id), ('hold', 'a'))
        self.assertEqual(self.policy.state.successful_cuts, 0)

    def test_one_v_many_prefers_lone_player_over_opposing_max_kill_probability(self):
        lone = Candidate('clutcher', 'T', direct_visibility='blocked', kill_probability=.01)
        defenders = [Candidate(f'ct{i}', 'CT', setup_interest=.8, kill_probability=.9) for i in range(4)]
        decision = self.decide(0, None, [lone, *defenders], alive_team_counts={'T': 1, 'CT': 4})
        self.assertEqual((decision.target_id, decision.reason), ('clutcher', 'follow_clutch'))
        # A quiet clutch can remain on its protagonist through a long setup.
        decision = self.decide(19, 'clutcher', [lone, *defenders], alive_team_counts={'T': 1, 'CT': 4})
        self.assertEqual((decision.action, decision.reason), ('hold', 'preserve_clutch'))

    def test_partial_roster_does_not_invent_a_one_v_many_situation(self):
        lone_looking = Candidate('a', 'T')
        others = [Candidate(f'b{i}', 'CT', setup_interest=.7) for i in range(4)]
        decision = self.decide(0, None, [lone_looking, *others])
        self.assertNotEqual(decision.target_id, 'a')
        self.assertNotEqual(decision.reason, 'follow_clutch')

    def test_five_attackers_approaching_one_anchor_prefers_anchor(self):
        anchor = Candidate('anchor', 'CT', is_anchor=True, approaching_enemy_count=5,
                           supporting_teammate_count=0, direct_visibility='blocked', kill_probability=.05)
        attackers = [Candidate(f't{i}', 'T', setup_interest=.9, kill_probability=.8) for i in range(5)]
        decision = self.decide(0, None, [anchor, *attackers])
        self.assertEqual((decision.target_id, decision.reason), ('anchor', 'anchor_under_pressure'))

    def test_active_engagement_continuity_survives_other_model_score_changes(self):
        held = Candidate('a', 'T', engagement_active=True, engagement_id='fight', trade_group_id='chain', kill_probability=.1)
        other = Candidate('b', 'CT', setup_interest=1, kill_probability=.99)
        for now in (0, 1, 10):
            decision = self.decide(now, 'a', [held, other])
            self.assertEqual((decision.action, decision.mode, decision.reason), ('hold', 'engagement', 'preserve_engagement'))

    def test_confirmed_death_follows_trade_chain_instead_of_unrelated_top_probability(self):
        self.decide(0, 'a', [Candidate('a', 'T', engagement_active=True, trade_group_id='fight')])
        trader = Candidate('b', 'CT', engagement_active=True, trade_group_id='fight', kill_probability=.1)
        unrelated = Candidate('c', 'T', engagement_active=True, trade_group_id='elsewhere', kill_probability=.99)
        decision = self.decide(.2, 'a', [trader, unrelated], dead_player_ids=frozenset({'a'}))
        self.assertEqual((decision.target_id, decision.mode, decision.reason), ('b', 'trade', 'confirmed_death'))
        self.assertTrue(decision.urgent)
        self.assertEqual(self.policy.state.successful_cuts, 1)

    def test_confirmed_trade_action_can_cut_quickly_without_penalizing_valid_return(self):
        self.policy = ObserverPolicy(PolicyConfig(minimum_shot_seconds=10, setup_hold_seconds=10))
        a = Candidate('a', 'T', engagement_active=True, trade_group_id='fight')
        self.decide(0, 'a', [a])
        b = Candidate('b', 'CT', engagement_active=True, trade_group_id='fight')
        self.decide(.2, 'a', [b], dead_player_ids=frozenset({'a'}))
        # A -> B -> another actor in the same trade is legitimate event coverage,
        # even though a generic shot-length rule would have blocked the cut.
        c = Candidate('c', 'T', engagement_active=True, trade_group_id='fight', last_damage_at=.4)
        decision = self.decide(.4, 'b', [replace(b, engagement_active=False), c])
        self.assertEqual((decision.action, decision.target_id, decision.reason), ('switch', 'c', 'follow_trade'))
        self.assertEqual(self.policy.state.successful_cuts, 2)

    def test_short_aftermath_then_new_story_without_constant_postkill_rotation(self):
        held = Candidate('a', 'T', last_kill_at=1)
        other = Candidate('b', 'CT', setup_interest=.9)
        self.decide(0, 'a', [replace(held, last_kill_at=None), other])
        decision = self.decide(1.1, 'a', [held, other])
        self.assertEqual((decision.action, decision.mode), ('hold', 'aftermath'))
        decision = self.decide(3, 'a', [held, other])
        self.assertEqual((decision.action, decision.target_id), ('switch', 'b'))

    def test_event_supported_a_b_a_trade_return_is_allowed(self):
        self.policy = ObserverPolicy(PolicyConfig(minimum_shot_seconds=10, setup_hold_seconds=10))
        a = Candidate('a', 'T', engagement_active=True, trade_group_id='fight')
        b = Candidate('b', 'CT', trade_group_id='fight')
        self.decide(0, 'a', [a, b])
        decision = self.decide(.2, 'a', [replace(a, engagement_active=False),
                                        replace(b, engagement_active=True, last_damage_at=.2)])
        self.assertEqual((decision.target_id, decision.reason), ('b', 'follow_trade'))
        decision = self.decide(.4, 'b', [replace(a, last_damage_at=.4), b])
        self.assertEqual((decision.target_id, decision.reason), ('a', 'follow_trade'))
        self.assertEqual(self.policy.state.successful_cuts, 2)

    def test_lone_survivor_of_a_trade_group_gets_aftermath_not_an_invented_trade(self):
        held = Candidate('a', 'T', setup_interest=.5, trade_group_id='fight', last_kill_at=1)
        decision = self.decide(1.1, 'a', [held])
        self.assertEqual((decision.action, decision.mode), ('hold', 'aftermath'))

    def test_confirmed_objective_breaks_setup_hold(self):
        self.policy = ObserverPolicy(PolicyConfig(minimum_shot_seconds=20, setup_hold_seconds=20))
        a, b = Candidate('a', 'T', setup_interest=.8), Candidate('b', 'CT')
        self.decide(0, 'a', [a, b])
        decision = self.decide(.2, 'a', [a, b], objective_event='defuse', objective_target_id='b')
        self.assertEqual((decision.action, decision.target_id, decision.reason), ('switch', 'b', 'confirmed_objective'))
        self.assertTrue(decision.urgent)

    def test_objective_requires_a_confirmed_actor_and_alive_target(self):
        a, b = Candidate('a', 'T', setup_interest=.8), Candidate('b', 'CT')
        self.decide(0, 'a', [a, b])
        decision = self.decide(2, 'a', [a, b], objective_event='plant', objective_target_id='missing')
        self.assertEqual((decision.action, decision.target_id), ('hold', 'a'))

    def test_missing_held_player_is_a_telemetry_gap_not_a_confirmed_death(self):
        other = Candidate('b', 'CT', setup_interest=1)
        decision = self.decide(0, 'missing', [other])
        self.assertEqual((decision.action, decision.target_id, decision.reason), ('hold', 'missing', 'telemetry_gap'))
        decision = self.decide(1, 'missing', [other], dead_player_ids=frozenset({'missing'}))
        self.assertEqual((decision.action, decision.target_id, decision.reason), ('switch', 'b', 'confirmed_death'))

    def test_failed_or_mismatched_acknowledgment_changes_neither_state_nor_budget(self):
        decision = self.policy.decide(Context(0, None, [Candidate('a', 'T', setup_interest=.9)]))
        before = self.policy.state
        self.assertFalse(self.policy.ack(decision, False, .1, None))
        self.assertEqual(self.policy.state, before)
        self.assertFalse(self.policy.ack(decision, True, .2, 'wrong'))
        self.assertEqual(self.policy.state, before)
        self.assertTrue(self.policy.ack(decision, True, .3, 'a'))
        self.assertEqual(self.policy.state.successful_cuts, 1)
        self.assertEqual(self.policy.state.held_since, .3)

    def test_stale_or_duplicate_ack_cannot_charge_a_cut(self):
        candidate = Candidate('a', 'T', setup_interest=.9)
        stale = self.policy.decide(Context(0, None, [candidate]))
        current = self.policy.decide(Context(.1, None, [candidate]))
        self.assertFalse(self.policy.ack(stale, True, .1, 'a'))
        self.assertTrue(self.policy.ack(current, True, .2, 'a'))
        self.assertFalse(self.policy.ack(current, True, .3, 'a'))
        self.assertEqual(self.policy.state.successful_cuts, 1)

    def test_global_budget_counts_all_successful_cuts_and_expires(self):
        self.policy = ObserverPolicy(PolicyConfig(minimum_shot_seconds=0, setup_hold_seconds=0,
                                                  max_cuts_per_window=1, cut_window_seconds=10))
        a, b = Candidate('a', 'T', setup_interest=.9), Candidate('b', 'CT', setup_interest=.9)
        self.decide(0, None, [a])
        decision = self.decide(1, 'a', [replace(a, setup_interest=0), b])
        self.assertEqual((decision.action, decision.reason), ('hold', 'cut_budget'))
        self.assertEqual(self.policy.state.successful_cuts, 1)
        decision = self.decide(11, 'a', [replace(a, setup_interest=0), b])
        self.assertEqual((decision.action, decision.target_id), ('switch', 'b'))
        self.assertEqual(self.policy.state.successful_cuts, 2)
        self.assertEqual(self.policy.state.cut_times, (11,))

    def test_urgent_budget_exception_is_configurable_and_still_counted(self):
        self.policy = ObserverPolicy(PolicyConfig(max_cuts_per_window=0, urgent_cuts_bypass_budget=True))
        a, b = Candidate('a', 'T', setup_interest=.9), Candidate('b', 'CT', setup_interest=.9)
        self.decide(0, 'a', [a, b])
        decision = self.decide(.1, 'a', [b], dead_player_ids=frozenset({'a'}))
        self.assertEqual(decision.action, 'switch')
        self.assertEqual(self.policy.state.successful_cuts, 1)
        self.policy = ObserverPolicy(PolicyConfig(max_cuts_per_window=0, urgent_cuts_bypass_budget=False))
        decision = self.decide(0, 'a', [b], dead_player_ids=frozenset({'a'}))
        self.assertEqual((decision.action, decision.reason), ('hold', 'cut_budget'))

    def test_overview_requires_capability_and_no_repeated_cuts_while_already_there(self):
        decision = self.decide(0, 'a', [], round_phase='over')
        self.assertEqual((decision.action, decision.target_id, decision.mode), ('hold', 'a', 'overview'))
        decision = self.decide(1, 'a', [], round_phase='over', overview_available=True)
        self.assertEqual((decision.action, decision.target_id), ('overview', None))
        decision = self.decide(2, None, [], round_phase='over', overview_available=True)
        self.assertEqual((decision.action, decision.reason), ('hold', 'preserve_overview'))
        self.assertEqual(self.policy.state.successful_cuts, 1)

    def test_future_event_timestamps_cannot_create_aftermath_or_trade_evidence(self):
        held = Candidate('a', 'T', last_kill_at=99, last_damage_at=99, trade_group_id='fight')
        decision = self.decide(1, 'a', [held])
        self.assertEqual(decision.mode, 'overview')
        self.assertIsNone(self.policy.state.last_conflict_at)
        self.assertIsNone(self.policy.state.last_kill_at)

    def test_time_and_contract_validation(self):
        with self.assertRaises(ValueError):
            Candidate('a', 'T', setup_interest=float('nan'))
        with self.assertRaises(ValueError):
            Candidate('a', 'T', direct_visibility='probably')
        with self.assertRaises(ValueError):
            Context(0, None, [Candidate('a', 'T'), Candidate('a', 'CT')])
        with self.assertRaises(ValueError):
            PolicyConfig(minimum_shot_seconds=-1)
        self.decide(10, 'a', [Candidate('a', 'T')])
        with self.assertRaises(ValueError):
            self.policy.decide(Context(9, 'a', [Candidate('a', 'T')]))

    def test_zero_setup_threshold_still_cannot_make_model_only_interest(self):
        self.policy = ObserverPolicy(PolicyConfig(minimum_setup_interest=0))
        decision = self.decide(0, None, [Candidate('a', 'T', kill_probability=.99)])
        self.assertEqual((decision.action, decision.target_id), ('hold', None))

    def test_dead_view_can_recover_to_confirmed_overview_under_urgent_budget_exception(self):
        self.policy = ObserverPolicy(PolicyConfig(max_cuts_per_window=0))
        decision = self.decide(0, 'a', [], dead_player_ids=frozenset({'a'}), overview_available=True)
        self.assertEqual((decision.action, decision.target_id), ('overview', None))
        self.assertTrue(decision.urgent)
        self.assertEqual(self.policy.state.successful_cuts, 1)

    def test_confirmed_round_boundary_clears_prior_trade_and_aftermath_memory(self):
        held = Candidate('a', 'T', engagement_active=True, trade_group_id='previous-round', last_kill_at=0)
        self.decide(0, 'a', [held])
        self.decide(.1, 'a', [held], round_phase='over')
        self.assertIsNone(self.policy.state.trade_group_id)
        self.assertIsNone(self.policy.state.last_conflict_at)
        self.assertIsNone(self.policy.state.last_kill_at)

    def test_explicit_story_reset_clears_encounter_but_preserves_view_and_cut_accounting(self):
        a = Candidate('a', 'T', engagement_active=True, trade_group_id='old', last_kill_at=0)
        b = Candidate('b', 'CT', engagement_active=True, trade_group_id='old', last_kill_at=.1)
        self.decide(0, 'a', [a, b])
        self.decide(.1, 'a', [a, b], objective_event='defuse', objective_target_id='b')
        pending = self.policy.decide(Context(.2, 'b', [b]))
        before = self.policy.state
        self.assertEqual(before.successful_cuts, 1)
        self.policy.reset_story(.3, 'b')
        after = self.policy.state
        self.assertEqual((after.held_target, after.held_since, after.view_kind),
                         (before.held_target, before.held_since, before.view_kind))
        self.assertEqual(after.cut_times, before.cut_times)
        self.assertEqual(after.successful_cuts, before.successful_cuts)
        self.assertEqual((after.mode, after.mode_since), ('overview', None))
        self.assertEqual((after.trade_group_id, after.last_conflict_at, after.last_kill_at), (None, None, None))
        self.assertFalse(self.policy.ack(pending, True, .3, 'b'))
        decision = self.policy.decide(Context(.3, 'b', [Candidate('b', 'CT')]))
        self.assertNotIn(decision.reason, {'engagement_grace', 'preserve_trade_chain', 'aftermath_hold'})

    def test_story_reset_external_view_sync_does_not_invent_a_successful_cut(self):
        self.decide(1, 'a', [Candidate('a', 'T', setup_interest=.5)])
        self.policy.reset_story(2, 'b')
        self.assertEqual((self.policy.state.held_target, self.policy.state.held_since), ('b', 2))
        self.assertEqual(self.policy.state.successful_cuts, 0)
        self.assertEqual(self.policy.state.cut_times, ())
        with self.assertRaises(ValueError):
            self.policy.reset_story(1, 'b')


if __name__ == '__main__':
    unittest.main()
