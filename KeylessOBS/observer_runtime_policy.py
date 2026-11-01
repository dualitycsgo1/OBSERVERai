"""Translate causal GSI observations into broadcast decisions and one actuator.

Spatial thresholds are configurable engineering defaults, not fitted VOD
parameters. Blocked direct sightlines remain eligible editorial setups. Missing
health is unknown telemetry, never a death or proof of a clutch.
"""
from dataclasses import asdict, fields
import math

from observer_model import vector
from observer_policy import Candidate, Context, ObserverPolicy, PolicyConfig


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _position(player):
    try:
        return vector(player.get("position"))
    except (KeyError, TypeError, ValueError):
        return None


def _distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _facing(player, source, destination):
    try:
        forward = vector(player.get("forward"))
        delta = tuple(b - a for a, b in zip(source, destination))
        denominator = math.sqrt(sum(x * x for x in forward)) * math.sqrt(sum(x * x for x in delta))
        return max(0.0, min(1.0, sum(a * b for a, b in zip(forward, delta)) / denominator)) if denominator else 0.0
    except (KeyError, TypeError, ValueError):
        return 0.0


def _active_weapon(player):
    for weapon in (player.get("weapons") or {}).values():
        if isinstance(weapon, dict) and weapon.get("state") == "active":
            return weapon.get("name"), _number(weapon.get("ammo_clip"))
    return None, None


class BroadcastObserverRuntime:
    def __init__(self, detector):
        self.detector = detector
        settings = detector.config.get("broadcast_policy", {})
        policy_keys = {field.name for field in fields(PolicyConfig)}
        self.policy = ObserverPolicy(PolicyConfig(**{key: value for key, value in settings.items() if key in policy_keys}))
        self.facts_config = settings.get("facts", {})
        self.previous = {}
        self.damage_times, self.kill_times, self.shot_times, self.groups = {}, {}, {}, {}
        self.round_key = None
        self.known_roster = {}
        self.confirmed_dead = set()
        self.previous_phase = None
        self.previous_time = None
        self.last_context = None

    def _recent(self, table, sid, now, window):
        return sid in table and 0 <= now - table[sid] <= window

    def _reset_facts(self):
        self.previous.clear()
        self.damage_times.clear()
        self.kill_times.clear()
        self.shot_times.clear()
        self.groups.clear()

    def build_context(self, gamestate, now):
        raw = gamestate.get("allplayers") or {}
        players = {str(sid): data for sid, data in raw.items()
                   if sid != "total" and isinstance(data, dict) and data.get("team") in ("T", "CT")}
        phase = (gamestate.get("round") or {}).get("phase", "unknown")
        map_info = gamestate.get("map") or {}
        previous_map, previous_round = self.round_key or (None, None)
        reported_map = map_info.get("name")
        reported_map = reported_map if isinstance(reported_map, str) and reported_map else None
        reported_round = _number(map_info.get("round"))
        # Missing identity fields are a telemetry gap, not a new round. Retain
        # individually known fields until a replacement is explicitly observed.
        round_key = (reported_map if reported_map is not None else previous_map,
                     reported_round if reported_round is not None else previous_round)
        map_changed = reported_map is not None and previous_map is not None and reported_map != previous_map
        round_changed = reported_round is not None and previous_round is not None and reported_round != previous_round
        freeze_started = phase == "freezetime" and self.previous_phase not in (None, "freezetime")
        freeze_ended = self.previous_phase == "freezetime" and phase == "live"
        new_segment = map_changed or (round_changed and reported_round < previous_round)
        if self.previous_time is not None and now < self.previous_time:
            self._reset_facts()
            self.policy = ObserverPolicy(self.policy.config)
        if new_segment:
            self.known_roster.clear()
        if map_changed or round_changed or freeze_started or freeze_ended:
            self._reset_facts()
            self.confirmed_dead.clear()
            # Clear the previous round's editorial story, retaining its actual
            # held-view observation and the rolling global dispatch cut budget.
            self.policy.reset_story(now, self.detector.current_target_sid)
        self.round_key = round_key
        dt = now - self.previous_time if self.previous_time is not None else 0.0
        if dt > float(self.facts_config.get("maximum_event_gap_seconds", 2.0)):
            # A kill somewhere in a long telemetry gap is not a just-observed
            # kill; do not manufacture a fresh trade/aftermath from it.
            self._reset_facts()
        health = {sid: _number((data.get("state") or {}).get("health")) for sid, data in players.items()}
        self.known_roster.update({sid: data["team"] for sid, data in players.items()})
        for sid, value in health.items():
            if value == 0:
                self.confirmed_dead.add(sid)
            elif value is not None and value > 0:
                self.confirmed_dead.discard(sid)
        alive = {sid: data for sid, data in players.items() if health[sid] is not None and health[sid] > 0}
        dead = frozenset(self.confirmed_dead)
        positions = {sid: _position(data) for sid, data in players.items()}
        counts = {"T": 0, "CT": 0}
        complete = bool(players) and all(value is not None for value in health.values())
        missing_roster = set(self.known_roster) - set(players)
        if missing_roster - self.confirmed_dead:
            complete = False
        reported_total = _number(raw.get("total"))
        if reported_total is not None and reported_total > len(players) + len(missing_roster & self.confirmed_dead):
            complete = False
        for sid in alive:
            counts[players[sid]["team"]] += 1

        new_events = set()
        for sid, data in players.items():
            previous = self.previous.get(sid)
            if not previous or phase != "live":
                continue
            hp = health[sid]
            if hp is not None and previous["health"] is not None and hp < previous["health"]:
                self.damage_times[sid] = now
                new_events.add(sid)
            kills = _number((data.get("state") or {}).get("round_kills"))
            if kills is not None and previous["kills"] is not None and kills > previous["kills"]:
                self.kill_times[sid] = now
                new_events.add(sid)
            weapon, ammo = _active_weapon(data)
            if (weapon and weapon == previous["weapon"] and ammo is not None and ammo >= 0
                    and previous["ammo"] is not None and previous["ammo"] > ammo):
                self.shot_times[sid] = now
                new_events.add(sid)

        # Score all alive players on one common scale, including blocked setups.
        probabilities = (dict(self.detector._rank_model_candidates(gamestate, list(alive)))
                         if complete else {})
        setup_distance = float(self.facts_config.get("setup_distance", 2000.0))
        local_radius = float(self.facts_config.get("local_support_radius", 700.0))
        approach_speed = float(self.facts_config.get("minimum_approach_speed", 20.0))
        activity_window = float(self.facts_config.get("combat_activity_seconds", 0.75))
        trade_window = self.policy.config.trade_window_seconds
        pair_data = {}
        for sid, data in alive.items():
            if positions[sid] is None:
                continue
            for other, enemy in alive.items():
                if enemy["team"] == data["team"] or positions[other] is None:
                    continue
                distance = _distance(positions[sid], positions[other])
                previous_me, previous_enemy = self.previous.get(sid), self.previous.get(other)
                closing = 0.0
                if (dt > 0 and previous_me and previous_enemy and previous_me["position"] is not None
                        and previous_enemy["position"] is not None):
                    closing = (_distance(previous_me["position"], previous_enemy["position"]) - distance) / dt
                pair_data[sid, other] = (distance, closing,
                    _facing(data, positions[sid], positions[other]),
                    self.detector._duel_visibility(data, enemy))

        # Event-supported local groups persist through a death for the trade window.
        for sid in sorted(new_events):
            seed_position = positions.get(sid) or (self.previous.get(sid) or {}).get("position")
            if seed_position is None:
                continue
            neighbors = [other for other in players if positions.get(other) is not None
                         and _distance(seed_position, positions[other]) <= local_radius]
            enemies = [(details[0], other) for (owner, other), details in pair_data.items() if owner == sid]
            if enemies:
                distance, nearest = min(enemies)
                if distance <= setup_distance:
                    neighbors.append(nearest)
                    neighbors.extend(other for other in players if positions.get(other) is not None
                                     and _distance(positions[nearest], positions[other]) <= local_radius)
            previous_groups = [self.groups[other][0] for other in neighbors if other in self.groups and self.groups[other][1] >= now]
            group = min(previous_groups) if previous_groups else f"encounter:{round_key}:{sid}:{now:.6f}"
            for other in set(neighbors + [sid]):
                self.groups[other] = (group, now + trade_window)

        candidates = []
        for sid, data in sorted(alive.items()):
            encounters = [(other, facts) for (owner, other), facts in pair_data.items() if owner == sid]
            visibility = ("visible" if any(facts[3] is True for _, facts in encounters) else
                          "blocked" if encounters and all(facts[3] is False for _, facts in encounters) else "unknown")
            setup = max((0.5 * facing / (1.0 + distance / 1000.0)
                         + 0.25 * min(1.0, max(0.0, closing) / 200.0)
                         for _, (distance, closing, facing, _) in encounters if distance <= setup_distance), default=0.0)
            event_active = self._recent(self.damage_times, sid, now, activity_window) or self._recent(self.kill_times, sid, now, activity_window)
            aimed_visible = any(facts[3] is True and facts[2] >= math.cos(math.radians(20)) for _, facts in encounters)
            event_active = event_active or (self._recent(self.shot_times, sid, now, activity_window) and aimed_visible)
            supporters = [other for other in alive if other != sid and players[other]["team"] == data["team"]]
            support_count = sum(positions[sid] is not None and positions[other] is not None and
                                _distance(positions[sid], positions[other]) <= local_radius for other in supporters)
            # Unknown/missing teammates cannot establish an unsupported anchor.
            support_known = complete and positions[sid] is not None and all(positions[other] is not None for other in supporters)
            approachers = 0
            for other, (distance, _, _, _) in encounters:
                old_enemy = (self.previous.get(other) or {}).get("position")
                if dt <= 0 or distance <= 0 or distance > setup_distance or old_enemy is None:
                    continue
                # Count opponents moving toward this player, not a lone player
                # walking toward stationary enemies and being mislabeled anchor.
                enemy_motion = tuple(b - a for a, b in zip(old_enemy, positions[other]))
                toward_me = tuple(a - b for a, b in zip(positions[sid], positions[other]))
                incoming_speed = sum(a * b for a, b in zip(enemy_motion, toward_me)) / (distance * dt)
                approachers += incoming_speed >= approach_speed
            group = self.groups.get(sid)
            candidates.append(Candidate(
                player_id=sid, team=data["team"], direct_visibility=visibility,
                setup_interest=min(1.0, setup), engagement_active=bool(event_active),
                engagement_id=group[0] if group and group[1] >= now else None,
                trade_group_id=group[0] if group and group[1] >= now else None,
                last_damage_at=self.damage_times.get(sid), last_kill_at=self.kill_times.get(sid),
                is_anchor=support_known and support_count == 0 and approachers >= 2,
                approaching_enemy_count=approachers, supporting_teammate_count=support_count,
                kill_probability=probabilities.get(sid)))

        objective, objective_target = None, None
        bomb = gamestate.get("bomb") or {}
        if bomb.get("state") in ("planting", "defusing") and str(bomb.get("player")) in alive:
            objective, objective_target = bomb["state"], str(bomb["player"])
        else:
            for sid, data in sorted(alive.items()):
                state = data.get("state") or {}
                for action in ("defusing", "planting"):
                    if state.get(action) is True:
                        objective, objective_target = action, sid
                        break
                if objective:
                    break

        self.previous = {sid: {"health": health[sid], "position": positions[sid],
                              "kills": _number((data.get("state") or {}).get("round_kills")),
                              "weapon": _active_weapon(data)[0], "ammo": _active_weapon(data)[1]}
                         for sid, data in players.items()}
        self.previous_time, self.previous_phase = now, phase
        known_phase = phase in ("live", "freezetime", "over", "match_over")
        context = Context(now=now, held_target=self.detector.current_target_sid,
                          candidates=candidates if known_phase else (),
                          round_phase=phase, objective_event=objective, objective_target_id=objective_target,
                          dead_player_ids=dead if known_phase else frozenset(),
                          alive_team_counts=counts if complete and known_phase else None,
                          overview_available=bool(getattr(self.detector, "campaths_enabled", False)) and known_phase)
        self.last_context = context
        return context

    def process(self, gamestate, now, send_overview):
        context = self.build_context(gamestate, now)
        decision = self.policy.decide(context)
        attempted = decision.action in ("switch", "overview")
        success = False
        if decision.action == "switch":
            slot = self.detector.get_player_slot(decision.target_id, gamestate)
            success = bool(slot is not None and self.detector.switch_to_player(slot))
            if success:
                candidate = next(item for item in context.candidates if item.player_id == decision.target_id)
                score = candidate.kill_probability if candidate.kill_probability is not None else candidate.setup_interest
                self.detector._update_current_target(decision.target_id, score)
        elif decision.action == "overview":
            success = bool(context.overview_available and send_overview())
            if success:
                self.detector.current_target_sid = None
                self.detector.current_target_start = None
                self.detector.current_target_score_ema = 0.0
        else:
            success = True
        # Windows input success confirms dispatch only. GSI camera feedback is
        # unavailable here; the internal target/cut budget track accepted input
        # requests rather than pretending the rendered POV was independently seen.
        acknowledged = self.policy.ack(decision, success, now, self.detector.current_target_sid)
        trace = asdict(decision)
        trace.update(success=success, acknowledged=acknowledged, actuator_attempted=attempted,
                     held_target=self.detector.current_target_sid,
                     successful_cuts=self.policy.state.successful_cuts,
                     successful_dispatch_cuts=self.policy.state.successful_cuts,
                     dispatch_succeeded=success if attempted else None,
                     camera_feedback_confirmed=False,
                     alive_team_counts=context.alive_team_counts)
        self.detector.last_observer_decision = trace
        return trace
