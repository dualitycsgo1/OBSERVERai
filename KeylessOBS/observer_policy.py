"""Deterministic broadcast camera policy, independent of models and geometry.

Inputs are causal facts. A blocked sightline *now* does not invalidate an
interesting setup. Kill probability is only a tie-breaker between editorially
relevant candidates; it cannot create interest or force top-two rotation.

Timer/budget defaults below are configurable engineering choices, not estimated
professional-observer parameters. VOD observations motivate phase-dependent
continuity, not a universal shot duration. No clocks, I/O or future labels occur
inside this module. The caller supplies monotonic match time and confirms the
actual camera result with ``ack`` (including hold decisions).
"""

from dataclasses import dataclass, replace
import math
from typing import FrozenSet, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Candidate:
    player_id: str
    team: str
    alive: bool = True
    direct_visibility: str = "unknown"
    setup_interest: float = 0.0
    engagement_id: Optional[str] = None
    engagement_active: bool = False
    trade_group_id: Optional[str] = None
    last_damage_at: Optional[float] = None
    last_kill_at: Optional[float] = None
    is_anchor: bool = False
    approaching_enemy_count: int = 0
    supporting_teammate_count: int = 0
    kill_probability: Optional[float] = None

    def __post_init__(self):
        if not self.player_id or not self.team:
            raise ValueError("Candidate requires a player identity and team")
        if self.direct_visibility not in {"visible", "blocked", "unknown"}:
            raise ValueError("direct_visibility must be visible, blocked or unknown")
        for name in ("setup_interest", "kill_probability"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be a finite value between zero and one")
        if self.approaching_enemy_count < 0 or self.supporting_teammate_count < 0:
            raise ValueError("Encounter player counts cannot be negative")
        for value in (self.last_damage_at, self.last_kill_at):
            if value is not None and not math.isfinite(value):
                raise ValueError("Event times must be finite")


@dataclass(frozen=True)
class Context:
    now: float
    held_target: Optional[str]
    candidates: Sequence[Candidate]
    round_phase: str = "live"
    objective_event: Optional[str] = None
    objective_target_id: Optional[str] = None
    dead_player_ids: FrozenSet[str] = frozenset()
    alive_team_counts: Optional[Mapping[str, int]] = None
    overview_available: bool = False

    def __post_init__(self):
        if not math.isfinite(self.now):
            raise ValueError("Context time must be finite")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "dead_player_ids", frozenset(self.dead_player_ids))
        ids = [candidate.player_id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("Candidate identities must be unique")
        if self.alive_team_counts is not None:
            counts = dict(self.alive_team_counts)
            if any(not isinstance(count, int) or count < 0 for count in counts.values()):
                raise ValueError("Alive team counts must be nonnegative integers")
            object.__setattr__(self, "alive_team_counts", counts)


@dataclass(frozen=True)
class PolicyConfig:
    minimum_shot_seconds: float = 1.0
    setup_hold_seconds: float = 3.0
    engagement_grace_seconds: float = 0.75
    trade_window_seconds: float = 2.0
    aftermath_seconds: float = 1.2
    cut_window_seconds: float = 10.0
    max_cuts_per_window: int = 3
    urgent_cuts_bypass_budget: bool = True
    minimum_setup_interest: float = 0.2
    setup_switch_margin: float = 0.15
    anchor_minimum_approachers: int = 2
    clutch_minimum_opponents: int = 2

    def __post_init__(self):
        for name in ("minimum_shot_seconds", "setup_hold_seconds", "engagement_grace_seconds",
                     "trade_window_seconds", "aftermath_seconds", "cut_window_seconds",
                     "minimum_setup_interest", "setup_switch_margin"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.cut_window_seconds == 0 or self.max_cuts_per_window < 0:
            raise ValueError("Cut window must be positive and budget nonnegative")
        if self.anchor_minimum_approachers < 1 or self.clutch_minimum_opponents < 1:
            raise ValueError("Encounter thresholds must be positive")


@dataclass(frozen=True)
class PolicyState:
    held_target: Optional[str] = None
    held_since: Optional[float] = None
    view_kind: str = "unknown"  # pov, overview, unknown; distinct from editorial mode
    mode: str = "overview"
    mode_since: Optional[float] = None
    trade_group_id: Optional[str] = None
    last_conflict_at: Optional[float] = None
    last_kill_at: Optional[float] = None
    cut_times: Tuple[float, ...] = ()
    successful_cuts: int = 0
    acknowledged_at: Optional[float] = None


@dataclass(frozen=True)
class Decision:
    decision_id: int
    action: str  # hold, switch, overview
    target_id: Optional[str]
    mode: str  # setup, engagement, trade, aftermath, overview
    reason: str
    issued_at: float
    from_target_id: Optional[str]
    urgent: bool = False
    trade_group_id: Optional[str] = None
    conflict_at: Optional[float] = None
    kill_at: Optional[float] = None


class ObserverPolicy:
    def __init__(self, config: Optional[PolicyConfig] = None):
        self.config = config or PolicyConfig()
        self.state = PolicyState()
        self._sequence = 0
        self._pending: Optional[Decision] = None

    def reset_story(self, now: float, held_target: Optional[str]) -> None:
        """Forget encounter history at a confirmed round/map boundary.

        This is an observation reset, not a camera cut. Preserve the observed
        view, its shot start, and all cut accounting. A changed external held
        target starts a newly observed shot without inventing an actuator cut.
        Any unacknowledged decision from the previous story becomes stale.
        """
        if not math.isfinite(now):
            raise ValueError("Story reset time must be finite")
        if self.state.acknowledged_at is not None and now < self.state.acknowledged_at:
            raise ValueError("Story reset time cannot move backwards")
        same_view = held_target == self.state.held_target
        self.state = replace(
            self.state, held_target=held_target,
            held_since=self.state.held_since if same_view else now,
            view_kind=self.state.view_kind if same_view else "pov" if held_target is not None else "unknown",
            mode="overview", mode_since=None, trade_group_id=None,
            last_conflict_at=None, last_kill_at=None, acknowledged_at=now)
        self._pending = None

    @staticmethod
    def _latest(*values):
        valid = [value for value in values if value is not None]
        return max(valid) if valid else None

    @staticmethod
    def _causal(value, now):
        # Future event timestamps are never used as evidence.
        return value if value is not None and value <= now else None

    def _recent(self, value, now, window):
        return value is not None and 0 <= now - value <= window

    def _role(self, candidate, context):
        counts = context.alive_team_counts
        if (counts and counts.get(candidate.team) == 1
                and sum(count for team, count in counts.items() if team != candidate.team)
                >= self.config.clutch_minimum_opponents):
            return 2  # Follow the lone player's clutch story, not the opposing max-p.
        if (candidate.is_anchor and candidate.supporting_teammate_count == 0
                and candidate.approaching_enemy_count >= self.config.anchor_minimum_approachers):
            return 1
        return 0

    def _editorial_key(self, candidate, context):
        return (self._role(candidate, context), int(candidate.engagement_active), candidate.setup_interest)

    def _interesting(self, candidate, context):
        return (self._role(candidate, context) > 0 or candidate.engagement_active
                or (candidate.setup_interest > 0 and candidate.setup_interest >= self.config.minimum_setup_interest))

    def _best(self, candidates, context):
        # Model scores only settle otherwise-equal editorial alternatives.
        return min(candidates, key=lambda candidate: (
            *(-value for value in self._editorial_key(candidate, context)),
            -(candidate.kill_probability if candidate.kill_probability is not None else -1),
            candidate.player_id)) if candidates else None

    def _make(self, context, action, target, mode, reason, urgent=False, candidate=None):
        same_observed_view = self.state.held_target == context.held_target
        conflict_at = self.state.last_conflict_at if same_observed_view else None
        kill_at = self.state.last_kill_at if same_observed_view else None
        group = self.state.trade_group_id if same_observed_view else None
        if candidate is not None:
            group = candidate.trade_group_id or (group if candidate.player_id == context.held_target else None)
            conflict_at = self._latest(self._causal(candidate.last_damage_at, context.now),
                                       self._causal(candidate.last_kill_at, context.now),
                                       context.now if candidate.engagement_active else None,
                                       conflict_at if candidate.player_id == context.held_target else None)
            kill_at = self._latest(self._causal(candidate.last_kill_at, context.now),
                                   kill_at if candidate.player_id == context.held_target else None)
        if action == "switch" and target == context.held_target:
            action = "hold"
        if mode == "overview" and reason.startswith("round_overview"):
            group, conflict_at, kill_at = None, None, None
        self._sequence += 1
        decision = Decision(self._sequence, action, target, mode, reason, context.now,
                            context.held_target, urgent, group, conflict_at, kill_at)
        self._pending = decision
        return decision

    def _hold(self, context, mode, reason, candidate=None):
        return self._make(context, "hold", context.held_target, mode, reason, candidate=candidate)

    def _cut(self, context, candidate, mode, reason, urgent=False):
        if candidate.player_id == context.held_target:
            return self._hold(context, mode, reason, candidate)
        recent_cuts = sum(0 <= context.now - at < self.config.cut_window_seconds for at in self.state.cut_times)
        if (recent_cuts >= self.config.max_cuts_per_window
                and not (urgent and self.config.urgent_cuts_bypass_budget)):
            current = next((item for item in context.candidates if item.player_id == context.held_target), None)
            return self._hold(context, self.state.mode, "cut_budget", current)
        return self._make(context, "switch", candidate.player_id, mode, reason, urgent, candidate)

    def _overview(self, context, reason, urgent=False):
        if not context.overview_available:
            return self._hold(context, "overview", reason + "_retain_pov")
        if self.state.view_kind == "overview" and context.held_target is None:
            return self._hold(context, "overview", "preserve_overview")
        cuts = sum(0 <= context.now - at < self.config.cut_window_seconds for at in self.state.cut_times)
        if (cuts >= self.config.max_cuts_per_window
                and not (urgent and self.config.urgent_cuts_bypass_budget)):
            return self._hold(context, self.state.mode, "cut_budget")
        return self._make(context, "overview", None, "overview", reason, urgent=urgent)

    def decide(self, context: Context) -> Decision:
        """Propose a decision without committing camera state or cut budgets.

        A caller must acknowledge even holds with the observed current target.
        Missing players are telemetry gaps unless explicitly reported dead.
        alive_team_counts must be supplied only when the counts are trustworthy.
        """
        if self.state.acknowledged_at is not None and context.now < self.state.acknowledged_at:
            raise ValueError("Policy context time cannot move backwards")
        dead = context.dead_player_ids | {item.player_id for item in context.candidates if not item.alive}
        alive = [item for item in context.candidates if item.alive and item.player_id not in dead]
        by_id = {item.player_id: item for item in alive}
        held = by_id.get(context.held_target)
        held_dead = context.held_target is not None and context.held_target in dead

        if context.round_phase in {"freezetime", "over", "match_over"}:
            return self._overview(context, "round_overview")
        objective = by_id.get(context.objective_target_id)
        if context.objective_event and objective is not None:
            return self._cut(context, objective, "engagement", "confirmed_objective", urgent=True)
        if context.held_target and held is None and not held_dead:
            return self._hold(context, self.state.mode, "telemetry_gap")

        same_view = self.state.held_target == context.held_target
        group = held.trade_group_id if held and held.trade_group_id else self.state.trade_group_id if same_view else None
        conflict = self._latest(self.state.last_conflict_at if same_view else None,
                                self._causal(held.last_damage_at, context.now) if held else None,
                                self._causal(held.last_kill_at, context.now) if held else None)
        traded = [item for item in alive if group and item.trade_group_id == group
                  and self._interesting(item, context)]

        if held_dead:
            follow = self._best(traded, context) or self._best([item for item in alive if self._interesting(item, context)], context)
            # A confirmed death requires leaving the dead view; an alive fallback
            # is a recovery decision, never claimed to predict the next kill.
            follow = follow or self._best(alive, context)
            return (self._cut(context, follow, "trade" if traded else "setup", "confirmed_death", urgent=True)
                    if follow else self._overview(context, "dead_view_no_target", urgent=True))

        if held and held.engagement_active:
            return self._hold(context, "engagement", "preserve_engagement", held)
        if held and self._role(held, context) == 2:
            return self._hold(context, "setup", "preserve_clutch", held)

        trade_alternatives = [item for item in traded if item.player_id != context.held_target]
        if self._recent(conflict, context.now, self.config.trade_window_seconds) and trade_alternatives:
            follow = self._best(trade_alternatives, context)
            if follow and follow.player_id != context.held_target and follow.engagement_active:
                return self._cut(context, follow, "trade", "follow_trade", urgent=True)
            if held and held.trade_group_id == group:
                return self._hold(context, "trade", "preserve_trade_chain", held)

        if (held and self.state.mode == "engagement" and same_view
                and self._recent(conflict, context.now, self.config.engagement_grace_seconds)):
            return self._hold(context, "engagement", "engagement_grace", held)
        kill_at = self._latest(self.state.last_kill_at if same_view else None,
                               self._causal(held.last_kill_at, context.now) if held else None)
        if held and self._recent(kill_at, context.now, self.config.aftermath_seconds):
            return self._hold(context, "aftermath", "aftermath_hold", held)

        interesting = [item for item in alive if self._interesting(item, context)]
        best = self._best(interesting, context)
        if not best:
            return self._overview(context, "no_active_story")
        if held and best.player_id == held.player_id:
            reason = "preserve_clutch" if self._role(held, context) == 2 else "anchor_under_pressure" if self._role(held, context) == 1 else "preserve_setup"
            return self._hold(context, "setup", reason, held)

        # Existing editorial interest wins ties even if model probabilities move.
        if held and self._interesting(held, context):
            held_key, best_key = self._editorial_key(held, context), self._editorial_key(best, context)
            if best_key[:2] < held_key[:2] or (best_key[:2] == held_key[:2]
                    and best.setup_interest <= held.setup_interest + self.config.setup_switch_margin):
                return self._hold(context, "setup", "preserve_setup", held)

        held_since = self.state.held_since if same_view and self.state.held_since is not None else context.now
        if context.held_target and context.now - held_since < self.config.minimum_shot_seconds:
            return self._hold(context, self.state.mode, "minimum_shot", held)
        if (held and self.state.mode == "setup" and same_view
                and self._interesting(held, context) and context.now - held_since < self.config.setup_hold_seconds):
            return self._hold(context, "setup", "setup_hold", held)
        mode = "engagement" if best.engagement_active else "setup"
        reason = "follow_clutch" if self._role(best, context) == 2 else "anchor_under_pressure" if self._role(best, context) == 1 else "follow_engagement" if best.engagement_active else "setup_interest"
        return self._cut(context, best, mode, reason)

    def ack(self, decision: Decision, success: bool, now: float,
            actual_target_id: Optional[str]) -> bool:
        """Commit only a confirmed result; False/failed/mismatched/stale keys do nothing.

        ``success=True`` means the requested view was actually attained, not just
        that an input request was queued. An acknowledged hold initializes an
        already-observed view without charging a cut. Overview requires actual
        overview capability and confirmation by the adapter. Urgent cuts still
        count, including configured exceptions to the ordinary cut budget.
        """
        if (self._pending != decision or not success or actual_target_id != decision.target_id):
            return False
        if not math.isfinite(now) or now < decision.issued_at:
            raise ValueError("Acknowledgment time precedes the decision")
        if self.state.acknowledged_at is not None and now < self.state.acknowledged_at:
            raise ValueError("Acknowledgment time cannot move backwards")
        is_cut = decision.action in {"switch", "overview"}
        changed = actual_target_id != self.state.held_target
        since = now if changed or self.state.held_since is None or is_cut else self.state.held_since
        kind = "overview" if decision.action == "overview" else "pov" if actual_target_id is not None else self.state.view_kind
        cut_times = tuple(at for at in self.state.cut_times if 0 <= now - at < self.config.cut_window_seconds)
        if is_cut:
            cut_times += (now,)
        self.state = PolicyState(
            held_target=actual_target_id, held_since=since, view_kind=kind,
            mode=decision.mode, mode_since=now if decision.mode != self.state.mode or self.state.mode_since is None else self.state.mode_since,
            trade_group_id=decision.trade_group_id, last_conflict_at=decision.conflict_at,
            last_kill_at=decision.kill_at, cut_times=cut_times,
            successful_cuts=self.state.successful_cuts + int(is_cut), acknowledged_at=now)
        self._pending = None
        return True
