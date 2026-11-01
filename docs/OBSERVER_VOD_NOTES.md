# Professional observer review

These observations motivate a first camera policy. They are not labels of the
observer's intention, a trained imitation model, or a broadcast quality benchmark.
Only the selected sequences below were visually reviewed, not the complete VODs.

## Evidence and review scope

1. [ESL Archives: Vitality–Spirit, IEM Katowice 2025 final](https://www.youtube.com/watch?v=ykyKLhnX6lw).
   The first 180 seconds were inspected as 180 sampled frames at one frame per
   second, with enlarged HUD/name checks. This includes the complete live first
   Dust2 round, approximately 0:39–1:51, and part of round two. Boundaries are
   approximate to a sampled second; shorter cuts can be missed. Audio was not
   reviewed, so caster coordination cannot be inferred.
2. [BLAST Premier: G2–NAVI, Fall Final 2024](https://www.youtube.com/watch?v=SqLtMKwLB0k&t=5452s).
   Selected one-frame-per-second contact sheets cover the first two Ancient live
   rounds in a four-minute excerpt starting near 1:30:52. Other sampled sheets
   show reactions, replay and timeout transitions. Partial-download keyframe
   preroll makes these timestamps approximate; duplicate preroll frames were
   excluded from duration conclusions. No full-match shot statistics are claimed.

The notes record visible POV names, HUD counts, killfeed and broadcast graphics.
An X-ray outline behind cover is visible broadcast evidence, not a geometric
proof of a blocked bullet trajectory. Mesh visibility is measured separately.
Local video/frame files are not redistributed in the repository.

## Observations translated into rules

| Visible sequence | Observation | Policy hypothesis |
| --- | --- | --- |
| [ESL ~1:12–1:21](https://www.youtube.com/watch?v=ykyKLhnX6lw&t=72s) | ZywOo's POV stays on a held angle before immediate contact; opponents are visible by X-ray behind cover during the setup. | A setup can be interesting before a direct duel is visible. Do not require a clear current sightline for every camera candidate. |
| [ESL ~1:22–1:29](https://www.youtube.com/watch?v=ykyKLhnX6lw&t=82s) | Several POV changes accompany a kill/death chain. | A confirmed death or local trade can justify a rapid cut, including returning to a previously shown player. A universal cooldown cannot express this. |
| [ESL ~1:33–1:51](https://www.youtube.com/watch?v=ykyKLhnX6lw&t=93s) | chopper is the last T against four CTs, at 10 HP. The camera stays with him through the final fight and his death to ropz. | The lone player's perspective can carry the situation even when he is the eventual victim. Kill probability alone is an incomplete editorial objective. |
| [BLAST Ancient round 1, ~1:31:25 onward](https://www.youtube.com/watch?v=SqLtMKwLB0k&t=5485s) | huNter's CT POV is held as several T X-ray outlines approach behind cover/smoke. His fight is followed before the view changes to malbsMd. | Show the defending angle and maintain one understandable engagement. Incoming pressure and nearby support are useful setup facts. |
| BLAST Ancient round 1, same sequence | The killfeed shows malbsMd's kills while huNter's POV remains on screen for part of the sequence. | Chasing every latest kill is not equivalent to coherent observing. A kill elsewhere does not always invalidate the current story. |
| [BLAST Ancient round 2, ~1:32:46–1:33:00](https://www.youtube.com/watch?v=SqLtMKwLB0k&t=5566s) | iM is the last T. His POV is retained through smoke/obstructed moments and the final fight, ending in death to Snax. | This second production also supports lone-player continuity and interest before unobstructed contact. |
| ESL first-round aftermath; BLAST round transitions | Player/arena reactions and explicit REPLAY graphics interrupt live POV footage. | Separate live selection, overview and replay/reaction material. Replayed kills must not become additional live events or training labels. |

These are observed choices plus our interpretation. We do not know which
information, director instructions or caster requests caused an individual cut.
The reviewed sample does not establish optimal hold durations or cut budgets.

## Implemented first policy

`KeylessOBS/observer_policy.py` is a deterministic policy with explicit setup,
engagement, trade, aftermath and overview modes. `observer_runtime_policy.py`
derives facts from past/current GSI snapshots and adapts a decision to one input
attempt. `observer.camera_policy` defaults to `broadcast`; `legacy` remains
available for comparison.

- All known living players can be considered for a setup. Direct visibility
  remains a separate `visible`/`blocked`/`unknown` fact. Static blocked geometry
  does not remove a player who may peek later.
- Facing, distance and measured closing movement provide setup interest.
  Observed health drops and round-kill increments support engagements and local
  trade groups. Decreases in the active weapon's clip ammunition are shooting
  proxies, not confirmed `weapon_fire` events. Missing health is unknown, not death.
  The forecast abstains when the known roster is incomplete or another reported
  playing-team player's health is unknown, rather than reducing alive counts.
- Prefer continuity during an active engagement, a local trade chain, or a
  confirmed lone-player situation. An isolated defender facing multiple moving
  opponents is another role cue. These approximations do not detect complete
  executes, utility plans, or reachable future sightlines.
- Confirmed death and identified planting/defusing actors can interrupt holds.
  Quiet low model scores do not cause automatic top-two rotation. The kill
  forecast only breaks ties among editorially relevant candidates.
- A rolling cut budget and phase-dependent holds control ordinary switches.
  Explicit urgent exceptions are counted. Failed input attempts change neither
  the internal target nor the policy budget. Freezetime follows the actual round
  phase rather than an unconditional 19-second lock.

All timer, distance and score defaults in `config.json` are engineering choices,
not values estimated from these few rounds. Overview only uses the existing
campath capability when enabled; player cameras, arena cameras and broadcast
replay production are not implemented by this policy.

The pure policy supports acknowledgments. The current keyboard adapter can
confirm only that an input call succeeded; it has no independent rendered-camera
feedback. Offline replay simulates successful camera selection. Live switching,
spectator-slot mapping and delivery delay still require a real spectator test.

## Evaluation needed

Regressions check the concrete mechanics: a blocked setup remains eligible, a
low-health lone player is retained, genuine rapid trades are allowed, missing
telemetry does not manufacture deaths/clutches, and failed keys do not consume
state or budget. These tests establish implementation behavior, not human quality.

The audit compares both policies using the same model, meshes and causal replay
inputs. Next-killer ranking, held-target coverage and camera continuity must be
reported separately. Human observers may intentionally show the victim or retain
a fight while another kill happens elsewhere. Ultimately a useful benchmark
needs aligned VOD/demo POV labels, more independent matches, and human review of
coherence, missed action, utility, clutches and transitions.

For complementary first-person context, sapphiRe describes role-dependent
choices such as the lone clutcher and the defending anchor in this
[PC Gamer interview](https://www.pcgamer.com/what-it-takes-to-be-a-counter-strike-global-offensive-observer/).
That supports the hypotheses; it does not supply per-cut ground truth for these VODs.
