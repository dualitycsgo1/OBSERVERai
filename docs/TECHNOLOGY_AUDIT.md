# Technology audit — 1 November 2026

Baseline inspected: `de89700` on `main`. This audit covers static collision
geometry, duel selection, demo parsing, model training/evaluation, and live model
integration. It does not certify complete broadcast automation.

## Findings and corrections

| Area | Baseline defect | Implemented behavior |
| --- | --- | --- |
| Prediction target | `AITrainer` trained `is_headshot` but observer used the result as a kill likelihood | Explicit `kill_within_horizon` target: at least one enemy kill within the next two seconds |
| Feature parity | Trainer and live analyzer used different 27-feature definitions | One shared, versioned 17-feature extractor based on information available before a kill |
| Evaluation | Random split within demos and duplicated batch/master data | SHA-256 demo identity, sample deduplication, persisted split by entire demo, positive and negative examples |
| Reproducibility | Parser map iteration could reorder samples and change a seeded forest after reparsing | Canonical `(demo_id, sample_tick, player_id)` order and stable feature tie-breaks; permutation regression verifies identical predictions and tree structure |
| Improvement claims | Data counts and headshot accuracy were presented as increasing AI intelligence | Baseline/incumbent comparisons, Brier/log-loss/ranking metrics, per-demo uncertainty, promotion rejection |
| Geometry | NAV LOS was distance-based; fallback blockmasks could generate artificial walls | Actual 3D collision triangles, BVH ray casting, explicit visible/blocked/unknown states |
| Angles | Facing flattened height and pitch into 2D | 3D forward-vector aim angle; eye-to-body visibility samples for partial cover |
| Missing data | Default positions/angles and header-only success could become training data | Missing quantities remain missing; failed/metadata-only parses do not count as prediction examples |
| Parser | Checked-in Go source was duplicated/corrupt; v5.0.2 failed on all five supplied demos | Buildable parser pinned to demoinfocs v6.0.0-rc.1; all five demos parse successfully |
| Compressed demos | `.dem.zst` could not be imported | Streaming decompression to temporary files, source preservation, canonical uncompressed content hash |
| Model reload | Initialization/order, incompatible estimators and cache/notification handling could prevent real live updates | File polling, bundle validation, atomic publication, last-valid-model retention, first model can arrive after startup |
| Duel locks | A premature return made later selection unreachable; blocked locks could persist; heuristic trigger scores suppressed low model probabilities | Shared model ranking in locked/normal paths, expiry-aware event holds, geometry invalidation before cooldown, comparisons on the same probability scale |
| Inference cost | One parallel forest inference per player per call | Batched candidate inference; training uses parallel fitting and saves a single-thread inference estimator |

The former README accuracy statements (85–96%) and AIIMS intelligence labels
were not valid evidence of future-kill or camera performance. The README and
trainer display now describe what is actually measured.

## Real demo experiment

Five user-supplied demos were read without modifying the originals. Player names,
Steam IDs, filenames, full snapshots and demo binaries are not included in this
public report. The files cover four maps and two patches/build-header numbers.

| Map | Registered kills | Valid player snapshots | Positive two-second labels |
| --- | ---: | ---: | ---: |
| Ancient | 181 | 19,207 | 604 |
| Inferno | 109 | 11,641 | 365 |
| Dust2 A (held out) | 172 | 23,260 | 595 |
| Cache (held out) | 133 | 16,700 | 450 |
| Dust2 B | 161 | 19,725 | 549 |
| Total | 756 | 90,533 | 2,563 |

Samples are taken every 0.5 seconds in live rounds, after the current frame.
Only kills in future frames and inside a complete two-second horizon in the same
round count as positive labels. Teamkills, warmup/freezetime and partial end-of-
round/EOF horizons are excluded. Consecutive samples overlap and are correlated;
90,533 rows are not 90,533 independent matches.

Phase filtering does not distinguish every match segment. Subsequent inspection
found an opening knife-only live interval in both held-out demos, followed by a
long gap and reuse of the first round number. Those intervals contribute 36 of
the 942 strict next-killer windows (3.82%). The original static model experiment
includes them; it has not been silently retrained or relabelled. Camera replay
also reports the remaining 906 windows separately and resets reconstruction at
observed telemetry gaps. This is a dataset limitation, not evidence that a missing
phase boundary can be inferred exactly.

The fixed split has three training demos (50,573 player rows) and two held-out
demos (39,960 rows, 1,045 positives). The estimator is a Random Forest with 150
trees, minimum leaf size 3, seed 42. No historical compatible kill model was
present, so the initial comparison is against explicit baselines.

| Metric on held-out demos | Model | Baseline |
| --- | ---: | ---: |
| Strict next-killer top-1, next two seconds | 48.83% (460/942) | Nearest enemy: 28.66% (270/942) |
| Top-1 player gets any enemy kill within two seconds | 50.80% (478/941) | Nearest enemy: 30.29% (285/941) |
| Brier score (lower is better) | 0.022900 | Constant training prevalence: 0.025482 |
| Log loss (lower is better) | 0.096182 | 0.121365 |
| Average precision (higher is better) | 0.217767 | Constant prevalence: 0.026151 |
| Accuracy at 0.5 | 97.35% | Predict no kill: 97.38% |

The accuracy row illustrates class imbalance: an always-negative model exceeds
97% while predicting no kills. It is not used as proof of useful learning.

Strict next-killer uses the first enemy kill event after each sample, within two
seconds and the same round; same-tick kills are excluded. An absent killer counts
as a miss. That produces one more opportunity than the candidate-label metric.
These are **static rankings across all eligible exported players**, without the
observer's mesh candidate filter, switch cooldown, camera delays or duel locks.
They do not mean that the broadcast camera captures half of all kills.

## Does adding demos improve the model?

An exploratory learning curve kept the same two validation demos and increased
training from one to two to three demos. This curve is for diagnosis; the trainer
itself still requires the independent-demo minimum documented in README.

| Training demos | Top-1 any-kill coverage | Brier | Average precision | Log loss |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 48.14% | 0.023269 | 0.20154 | 0.098472 |
| 2 | 52.28% | 0.022978 | 0.21784 | 0.097033 |
| 3 | 50.80% | 0.022900 | 0.21777 | 0.096182 |

Adding the second demo improved these metrics. Adding the third improved Brier
and log loss, slightly lowered average precision and reduced top-1 coverage.
More data did not produce monotonic improvement across the relevant measures.

The actual promotion evaluator accepted the 1→2 comparison and rejected 2→3.
For 2→3, mean per-demo Brier improvement was +0.00007281, with paired per-demo
bootstrap interval [+0.00004102, +0.00010460]. Nevertheless top-1 coverage fell
from 52.28% to 50.80%, so the promotion gate rejected the candidate. A sequential
deployment therefore retains the incumbent instead of replacing it on probability
error alone. Experiment models remain isolated from the live installation.

Only **two independent validation demos** were available. Bootstrap intervals are
therefore fragile, and repeated evaluation on this fixed set can overfit model
selection. A general accuracy claim needs substantially more fresh demos, maps,
patches and skill levels kept untouched until final testing.

## Camera selection: visibility, VODs and fair replay

An initial replay used the living-player export without restored deaths or
round-kill counters. The legacy policy held the next killer in 25.16% of 942
windows with matched meshes, versus 27.92% in unknown-geometry fallback. The raw
model ranked the next killer first in 48.83%. Current direct-duel filtering put
the next killer in its pool in only 54.99% of windows. This diagnosed a mismatch:
current occlusion cannot rule out an upcoming peek. Those historical runs are
retained in the JSON, but are not a fair baseline for the enhanced replay below.

Selected professional sequences were then visually reviewed: one complete ESL
Dust2 round and the first two BLAST Ancient live rounds at one sampled frame per
second. [Timestamped facts and interpretations](OBSERVER_VOD_NOTES.md) document
holds before contact, continuity with a lone player, and rapid event-driven cuts.
The resulting hand-written policy separates setup, engagement, trade, aftermath
and overview. It keeps blocked setups eligible and does not rotate on quiet
model scores. This is not imitation learning, and default timers were not fitted
to the VODs or the two validation demos.

Both policies were compared again with identical canonical model, matching patch
meshes and causal event reconstruction. A death/kill counter is restored only
from an already-observed event in the same round (both tick and time checks).
Positions and ammunition come only from current/past snapshots. Unconfirmed
missing players remain unknown. Gaps longer than two seconds start a new observed
interval so reused round numbers do not carry opening knife deaths into later
play. This conservative reset also drops events in unobserved gaps.
The shared feature extractor abstains when a reported playing-team player's
health is unknown; the broadcast adapter also abstains when its known roster is
incomplete. It does not reduce alive counts to manufacture a model forecast.
Broadcast model scores were available in
5408/5558 snapshots; the
remaining snapshots were processed without a forecast.

| Metric, same 5,558 sampled states / 942 action windows | Legacy policy | Broadcast policy |
| --- | ---: | ---: |
| Held target is the next killer, within two seconds | 22.82% (215/942) | 26.01% (245/942) |
| Held target is killer or victim of that next kill | 46.82% (441/942) | 52.12% (491/942) |
| Simulated target changes | 504 | 583 |
| Sampled held target confirmed dead | 9 | 0 |
| Mean process time on this machine | 30.2 ms | 62.9 ms |

The increase is concentrated on Dust2: +28 next-killer hits and +49
killer-or-victim hits. Cache changes by only +2 and +1, respectively. Total
simulated cuts rise from 504 to 583 (15.7%), and measured computation roughly
doubles. These are useful tradeoffs to review, not uniform improvement across
matches or evidence that more cutting makes a better broadcast.

Excluding the first knife-only intervals leaves 906 windows. Next-killer coverage
is 23.18%
for legacy and 26.05%
for broadcast. Killer-or-victim coverage is
47.24%
and 52.21%,
respectively. This is a diagnostic subset, not a fresh test set.

All four enhanced runs had 0 processing
errors, 0 native input
attempts and 0 simulated
camera/internal-target mismatches. Raw per-map and aggregate counts plus source
fingerprints are saved in [measurements.json](measurements.json). The direct-duel
pool diagnostics describe the legacy filter; the broadcast setup candidate pool
is recorded separately.

These metrics describe the held player at sampled times before an upcoming kill,
not rendered POV kill capture, complete action coverage or professional quality.
Killer-or-victim inclusion does not establish a readable shot or trade sequence.
The export still lacks freeze phases, bomb objectives, omitted world/suicide
deaths and final partial horizons. Only two independent demos are available.
Processing timings include offline concurrent load and are not live latency.

The pure policy commits state on acknowledgment. Its keyboard adapter only knows
whether dispatch succeeded; it has no independent spectator-camera feedback.
Failed dispatch changes neither target nor cut budget. A live spectator trial
must verify slot mapping, delivery delays and actual camera results. Static
visibility and movement-based setup interest also do not yet solve reachable
future-peek geometry or wall penetration. This remains an audited draft.

## Local encounter revision after a real trial

A user's tunnel-rush recording exposed a policy failure. A distant defender's
molotov damage and anchor-role transition stole the camera from an already
selected local multi-player attack. The controlled causal replay reproduces the
branch; it does not reconstruct exact live GSI cadence. The case uses a training
demo and is kept separate from the held-out regression.

The new policy commits to the selected local encounter, retains it through firing
pauses and follows a local participant after confirmed death. It uses the existing
700-unit radius without map/player hardcoding. Unknown contacts do not become
confirmed disengagement. The obsolete separate Positional AI integration is also
removed; the validated future-kill model's weights and feature schema are unchanged.

| Same 942 overlapping windows | Previous broadcast | Local encounter revision |
| --- | ---: | ---: |
| Held next killer | 245 (26.01%) | 274 (29.09%) |
| Held killer or victim | 491 (52.12%) | 520 (55.20%) |
| Simulated target changes | 583 | 507 |
| Completed POV spans below one second | 25 | 18 |

All 5,558 per-snapshot inputs, raw model rankings and future diagnostic labels
were checked identical before/after. New runs have zero processing errors,
native input attempts, simulated camera/internal-target mismatches or held targets
confirmed dead. Excluding opening knife intervals gives 262/906 next-killer hits
and 502/906 killer-or-victim hits. The earlier legacy run is explicitly a
reference from its recorded source, not a new run on the current source.

Results differ by map: Cache improves from 99 to 120 next-killer hits and 191 to
226 duel hits; Dust2 improves from 146 to 154 next-killer hits but drops from 300
to 294 duel hits. Repeated use of these same two demos is a regression check, not
fresh independent validation. The new measured process time is not a controlled
before/after benchmark because host/game and parallel load changed.

[Case mechanism and installed VPK inspection](LOCAL_ENCOUNTERS_AND_VPK.md) also
document read-only extraction of NAV, physics/material tags and 43 named place
entities. These native exports do not automatically replace runtime geometry.
The local trial records future diagnostic GSI/decision sessions on its next
manual start; no application is launched by the update.

## Geometry verification and practical limits

Native AWPY AWMH v1 `.mesh` files and numeric NPZ collision triangles are supported.
NAV meshes and the old 2D blockmasks are not treated as visibility evidence.
The setup CLI pins a ClientVersion, checks the upstream archive hash, validates
mesh structure and records provenance. It performs no automatic runtime download.

Official assets from release 2000939 were benchmarked on this machine:

| Map | Triangles | BVH load/build | Mean single ray | p95 single ray | Mean sampled duel pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dust2 | 239,848 | 0.558 s | 0.261 ms | 0.598 ms | 2.900 ms |
| Inferno | 547,829 | 1.280 s | 0.224 ms | 0.657 ms | 2.331 ms |

500 randomized rays and 50 randomized duel pairs per map were measured. Official
Inferno reference points returned the documented visible and occluded results.
These checks establish intersection behavior and a local timing measurement,
not map-wide precision or a real-time latency guarantee.

Static occlusion only filters **direct** engagements. Penetrable walls, material
thickness and damage, smoke, dynamic doors/destructibles, exact crouch/animated
hitboxes and future peeks require additional models. Multiple eye-to-body rays
are an approximation; upstream meshes may be decimated. `blocked` is not an
absolute statement that a kill is impossible. Range/facing thresholds are
selection policy, not physical weapon limits.

Use geometry matching the game/map version. For the held-out demos, matching
patch manifests are available as Dust2 2000927 (1.41.8.9) and Cache 2000908
(1.41.8.1). Header BuildNum and Steam buildid are different identifiers; patch
matching alone does not certify exact source-asset identity.

## Reproduction and remaining work

1. Build the pinned parser, import `.dem`/`.dem.zst`, and inspect the parser's
   success/data-quality fields before accepting the samples.
2. Keep the persistent `validation_manifest.json` with the dataset. Do not move
   validation demos into training and reuse their results as independent tests.
3. Train via `AITrainer.train_model()` or the GUI; inspect `latest_evaluation.json`.
   Candidate publishing is gated and atomic; rejected candidates keep the model.
4. Install matching map collision assets, start the observer, and inspect `status`.
5. Use `replay_observer.py` with keyboard disabled for sampled state-machine
   checks. A complete broadcast benchmark still needs all game phases/players,
   event timing, GSI delivery delays and actual camera switching observations.

The isolated evaluation can be reproduced from raw parser JSON exports with:

```powershell
python KeylessOBS/evaluate_demos.py --results-dir work/parser-results --output-dir work/evaluation-new --learning-curve --strict-next-killer
```

Aggregated results without player identifiers are in [measurements.json](measurements.json).

Local checks passed: 169 Python tests and 4 subtests, 7 Go tests, a source parser
build, all five compressed real demos, the isolated evaluation CLI, and four
initial and four enhanced held-out replay runs. Automated validation commands are in README and
the Windows CI workflow. The
source build is verified. Packaged observer/trainer EXEs and interaction with a
running CS2 spectator session were not validated in this audit.

Official references:
[AWPY visibility](https://awpy.readthedocs.io/en/stable/visibility.html),
[versioned map data](https://github.com/pnxenopoulos/awpy-data),
[pinned parser release](https://github.com/markus-wa/demoinfocs-golang/releases/tag/v6.0.0-rc.1).
