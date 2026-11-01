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

## Sampled camera replay: the remaining gap

The actual `process_gamestate` state machine was replayed on both held-out demos
using their sample times, the exact canonical model, simulated spectator slots,
and blocked native keyboard APIs. Matching patch meshes were installed for each
map. A control run used the declared unknown-geometry fallback.

| Strict next-killer opportunity, 942 windows | Matched collision mesh | Unknown geometry fallback |
| --- | ---: | ---: |
| Raw model's top-ranked player | 48.83% | 48.83% |
| Next killer is present in the current duel candidate pool | 54.99% | 98.94% |
| Model's top-ranked player inside that pool | 31.32% | 48.83% |
| Camera's held target at the sample time | 25.16% (237/942) | 27.92% (263/942) |

Each variant processed 5,558 snapshots. All four complete runs had zero processing
errors, zero native input attempts, and zero mismatches between the camera target
and the detector's internal target. Matched-mesh runs emitted 399 switch keys for
399 actual target changes; fallback runs emitted 615 for 615. The former no-op
bug, which reset cooldown while reselecting the same player, is covered by
regression tests and absent from these runs.

This exposes a real architectural gap. A currently blocked sightline does not
mean that a kill two seconds later is impossible: players may peek. The current
duel pool combines current visibility with range/facing policy, excluding the
next killer from roughly 45% of action windows. The table separates that loss
from the additional effect of camera holds, switching cooldown and other policy.
The default 4.2-second switch cooldown is also longer than the model's two-second
forecast horizon. These policies were not tuned to the two validation demos.

Mean processing time was 54.7 ms with mesh and 12.1 ms with fallback on this
machine during concurrent runs, including map loading. These are offline
processing measurements, not end-to-end live latency.

The export contains only living players and complete live-round horizons;
freeze/death transition states, round kill counters, final partial horizons and
real GSI/input delivery are absent. There were 233 matched-mesh snapshots whose
held target was absent from the living-player export, versus 5 in fallback.
Consequently the percentages measure held targets at sampled times, not rendered
POV kill capture. They do not establish an improvement over the old camera system.

The next engineering work should distinguish a verified **current direct duel**
from a **feasible future peek** using movement/reachable positions and geometry,
then evaluate a camera policy with a compatible forecast horizon on fresh demos.
This revision is an audited foundation and a draft for review, not evidence of a
production-ready broadcast observer.

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

Local checks passed: 86 Python tests and 4 subtests, 7 Go tests, a source parser
build, all five compressed real demos, the isolated evaluation CLI, and four
complete held-out replay runs. Automated validation commands are in README and
the Windows CI workflow. The
source build is verified. Packaged observer/trainer EXEs and interaction with a
running CS2 spectator session were not validated in this audit.

Official references:
[AWPY visibility](https://awpy.readthedocs.io/en/stable/visibility.html),
[versioned map data](https://github.com/pnxenopoulos/awpy-data),
[pinned parser release](https://github.com/markus-wa/demoinfocs-golang/releases/tag/v6.0.0-rc.1).
