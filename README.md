# OBSERVERai

A Windows CS2 observer that selects players using game-state data, static map
collision visibility, and an optional trained two-second kill forecast.

This source checkout does **not** contain a trained kill model or collision
meshes. Without them the observer uses explicitly identified heuristic fallbacks.
Historical claims of 85–96% kill prediction accuracy were not established by the
old code: its trainer predicted headshots and used different features from live
inference. See [the technology audit](docs/TECHNOLOGY_AUDIT.md) for evidence,
changes, measurements, and remaining limitations.

The final two-demo test gives 48.8% static next-killer ranking versus 28.7% for a
nearest-enemy baseline. Sampled camera replay with mesh reaches 25.2%; current
duel filtering and camera policy remain substantial limits. This revision is a
draft for review, not a validated broadcast release.

## Run from source

Use Windows, Python 3.12, and Go 1.27 or newer. Commands below start at the
repository root:

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r KeylessOBS/requirements.txt
cd KeylessOBS
go build -o demo_parser_with_positions.exe .
python ai_trainer_gui.py
# In a second terminal with the same environment:
python cs2_duel_detector.py
```

The parser pins demoinfocs v6.0.0-rc.1 because the previous v5.0.2 dependency failed
on the real demo files used in this audit. This dependency is a release candidate;
source build and real-demo compatibility were checked, but a packaged EXE release
has not been validated. Older release/build scripts reference historical assets
absent from this checkout. Use the source workflow above for this revision.

Copy `KeylessOBS/gamestate_integration_obs_updated.cfg` to your CS2 installation's
`game/csgo/cfg` folder and restart CS2. The observer requires spectator/GOTV GSI
data including `allplayers`, positions, and forward vectors. Its console `status`
command identifies whether collision geometry and a validated model are active.

## Map geometry

Install collision data for the **same map/game version** as the match:

```powershell
python KeylessOBS/setup_map_geometry.py --version 2000939 --maps de_dust2 de_inferno
```

`2000939` is the version verified in the audit, not a promise that it matches
every demo. The setup command verifies archive hashes and records provenance.
The observer never downloads geometry while a match runs.

- `.nav` files describe walking surfaces and cannot establish line of sight.
- Actual triangle meshes provide 3D ray intersections. Eye-to-body samples allow
  partial cover; aiming scores include pitch and yaw.
- Missing meshes produce `unknown`, separate from `visible` and `blocked`.
- `geometry.require_verified_visibility` in `KeylessOBS/config.json` defaults to
  `false`, allowing a declared heuristic fallback. Set it to `true` to require
  verified visibility for direct-duel candidates.
- Blocked static sightlines do not prove a kill impossible: bullet penetration,
  smoke, moving doors, and precise animated hitboxes are not simulated.

See [collision setup and API details](KeylessOBS/maps/COLLISION_GEOMETRY.md).

## Demo training and evaluation

Import `.dem` or `.dem.zst` files in the trainer. Archives are streamed to a
temporary file; source demos are preserved. The parser samples live rounds every
0.5 seconds and labels each living player by whether they get at least one enemy
kill in the **next two seconds**. Same-tick outcomes, teamkills, warmup/freezetime,
and incomplete round-end/EOF horizons are excluded from prediction labels.

Training and live inference use one versioned 17-feature schema. Kill-only and
legacy headshot models are rejected. The Random Forest is refitted on accumulated,
deduplicated data; it is not an incremental-learning algorithm.

At least five independent demos and 100 valid rows are required. A persistent
split keeps entire demos out of training. Candidate evaluation compares Brier
score, log loss, average precision, and top-ranked player kill coverage with
baselines and, where present, the deployed model. Improvement intervals resample
whole demos. Rejected candidates leave the deployed model unchanged.

Accepted bundles are saved atomically to
`KeylessOBS/models/observer_ai_model.pkl`. The observer polls the file every
30 seconds and validates the target, feature order, classes, and evaluation
before swapping. A failed reload retains the last valid model.

More data is **not** evidence of improvement. Review
`KeylessOBS/training_data/latest_evaluation.json` after training. Fixed validation
data can itself be overfit by repeated selection; use fresh, untouched demos for
a final performance claim. Forecast scores and snapshot ranking do not measure
the complete camera system's kill capture rate under delays and camera locks.

For a reproducible experiment in a new, isolated output directory, using raw
parser JSON exports:

```powershell
python KeylessOBS/evaluate_demos.py --results-dir work/parser-results --output-dir work/evaluation-new --learning-curve --strict-next-killer
```

See [the recorded measurements](docs/measurements.json) for the audit's model
comparison and the audit report for its limits.

## Checks

```powershell
cd KeylessOBS
python -m pytest tests test_demo_processing.py -q
go test ./...
go build -o demo_parser_with_positions.exe .
```

Tests cover real geometry intersections and unknown states, causal label
boundaries, compressed imports, training/inference parity, grouped evaluation,
model rejection/promotion, and live hot reload. Synthetic test data verifies
mechanics; it is never reported as empirical match accuracy.
