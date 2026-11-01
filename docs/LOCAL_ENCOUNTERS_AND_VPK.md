# Local encounter continuity and installed-map inspection

## Reported tunnel-rush case

A locally supplied 29.3-second trial recording showed the camera leaving a
living Dual Berettas defender on Dust2 B as multiple opponents emerged from
tunnels. It switched to a defender at mid, missing the first defender's death,
then returned to B later. The original trial log contained slot dispatches but
no GSI history or decision reasons, so the exact live input sequence cannot be
reconstructed from that log.

A controlled replay of the corresponding demo round reproduces the same bad
choice. It uses causal exported snapshots every approximately 0.5 seconds,
the same model/mesh and an explicitly selected initial POV. This demo was part
of training: it is a bug diagnostic, not an independent accuracy experiment.

At the critical sample, two attackers were approximately 166 and 333 world units
from the B defender and closing. The mid defender's nearest opponent was about
1,194 units away. Both defenders met the old `anchor` role rule. A nearby teammate
had just moved beyond the mid defender's 700-unit support threshold, while that
defender's health fell from 100 to 67 beside a molotov. The active-damage flag
then won the role tie before setup quality or model probability was considered.
The model itself ranked the B defender higher; this was an editorial policy
failure. Subsequent engagement/trade holds prolonged the view at mid.

The controlled baseline switches at tick 11054; the video first clearly shows
mid around tick 11087. This establishes the reproduced mechanism and nearby
timing, not exact live GSI cadence or input/render synchronization.

## Correction

The policy now remembers a local encounter for the selected player when at least
two known opponents are inside the existing local radius. It stays with that
encounter while an original local opponent remains nearby, even before shots,
during a firing pause or when a teammate is nearby. The radius remains 700 units;
there are no map, player or weapon-specific exceptions.

Unknown local telemetry is distinct from a confirmed empty set. Confirmed
disengagement, dead opponents and round boundaries release the encounter.
Identified objective actions can still interrupt. A confirmed death leaves the
dead POV and first follows a local opponent or linked defender, rather than
letting an unrelated damage event select another area.

In the same controlled case, the corrected policy holds the B defender at ticks
11021 and 11054, preserves his engagement at 11087, and follows a local B attacker
after the confirmed death at 11119. It later follows another B defender. This
fixes the reproduced case; it does not establish general broadcast quality.

Regressions cover remote damage, stopped attackers, reload pauses, nearby
support, partial telemetry, confirmed deaths, disengagement, objective overrides,
round resets and failed dispatch. Local proximity remains an approximation:
players close in Euclidean space can be separated by walls or floors.

The trial wrapper also records buffered local gzip JSONL sessions on the next
manual start, including GSI without auth/token fields, timestamps, decisions,
dispatch outcomes and model/source fingerprints. It adds no network destination.
The application is not automatically launched or restarted by the update.

## Obsolete Positional AI integration

The former runtime tried to load a separate `enhanced_positional_models` folder.
That model family was absent. Its legacy call to `predict_engagement_quality`
also did not match the historical class's API. The import, initialization,
readiness label and scoring hook have been removed from the observer runtime.
The standalone historical file is retained but is no longer imported there.

The validated `observer_model.py` future-kill model continues to use position,
3D facing, distance and alive-player context. Collision raycasting is a separate
map service. Removing the obsolete component neither invents another trained
model nor changes the current model's weights or schema.

## What the installed VPK contains

The local Dust2 VPK was inspected read-only using the official Source 2 Viewer
CLI 20.1, with its downloaded archive SHA-256 verified as
`67aaf8a67a85ca60fd09ff343e62103a4cdf3a2d177fc656231d7d734c5b3aa3`.
The installed game reports ClientVersion 2000939 / patch 1.41.9.4.

Observed assets include:

- `maps/de_dust2.nav`: 481,872 bytes, navigation version 36, 2,242 areas.
- `maps/de_dust2/world_physics.vmdl_c`: 21,843,224 bytes compiled physics.
- World and entity resources alongside those files.

Exported physics contains 435,649 triangles in 39 groups. Surface properties
include concrete, wood, glass and metal. Collision tags include `playerclip`,
`npcclip`, `csgo_grenadeclip`, `sky` and `passbullets`. This is richer provenance
and material data than an anonymous triangle soup. The inspector records these
facts and hashes in a local manifest; game binaries are not committed here.

The entity export also contains **43 actual `env_cs_place` entities**, including
`BombsiteB`, `UpperTunnel` and `MidDoors`, with origins, rotations and referenced
volume models. Those three referenced models were independently exported and
checked as closed, convex volumes (12 triangles, eight unique vertices). For
these zero-rotation, unit-scale examples, raw local vertices plus entity origin
give world-space volumes. This provides a concrete route from GSI coordinates to
named places without guessing a label from the nearest map point. The inspector
exports the entity inventory; automatic runtime containment labels and general
entity rotations are not yet implemented or validated.

GSI positions already use game-world coordinates. Useful additional layers are
mapping those coordinates to navigation polygons and routes, separating nearby
players through floors/walls, recognizing connected approach corridors, and
using material/thickness information in a separately validated penetration
model. Navigation describes movement; it is not line-of-sight evidence.

The glTF export also contains axis/unit transforms (including a 0.0254 scale),
and collision groups that should not block bullet sightlines. Importing it
blindly would produce incorrect walls and coordinates. The exported local assets
therefore do **not** automatically replace the tested collision mesh. NAV v36
also needs a compatible parser before routing can be used. Material names alone
do not establish CS2 penetration damage or a kill probability.

References: [official CLI documentation](https://github.com/ValveResourceFormat/ValveResourceFormat/blob/20.1/docs/guides/command-line.md),
[Source 2 format/export limitations](https://s2v.app/ValveResourceFormat/guides/format-support),
[versioned navigation reader](https://github.com/ValveResourceFormat/ValveResourceFormat/blob/20.1/ValveResourceFormat/NavMesh/NavMeshFile.cs).
