# Collision geometry for direct engagements

The checked-in `.nav` files describe walking surfaces. They do not contain the
visual occluders required to decide whether two players can see each other.
The observer no longer creates artificial walls or substitutes a distance
threshold for line of sight. Without a collision asset it reports `unknown`.

Place one of these files in this directory, using the actual map basename:

- `de_dust2.mesh`: a modern AWPY AWMH v1 collision mesh. The native parser needs
  only NumPy; installing AWPY is not required. The observer loads this explicit
  local path; it does not download assets during a match.
- `de_dust2_collision.npz`: a NumPy archive containing `triangles`, a finite numeric
  array of shape `(N, 3, 3)`, or `vertices` of shape `(V, 3)` and integer `faces` of
  shape `(N, 3)`. Face indices must be valid and nonnegative. This format needs
  only NumPy and uses the built-in triangle BVH.

Both formats use world coordinates in Hammer/Source units, with Z up. Export
**visual occluders**, excluding sky and invisible player clips. Use geometry from
the same map/game version as the match or demo. Record the source, game version
and hash alongside imported assets; an old map mesh can produce wrong decisions.

The setup CLI installs selected maps, pins a ClientVersion, verifies the
upstream archive's SHA-256, validates each mesh, and records per-map provenance
and checksums. Run it explicitly before starting the observer:

```text
python KeylessOBS/setup_map_geometry.py --version 2000939 --maps de_dust2 de_inferno
```

`2000939` is an example verified during this audit; choose the ClientVersion
matching the match/demo. Existing assets require `--force` to replace. Use only
one collision format per map to avoid ambiguous precedence. The same command
can install a pre-downloaded release offline:

```text
python KeylessOBS/setup_map_geometry.py --version 2000939 --maps de_dust2 --archive geometry.zip --manifest manifest.json
```

Use `--output <directory>` for another destination and `--format npz` to convert
the validated upstream mesh to the NumPy format. Generated asset files are
runtime data and should not be committed automatically.

Official AWPY data and acquisition instructions:
[visibility and map data](https://awpy.readthedocs.io/en/stable/visibility.html),
[awpy-data](https://github.com/pnxenopoulos/awpy-data).
Alternatively, AWPY's documented `awpy get <ClientVersion>` fetches an asset release;
copy the desired `geometry/<map>.mesh` into this directory. Old AWPY `.tri`
assets are a different format and are not accepted as `.mesh` files.

For an already extracted collision triangle array:

```python
import numpy as np

# triangles must come from the real map export, not a navmesh or radar image.
np.savez_compressed("maps/de_dust2_collision.npz", triangles=triangles)
```

Restart the observer after adding/replacing assets. Programmatic callers can
explicitly reload using `manager.load_map_geometry("de_dust2")` after a previously
missing asset is installed. Loaded meshes are cached for repeated queries.

`MapGeometryManager.visibility(a, b)` accepts exact 3D ray endpoints and returns
`visible`, `blocked` or `unknown`, plus a reason and source path. Its compatibility
method `has_line_of_sight` returns `True`, `False` or `None`; callers must not treat
`None` as a proven obstruction.

`duel_geometrically_possible` accepts player **feet** positions. It casts from
each player's eye to sampled head/body positions, including torso sides. Standing
eye height is 64 units; the optional crouch flags use 46 units. This handles
elevation and partial cover better than one 2D line. The samples approximate the
player body; exact animated hitboxes are not modeled.

`blocked` means sampled **direct** sightlines are blocked by static geometry.
It does not mean a kill is physically impossible: bullet penetration, material
thickness, weapon damage, smoke, moving doors and destructible objects are not
modeled. Keep wallbang/event handling separate from direct-duel selection.
The upstream geometry may be decimated, so even static visibility can differ
slightly around thin edges and small gaps. Benchmark timings are measurements
on one machine and do not establish an accuracy score.

Run regression tests from the repository root:

```text
python -m unittest discover -s KeylessOBS/tests -p test_geometry.py -v
```

The tests cover wall intersections, both triangle windings, finite segments,
seams, coplanar rays, floors between levels, partial cover/crouching, missing
assets, map changes, invalid meshes, navigation-only data and full 3D aim angles.
