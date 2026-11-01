"""Inspect local CS2 map assets without modifying game files or enabling LOS.

Requires a separately supplied Source2Viewer CLI. No downloads, shell commands,
runtime activation, nav-to-visibility conversion or coordinate conversion occur.
Exports are staged under --output and validated before final files are written.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile


NAV_MAGIC = 0xFEEDFACE
VPK_MAGIC = 0x55AA1234
TRANSFORM_CAVEAT = (
    "Raw POSITION accessor coordinates and composed glTF node/scene transforms are separate. "
    "Source2Viewer physics exports may contain axis rotation and a 0.0254 scale. "
    "Do not apply or remove transforms blindly. Validate world-coordinate alignment against "
    "known NAV/demo points before any collision use. This inventory does not establish a "
    "validated visibility mesh, wall materials/penetration model, route graph or semantic map."
)


def file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def validate_paths(cli, vpk, output):
    cli, vpk, output = (Path(value).resolve() for value in (cli, vpk, output))
    if not cli.is_file() or not vpk.is_file():
        raise ValueError("--cli and --vpk must name existing files")
    if vpk.suffix.lower() != ".vpk":
        raise ValueError("--vpk must be a local .vpk file")
    if output.exists() and not output.is_dir():
        raise ValueError("--output must be a directory, not a file")
    protected = [vpk.parent]
    if vpk.parent.name.lower() == "maps":
        protected.append(vpk.parent.parent)
    protected.extend(parent for parent in vpk.parents if parent.name.lower() in {"csgo", "game"})
    protected.extend(parent.parent for parent in vpk.parents if parent.name.lower() == "game")
    if any(output == root or output.is_relative_to(root) for root in protected):
        raise ValueError("--output must be outside the map/game input directory")
    with vpk.open("rb") as stream:
        header = stream.read(28)
    if len(header) < 12:
        raise ValueError("Truncated VPK header")
    magic, version = struct.unpack_from("<II", header)
    if magic != VPK_MAGIC or version not in (1, 2) or (version == 2 and len(header) < 28):
        raise ValueError("Expected a VPK version 1 or 2 header")
    return cli, vpk, output, version


def inspect_nav(path, cli_dump):
    with Path(path).open("rb") as stream:
        header = stream.read(12)
    if len(header) < 12:
        raise ValueError("Truncated NAV header")
    magic, version, subversion = struct.unpack("<III", header)
    if magic != NAV_MAGIC:
        raise ValueError("Invalid NAV magic; CLI output is not a navigation file")
    parsed_version = re.search(r"^Version:\s*(\d+)\s*$", cli_dump, re.MULTILINE)
    areas = re.search(r"^Number of areas:\s*(\d+)\s*$", cli_dump, re.MULTILINE)
    if not parsed_version or not areas or int(parsed_version.group(1)) != version:
        raise ValueError("Source2Viewer --all did not confirm the NAV version and area count")
    ladders = re.search(r"^Number of ladders:\s*(\d+)\s*$", cli_dump, re.MULTILINE)
    return {"version": version, "subversion": subversion, "area_count": int(areas.group(1)),
            "ladder_count": int(ladders.group(1)) if ladders else None,
            "scope": "Navigation inventory only; areas are not walls or validated sightlines"}


def inspect_place_entities(path):
    """Inventory actual env_cs_place entity transforms and brush-model references."""
    text = Path(path).read_text(encoding="utf-8")
    blocks = re.split(r"^====(\d+)====\s*$", text, flags=re.MULTILINE)
    places = []
    for index in range(1, len(blocks), 2):
        entity_index, body = int(blocks[index]), blocks[index + 1]

        def field(name):
            found = re.search(r"^" + re.escape(name) + r"\s+(.+)$", body, re.MULTILINE)
            return found.group(1).strip() if found else None

        if field("classname") != '"env_cs_place"':
            continue
        record = {"entity_index": entity_index}
        for name in ("place_name", "hammeruniqueid"):
            try:
                record[name] = json.loads(field(name))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Malformed place entity {entity_index}: {name}") from error
        for name in ("origin", "angles", "scales"):
            try:
                values = json.loads(field(name))
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"Malformed place entity {entity_index}: {name}") from error
            if (not isinstance(values, list) or len(values) != 3
                    or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values)):
                raise ValueError(f"Malformed place entity transform: {name}")
            record[name] = values
        model = field("model") or ""
        match = re.fullmatch(r'resource_name:(".*")', model)
        if not match:
            raise ValueError(f"Place entity {entity_index} has no brush-model reference")
        record["model"] = json.loads(match.group(1))
        record["compiled_model"] = record["model"] + "_c" if record["model"].endswith(".vmdl") else record["model"]
        places.append(record)
    return {"available": True, "place_entity_count": len(places),
            "place_name_counts": dict(Counter(place["place_name"] for place in places)),
            "places": places,
            "scope": "Actual place brush entities and model references, not nearest-label guesses. Volumes are not evaluated or activated here.",
            "coordinate_transform_caveat": "Entity origin/angles/scales must compose with model-local geometry. glTF node transforms additionally rotate/scale exported display coordinates. Nonzero entity rotations and containment require independent validation."}


def _integer(value, label):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"Invalid nonnegative integer: {label}")
    return value


def inspect_glb(path):
    """Validate GLB structure/accessor bounds and inventory declared triangles.

    Counts describe exported mesh primitives and node instances. They do not
    certify which surfaces block visibility or whether a transform is correct.
    """
    path = Path(path)
    size = path.stat().st_size
    chunks, document, binary_bytes = [], None, None
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError(f"Truncated GLB header: {path.name}")
        magic, version, declared_size = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2 or declared_size != size:
            raise ValueError(f"Invalid GLB magic/version/declared length: {path.name}")
        while stream.tell() < size:
            chunk_header = stream.read(8)
            if len(chunk_header) != 8:
                raise ValueError("Truncated GLB chunk header")
            length, kind = struct.unpack("<II", chunk_header)
            if length % 4 or stream.tell() + length > size:
                raise ValueError("GLB chunk length is misaligned or exceeds file bounds")
            if not chunks and kind != 0x4E4F534A:
                raise ValueError("GLB first chunk must be JSON")
            if kind == 0x4E4F534A:
                if document is not None or length > 64 * 1024 * 1024:
                    raise ValueError("Invalid duplicate or oversized GLB JSON chunk")
                document = json.loads(stream.read(length).decode("utf-8"))
            else:
                if kind == 0x004E4942:
                    if binary_bytes is not None:
                        raise ValueError("Duplicate GLB binary chunk")
                    binary_bytes = length
                stream.seek(length, 1)
            chunks.append({"kind": hex(kind), "bytes": length})
    if not isinstance(document, dict) or document.get("asset", {}).get("version") != "2.0":
        raise ValueError("Missing glTF 2.0 asset document")
    buffers = document.get("buffers", [])
    for index, buffer in enumerate(buffers):
        length = _integer(buffer.get("byteLength"), "buffer byteLength")
        if "uri" in buffer:
            raise ValueError("External GLB buffers are not inspected; expected a self-contained physics export")
        if index != 0 or binary_bytes is None or not length <= binary_bytes <= length + 3:
            raise ValueError("GLB embedded buffer length is inconsistent")
    views = document.get("bufferViews", [])
    for view in views:
        buffer_index = _integer(view.get("buffer"), "bufferView buffer")
        if buffer_index >= len(buffers):
            raise ValueError("GLB bufferView refers to a missing buffer")
        offset = _integer(view.get("byteOffset", 0), "bufferView byteOffset")
        length = _integer(view.get("byteLength"), "bufferView byteLength")
        if offset + length > buffers[buffer_index]["byteLength"]:
            raise ValueError("GLB bufferView exceeds its buffer")
    accessors = document.get("accessors", [])
    components = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
    for accessor in accessors:
        count = _integer(accessor.get("count"), "accessor count")
        if "sparse" in accessor:
            raise ValueError("Sparse accessor inspection is unsupported")
        view_index = _integer(accessor.get("bufferView"), "accessor bufferView")
        if view_index >= len(views):
            raise ValueError("GLB accessor refers to a missing bufferView")
        if accessor.get("componentType") not in components or accessor.get("type") not in widths:
            raise ValueError("Unsupported GLB accessor component/type")
        view = views[view_index]
        width = components[accessor["componentType"]] * widths[accessor["type"]]
        stride = _integer(view.get("byteStride", width), "accessor stride")
        offset = _integer(accessor.get("byteOffset", 0), "accessor offset")
        required = offset + (count - 1) * stride + width if count else offset
        if stride < width or required > view["byteLength"]:
            raise ValueError("GLB accessor exceeds its bufferView")

    def accessor_at(index):
        index = _integer(index, "primitive accessor")
        if index >= len(accessors):
            raise ValueError("GLB primitive refers to a missing accessor")
        return accessors[index]

    meshes, inventory, mode_counts = document.get("meshes", []), [], Counter()
    for mesh in meshes:
        triangles = 0
        for primitive in mesh.get("primitives", []):
            position = accessor_at(primitive.get("attributes", {}).get("POSITION"))
            if position["type"] != "VEC3":
                raise ValueError("Mesh POSITION accessor must be VEC3")
            indices = accessor_at(primitive["indices"]) if "indices" in primitive else position
            if "indices" in primitive and (indices["type"] != "SCALAR" or indices["componentType"] not in (5121, 5123, 5125)):
                raise ValueError("Invalid triangle index accessor type")
            count, mode = indices["count"], primitive.get("mode", 4)
            mode_counts[str(mode)] += 1
            if mode == 4:
                if count % 3:
                    raise ValueError("Triangle primitive index/vertex count is not divisible by three")
                triangles += count // 3
            elif mode in (5, 6):
                triangles += max(0, count - 2)
        inventory.append({"name": mesh.get("name"), "triangles": triangles})
    surface, tags, extra_keys, transforms = Counter(), Counter(), Counter(), Counter()
    instances = 0
    nodes = document.get("nodes", [])
    for node in nodes:
        if "mesh" in node:
            index = _integer(node["mesh"], "node mesh")
            if index >= len(inventory):
                raise ValueError("GLB node refers to a missing mesh")
            instances += inventory[index]["triangles"]
        extras = node.get("extras", {})
        if isinstance(extras, dict):
            extra_keys.update(extras.keys())
            if "SurfaceProperty" in extras:
                surface[str(extras["SurfaceProperty"])] += 1
            interaction = extras.get("InteractAs", [])
            if not isinstance(interaction, list):
                interaction = [interaction]
            tags.update(str(tag) for tag in interaction)
        transform = {key: node[key] for key in ("matrix", "translation", "rotation", "scale") if key in node}
        expected_lengths = {"matrix": 16, "translation": 3, "rotation": 4, "scale": 3}
        for key, values in transform.items():
            if (not isinstance(values, list) or len(values) != expected_lengths[key]
                    or any(not isinstance(value, (float, int)) or not math.isfinite(value) for value in values)):
                raise ValueError("Malformed GLB node transform")
        transforms[json.dumps(transform, sort_keys=True)] += 1
    triangles = sum(item["triangles"] for item in inventory)
    return {"version": version, "chunks": chunks, "mesh_count": len(meshes), "node_count": len(nodes),
            "mesh_triangle_count": triangles, "node_instance_triangle_count": instances,
            "has_geometry": triangles > 0, "meshes": inventory,
            "material_names": [material.get("name") for material in document.get("materials", [])],
            "surface_properties": dict(surface), "interaction_tags": dict(tags),
            "node_extra_keys": dict(extra_keys), "primitive_modes": dict(mode_counts),
            "node_transforms": [{"transform": json.loads(key), "node_count": count} for key, count in transforms.items()],
            "coordinate_transform_caveat": TRANSFORM_CAVEAT}


def inspect_map_vpk(cli, vpk, output, map_name=None, timeout=300):
    cli, vpk, output, vpk_version = validate_paths(cli, vpk, output)
    map_name = map_name or vpk.stem.removesuffix("_dir")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", map_name):
        raise ValueError("Map name must be a simple basename")
    output.mkdir(parents=True, exist_ok=True)
    names = [f"{map_name}.nav", f"{map_name}-physics.glb", f"{map_name}-physics_physics.glb",
             "nav-export.log", "physics-export.log", "nav-inspection.log", "cli-version.log", "inspection_manifest.json",
             "place-entities.txt", "place-export.log"]
    if any((output / name).exists() for name in names):
        raise FileExistsError("Use an empty output directory; inspection does not overwrite existing artifacts")
    source_vpk, source_cli = file_record(vpk), file_record(cli)
    commands = []
    with tempfile.TemporaryDirectory(prefix="vpk-inspection-", dir=output) as temp:
        stage = Path(temp)

        def run(arguments, log_name):
            command = [str(cli), *map(str, arguments)]
            result = subprocess.run(command, cwd=stage, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=timeout, check=False)
            log = result.stdout + ("\nSTDERR:\n" + result.stderr if result.stderr else "")
            (stage / log_name).write_text(log, encoding="utf-8")
            commands.append({"arguments": command, "exit_code": result.returncode, "log": log_name})
            if result.returncode:
                raise RuntimeError(f"Source2Viewer failed ({result.returncode}); {log[-2000:]}")
            return log

        nav = stage / names[0]
        physics = stage / names[1]
        companion = stage / names[2]
        version_dump = run(["--version"], "cli-version.log")
        cli_version = re.search(r"^Version:\s*(.+)$", version_dump, re.MULTILINE)
        if not cli_version:
            raise ValueError("Source2Viewer did not report its version")
        run(["-i", vpk, "-f", f"maps/{map_name}.nav", "-o", nav], "nav-export.log")
        if not nav.is_file():
            raise ValueError("CLI exited successfully without producing the expected NAV file")
        nav_dump = run(["-i", nav, "--all"], "nav-inspection.log")
        nav_data = inspect_nav(nav, nav_dump)
        run(["-i", vpk, "-f", f"maps/{map_name}/world_physics.vmdl_c", "-o", physics,
             "-d", "--gltf_export_format", "glb", "--gltf_export_extras"], "physics-export.log")
        if not physics.is_file():
            raise ValueError("CLI exited successfully without producing the expected physics GLB")
        glbs = [path for path in (physics, companion) if path.is_file()]
        glb_data = {path.name: inspect_glb(path) for path in glbs}
        if not any(data["has_geometry"] for data in glb_data.values()):
            raise ValueError("Export contains no triangles; an empty main GLB is not a usable physics export")
        place_dump = stage / "place-entities.txt"
        place_member = f"maps/{map_name}/entities/default_ents.vents_c"
        try:
            run(["-i", vpk, "-f", place_member, "-o", place_dump, "-d"], "place-export.log")
            if not place_dump.is_file():
                raise ValueError("Expected entity dump was not produced")
            place_data = inspect_place_entities(place_dump)
        except Exception as error:
            place_data = {"available": False, "reason": str(error), "source_member": place_member}
        if file_record(vpk)["sha256"] != source_vpk["sha256"] or file_record(cli)["sha256"] != source_cli["sha256"]:
            raise ValueError("An input file changed during inspection; refusing to certify provenance")
        manifest = {
            "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "map_name": map_name,
            "input_vpk": dict(source_vpk, vpk_version=vpk_version),
            "source2viewer_cli": dict(source_cli, reported_version=cli_version.group(1).strip()),
            "input_hashes_unchanged_after_export": True,
            "source_members": {"nav": f"maps/{map_name}.nav", "physics": f"maps/{map_name}/world_physics.vmdl_c", "places": place_member},
            "place_inventory": place_data,
            "commands": commands, "nav": nav_data, "glb_inventory": glb_data,
            "outputs": [], "coordinate_transform_caveat": TRANSFORM_CAVEAT,
            "runtime_activated": False, "game_files_modified": False,
            "limitations": ["Inspection and declared mesh topology only; no visibility, penetration or navigation graph is built.",
                            "A zero CLI exit code is insufficient; expected output files, NAV parsing and GLB bounds were validated.",
                            "Physics mesh names, SurfaceProperty and InteractAs tags are inventory, not an occluder classification."],
        }
        staged = [nav, *glbs, *(stage / name for name in ("nav-export.log", "physics-export.log", "nav-inspection.log", "cli-version.log"))]
        staged.extend(path for path in (place_dump, stage / "place-export.log") if path.is_file())
        for path in staged:
            record = file_record(path)
            record["path"] = str((output / path.name).resolve())
            manifest["outputs"].append(record)
        for path in staged:
            shutil.copyfile(path, output / path.name)
        destination = output / "inspection_manifest.json"
        destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cli", type=Path, required=True, help="Existing Source2Viewer-CLI executable")
    parser.add_argument("--vpk", type=Path, required=True, help="Local CS2 map VPK to read")
    parser.add_argument("--output", type=Path, required=True, help="Separate output directory outside the map/game directory")
    parser.add_argument("--map-name", help="Map basename if it differs from the VPK filename")
    parser.add_argument("--timeout", type=float, default=300, help="Per-CLI-call timeout in seconds")
    args = parser.parse_args(argv)
    try:
        result = inspect_map_vpk(args.cli, args.vpk, args.output, args.map_name, args.timeout)
        print(json.dumps({"manifest": str((args.output / 'inspection_manifest.json').resolve()),
                          "nav": result["nav"], "triangle_counts": {
                              name: data["mesh_triangle_count"] for name, data in result["glb_inventory"].items()},
                          "runtime_activated": False}, indent=2))
    except Exception as error:
        parser.exit(1, f"VPK inspection failed: {error}\n")


if __name__ == "__main__":
    main()
