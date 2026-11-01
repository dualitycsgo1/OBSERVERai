"""Explicitly install selected, versioned AWPY collision assets.

Never imported by the observer. Downloads only occur when this CLI is run.
The upstream manifest's SHA-256 is checked before any mesh is installed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.request import Request, urlopen
import zipfile

import numpy as np

from line_of_sight import TriangleMeshLOS, normalize_map_name

_RELEASE_ROOT = "https://github.com/pnxenopoulos/awpy-data/releases/download"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url, target, max_bytes):
    request = Request(url, headers={"User-Agent": "OBSERVERai-geometry-setup"})
    total = 0
    with urlopen(request, timeout=60) as response, Path(target).open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError("Downloaded geometry asset exceeds the supported size")
            output.write(chunk)


def install_geometry(version, maps, output_dir, archive_path, manifest_path,
                     output_format="mesh", force=False):
    """Validate a local release pair, then install selected map meshes atomically."""
    version = str(version)
    if not re.fullmatch(r"[0-9]+", version):
        raise ValueError("version must be an explicit numeric CS2 ClientVersion")
    if output_format not in {"mesh", "npz"}:
        raise ValueError("format must be mesh or npz")
    selected = list(dict.fromkeys(normalize_map_name(name) for name in maps))
    if not selected:
        raise ValueError("Select at least one map")
    output_dir = Path(output_dir)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if str(manifest.get("release_key")) != version or str(manifest.get("client_version")) != version:
        raise ValueError("Manifest ClientVersion/release_key does not match requested version")
    artifact = manifest.get("artifacts", {}).get("geometry.zip", {})
    expected = artifact.get("sha256", "")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
        raise ValueError("Manifest is missing a valid geometry.zip SHA-256")
    actual = sha256_file(archive_path)
    if actual != expected.lower():
        raise ValueError("geometry.zip SHA-256 does not match the release manifest")
    if Path(archive_path).stat().st_size != artifact.get("bytes"):
        raise ValueError("geometry.zip byte length does not match the release manifest")
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with zipfile.ZipFile(archive_path) as archive, tempfile.TemporaryDirectory(dir=output_dir) as scratch_dir:
        scratch = Path(scratch_dir)
        # Validate every selected asset before changing any installed file.
        for map_name in selected:
            member = f"{map_name}.mesh"
            info = archive.getinfo(member)
            if info.file_size > 256 * 1024 * 1024:
                raise ValueError(f"Collision mesh is too large: {member}")
            raw = archive.read(member)
            local = scratch / member
            local.write_bytes(raw)
            mesh = TriangleMeshLOS.from_awpy_mesh(local)
            filename = member if output_format == "mesh" else f"{map_name}_collision.npz"
            destination = output_dir / filename
            sidecar = output_dir / f"{map_name}_collision_source.json"
            other_format = output_dir / (f"{map_name}_collision.npz" if output_format == "mesh" else member)
            if other_format.exists():
                raise FileExistsError(f"Remove the other collision format explicitly to avoid ambiguous precedence: {other_format}")
            if not force and (destination.exists() or sidecar.exists()):
                raise FileExistsError(f"Asset already installed; use --force to replace: {destination}")
            if output_format == "npz":
                local = scratch / filename
                np.savez_compressed(local, triangles=mesh.triangles)
            metadata = {
                "map": map_name,
                "client_version": version,
                "patch_version": manifest.get("patch_version"),
                "source_revision": manifest.get("source_revision"),
                "buildid": manifest.get("buildid"),
                "release_url": f"https://github.com/pnxenopoulos/awpy-data/releases/tag/{version}",
                "archive_sha256": actual,
                "upstream_mesh_sha256": hashlib.sha256(raw).hexdigest(),
                "installed_sha256": sha256_file(local),
                "triangle_count": len(mesh.triangles),
                "installed_at": datetime.now(timezone.utc).isoformat(),
                "limitations": "Static visual occluders; no bullet penetration, smoke or moving objects",
            }
            metadata_file = scratch / sidecar.name
            metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            results.append((local, destination, metadata_file, sidecar, metadata))
        for local, destination, metadata_file, sidecar, _ in results:
            local.replace(destination)
            metadata_file.replace(sidecar)
    return [metadata for _, _, _, _, metadata in results]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Explicit CS2 ClientVersion / awpy-data release tag")
    parser.add_argument("--maps", nargs="+", required=True, help="Map basenames, e.g. de_dust2 de_nuke")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "maps")
    parser.add_argument("--format", choices=("mesh", "npz"), default="mesh")
    parser.add_argument("--archive", type=Path, help="Existing geometry.zip (offline installation)")
    parser.add_argument("--manifest", type=Path, help="Matching manifest.json (required with --archive)")
    parser.add_argument("--force", action="store_true", help="Replace existing assets of the selected format")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[0-9]+", args.version):
        parser.error("--version must be a numeric ClientVersion")
    if bool(args.archive) != bool(args.manifest):
        parser.error("--archive and --manifest must be supplied together")
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output) as scratch_dir:
            scratch = Path(scratch_dir)
            archive, manifest = args.archive, args.manifest
            if archive is None:
                manifest, archive = scratch / "manifest.json", scratch / "geometry.zip"
                _download(f"{_RELEASE_ROOT}/{args.version}/manifest.json", manifest, 1024 * 1024)
                _download(f"{_RELEASE_ROOT}/{args.version}/geometry.zip", archive, 512 * 1024 * 1024)
            results = install_geometry(args.version, args.maps, args.output, archive, manifest,
                                       output_format=args.format, force=args.force)
        print(json.dumps(results, indent=2))
    except Exception as exc:
        parser.exit(1, f"Geometry installation failed: {exc}\n")


if __name__ == "__main__":
    main()
