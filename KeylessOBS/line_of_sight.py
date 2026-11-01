"""Static 3D collision visibility for CS2 (world units, Z up).

A navigation mesh describes walkable surfaces, not walls that block shots.
Only collision meshes are accepted here. Missing assets produce ``unknown``.
Visibility is not bullet penetration, smoke visibility, or a guarantee of a kill.
"""

from dataclasses import dataclass
import logging
from math import atan2, degrees, hypot
from pathlib import Path
import re
import struct
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
EYE_HEIGHT = 64.0
CROUCH_EYE_HEIGHT = 46.0
_EPSILON = 1e-6


def _point(value):
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("A position must contain three finite world coordinates")
    return point


def normalize_map_name(map_name: str) -> str:
    """Accept a basename or workshop path without allowing filesystem traversal."""
    name = str(map_name or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    for suffix in (".vpk", ".bsp"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    if not re.fullmatch(r"[a-z0-9_-]+", name):
        raise ValueError("Invalid map name")
    if name in {"dust2", "mirage", "inferno", "nuke", "ancient", "anubis",
                "overpass", "train", "vertigo", "cache"}:
        name = "de_" + name
    return name


@dataclass(frozen=True)
class VisibilityResult:
    state: str  # visible, blocked, unknown
    reason: str
    source: Optional[str] = None

    @property
    def visible(self) -> Optional[bool]:
        return {"visible": True, "blocked": False, "unknown": None}[self.state]


@dataclass
class _Node:
    lower: np.ndarray
    upper: np.ndarray
    indices: Optional[np.ndarray] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None


class TriangleMeshLOS:
    """Double-sided finite-segment ray casting with a triangle BVH.

    ``triangles`` has shape (N, 3, 3), in the same coordinate frame as GSI.
    The mesh must contain visual occluders; player clips/sky must be excluded.
    """

    def __init__(self, triangles, leaf_size: int = 24):
        triangles = np.asarray(triangles, dtype=np.float64)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
            raise ValueError("Collision triangles must have nonempty shape (N, 3, 3)")
        if not np.isfinite(triangles).all():
            raise ValueError("Collision triangles contain non-finite coordinates")
        if leaf_size < 1:
            raise ValueError("leaf_size must be positive")
        edge1 = triangles[:, 1] - triangles[:, 0]
        edge2 = triangles[:, 2] - triangles[:, 0]
        valid = np.linalg.norm(np.cross(edge1, edge2), axis=1) > _EPSILON
        self.triangles = triangles[valid].copy()
        if not len(self.triangles):
            raise ValueError("Collision mesh contains no nondegenerate triangles")
        self._lower = self.triangles.min(axis=1)
        self._upper = self.triangles.max(axis=1)
        self._centers = (self._lower + self._upper) * 0.5
        self._leaf_size = leaf_size
        self.root = self._build(np.arange(len(self.triangles)))

    @classmethod
    def from_awpy_mesh(cls, path):
        """Read AWPY AWMH v1 indexed collision meshes without optional AWPY.

        Format: https://github.com/pnxenopoulos/awpy-data#collision-meshes
        All integer/float fields are little endian; coordinates are Z-up.
        """
        raw = Path(path).read_bytes()
        if len(raw) < 16:
            raise ValueError("Truncated AWMH collision mesh")
        magic, version, vertex_count, triangle_count = struct.unpack_from("<4sIII", raw)
        if magic != b"AWMH" or version != 1:
            raise ValueError("Expected an AWPY AWMH version 1 collision mesh")
        expected = 16 + vertex_count * 12 + triangle_count * 12
        if len(raw) != expected:
            raise ValueError("AWMH byte length does not match its vertex/triangle counts")
        vertices = np.frombuffer(raw, dtype="<f4", count=vertex_count * 3, offset=16).reshape(-1, 3)
        faces = np.frombuffer(raw, dtype="<u4", count=triangle_count * 3,
                              offset=16 + vertex_count * 12).reshape(-1, 3)
        if not vertex_count or not triangle_count or (faces >= vertex_count).any():
            raise ValueError("AWMH mesh has empty or out-of-range triangle indices")
        return cls(vertices[faces])

    def _build(self, indices):
        node = _Node(self._lower[indices].min(axis=0), self._upper[indices].max(axis=0))
        if len(indices) <= self._leaf_size:
            node.indices = indices
            return node
        axis = int(np.argmax(np.ptp(self._centers[indices], axis=0)))
        split = len(indices) // 2
        order = np.argpartition(self._centers[indices, axis], split)
        node.left = self._build(indices[order[:split]])
        node.right = self._build(indices[order[split:]])
        return node

    @staticmethod
    def _box_hit(node, origin, direction, length):
        near, far = 0.0, length
        for axis in range(3):
            if abs(direction[axis]) < 1e-12:
                if origin[axis] < node.lower[axis] - _EPSILON or origin[axis] > node.upper[axis] + _EPSILON:
                    return False
            else:
                a = (node.lower[axis] - origin[axis]) / direction[axis]
                b = (node.upper[axis] - origin[axis]) / direction[axis]
                near, far = max(near, min(a, b)), min(far, max(a, b))
                if far < near - _EPSILON:
                    return False
        return True

    @staticmethod
    def _coplanar_hit(triangle, origin, direction, length):
        """Resolve rays along a triangle plane instead of silently skipping them."""
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        if abs(np.dot(normal, origin - triangle[0])) > _EPSILON * np.linalg.norm(normal):
            return False
        drop = int(np.argmax(np.abs(normal)))
        tri = np.delete(triangle, drop, axis=1)
        a = np.delete(origin + direction * _EPSILON, drop)
        b = np.delete(origin + direction * (length - _EPSILON), drop)

        def cross(u, v):
            return u[0] * v[1] - u[1] * v[0]

        def inside(p):
            signs = [cross(tri[(i + 1) % 3] - tri[i], p - tri[i]) for i in range(3)]
            return min(signs) >= -_EPSILON or max(signs) <= _EPSILON

        if inside(a) or inside(b):
            return True
        for i in range(3):
            c, d = tri[i], tri[(i + 1) % 3]
            ab, cd = b - a, d - c
            denominator = cross(ab, cd)
            if abs(denominator) > _EPSILON:
                t, u = cross(c - a, cd) / denominator, cross(c - a, ab) / denominator
                if 0 <= t <= 1 and 0 <= u <= 1:
                    return True
            elif abs(cross(c - a, ab)) <= _EPSILON:
                axis = int(np.argmax(np.abs(ab)))
                if max(min(a[axis], b[axis]), min(c[axis], d[axis])) <= min(max(a[axis], b[axis]), max(c[axis], d[axis])):
                    return True
        return False

    def _triangles_hit(self, indices, origin, direction, length):
        triangles = self.triangles[indices]
        edge1 = triangles[:, 1] - triangles[:, 0]
        edge2 = triangles[:, 2] - triangles[:, 0]
        h = np.cross(direction, edge2)
        determinant = np.einsum("ij,ij->i", edge1, h)
        scale = np.linalg.norm(np.cross(edge1, edge2), axis=1)
        parallel = np.abs(determinant) <= 1e-10 * scale
        usable = ~parallel
        if usable.any():
            inverse = 1.0 / determinant[usable]
            s = origin - triangles[usable, 0]
            u = inverse * np.einsum("ij,ij->i", s, h[usable])
            q = np.cross(s, edge1[usable])
            v = inverse * (q @ direction)
            distance = inverse * np.einsum("ij,ij->i", edge2[usable], q)
            if np.any((u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10)
                      & (distance > _EPSILON) & (distance < length - _EPSILON)):
                return True
        return any(self._coplanar_hit(t, origin, direction, length) for t in triangles[parallel])

    def has_line_of_sight(self, pos1, pos2) -> bool:
        origin, target = _point(pos1), _point(pos2)
        delta = target - origin
        length = float(np.linalg.norm(delta))
        if length <= 2 * _EPSILON:
            return True
        direction = delta / length
        stack = [self.root]
        while stack:
            node = stack.pop()
            if not self._box_hit(node, origin, direction, length):
                continue
            if node.indices is not None:
                if self._triangles_hit(node.indices, origin, direction, length):
                    return False
            else:
                stack.extend((node.left, node.right))
        return True


class MapGeometryManager:
    """Load real local collision assets; never generate fictional geometry.

    Supports ``<map>_collision.npz`` (``triangles`` or ``vertices`` + integer
    ``faces``) and AWPY ``<map>.mesh``. Default paths resolve
    beside this module, independent of the process working directory.
    """

    def __init__(self, maps_dir: Optional[str] = None):
        self.maps_dir = Path(maps_dir) if maps_dir is not None else Path(__file__).resolve().parent / "maps"
        self.los_checkers: Dict[str, object] = {}
        self.sources: Dict[str, str] = {}
        self.errors: Dict[str, str] = {}
        self.current_map: Optional[str] = None
        self.use_real_geometry = False

    def load_map_geometry(self, map_name: str) -> bool:
        map_name = normalize_map_name(map_name)
        self.current_map = map_name
        if map_name in self.los_checkers:
            self.use_real_geometry = True
            return True
        npz_path = self.maps_dir / f"{map_name}_collision.npz"
        mesh_path = self.maps_dir / f"{map_name}.mesh"
        try:
            if npz_path.is_file():
                with np.load(npz_path, allow_pickle=False) as data:
                    if "triangles" in data:
                        triangles = data["triangles"]
                    else:
                        vertices, faces = data["vertices"], data["faces"]
                        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
                            raise ValueError("vertices/faces must have shape (N, 3)")
                        if not np.issubdtype(faces.dtype, np.integer) or (faces < 0).any() or (faces >= len(vertices)).any():
                            raise ValueError("faces must contain valid nonnegative integer vertex indices")
                        triangles = vertices[faces]
                    checker = TriangleMeshLOS(triangles)
                source = str(npz_path)
            elif mesh_path.is_file():
                # Native parsing needs no AWPY install or runtime download.
                checker = TriangleMeshLOS.from_awpy_mesh(mesh_path)
                source = str(mesh_path)
            else:
                raise FileNotFoundError(f"No collision mesh for {map_name}; .nav and 2D blockmasks cannot establish 3D visibility")
            self.los_checkers[map_name] = checker
            self.sources[map_name] = source
            self.errors.pop(map_name, None)
            self.use_real_geometry = True
            logger.info("Loaded collision geometry for %s from %s", map_name, source)
            return True
        except Exception as exc:
            self.errors[map_name] = str(exc)
            self.use_real_geometry = False
            logger.warning("Geometry unavailable for %s: %s", map_name, exc)
            return False

    def set_current_map(self, map_name: str):
        try:
            normalized = normalize_map_name(map_name)
        except ValueError:
            self.current_map = None
            self.use_real_geometry = False
            return False
        # Failed loads remain unknown; switching never retains old geometry.
        if normalized == self.current_map and normalized in self.errors:
            return False
        return self.load_map_geometry(normalized)

    def visibility(self, pos1, pos2) -> VisibilityResult:
        try:
            _point(pos1)
            _point(pos2)
        except (TypeError, ValueError):
            return VisibilityResult("unknown", "Invalid or missing world position")
        checker = self.los_checkers.get(self.current_map)
        if checker is None:
            return VisibilityResult("unknown", self.errors.get(self.current_map, "No current collision mesh"))
        try:
            if hasattr(checker, "is_visible"):
                clear = bool(checker.is_visible(tuple(pos1), tuple(pos2)))
            else:
                clear = checker.has_line_of_sight(pos1, pos2)
            return VisibilityResult("visible" if clear else "blocked", "Static collision mesh ray cast", self.sources.get(self.current_map))
        except Exception as exc:
            logger.warning("Collision query failed: %s", exc)
            return VisibilityResult("unknown", f"Collision query failed: {exc}", self.sources.get(self.current_map))

    def has_line_of_sight(self, pos1, pos2) -> Optional[bool]:
        """Tri-state API. Callers must distinguish None from False."""
        return self.visibility(pos1, pos2).visible


def aim_angles(origin, target) -> Tuple[float, float]:
    """Return Source-style (pitch, yaw) degrees for world positions."""
    dx, dy, dz = _point(target) - _point(origin)
    return -degrees(atan2(dz, hypot(dx, dy))), degrees(atan2(dy, dx))


def aim_error_degrees(forward, origin, target) -> Optional[float]:
    """3D angle from a GSI forward vector to a target; None for zero vectors."""
    try:
        view, delta = _point(forward), _point(target) - _point(origin)
    except (TypeError, ValueError):
        return None
    magnitude = float(np.linalg.norm(view) * np.linalg.norm(delta))
    if magnitude <= _EPSILON:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(view, delta) / magnitude, -1.0, 1.0))))


def duel_geometrically_possible(pos_a, pos_b, geometry_manager: MapGeometryManager,
                               max_2d_distance: Optional[float] = None,
                               min_2d_distance: float = 0.0,
                               crouched_a: bool = False, crouched_b: bool = False) -> Optional[bool]:
    """Whether sampled direct-fire sightlines exist in either direction.

    Inputs are feet positions. Sample head, torso, pelvis and torso sides from
    each player's eye to allow partial cover and asymmetric height advantages.
    False means sampled rays are occluded, not that wallbangs are impossible.
    None means geometry/positions are unknown. Optional range limits are caller
    policy rather than a claim about physically possible weapon kills.
    """
    try:
        a, b = _point(pos_a), _point(pos_b)
    except (TypeError, ValueError):
        return None
    distance = hypot(*(a[:2] - b[:2]))
    if distance < min_2d_distance or (max_2d_distance is not None and distance > max_2d_distance):
        return False
    unknown = False
    for source, target, crouched_source, crouched_target in ((a, b, crouched_a, crouched_b), (b, a, crouched_b, crouched_a)):
        eye_height = CROUCH_EYE_HEIGHT if crouched_source else EYE_HEIGHT
        target_eye = CROUCH_EYE_HEIGHT if crouched_target else EYE_HEIGHT
        eye = source + (0, 0, eye_height)
        for offset in ((0, 0, target_eye), (0, 0, target_eye * .72), (0, 0, target_eye * .45),
                       (-12, 0, target_eye * .72), (12, 0, target_eye * .72),
                       (0, -12, target_eye * .72), (0, 12, target_eye * .72)):
            result = geometry_manager.has_line_of_sight(eye, target + offset)
            if result is True:
                return True
            unknown = unknown or result is None
    return None if unknown else False


_geometry_manager = None


def get_geometry_manager() -> MapGeometryManager:
    global _geometry_manager
    if _geometry_manager is None:
        _geometry_manager = MapGeometryManager()
    return _geometry_manager
