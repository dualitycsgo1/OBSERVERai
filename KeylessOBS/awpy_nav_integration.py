"""Navigation availability, kept separate from collision visibility.

The former adapter claimed to load AWPY geometry but used distance/height
heuristics. A .nav file alone cannot answer line-of-sight queries. Consumers
needing visibility must use line_of_sight.MapGeometryManager with collision data.
"""

import math
from pathlib import Path
from typing import Optional, Tuple

from line_of_sight import normalize_map_name


class AWPYNavGeometryManager:
    """Compatibility adapter that explicitly reports unavailable nav queries."""

    def __init__(self, maps_dir: Optional[str] = None):
        self.maps_dir = Path(maps_dir) if maps_dir else Path(__file__).resolve().parent / "maps"
        self.current_map = None
        self.nav_loader = None
        self.nav_data = {}
        self.awpy_available = False
        try:
            import awpy
            self.awpy = awpy
            self.awpy_available = True
        except ImportError:
            self.awpy = None

    def set_current_map(self, map_name: str) -> bool:
        try:
            self.current_map = normalize_map_name(map_name)
        except ValueError:
            self.current_map = None
            return False
        # Asset availability does not mean a navigation mesh was parsed.
        return (self.maps_dir / f"{self.current_map}.nav").is_file()

    def has_line_of_sight(self, pos1, pos2) -> Optional[bool]:
        """Unknown: walking surfaces do not encode visual occluders."""
        return None

    def get_nearest_navmesh_point(self, position) -> Optional[Tuple[float, float, float]]:
        return None

    def is_position_valid(self, position) -> Optional[bool]:
        return None

    def get_distance_along_navmesh(self, pos1, pos2) -> Optional[float]:
        """Unknown without a parsed graph; Euclidean distance is not a route."""
        return None

    @staticmethod
    def euclidean_distance(pos1, pos2) -> float:
        return math.dist(pos1, pos2)
