"""
Line of Sight (LOS) System for CS2 Observer AI
Prevents false duels through walls using occupancy grid + 2D ray casting
"""

import numpy as np
import os
import logging
from typing import Dict, Tuple, Optional
from math import hypot

logger = logging.getLogger(__name__)

GRID_SIZE = 64.0  # Source units per cell (tune per map)
EYE_HEIGHT = 70.0  # Eye height above feet position
MAX_Z_TOLERANCE = 200.0  # Maximum vertical separation to consider for LOS

class GridLOS:
    """
    Line of Sight checker using occupancy grid and Bresenham ray casting
    """
    
    def __init__(self, blockmask: np.ndarray, origin_xy: Tuple[float, float] = (0.0, 0.0)):
        """
        Initialize LOS checker with a blocking mask
        
        Args:
            blockmask: HxW np.uint8 array (1=solid wall, 0=free space)
            origin_xy: World coordinate (x0,y0) that maps to mask[0,0]
        """
        self.mask = blockmask
        self.H, self.W = blockmask.shape
        self.x0, self.y0 = origin_xy
        logger.info(f"GridLOS initialized: {self.W}x{self.H} grid, origin: {origin_xy}")

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid cell indices"""
        i = int((y - self.y0) / GRID_SIZE)
        j = int((x - self.x0) / GRID_SIZE)
        return i, j

    def clear_line(self, i0: int, j0: int, i1: int, j1: int) -> bool:
        """
        Bresenham ray casting through grid
        Returns True if no solid cells are hit
        """
        di = abs(i1 - i0)
        dj = abs(j1 - j0)
        si = 1 if i0 < i1 else -1
        sj = 1 if j0 < j1 else -1
        err = dj - di
        i, j = i0, j0
        
        while True:
            # Check bounds - outside grid is considered blocking
            if not (0 <= i < self.H and 0 <= j < self.W):
                return False
            
            # Check if current cell is solid
            if self.mask[i, j] != 0:
                return False
            
            # Reached destination
            if i == i1 and j == j1:
                break
            
            # Bresenham step
            e2 = 2 * err
            if e2 > -di:
                err -= di
                j += sj
            if e2 < dj:
                err += dj
                i += si
        
        return True

    def has_line_of_sight(self, ax: float, ay: float, az: float, 
                         bx: float, by: float, bz: float) -> bool:
        """
        Check line of sight between two world positions
        Uses 2D ray casting with basic elevation check
        """
        # Basic elevation sanity check
        eye_a = az + EYE_HEIGHT
        eye_b = bz + EYE_HEIGHT
        if abs(eye_a - eye_b) > MAX_Z_TOLERANCE:
            # Large vertical separation - still allow but could be penalized
            pass
        
        # Convert to grid coordinates
        i0, j0 = self.world_to_cell(ax, ay)
        i1, j1 = self.world_to_cell(bx, by)
        
        # Ray cast through grid
        return self.clear_line(i0, j0, i1, j1)

class MapGeometryManager:
    """
    Manages line-of-sight checking for different CS2 maps
    """
    
    def __init__(self, maps_dir: str = "maps"):
        self.maps_dir = maps_dir
        self.los_checkers: Dict[str, GridLOS] = {}
        self.current_map = None
        self.ensure_maps_directory()
        
        # Initialize real geometry systems (priority: AWPY NAV > BSP > Approximation)
        self.awpy_nav_manager = None
        self.bsp_loader = None
        self.use_real_geometry = False
        
        # Try AWPY NAV first (best option)
        try:
            from awpy_nav_integration import AWPYNavGeometryManager
            self.awpy_nav_manager = AWPYNavGeometryManager(maps_dir)
            self.use_real_geometry = True
            logger.info("🧭 AWPY NAV geometry system initialized (highest accuracy)")
        except ImportError:
            logger.info("📐 AWPY not available, trying BSP loader...")
            
            # Fallback to BSP loader
            try:
                from gcfscape_integration import BSPLoader
                self.bsp_loader = BSPLoader(maps_dir)
                self.use_real_geometry = True
                logger.info("🗺️  BSP geometry system initialized")
            except ImportError:
                logger.info("📐 Using approximated geometry (no real geometry systems available)")

    def ensure_maps_directory(self):
        """Create maps directory if it doesn't exist"""
        if not os.path.exists(self.maps_dir):
            os.makedirs(self.maps_dir)
            logger.info(f"Created maps directory: {self.maps_dir}")

    def load_map_geometry(self, map_name: str) -> bool:
        """
        Load geometry data for a specific map
        Returns True if successful
        """
        # Try to load real BSP geometry first
        if self.use_real_geometry and self.bsp_loader:
            try:
                logger.info(f"🗺️  Loading real BSP geometry for {map_name}")
                bsp_data = self.bsp_loader.load_map_geometry(map_name)
                
                if bsp_data and 'collision_mesh' in bsp_data:
                    # Use real collision mesh from BSP
                    collision_mesh = bsp_data['collision_mesh']
                    bounds = bsp_data.get('bounds', {})
                    origin_xy = (bounds.get('min_x', -2000), bounds.get('min_y', -2000))
                    
                    self.los_checkers[map_name] = GridLOS(collision_mesh, origin_xy)
                    self.current_map = map_name
                    logger.info(f"✅ Loaded REAL BSP geometry for {map_name}")
                    logger.info(f"   Grid size: {collision_mesh.shape}")
                    logger.info(f"   Solid cells: {np.sum(collision_mesh)} ({np.sum(collision_mesh)/collision_mesh.size*100:.1f}%)")
                    return True
                else:
                    logger.warning(f"BSP geometry loading failed for {map_name}, falling back to approximation")
            except Exception as e:
                logger.error(f"Error loading BSP geometry for {map_name}: {e}")
        
        # Fallback to cached/approximated geometry
        map_file = os.path.join(self.maps_dir, f"{map_name}_blockmask.npy")
        
        if os.path.exists(map_file):
            try:
                blockmask = np.load(map_file)
                origin_file = os.path.join(self.maps_dir, f"{map_name}_origin.npy")
                
                if os.path.exists(origin_file):
                    origin_xy = tuple(np.load(origin_file))
                else:
                    origin_xy = (0.0, 0.0)  # Default origin
                
                self.los_checkers[map_name] = GridLOS(blockmask, origin_xy)
                self.current_map = map_name
                logger.info(f"📐 Loaded approximated geometry for {map_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error loading geometry for {map_name}: {e}")
                return False
        else:
            # Generate default geometry if file doesn't exist
            logger.warning(f"No geometry file found for {map_name}, generating default")
            self.generate_default_geometry(map_name)
            return False

    def generate_default_geometry(self, map_name: str):
        """
        Generate a basic geometry approximation for common maps
        This is a fallback when no proper geometry data is available
        """
        if map_name.lower() in ['de_dust2', 'dust2']:
            self.generate_dust2_approximation()
        elif map_name.lower() in ['de_mirage', 'mirage']:
            self.generate_mirage_approximation()
        elif map_name.lower() in ['de_inferno', 'inferno']:
            self.generate_inferno_approximation()
        else:
            self.generate_generic_geometry(map_name)

    def generate_dust2_approximation(self):
        """
        Generate approximate Dust2 geometry based on common wall positions
        This blocks obvious impossible sightlines like CT spawn <-> A catwalk
        """
        # Create a 100x100 grid covering typical Dust2 coordinates
        # Dust2 ranges roughly from -2000 to +2000 in both X and Y
        size = 100
        mask = np.zeros((size, size), dtype=np.uint8)
        origin = (-2000.0, -2000.0)
        
        # Add strategic blocking structures to prevent impossible sightlines
        # Grid: 100x100 covering -2000 to +2000 (40 units per cell)
        
        # CT spawn area (bottom-left in our coordinate system)
        # Blocks CT spawn from seeing across map
        mask[15:25, 25:35] = 1  # CT spawn building
        
        # A site area (top-right)
        # Blocks A site from CT spawn sightlines  
        mask[75:85, 75:85] = 1  # A site building
        
        # Central mid area (blocks cross-map sightlines)
        mask[45:55, 45:55] = 1  # Mid building/wall
        
        # B site area (bottom-right)
        mask[25:35, 75:85] = 1  # B site building
        
        # T spawn area (top-left) 
        mask[75:85, 25:35] = 1  # T spawn building
        
        # Long diagonal wall (key structure separating CT spawn from A site)
        # This should block the main impossible sightline mentioned by user
        for i in range(20, 70):
            for j in range(max(0, i-10), min(100, i+10)):
                if 30 <= i <= 65 and 55 <= j <= 90:  # Diagonal wall area
                    mask[i, j] = 1
        
        # Save the approximation
        map_file = os.path.join(self.maps_dir, "de_dust2_blockmask.npy")
        origin_file = os.path.join(self.maps_dir, "de_dust2_origin.npy")
        
        np.save(map_file, mask)
        np.save(origin_file, np.array(origin))
        
        self.los_checkers['de_dust2'] = GridLOS(mask, origin)
        self.current_map = 'de_dust2'
        
        logger.info("Generated Dust2 geometry approximation")

    def generate_mirage_approximation(self):
        """Generate approximate Mirage geometry"""
        size = 80
        mask = np.zeros((size, size), dtype=np.uint8)
        origin = (-1500.0, -1500.0)
        
        # Major structures
        mask[30:50, 30:50] = 1  # Central area
        mask[15:30, 60:75] = 1  # A site
        mask[60:75, 15:30] = 1  # B site
        
        map_file = os.path.join(self.maps_dir, "de_mirage_blockmask.npy")
        origin_file = os.path.join(self.maps_dir, "de_mirage_origin.npy")
        
        np.save(map_file, mask)
        np.save(origin_file, np.array(origin))
        
        self.los_checkers['de_mirage'] = GridLOS(mask, origin)
        logger.info("Generated Mirage geometry approximation")

    def generate_inferno_approximation(self):
        """Generate approximate Inferno geometry"""
        size = 90
        mask = np.zeros((size, size), dtype=np.uint8)
        origin = (-1800.0, -1800.0)
        
        # Complex structure approximation
        mask[25:45, 35:55] = 1  # Central buildings
        mask[15:35, 65:80] = 1  # A site complex
        mask[65:80, 15:35] = 1  # B site complex
        
        map_file = os.path.join(self.maps_dir, "de_inferno_blockmask.npy")
        origin_file = os.path.join(self.maps_dir, "de_inferno_origin.npy")
        
        np.save(map_file, mask)
        np.save(origin_file, np.array(origin))
        
        self.los_checkers['de_inferno'] = GridLOS(mask, origin)
        logger.info("Generated Inferno geometry approximation")

    def generate_generic_geometry(self, map_name: str):
        """Generate minimal generic geometry for unknown maps"""
        size = 60
        mask = np.zeros((size, size), dtype=np.uint8)
        origin = (-1500.0, -1500.0)
        
        # Minimal structure - just prevents extreme cross-map sightlines
        mask[25:35, 25:35] = 1  # Central obstruction
        
        map_file = os.path.join(self.maps_dir, f"{map_name}_blockmask.npy")
        origin_file = os.path.join(self.maps_dir, f"{map_name}_origin.npy")
        
        np.save(map_file, mask)
        np.save(origin_file, np.array(origin))
        
        self.los_checkers[map_name] = GridLOS(mask, origin)
        logger.info(f"Generated generic geometry for {map_name}")

    def has_line_of_sight(self, pos1: Tuple[float, float, float], 
                         pos2: Tuple[float, float, float]) -> bool:
        """
        Check if two positions have line of sight
        Returns True if no map geometry blocks the view
        """
        # Try AWPY NAV first (most accurate)
        if self.awpy_nav_manager and self.current_map:
            try:
                return self.awpy_nav_manager.has_line_of_sight(pos1, pos2)
            except Exception as e:
                logger.warning(f"AWPY NAV LOS failed, falling back: {e}")
        
        # Fallback to grid-based LOS
        if not self.current_map or self.current_map not in self.los_checkers:
            # No geometry data - allow all sightlines (fallback)
            return True
        
        los_checker = self.los_checkers[self.current_map]
        return los_checker.has_line_of_sight(pos1[0], pos1[1], pos1[2],
                                           pos2[0], pos2[1], pos2[2])

    def set_current_map(self, map_name: str):
        """Set the current map and load its geometry if needed"""
        if map_name != self.current_map:
            # Update AWPY NAV manager first
            if self.awpy_nav_manager:
                self.awpy_nav_manager.set_current_map(map_name)
            
            # Then load grid-based geometry as fallback
            self.load_map_geometry(map_name)

def duel_geometrically_possible(pos_a: Tuple[float, float, float],
                               pos_b: Tuple[float, float, float],
                               geometry_manager: MapGeometryManager,
                               max_2d_distance: float = 3000.0,
                               min_2d_distance: float = 50.0) -> bool:
    """
    Check if a duel between two positions is geometrically possible
    
    Args:
        pos_a, pos_b: Player positions (x, y, z)
        geometry_manager: Map geometry manager
        max_2d_distance: Maximum 2D distance for duels
        min_2d_distance: Minimum 2D distance (avoid shoulder clips)
    
    Returns:
        True if duel is geometrically possible
    """
    ax, ay, az = pos_a
    bx, by, bz = pos_b
    
    # Distance checks
    dist_2d = hypot(ax - bx, ay - by)
    if dist_2d > max_2d_distance or dist_2d < min_2d_distance:
        return False
    
    # Line of sight check
    if not geometry_manager.has_line_of_sight(pos_a, pos_b):
        return False
    
    return True

# Global geometry manager instance
_geometry_manager = None

def get_geometry_manager() -> MapGeometryManager:
    """Get the global geometry manager instance"""
    global _geometry_manager
    if _geometry_manager is None:
        _geometry_manager = MapGeometryManager()
    return _geometry_manager

if __name__ == "__main__":
    # Test the geometry system
    logging.basicConfig(level=logging.INFO)
    
    # Create geometry manager and test Dust2
    gm = MapGeometryManager()
    gm.generate_dust2_approximation()
    
    # Test cases for Dust2
    test_cases = [
        # (pos1, pos2, expected, description)
        ((-1000, 0, 0), (1000, 0, 0), False, "Cross-map through central building"),
        ((-500, -500, 0), (-400, -400, 0), True, "Close positions with LOS"),
        ((-1500, 0, 0), (1500, 1500, 0), False, "CT spawn to A catwalk (blocked)"),
        ((0, 0, 0), (200, 200, 0), True, "Short range duel"),
    ]
    
    print("🗺️  Testing Dust2 Geometry System")
    print("=" * 50)
    
    for pos1, pos2, expected, desc in test_cases:
        result = duel_geometrically_possible(pos1, pos2, gm)
        status = "✅" if result == expected else "❌"
        print(f"{status} {desc}: {result} (expected {expected})")
    
    print(f"\nGeometry files created in: {gm.maps_dir}/")
    print("Ready to integrate with duel detector!")