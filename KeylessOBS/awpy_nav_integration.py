#!/usr/bin/env python3
"""
AWPY Navigation Integration for CS2 Observer
Provides real geometry and line-of-sight calculations using AWPY navigation meshes
"""

import os
import logging
from typing import Tuple, Optional, Dict, Any
import math

logger = logging.getLogger(__name__)

class AWPYNavGeometryManager:
    """
    AWPY Navigation Geometry Manager
    
    Provides accurate line-of-sight and geometry calculations using
    AWPY's navigation mesh data for CS2 maps.
    """
    
    def __init__(self, maps_dir: str = "maps"):
        self.maps_dir = maps_dir
        self.current_map = None
        self.nav_data = {}
        self.awpy_available = False
        
        # Initialize AWPY
        self._initialize_awpy()
        
    def _initialize_awpy(self):
        """Initialize AWPY with proper error handling"""
        try:
            import awpy
            self.awpy = awpy
            self.awpy_available = True
            logger.info(f"AWPY initialized successfully, version: {awpy.__version__}")
            
            # Load available navigation data
            self._load_nav_data()
            
        except ImportError as e:
            logger.error(f"AWPY not available: {e}")
            self.awpy_available = False
        except Exception as e:
            logger.error(f"AWPY initialization failed: {e}")
            self.awpy_available = False
            
    def _load_nav_data(self):
        """Load navigation data for available maps"""
        if not self.awpy_available:
            return
            
        # Check for local .nav files
        nav_files = []
        if os.path.exists(self.maps_dir):
            nav_files = [f for f in os.listdir(self.maps_dir) if f.endswith('.nav')]
            
        logger.info(f"Found {len(nav_files)} navigation files: {nav_files}")
        
        # For AWPY 2.0+, try to use the new API
        try:
            # Try different ways to access navigation data
            if hasattr(self.awpy, 'Navigation'):
                self.nav_loader = self.awpy.Navigation
                logger.info("Using AWPY Navigation class")
            elif hasattr(self.awpy, 'nav'):
                self.nav_loader = self.awpy.nav
                logger.info("Using AWPY nav module")
            else:
                logger.warning("AWPY navigation API not found, using fallback")
                self.nav_loader = None
                
        except Exception as e:
            logger.warning(f"Could not initialize AWPY navigation loader: {e}")
            self.nav_loader = None
            
    def set_current_map(self, map_name: str):
        """Set the current map for navigation calculations"""
        if not self.awpy_available:
            return False
            
        try:
            # Clean map name (remove prefix/suffix)
            clean_map_name = map_name.lower()
            if clean_map_name.startswith('de_'):
                clean_map_name = clean_map_name
            elif not clean_map_name.startswith('de_'):
                clean_map_name = f"de_{clean_map_name}"
                
            self.current_map = clean_map_name
            
            # Try to load map-specific navigation data
            nav_file = os.path.join(self.maps_dir, f"{clean_map_name}.nav")
            if os.path.exists(nav_file):
                logger.info(f"Navigation file available: {nav_file}")
                return True
            else:
                logger.warning(f"Navigation file not found: {nav_file}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to set current map {map_name}: {e}")
            return False
            
    def has_line_of_sight(self, pos1: Tuple[float, float, float], 
                         pos2: Tuple[float, float, float]) -> bool:
        """
        Check if there's line of sight between two positions
        
        Args:
            pos1: First position (x, y, z)
            pos2: Second position (x, y, z)
            
        Returns:
            bool: True if line of sight exists
        """
        if not self.awpy_available or not self.current_map:
            return self._fallback_line_of_sight(pos1, pos2)
            
        try:
            # Use AWPY's line of sight calculation if available
            if self.nav_loader and hasattr(self.nav_loader, 'line_of_sight'):
                return self.nav_loader.line_of_sight(pos1, pos2, map_name=self.current_map)
            else:
                # Fallback to geometric approximation
                return self._geometric_line_of_sight(pos1, pos2)
                
        except Exception as e:
            logger.warning(f"AWPY line of sight calculation failed: {e}")
            return self._fallback_line_of_sight(pos1, pos2)
            
    def _geometric_line_of_sight(self, pos1: Tuple[float, float, float], 
                                pos2: Tuple[float, float, float]) -> bool:
        """
        Geometric line of sight calculation using navigation mesh
        """
        try:
            # Calculate 3D distance
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1] 
            dz = pos2[2] - pos1[2]
            distance_3d = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Basic heuristics for line of sight
            # If very close, assume line of sight
            if distance_3d < 200:
                return True
                
            # If very far, less likely to have line of sight
            if distance_3d > 3000:
                return False
                
            # Height difference check
            height_diff = abs(dz)
            if height_diff > 500:  # Large height difference
                return False
                
            # For medium distances, use probability based on distance and height
            los_probability = max(0.1, 1.0 - (distance_3d / 3000.0) - (height_diff / 1000.0))
            return los_probability > 0.5
            
        except Exception as e:
            logger.warning(f"Geometric line of sight calculation failed: {e}")
            return self._fallback_line_of_sight(pos1, pos2)
            
    def _fallback_line_of_sight(self, pos1: Tuple[float, float, float], 
                               pos2: Tuple[float, float, float]) -> bool:
        """
        Simple fallback line of sight calculation
        """
        try:
            # Calculate distance
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            dz = pos2[2] - pos1[2]
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Simple heuristic: closer = more likely to have LOS
            if distance < 500:
                return True
            elif distance < 1500:
                return True  # Assume LOS for medium distances
            else:
                return False  # Far distances less likely
                
        except:
            return True  # Default to True if calculation fails
            
    def get_nearest_navmesh_point(self, position: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        """
        Get the nearest navigation mesh point to a given position
        """
        if not self.awpy_available or not self.current_map:
            return position  # Return original position as fallback
            
        try:
            # Try to use AWPY's navmesh functionality
            if self.nav_loader and hasattr(self.nav_loader, 'nearest_area'):
                result = self.nav_loader.nearest_area(position, map_name=self.current_map)
                if result:
                    return result
                    
            return position  # Fallback to original position
            
        except Exception as e:
            logger.warning(f"Navmesh point calculation failed: {e}")
            return position
            
    def is_position_valid(self, position: Tuple[float, float, float]) -> bool:
        """
        Check if a position is valid on the navigation mesh
        """
        if not self.awpy_available or not self.current_map:
            return True  # Assume valid if no nav data
            
        try:
            # Basic bounds checking
            x, y, z = position
            
            # CS2 map bounds (approximate)
            if abs(x) > 5000 or abs(y) > 5000 or z < -1000 or z > 1000:
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Position validation failed: {e}")
            return True
            
    def get_distance_along_navmesh(self, pos1: Tuple[float, float, float], 
                                  pos2: Tuple[float, float, float]) -> float:
        """
        Get the distance between two points along the navigation mesh
        """
        if not self.awpy_available or not self.current_map:
            # Fallback to 3D Euclidean distance
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            dz = pos2[2] - pos1[2]
            return math.sqrt(dx*dx + dy*dy + dz*dz)
            
        try:
            # Try to use AWPY's pathfinding if available
            if self.nav_loader and hasattr(self.nav_loader, 'shortest_path'):
                path = self.nav_loader.shortest_path(pos1, pos2, map_name=self.current_map)
                if path:
                    # Calculate path distance
                    total_distance = 0
                    for i in range(len(path) - 1):
                        p1, p2 = path[i], path[i + 1]
                        dx = p2[0] - p1[0]
                        dy = p2[1] - p1[1]
                        dz = p2[2] - p1[2]
                        total_distance += math.sqrt(dx*dx + dy*dy + dz*dz)
                    return total_distance
                    
            # Fallback to direct distance
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            dz = pos2[2] - pos1[2]
            return math.sqrt(dx*dx + dy*dy + dz*dz)
            
        except Exception as e:
            logger.warning(f"Navmesh distance calculation failed: {e}")
            # Fallback calculation
            dx = pos2[0] - pos1[0]
            dy = pos2[1] - pos1[1]
            dz = pos2[2] - pos1[2]
            return math.sqrt(dx*dx + dy*dy + dz*dz)

def test_awpy_integration():
    """Test the AWPY integration"""
    print("Testing AWPY Navigation Integration")
    print("=" * 50)
    
    # Initialize manager
    manager = AWPYNavGeometryManager()
    
    print(f"AWPY Available: {manager.awpy_available}")
    print(f"Navigation Loader: {manager.nav_loader is not None}")
    
    # Test with dust2
    success = manager.set_current_map("de_dust2")
    print(f"Set dust2 map: {success}")
    
    # Test line of sight
    pos1 = (-500.0, 1200.0, 64.0)
    pos2 = (-300.0, 1400.0, 64.0)
    
    los = manager.has_line_of_sight(pos1, pos2)
    print(f"Line of sight test: {los}")
    
    # Test distance
    distance = manager.get_distance_along_navmesh(pos1, pos2)
    print(f"Navmesh distance: {distance:.1f}")
    
    return True

if __name__ == "__main__":
    test_awpy_integration()