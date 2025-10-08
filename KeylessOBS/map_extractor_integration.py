#!/usr/bin/env python3
"""
Map Extractor Integration - Extract map names from CS2 demo files
Uses the Go-based map_extractor.exe to get demo information including map names
"""

import json
import os
import subprocess
import logging
import sys
from typing import Dict, Optional

def get_base_dir():
    """Get the base directory for the application (same as ai_trainer.py)"""
    if hasattr(get_base_dir, '_cached_dir'):
        return get_base_dir._cached_dir
    
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable
        base_dir = os.path.dirname(sys.executable)
    else:
        # Running as script
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    get_base_dir._cached_dir = base_dir
    return base_dir

def extract_map_from_demo(demo_path: str) -> Optional[Dict]:
    """
    Extract map information from a demo file using the Go map extractor
    
    Args:
        demo_path: Path to the demo file
        
    Returns:
        Dictionary containing demo information including map_name, or None if failed
    """
    try:
        base_dir = get_base_dir()
        extractor_path = os.path.join(base_dir, "cs2_header_extractor.exe")
        
        # Check if extractor exists
        if not os.path.exists(extractor_path):
            logging.warning(f"Header extractor not found at {extractor_path}")
            return None
        
        # Check if demo file exists
        if not os.path.exists(demo_path):
            logging.warning(f"Demo file not found: {demo_path}")
            return None
        
        # Run the map extractor
        result = subprocess.run(
            [extractor_path, demo_path],
            capture_output=True,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        if result.returncode != 0:
            logging.error(f"Map extractor failed with code {result.returncode}: {result.stderr}")
            return None
        
        # Parse JSON output
        try:
            demo_info = json.loads(result.stdout)
            if demo_info.get('success', False):
                return demo_info
            else:
                logging.error(f"Map extraction failed: {demo_info.get('error_message', 'Unknown error')}")
                return None
                
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse map extractor output: {e}")
            logging.debug(f"Raw output: {result.stdout}")
            return None
        
    except subprocess.TimeoutExpired:
        logging.error(f"Map extraction timed out for {demo_path}")
        return None
    except Exception as e:
        logging.error(f"Error extracting map from {demo_path}: {e}")
        return None

def get_demo_map_name(demo_path: str) -> str:
    """
    Get just the map name from a demo file
    
    Args:
        demo_path: Path to the demo file
        
    Returns:
        Map name string, or "Unknown" if extraction failed
    """
    demo_info = extract_map_from_demo(demo_path)
    if demo_info and 'map_name' in demo_info:
        map_name = demo_info['map_name']
        
        # Clean up the map name if it's a debug/placeholder value
        if map_name in ['match_started_no_convars', 'convars_available', 'round_started_no_map']:
            return "Unknown"
        
        # Extract actual map name from hostname if needed
        if map_name.startswith("from_hostname:"):
            hostname = map_name.replace("from_hostname:", "").strip()
            # Try to extract map name from hostname patterns
            for potential_map in ['dust2', 'mirage', 'inferno', 'cache', 'overpass', 'train', 'nuke', 'vertigo', 'ancient']:
                if potential_map in hostname.lower():
                    return potential_map.capitalize()
            return "Unknown"
        
        return map_name
    
    return "Unknown"

def test_map_extraction():
    """Test function to verify map extraction works"""
    import sys
    
    # Add current directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    logging.basicConfig(level=logging.DEBUG)
    
    # Test with existing demo files
    demos_dir = os.path.join(get_base_dir(), "demos")
    if os.path.exists(demos_dir):
        for demo_file in os.listdir(demos_dir):
            if demo_file.endswith('.dem'):
                demo_path = os.path.join(demos_dir, demo_file)
                print(f"Testing: {demo_file}")
                
                demo_info = extract_map_from_demo(demo_path)
                if demo_info:
                    print(f"  Map: {demo_info.get('map_name', 'Unknown')}")
                    print(f"  Rounds: {demo_info.get('total_rounds', 0)}")
                    print(f"  Kills: {demo_info.get('total_kills', 0)}")
                else:
                    print("  Extraction failed")
                print()

if __name__ == "__main__":
    test_map_extraction()