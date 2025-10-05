#!/usr/bin/env python3
"""
Comprehensive Demo Processor - Combines header extraction and full demo parsing
Uses both cs2_header_extractor.exe and demo_parser_with_positions.exe to get complete data
"""

import json
import os
import subprocess
import logging
from typing import Dict, List, Optional, Tuple
import sys

def get_base_dir():
    """Get the base directory for the application"""
    if hasattr(get_base_dir, '_cached_dir'):
        return get_base_dir._cached_dir
    
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    get_base_dir._cached_dir = base_dir
    return base_dir

class ComprehensiveDemoProcessor:
    """Processes demos with both header extraction and full parsing"""
    
    def __init__(self):
        self.base_dir = get_base_dir()
        self.header_extractor = os.path.join(self.base_dir, "cs2_header_extractor.exe")
        self.demo_parser = os.path.join(self.base_dir, "demo_parser_with_positions.exe")
        
        self.logger = logging.getLogger(__name__)
        
        # Check if both extractors exist
        if not os.path.exists(self.header_extractor):
            self.logger.warning(f"Header extractor not found: {self.header_extractor}")
        if not os.path.exists(self.demo_parser):
            self.logger.warning(f"Demo parser not found: {self.demo_parser}")
    
    def extract_header_info(self, demo_path: str) -> Optional[Dict]:
        """Extract header information using cs2_header_extractor.exe"""
        try:
            if not os.path.exists(self.header_extractor):
                return None
            
            result = subprocess.run(
                [self.header_extractor, demo_path], 
                capture_output=True, 
                text=True, 
                timeout=30
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                self.logger.warning(f"Header extraction failed for {demo_path}: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting header from {demo_path}: {e}")
            return None
    
    def parse_demo_content(self, demo_path: str) -> Optional[Dict]:
        """Parse demo content using demo_parser_with_positions.exe"""
        try:
            if not os.path.exists(self.demo_parser):
                return None
            
            result = subprocess.run(
                [self.demo_parser, demo_path], 
                capture_output=True, 
                text=True, 
                timeout=60  # Reasonable timeout for demo parsing
            )
            
            if result.returncode == 0:
                try:
                    parsed_data = json.loads(result.stdout)
                    # Debug: log the structure we got
                    self.logger.info(f"Demo parser output keys: {list(parsed_data.keys()) if isinstance(parsed_data, dict) else 'Not a dict'}")
                    return parsed_data
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON from demo parser: {e}")
                    self.logger.debug(f"Raw output: {result.stdout[:500]}...")
                    return None
            else:
                self.logger.warning(f"Demo parsing failed for {demo_path}: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error parsing demo {demo_path}: {e}")
            return None
    
    def process_demo_comprehensive(self, demo_path: str) -> Dict:
        """
        Process a demo with both extractors and combine the results
        
        Returns:
            Combined demo data with header info and kill events
        """
        demo_file = os.path.basename(demo_path)
        
        # Initialize result structure
        result = {
            "demo_file": demo_file,
            "demo_path": demo_path,
            "success": False,
            "header_info": None,
            "kill_data": None,
            "map_name": None,
            "server_name": None,
            "demo_type": None,
            "kills": [],
            "kill_count": 0,
            "error": None
        }
        
        self.logger.info(f"Processing demo comprehensively: {demo_file}")
        
        # Step 1: Extract header information
        header_info = self.extract_header_info(demo_path)
        if header_info and header_info.get('success'):
            result["header_info"] = header_info
            result["map_name"] = header_info.get("map_name")
            result["server_name"] = header_info.get("server_name")
            result["demo_type"] = header_info.get("demo_type")
            self.logger.info(f"Header extracted: {header_info.get('map_name')} ({header_info.get('demo_type')})")
        else:
            self.logger.warning(f"Failed to extract header from {demo_file}")
        
        # Step 2: Parse full demo content
        kill_data = self.parse_demo_content(demo_path)
        if kill_data:
            result["kill_data"] = kill_data
            
            # Debug: log what we got from demo parser
            self.logger.info(f"Demo parser returned: {type(kill_data)} with keys: {list(kill_data.keys()) if isinstance(kill_data, dict) else 'No keys'}")
            
            # Handle different demo parser formats
            events = kill_data.get("events", [])
            kills = kill_data.get("kills", [])  # Original format
            
            # Log what we found
            self.logger.info(f"Found {len(events)} events and {len(kills)} kills")
            
            # Filter and convert kill events from the events list
            if events:
                kill_events = [event for event in events if event.get('type') == 'kill']
                converted_kills = []
                for event in kill_events:
                    converted_kill = self.convert_event_to_kill_format(event, result.get("map_name", "unknown"))
                    if converted_kill:
                        converted_kills.append(converted_kill)
                result["kills"] = converted_kills
            else:
                result["kills"] = kills if kills else []
            
            result["kill_count"] = len(result["kills"])
            
            # If header extraction failed, try to get map name from demo parser
            if not result["map_name"] and kill_data.get("map_name"):
                result["map_name"] = kill_data.get("map_name")
            
            self.logger.info(f"Demo parsed: {result['kill_count']} kills found")
            
            # Log sample event data if available
            if result["kills"]:
                sample_event = result["kills"][0]
                self.logger.info(f"Sample event keys: {list(sample_event.keys()) if isinstance(sample_event, dict) else 'Not a dict'}")
                if isinstance(sample_event, dict) and 'data' in sample_event:
                    sample_data = sample_event.get('data', {})
                    self.logger.info(f"Sample event data keys: {list(sample_data.keys()) if isinstance(sample_data, dict) else 'No data keys'}")
                
                # Look for kill events specifically
                kill_events = [event for event in result["kills"] if event.get('type') == 'kill']
                self.logger.info(f"Found {len(kill_events)} kill events out of {len(result['kills'])} total events")
                
                if kill_events:
                    sample_kill = kill_events[0] 
                    kill_data = sample_kill.get('data', {})
                    self.logger.info(f"Sample kill event data keys: {list(kill_data.keys()) if isinstance(kill_data, dict) else 'No kill data'}")
                
        else:
            self.logger.warning(f"Failed to parse demo content from {demo_file}")
        
        # Step 3: Enhance kill events with header information
        if result["kills"] and result["header_info"]:
            for kill_event in result["kills"]:
                # Override map name with header extraction (more reliable)
                if result["map_name"]:
                    kill_event["map_name"] = result["map_name"]
                
                # Add server and demo type information
                kill_event["server_name"] = result.get("server_name", "")
                kill_event["demo_type"] = result.get("demo_type", "Unknown")
        
        # Determine overall success
        result["success"] = bool(result["header_info"] or result["kill_data"])
        
        if not result["success"]:
            result["error"] = "Both header extraction and demo parsing failed"
        
        return result
    
    def process_multiple_demos(self, demo_paths: List[str]) -> List[Dict]:
        """Process multiple demos comprehensively"""
        results = []
        
        for demo_path in demo_paths:
            try:
                result = self.process_demo_comprehensive(demo_path)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process {demo_path}: {e}")
                results.append({
                    "demo_file": os.path.basename(demo_path),
                    "demo_path": demo_path,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def get_training_data_from_results(self, results: List[Dict]) -> List[Dict]:
        """
        Convert comprehensive demo results into training data format
        with proper map names from header extraction
        """
        training_data = []
        
        for result in results:
            if result.get("success") and result.get("kills"):
                demo_data = {
                    "demo_file": result["demo_file"],
                    "map_name": result.get("map_name", "unknown"),
                    "server_name": result.get("server_name", ""),
                    "demo_type": result.get("demo_type", "Unknown"),
                    "kills": result["kills"],
                    "kill_count": result["kill_count"]
                }
                training_data.append(demo_data)
        
        return training_data

    def convert_event_to_kill_format(self, event: Dict, map_name: str) -> Optional[Dict]:
        """
        Convert the new event format to the original kill format expected by AI trainer
        """
        try:
            event_data = event.get('data', {})
            
            # Handle different data structures
            def safe_get(obj, key, default=None):
                if isinstance(obj, dict):
                    return obj.get(key, default)
                else:
                    return default
            
            # Extract killer and victim info (they might be strings or dicts)
            killer = safe_get(event_data, 'killer', {})
            victim = safe_get(event_data, 'victim', {})
            
            # Extract positions
            killer_pos = safe_get(event_data, 'killer_pos', {})
            victim_pos = safe_get(event_data, 'victim_pos', {})
            killer_angle = safe_get(event_data, 'killer_angle', {})
            
            # Convert to the expected format
            kill_event = {
                'tick': event.get('tick', 0),
                'attacker_name': safe_get(killer, 'name', str(killer) if isinstance(killer, str) else 'Unknown'),
                'attacker_steam_id': str(safe_get(killer, 'steam_id', 0)),
                'attacker_pos': {
                    'x': safe_get(killer_pos, 'x', 0),
                    'y': safe_get(killer_pos, 'y', 0),
                    'z': safe_get(killer_pos, 'z', 0)
                },
                'victim_name': safe_get(victim, 'name', str(victim) if isinstance(victim, str) else 'Unknown'),
                'victim_steam_id': str(safe_get(victim, 'steam_id', 0)),
                'victim_pos': {
                    'x': safe_get(victim_pos, 'x', 0),
                    'y': safe_get(victim_pos, 'y', 0),
                    'z': safe_get(victim_pos, 'z', 0)
                },
                'weapon': safe_get(event_data, 'weapon', 'unknown'),
                'is_headshot': safe_get(event_data, 'headshot', False),
                'attacker_team': safe_get(killer, 'team', 'UNKNOWN'),
                'victim_team': safe_get(victim, 'team', 'UNKNOWN'),
                'map_name': map_name,
                'round_number': safe_get(event_data, 'round_number', 1),
                'distance': safe_get(event_data, 'distance', 0),
                'angle_diff': safe_get(killer_angle, 'yaw', 0),
                'attacker_view_x': safe_get(killer_angle, 'yaw', 0),
                'attacker_view_y': safe_get(killer_angle, 'pitch', 0),
                'victim_view_x': 0,  # Not available in new format
                'victim_view_y': 0,  # Not available in new format
                'demo_file': os.path.basename(map_name) if map_name else 'unknown'
            }
            
            return kill_event
            
        except Exception as e:
            self.logger.error(f"Error converting event to kill format: {e}")
            self.logger.debug(f"Event data: {event}")
            return None

def test_comprehensive_processing():
    """Test the comprehensive demo processing"""
    processor = ComprehensiveDemoProcessor()
    
    demo_dir = "demos"
    if not os.path.exists(demo_dir):
        print("No demos directory found")
        return
    
    demo_files = [f for f in os.listdir(demo_dir) if f.endswith('.dem')]
    if not demo_files:
        print("No demo files found")
        return
    
    # Test with first demo
    demo_path = os.path.join(demo_dir, demo_files[0])
    print(f"Testing comprehensive processing with: {demo_files[0]}")
    
    result = processor.process_demo_comprehensive(demo_path)
    
    print(f"Success: {result['success']}")
    print(f"Map Name: {result.get('map_name', 'N/A')}")
    print(f"Server: {result.get('server_name', 'N/A')}")
    print(f"Demo Type: {result.get('demo_type', 'N/A')}")
    print(f"Kill Count: {result.get('kill_count', 0)}")
    
    if result.get('kills'):
        sample_kill = result['kills'][0]
        print(f"Sample Kill Map: {sample_kill.get('map_name', 'N/A')}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_comprehensive_processing()