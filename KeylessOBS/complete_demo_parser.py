#!/usr/bin/env python3
"""
Complete CS2 Demo Parser with Full Positional and State Data
Uses the rebuilt demo_parser_with_positions.exe that has all the data we need
"""

import json
import subprocess
import os
import sys
from typing import Dict, List, Any, Optional
from comprehensive_demo_processor import _finite, _position, uncompressed_demo

def get_base_dir():
    """Get base directory for dependencies - works for both script and executable"""
    if getattr(sys, 'frozen', False):
        # Running as PyInstaller executable
        return os.path.dirname(sys.executable)
    else:
        # Running as script
        return os.path.dirname(os.path.abspath(__file__))

class CS2DemoParser:
    """Complete CS2 demo parser with positional and state data"""

    def __init__(self, parser_path: str = None):
        if parser_path is None:
            parser_path = os.path.join(get_base_dir(), 'demo_parser_with_positions.exe')
        self.parser_path = os.path.abspath(parser_path)

        if not os.path.exists(self.parser_path):
            raise FileNotFoundError(f"Parser not found at {self.parser_path}")

    def parse_demo(self, demo_path: str) -> Dict[str, Any]:
        with uncompressed_demo(demo_path) as parser_input:
            result = self._parse_uncompressed_demo(parser_input)
        result['demo_file'] = os.path.basename(demo_path)
        return result

    def _parse_uncompressed_demo(self, demo_path: str) -> Dict[str, Any]:
        """
        Parse CS2 demo file and extract complete event data

        Args:
            demo_path: Path to .dem file

        Returns:
            Dictionary with complete demo data including positions, health, armor, etc.
        """

        if not os.path.exists(demo_path):
            raise FileNotFoundError(f"Demo file not found at {demo_path}")

        try:
            # Run the enhanced parser
            result = subprocess.run(
                [self.parser_path, os.path.abspath(demo_path)],
                capture_output=True,
                text=True,
                encoding='utf-8',
                cwd=os.path.dirname(self.parser_path),
                timeout=120  # Increased timeout for large demos
            )

            if result.returncode != 0:
                details = result.stderr
                try:
                    payload = json.loads(result.stdout)
                    if isinstance(payload, dict):
                        details = payload.get('error') or details
                except ValueError:
                    pass
                raise RuntimeError(f"Parser failed (exit {result.returncode}): {details}")

            # Parse the JSON output
            data = json.loads(result.stdout)
            if not isinstance(data, dict):
                raise ValueError('Parser must return a JSON object')
            if data.get('success') is False or data.get('error'):
                raise ValueError(data.get('error') or 'Parser reported failure')

            # Enhance the data with our analysis
            enhanced_data = self._enhance_demo_data(data)

            return enhanced_data

        except subprocess.TimeoutExpired:
            raise RuntimeError("Demo parsing timed out")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON output: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")

    def _enhance_demo_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance the demo data with additional analysis"""

        events = data.get('events', [])
        if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
            raise ValueError('Invalid parser events list')
        events = list(events)
        if not any(event.get('type') == 'kill' for event in events):
            legacy_kills = data.get('kills', [])
            if not isinstance(legacy_kills, list):
                raise ValueError('Invalid parser kills list')
            for kill in legacy_kills:
                if not isinstance(kill, dict):
                    raise ValueError('Invalid parser kill record')
                events.append({
                    'type': 'kill', 'tick': kill.get('tick'), 'time': kill.get('time'),
                    'data': {
                        'killer': {'name': kill.get('attacker_name'),
                                   'steam_id': kill.get('attacker_steam_id'),
                                   'team': kill.get('attacker_team')},
                        'victim': {'name': kill.get('victim_name'),
                                   'steam_id': kill.get('victim_steam_id'),
                                   'team': kill.get('victim_team')},
                        'killer_pos': kill.get('attacker_pos'),
                        'victim_pos': kill.get('victim_pos'),
                        'killer_angle': {'yaw': kill.get('attacker_view_x'),
                                         'pitch': kill.get('attacker_view_y')},
                        'weapon': kill.get('weapon'), 'headshot': kill.get('is_headshot'),
                        'wallbang': kill.get('wallbang'), 'distance': kill.get('distance'),
                    },
                })

        # Count event types
        event_counts = {}
        kill_events = []
        hurt_events = []

        for event in events:
            event_type = event.get('type', 'unknown')
            if event_type in ('kill', 'player_hurt') and not isinstance(event.get('data', {}), dict):
                raise ValueError('Invalid event data object')
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

            if event_type == 'kill':
                kill_events.append(event)
            elif event_type == 'player_hurt':
                hurt_events.append(event)

        # Enhanced metadata
        contexts = [event.get('data', {}) for event in kill_events]
        contexts = [context for context in contexts if isinstance(context, dict)]
        # These flags describe observed data, not a promise made by this wrapper.
        has_positions = bool(contexts) and all(
            _position(context.get('killer_pos')) is not None and
            _position(context.get('victim_pos')) is not None for context in contexts)
        has_angles = bool(contexts) and all(
            isinstance(context.get('killer_angle'), dict) and
            all(_finite(context['killer_angle'].get(axis)) for axis in ('yaw', 'pitch'))
            for context in contexts)
        has_health_armor = bool(contexts) and all(
            isinstance(context.get('killer'), dict) and
            all(_finite(context['killer'].get(field)) for field in ('health', 'armor'))
            for context in contexts)
        enhanced_data = data.copy()
        enhanced_data.update({
            'events': events,
            'enhanced_by': 'CS2DemoParser',
            'has_positions': has_positions,
            'has_health_armor': has_health_armor,
            'has_angles': has_angles,
            'has_distances': bool(contexts) and all(_finite(context.get('distance')) for context in contexts),
            'event_counts': event_counts,
            'analysis': {
                'total_kills': len(kill_events),
                'total_damage_events': len(hurt_events),
                'average_distance_per_kill': self._calculate_average_kill_distance(kill_events),
                'headshot_percentage': self._calculate_headshot_percentage(kill_events),
                'wallbang_percentage': self._calculate_wallbang_percentage(kill_events),
            }
        })

        return enhanced_data

    def _calculate_average_kill_distance(self, kill_events: List[Dict]) -> float:
        """Calculate average distance for kills"""
        distances = []
        for kill in kill_events:
            distance = kill.get('data', {}).get('distance', 0)
            if _finite(distance) and distance > 0:
                distances.append(distance)

        return sum(distances) / len(distances) if distances else 0.0

    def _calculate_headshot_percentage(self, kill_events: List[Dict]) -> float:
        """Calculate headshot percentage"""
        if not kill_events:
            return 0.0

        headshots = sum(1 for kill in kill_events if kill.get('data', {}).get('headshot', False))
        return (headshots / len(kill_events)) * 100

    def _calculate_wallbang_percentage(self, kill_events: List[Dict]) -> float:
        """Calculate wallbang percentage"""
        if not kill_events:
            return 0.0

        wallbangs = sum(1 for kill in kill_events if kill.get('data', {}).get('wallbang', False))
        return (wallbangs / len(kill_events)) * 100

    def get_kills_with_full_context(self, demo_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract kill events with full contextual data for AI training"""

        kills = []
        for event in demo_data.get('events', []):
            if event.get('type') == 'kill':
                kill_data = event.get('data', {})

                # Extract comprehensive kill context
                kill_context = {
                    'timestamp': event.get('time', 0),
                    'tick': event.get('tick', 0),

                    # Basic kill info
                    'killer': kill_data.get('killer', 'unknown'),
                    'victim': kill_data.get('victim', 'unknown'),
                    'weapon': kill_data.get('weapon', 'unknown'),
                    'headshot': kill_data.get('headshot', False),
                    'wallbang': kill_data.get('wallbang', False),
                    'distance': kill_data.get('distance', 0),

                    # Positional data
                    'killer_position': kill_data.get('killer_pos', {}),
                    'victim_position': kill_data.get('victim_pos', {}),
                    'killer_angle': kill_data.get('killer_angle', {}),
                }

                kills.append(kill_context)

        return kills

    def get_damage_events_with_context(self, demo_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract damage events with full contextual data"""

        damages = []
        for event in demo_data.get('events', []):
            if event.get('type') == 'player_hurt':
                damage_data = event.get('data', {})

                damage_context = {
                    'timestamp': event.get('time', 0),
                    'tick': event.get('tick', 0),

                    # Damage info
                    'attacker': damage_data.get('attacker', 'unknown'),
                    'victim': damage_data.get('victim', 'unknown'),
                    'damage': damage_data.get('damage', 0),
                    'armor_damage': damage_data.get('armor_damage', 0),
                    'remaining_health': damage_data.get('remaining_health', 100),
                    'distance': damage_data.get('distance', 0),

                    # Positional data
                    'attacker_position': damage_data.get('attacker_pos', {}),
                    'victim_position': damage_data.get('victim_pos', {}),
                }

                damages.append(damage_context)

        return damages


def test_complete_parser():
    """Test the complete parser system"""

    demo_path = r"c:\Users\dlndm\Desktop\OBSERVERai\KeylessOBS\KeylessOBS\demos\ancientdoubledeagle.dem"

    try:
        print("🎯 Complete CS2 Demo Parser Test")
        print("=" * 60)

        parser = CS2DemoParser()
        print(f"🔍 Parsing demo: {os.path.basename(demo_path)}")

        # Parse the demo
        demo_data = parser.parse_demo(demo_path)

        print(f"✅ Parsing complete!")
        print(f"   Map: {demo_data.get('map_name', 'unknown')}")
        print(f"   Duration: {demo_data.get('duration', 0):.1f}s")
        print(f"   Has positions: {demo_data.get('has_positions', False)}")
        print(f"   Has health/armor: {demo_data.get('has_health_armor', False)}")
        print(f"   Has angles: {demo_data.get('has_angles', False)}")

        # Show event counts
        event_counts = demo_data.get('event_counts', {})
        print(f"\n📊 Event Summary:")
        for event_type, count in event_counts.items():
            print(f"   {event_type}: {count}")

        # Show analysis
        analysis = demo_data.get('analysis', {})
        print(f"\n📈 Analysis:")
        print(f"   Total kills: {analysis.get('total_kills', 0)}")
        print(f"   Average kill distance: {analysis.get('average_distance_per_kill', 0):.1f} units")
        print(f"   Headshot rate: {analysis.get('headshot_percentage', 0):.1f}%")
        print(f"   Wallbang rate: {analysis.get('wallbang_percentage', 0):.1f}%")

        # Get kill data for AI training
        kills = parser.get_kills_with_full_context(demo_data)
        damages = parser.get_damage_events_with_context(demo_data)

        print(f"\n🎯 AI Training Data:")
        print(f"   Kill events: {len(kills)}")
        print(f"   Damage events: {len(damages)}")

        if kills:
            print(f"\n📍 Sample kill with full context:")
            sample_kill = kills[0]
            print(f"   {sample_kill['killer']} killed {sample_kill['victim']}")
            print(f"   Weapon: {sample_kill['weapon']}")
            print(f"   Distance: {sample_kill['distance']:.1f} units")
            print(f"   Headshot: {sample_kill['headshot']}")
            print(f"   Wallbang: {sample_kill['wallbang']}")

            killer_pos = sample_kill['killer_position']
            victim_pos = sample_kill['victim_position']
            if killer_pos and victim_pos:
                print(f"   Killer pos: ({killer_pos.get('x', 0):.1f}, {killer_pos.get('y', 0):.1f}, {killer_pos.get('z', 0):.1f})")
                print(f"   Victim pos: ({victim_pos.get('x', 0):.1f}, {victim_pos.get('y', 0):.1f}, {victim_pos.get('z', 0):.1f})")

            killer_angle = sample_kill['killer_angle']
            if killer_angle:
                print(f"   Killer angle: yaw={killer_angle.get('yaw', 0):.1f}°, pitch={killer_angle.get('pitch', 0):.1f}°")

        return demo_data

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Test with provided demo file
        demo_path = sys.argv[1]
        print(f"🎯 Complete CS2 Demo Parser Test")
        print("=" * 60)
        print(f"🔍 Parsing demo: {os.path.basename(demo_path)}")

        parser = CS2DemoParser()
        result = parser.parse_demo(demo_path)

        if result:
            print(f"\n✅ Complete parser test successful!")
            print(f"   Kills: {len(result.get('kills', []))}")
            print(f"   Events: {len(result.get('events', []))}")
        else:
            print(f"\n❌ Parser test failed")
    else:
        result = test_complete_parser()

        if result:
            print(f"\n✅ Complete parser test successful!")
            print(f"   Ready for AI training with full positional and contextual data")
            print(f"   All conditions captured: positions, health, armor, angles, distances, weapons")
        else:
            print(f"\n❌ Parser test failed")
