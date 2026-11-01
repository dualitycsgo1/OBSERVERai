#!/usr/bin/env python3
"""Run the demo tools without turning missing or failed data into training data."""

import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import shutil
from contextlib import contextmanager
from typing import Dict, List, Optional


def get_base_dir():
    return os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _position(value):
    if isinstance(value, dict) and all(_finite(value.get(axis)) for axis in "xyz"):
        return {axis: float(value[axis]) for axis in "xyz"}
    return None


@contextmanager
def uncompressed_demo(demo_path):
    """Expose .dem or stream .dem.zst into a temporary file; preserve the source."""
    demo_path = os.path.abspath(demo_path)
    if not demo_path.lower().endswith(".dem.zst"):
        yield demo_path
        return
    try:
        import zstandard
    except ImportError as exc:
        raise RuntimeError("Reading .dem.zst requires zstandard; install the project requirements") from exc
    with tempfile.TemporaryDirectory(prefix="observer-demo-") as directory:
        output = os.path.join(directory, os.path.basename(demo_path)[:-4])
        with open(demo_path, "rb") as compressed, open(output, "wb") as destination:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                shutil.copyfileobj(reader, destination, length=1024 * 1024)
        yield output


def _empty_result(demo_path):
    return {
        "demo_file": os.path.basename(demo_path), "demo_path": demo_path,
        "success": False, "header_info": None, "kill_data": None,
        "map_name": None, "server_name": None, "demo_type": None,
        "kills": [], "kill_count": 0, "prediction_samples": [],
        "training_ready": False, "error": None,
    }


class ComprehensiveDemoProcessor:
    def __init__(self):
        self.base_dir = get_base_dir()
        self.header_extractor = os.path.join(self.base_dir, "cs2_header_extractor.exe")
        self.demo_parser = os.path.join(self.base_dir, "demo_parser_with_positions.exe")
        self.logger = logging.getLogger(__name__)
        self.last_errors = []

    def _run_tool(self, executable, demo_path, timeout):
        if not os.path.isfile(executable):
            error = f"Required tool not found: {executable}"
            self.last_errors.append(error)
            self.logger.warning(error)
            return None
        try:
            # Resolve before changing cwd: relative demo paths must name the same file.
            completed = subprocess.run(
                [os.path.abspath(executable), os.path.abspath(demo_path)],
                capture_output=True, text=True, encoding="utf-8", errors="strict",
                timeout=timeout, cwd=self.base_dir,
            )
            if completed.returncode:
                details = completed.stderr[:500]
                try:
                    payload = json.loads(completed.stdout)
                    if isinstance(payload, dict):
                        details = payload.get("error") or payload.get("error_message") or details
                except ValueError:
                    pass
                raise ValueError(f"exit {completed.returncode}: {details}")
            data = json.loads(completed.stdout)
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            if data.get("success") is False or data.get("error") or data.get("error_message"):
                raise ValueError(data.get("error") or data.get("error_message") or "tool reported failure")
            return data
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            error = f"{os.path.basename(executable)} failed: {exc}"
            self.last_errors.append(error)
            self.logger.error(error)
            return None

    def extract_header_info(self, demo_path: str) -> Optional[Dict]:
        # The full Go parser already decodes the Source 2 protobuf header.
        # This legacy helper is optional and its absence is not a demo error.
        if not os.path.isfile(self.header_extractor):
            return None
        return self._run_tool(self.header_extractor, demo_path, 60)

    def parse_demo_content(self, demo_path: str) -> Optional[Dict]:
        return self._run_tool(self.demo_parser, demo_path, 300)

    def process_demo_comprehensive(self, demo_path: str) -> Dict:
        source_path = os.path.abspath(demo_path)
        try:
            with uncompressed_demo(source_path) as parser_input:
                result = self._process_uncompressed_demo(parser_input)
            result["demo_path"] = source_path
            result["demo_file"] = os.path.basename(source_path)
            for kill in result.get("kills", []):
                kill["demo_file"] = result["demo_file"]
            return result
        except Exception as exc:
            result = _empty_result(source_path)
            result["error"] = f"Demo decompression/processing failed: {exc}"
            self.logger.error(result["error"])
            return result

    def _process_uncompressed_demo(self, demo_path: str) -> Dict:
        self.last_errors = []
        demo_path = os.path.abspath(demo_path)
        result = _empty_result(demo_path)
        if not os.path.isfile(demo_path):
            result["error"] = f"Demo file not found: {demo_path}"
            return result

        header = self.extract_header_info(demo_path)
        parsed = self.parse_demo_content(demo_path)
        if header:
            result["header_info"] = header
            for field in ("map_name", "server_name", "demo_type"):
                result[field] = header.get(field)
        # A header is metadata only. It cannot establish that a demo was parsed.
        if not isinstance(parsed, dict) or parsed.get("success") is False:
            result["error"] = "; ".join(self.last_errors) or "Demo content parsing failed"
            return result
        result["kill_data"] = parsed
        for field in ("map_name", "server_name", "demo_type"):
            # The full parser uses demoinfocs' protobuf decoder; prefer it to
            # the optional legacy header extractor's simplified byte parser.
            if parsed.get(field):
                result[field] = parsed.get(field)

        raw_events = parsed.get("events", [])
        raw_kills = parsed.get("kills", [])
        samples = parsed.get("prediction_samples", [])
        if not all(isinstance(value, list) for value in (raw_events, raw_kills, samples)):
            result["error"] = "Invalid demo schema: events, kills and prediction_samples must be lists"
            return result
        events = [e for e in raw_events if isinstance(e, dict) and e.get("type") == "kill"]
        # Non-kill events must not hide a parser's legacy kill list.
        if events:
            result["kills"] = [converted for event in events if (
                converted := self.convert_event_to_kill_format(event, result["map_name"], result["demo_file"])
            ) is not None]
        else:
            result["kills"] = [dict(kill, map_name=result["map_name"], demo_file=result["demo_file"])
                               for kill in raw_kills if isinstance(kill, dict)]
        result["kill_count"] = len(result["kills"])
        result["prediction_samples"] = samples
        result["demo_id"] = parsed.get("demo_id")
        result["data_quality"] = parsed.get("data_quality", {})
        for field in ("patch_version", "build_num", "demo_version_name"):
            if field in parsed:
                result[field] = parsed[field]
        result["training_ready"] = bool(samples)
        result["success"] = bool(result["kills"] or samples)
        if not result["success"]:
            result["error"] = "Parser produced no kill events or prediction samples"
        elif not samples:
            result["training_warning"] = (
                "Kill events alone cannot train future-kill prediction. Rebuild the parser "
                "to extract pre-event snapshots with positive and negative labels."
            )
        return result

    def process_multiple_demos(self, demo_paths: List[str]) -> List[Dict]:
        return [self.process_demo_comprehensive(path) for path in demo_paths]

    def get_training_data_from_results(self, results: List[Dict]) -> List[Dict]:
        """Only prediction snapshots, never post-outcome kill-only records, are trainable."""
        return [sample for result in results if result.get("success")
                for sample in result.get("prediction_samples", [])]

    def convert_event_to_kill_format(self, event: Dict, map_name: str,
                                     demo_file: str = "unknown") -> Optional[Dict]:
        data = event.get("data")
        if not isinstance(data, dict):
            return None
        killer = data.get("killer")
        victim = data.get("victim")
        killer_info = killer if isinstance(killer, dict) else {}
        victim_info = victim if isinstance(victim, dict) else {}
        attacker_pos = _position(data.get("killer_pos"))
        victim_pos = _position(data.get("victim_pos"))
        angle = data.get("killer_angle") if isinstance(data.get("killer_angle"), dict) else {}
        yaw = angle.get("yaw") if _finite(angle.get("yaw")) else None
        pitch = angle.get("pitch") if _finite(angle.get("pitch")) else None
        distance = angle_diff = None
        if attacker_pos is not None and victim_pos is not None:
            delta = {axis: victim_pos[axis] - attacker_pos[axis] for axis in "xyz"}
            distance = math.sqrt(sum(value * value for value in delta.values()))
            if yaw is not None and (delta["x"] or delta["y"]):
                target_yaw = math.degrees(math.atan2(delta["y"], delta["x"]))
                angle_diff = abs((target_yaw - yaw + 180) % 360 - 180)
        return {
            "tick": event.get("tick"),
            "attacker_name": killer_info.get("name", killer if isinstance(killer, str) else None),
            "attacker_steam_id": str(killer_info["steam_id"]) if "steam_id" in killer_info else None,
            "attacker_pos": attacker_pos, "victim_pos": victim_pos,
            "victim_name": victim_info.get("name", victim if isinstance(victim, str) else None),
            "victim_steam_id": str(victim_info["steam_id"]) if "steam_id" in victim_info else None,
            "weapon": data.get("weapon"), "is_headshot": data.get("headshot"),
            "wallbang": data.get("wallbang"), "attacker_team": killer_info.get("team"),
            "victim_team": victim_info.get("team"), "map_name": map_name,
            "round_number": data.get("round_number"), "distance": distance,
            "angle_diff": angle_diff, "attacker_view_x": yaw, "attacker_view_y": pitch,
            "victim_view_x": None, "victim_view_y": None, "demo_file": demo_file,
            "snapshot_phase": "kill_event",
        }
