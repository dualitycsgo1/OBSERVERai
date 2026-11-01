"""Regression tests for truthful demo parsing and causal training data boundaries."""

import math
import os
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from comprehensive_demo_processor import ComprehensiveDemoProcessor
from comprehensive_demo_processor import uncompressed_demo
from complete_demo_parser import CS2DemoParser


class DemoProcessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.demo_path = os.path.join(self.temp.name, "match.dem")
        with open(self.demo_path, "wb") as demo:
            demo.write(b"fixture, not a real demo")
        self.processor = ComprehensiveDemoProcessor()

    def process(self, parsed, header=None):
        with patch.object(self.processor, "extract_header_info", return_value=header), \
                patch.object(self.processor, "parse_demo_content", return_value=parsed):
            return self.processor.process_demo_comprehensive(self.demo_path)

    def test_header_only_is_not_success(self):
        result = self.process(None, {"success": True, "map_name": "de_nuke"})
        self.assertFalse(result["success"])
        self.assertFalse(result["training_ready"])
        self.assertIsNotNone(result["error"])

    def test_parser_failure_json_is_not_success(self):
        result = self.process({"success": False, "error": "corrupt demo", "kills": [{}]})
        self.assertFalse(result["success"])

    def test_legacy_kills_not_hidden_by_other_events(self):
        result = self.process({"events": [{"type": "round_start"}], "kills": [{"tick": 8}],
                               "map_name": "de_nuke"})
        self.assertEqual(result["kill_count"], 1)
        self.assertEqual(result["kills"][0]["demo_file"], "match.dem")
        self.assertFalse(result["training_ready"])
        self.assertIn("Kill events alone", result["training_warning"])

    def test_parser_map_resolved_before_event_conversion(self):
        result = self.process({"map_name": "de_nuke", "events": [{"type": "kill", "data": {}}]})
        self.assertEqual(result["kills"][0]["map_name"], "de_nuke")

    def test_absent_data_not_replaced_with_zero_coordinates(self):
        kill = self.processor.convert_event_to_kill_format({"data": {}}, "de_nuke")
        for key in ("attacker_pos", "victim_pos", "distance", "angle_diff", "attacker_view_x"):
            self.assertIsNone(kill[key])
        self.assertEqual(kill["snapshot_phase"], "kill_event")

    def test_distance_and_wrapped_aim_error_from_real_positions(self):
        kill = self.processor.convert_event_to_kill_format({"data": {
            "killer_pos": {"x": 0, "y": 0, "z": 0},
            "victim_pos": {"x": 3, "y": 4, "z": 12},
            "killer_angle": {"yaw": math.degrees(math.atan2(4, 3)) + 360, "pitch": 0},
            "distance": 169,
        }}, "de_nuke")
        self.assertEqual(kill["distance"], 13)
        self.assertAlmostEqual(kill["angle_diff"], 0)

    def test_malformed_coordinates_are_unknown(self):
        kill = self.processor.convert_event_to_kill_format({"data": {
            "killer_pos": {"x": float("nan"), "y": 0, "z": 0},
            "victim_pos": {"x": 0, "y": 0},
        }}, "de_nuke")
        self.assertIsNone(kill["attacker_pos"])
        self.assertIsNone(kill["victim_pos"])

    def test_prediction_samples_pass_through_without_kills(self):
        samples = [{"sample_tick": 64, "label": 0}, {"sample_tick": 64, "label": 1}]
        result = self.process({"success": True, "prediction_samples": samples})
        self.assertTrue(result["success"])
        self.assertTrue(result["training_ready"])
        self.assertEqual(self.processor.get_training_data_from_results([result]), samples)

    def test_invalid_schema_rejected(self):
        result = self.process({"kills": {"attacker": 1}})
        self.assertFalse(result["success"])
        self.assertIn("Invalid demo schema", result["error"])

    def test_tool_rejects_failure_even_with_zero_exit_code(self):
        completed = SimpleNamespace(returncode=0, stdout='{"success": false, "error": "bad demo"}', stderr="")
        with patch("comprehensive_demo_processor.subprocess.run", return_value=completed):
            self.assertIsNone(self.processor._run_tool(self.demo_path, self.demo_path, 1))
        self.assertIn("bad demo", self.processor.last_errors[0])

    def test_relative_demo_path_resolved_before_tool_changes_cwd(self):
        completed = SimpleNamespace(returncode=0, stdout='{"success": true}', stderr="")
        with patch("comprehensive_demo_processor.subprocess.run", return_value=completed) as run:
            self.processor._run_tool(self.demo_path, "relative.dem", 1)
        self.assertEqual(run.call_args.args[0][1], os.path.abspath("relative.dem"))

    def test_nonzero_parser_exit_preserves_json_error_details(self):
        completed = SimpleNamespace(returncode=1, stdout='{"success": false, "error": "entity desync"}', stderr="")
        with patch("comprehensive_demo_processor.subprocess.run", return_value=completed):
            self.assertIsNone(self.processor._run_tool(self.demo_path, self.demo_path, 1))
        self.assertIn("entity desync", self.processor.last_errors[0])

    def test_missing_tool_has_actionable_error(self):
        self.assertIsNone(self.processor._run_tool(os.path.join(self.temp.name, "absent.exe"), self.demo_path, 1))
        self.assertIn("Required tool not found", self.processor.last_errors[0])

    def test_complete_parser_capability_flags_follow_observed_data(self):
        parser = CS2DemoParser.__new__(CS2DemoParser)
        result = parser._enhance_demo_data({"events": [{"type": "kill", "data": {}}]})
        for flag in ("has_positions", "has_angles", "has_health_armor", "has_distances"):
            self.assertFalse(result[flag])

    def test_complete_parser_no_events_does_not_claim_full_data(self):
        parser = CS2DemoParser.__new__(CS2DemoParser)
        result = parser._enhance_demo_data({})
        self.assertFalse(result["has_positions"])

    def test_complete_parser_supports_actual_legacy_kill_schema(self):
        parser = CS2DemoParser.__new__(CS2DemoParser)
        result = parser._enhance_demo_data({"kills": [{
            "attacker_pos": {"x": 0, "y": 0, "z": 0},
            "victim_pos": {"x": 3, "y": 4, "z": 0},
            "attacker_view_x": 0, "attacker_view_y": 0, "distance": 5,
        }]})
        self.assertTrue(result["has_positions"])
        self.assertTrue(result["has_angles"])
        self.assertFalse(result["has_health_armor"])
        self.assertEqual(result["analysis"]["total_kills"], 1)
        self.assertEqual(len(parser.get_kills_with_full_context(result)), 1)

    def test_zstandard_demo_is_streamed_and_source_preserved(self):
        import zstandard
        contents = b"demo-fixture" * 200000
        compressed = self.demo_path + ".zst"
        with open(compressed, "wb") as destination:
            destination.write(zstandard.ZstdCompressor().compress(contents))
        with uncompressed_demo(compressed) as unpacked:
            self.assertNotEqual(unpacked, compressed)
            with open(unpacked, "rb") as source:
                self.assertEqual(source.read(), contents)
        self.assertFalse(os.path.exists(unpacked))
        self.assertTrue(os.path.exists(compressed))

    def test_compressed_result_keeps_original_identity(self):
        import zstandard
        compressed = self.demo_path + ".zst"
        with open(compressed, "wb") as destination:
            destination.write(zstandard.ZstdCompressor().compress(b"demo"))
        with patch.object(self.processor, "extract_header_info", return_value=None), \
                patch.object(self.processor, "parse_demo_content", return_value={
                    "success": True, "kills": [{"tick": 1}], "demo_id": "uncompressed-sha",
                }):
            result = self.processor.process_demo_comprehensive(compressed)
        self.assertTrue(result["success"])
        self.assertEqual(result["demo_file"], "match.dem.zst")
        self.assertEqual(result["demo_path"], compressed)
        self.assertEqual(result["kills"][0]["demo_file"], "match.dem.zst")
        self.assertEqual(result["demo_id"], "uncompressed-sha")

    def test_corrupt_zstandard_fails_without_training_data(self):
        compressed = self.demo_path + ".zst"
        with open(compressed, "wb") as destination:
            destination.write(b"not-zstd")
        result = self.processor.process_demo_comprehensive(compressed)
        self.assertFalse(result["success"])
        self.assertEqual(result["prediction_samples"], [])
        self.assertIn("decompression", result["error"])

    def test_optional_header_tool_absence_is_not_an_error(self):
        self.processor.header_extractor = os.path.join(self.temp.name, "absent.exe")
        self.assertIsNone(self.processor.extract_header_info(self.demo_path))
        self.assertEqual(self.processor.last_errors, [])


if __name__ == "__main__":
    unittest.main()
