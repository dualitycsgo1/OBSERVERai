"""Reject malformed exports and unsafe destinations without invoking game tools."""

import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspect_map_vpk import NAV_MAGIC, VPK_MAGIC, inspect_glb, inspect_map_vpk, inspect_nav, inspect_place_entities, validate_paths


class InspectorValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_nav_requires_real_magic_and_successfully_parsed_area_count(self):
        path = self.root / 'map.nav'
        path.write_bytes(struct.pack('<III', 0, 36, 1))
        with self.assertRaisesRegex(ValueError, 'magic'):
            inspect_nav(path, 'Version: 36\nNumber of areas: 2242\n')
        path.write_bytes(struct.pack('<III', NAV_MAGIC, 36, 1))
        with self.assertRaisesRegex(ValueError, 'area count'):
            inspect_nav(path, 'Version: 36\nException: truncated body\n')
        self.assertEqual(inspect_nav(path, 'Version: 36\nNumber of areas: 2242\n')['area_count'], 2242)

    def test_glb_requires_matching_length_and_accessor_bounds(self):
        path = self.root / 'physics.glb'
        path.write_bytes(struct.pack('<4sII', b'glTF', 2, 200))
        with self.assertRaisesRegex(ValueError, 'declared length'):
            inspect_glb(path)
        document = {'asset': {'version': '2.0'}, 'buffers': [{'byteLength': 4}],
                    'bufferViews': [{'buffer': 0, 'byteLength': 4}],
                    'accessors': [{'bufferView': 0, 'componentType': 5126, 'type': 'VEC3', 'count': 3}]}
        raw = json.dumps(document).encode()
        raw += b' ' * (-len(raw) % 4)
        chunks = struct.pack('<II', len(raw), 0x4E4F534A) + raw + struct.pack('<II', 4, 0x004E4942) + bytes(4)
        path.write_bytes(struct.pack('<4sII', b'glTF', 2, 12 + len(chunks)) + chunks)
        with self.assertRaisesRegex(ValueError, 'accessor exceeds'):
            inspect_glb(path)

    def test_cli_zero_exit_and_protected_game_output_are_not_accepted(self):
        game = self.root / 'install' / 'game' / 'csgo'
        maps = game / 'maps'
        maps.mkdir(parents=True)
        vpk, cli = maps / 'de_test.vpk', self.root / 'Source2Viewer-CLI.exe'
        vpk.write_bytes(struct.pack('<III', VPK_MAGIC, 1, 0))
        cli.write_bytes(b'fixture-not-executed')
        with self.assertRaisesRegex(ValueError, 'outside'):
            validate_paths(cli, vpk, game / 'inspection')
        output = self.root / 'separate-output'
        result = SimpleNamespace(returncode=0, stdout='Version: 20.1\nException: export failed\n', stderr='')
        with patch('inspect_map_vpk.subprocess.run', return_value=result):
            with self.assertRaisesRegex(ValueError, 'expected NAV'):
                inspect_map_vpk(cli, vpk, output)
        self.assertFalse((output / 'inspection_manifest.json').exists())
        self.assertEqual(vpk.read_bytes(), struct.pack('<III', VPK_MAGIC, 1, 0))

    def test_place_inventory_preserves_real_entity_transforms_and_model_references(self):
        path = self.root / 'entities.txt'
        path.write_text('====7====\nplace_name "UpperTunnel"\nclassname "env_cs_place"\n'
                        'origin [ -1984, 1604, 160 ]\nangles [ 0, 0, 0 ]\nscales [ 1, 1, 1 ]\n'
                        'hammeruniqueid "2:85"\nmodel resource_name:"maps/de_test/entities/volume.vmdl"\n')
        record = inspect_place_entities(path)
        self.assertEqual(record['place_entity_count'], 1)
        self.assertEqual(record['places'][0]['origin'], [-1984, 1604, 160])
        self.assertEqual(record['places'][0]['compiled_model'], 'maps/de_test/entities/volume.vmdl_c')
        path.write_text(path.read_text().replace('[ 0, 0, 0 ]', '[ 0, 0 ]'))
        with self.assertRaisesRegex(ValueError, 'transform'):
            inspect_place_entities(path)


if __name__ == '__main__':
    unittest.main()
