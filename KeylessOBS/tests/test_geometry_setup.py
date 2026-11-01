"""Offline tests for collision-asset provenance and installation integrity."""

import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from line_of_sight import MapGeometryManager
from setup_map_geometry import install_geometry


class GeometrySetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = self.root / 'maps'
        self.archive = self.root / 'geometry.zip'
        self.manifest = self.root / 'manifest.json'
        self.version = '1234567'
        triangles = np.array([[(0, -100, 0), (0, 100, 0), (0, 0, 100)]], dtype='<f4')
        self.mesh = (struct.pack('<4sIII', b'AWMH', 1, 3, 1) + triangles.tobytes()
                     + np.arange(3, dtype='<u4').tobytes())
        self.write_release({'de_dust2.mesh': self.mesh})

    def tearDown(self):
        self.temp.cleanup()

    def write_release(self, members):
        with zipfile.ZipFile(self.archive, 'w') as archive:
            for name, raw in members.items():
                archive.writestr(name, raw)
        self.manifest.write_text(json.dumps({
            'release_key': self.version, 'client_version': self.version,
            'source_revision': 'abc', 'buildid': '456',
            'artifacts': {'geometry.zip': {
                'sha256': hashlib.sha256(self.archive.read_bytes()).hexdigest(),
                'bytes': self.archive.stat().st_size}}}), encoding='utf-8')

    def install(self, maps=('de_dust2',), **kwargs):
        return install_geometry(self.version, maps, self.output, self.archive, self.manifest, **kwargs)

    def test_installs_selected_map_with_version_and_both_checksums(self):
        result = self.install()[0]
        self.assertEqual((self.output / 'de_dust2.mesh').read_bytes(), self.mesh)
        metadata = json.loads((self.output / 'de_dust2_collision_source.json').read_text())
        self.assertEqual(metadata, result)
        self.assertEqual(metadata['client_version'], self.version)
        self.assertEqual(metadata['upstream_mesh_sha256'], hashlib.sha256(self.mesh).hexdigest())
        self.assertEqual(metadata['installed_sha256'], metadata['upstream_mesh_sha256'])
        self.assertEqual(metadata['triangle_count'], 1)

    def test_checksum_mismatch_cannot_install_any_mesh(self):
        self.archive.write_bytes(self.archive.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            self.install()
        self.assertFalse(self.output.exists())

    def test_manifest_version_must_match_requested_version(self):
        with self.assertRaisesRegex(ValueError, 'ClientVersion'):
            install_geometry('999', ['de_dust2'], self.output, self.archive, self.manifest)

    def test_all_selected_assets_validate_before_first_file_is_installed(self):
        self.write_release({'de_dust2.mesh': self.mesh, 'de_nuke.mesh': b'bad mesh'})
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            self.install(('de_dust2', 'de_nuke'))
        self.assertFalse((self.output / 'de_dust2.mesh').exists())
        self.assertFalse((self.output / 'de_nuke.mesh').exists())

    def test_existing_asset_requires_force(self):
        self.install()
        with self.assertRaises(FileExistsError):
            self.install()
        self.install(force=True)

    def test_npz_conversion_loads_in_observer(self):
        self.install(output_format='npz')
        self.assertTrue((self.output / 'de_dust2_collision.npz').is_file())
        manager = MapGeometryManager(self.output)
        self.assertTrue(manager.set_current_map('de_dust2'))
        self.assertFalse(manager.has_line_of_sight((-100, 0, 50), (100, 0, 50)))

    def test_mixed_formats_require_explicit_cleanup(self):
        self.install(output_format='npz')
        with self.assertRaisesRegex(FileExistsError, 'other collision format'):
            self.install(output_format='mesh', force=True)


if __name__ == '__main__':
    unittest.main()
