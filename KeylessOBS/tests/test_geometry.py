"""Regression tests for real collision visibility, unknown data and Source angles."""

from pathlib import Path
import sys
import struct
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from line_of_sight import (MapGeometryManager, TriangleMeshLOS, aim_angles,
                           aim_error_degrees, duel_geometrically_possible,
                           normalize_map_name)
from awpy_nav_integration import AWPYNavGeometryManager


def wall(x=0, low=-100, high=200, half_width=200):
    return np.array([[(x, -half_width, low), (x, half_width, low), (x, half_width, high)],
                     [(x, -half_width, low), (x, half_width, high), (x, -half_width, high)]], dtype=float)


class TriangleVisibilityTests(unittest.TestCase):
    def setUp(self):
        self.mesh = TriangleMeshLOS(wall())

    def test_solid_wall_blocks_in_both_directions_and_windings(self):
        a, b = (-100, 0, 64), (100, 0, 64)
        for mesh in (self.mesh, TriangleMeshLOS(wall()[:, ::-1])):
            self.assertFalse(mesh.has_line_of_sight(a, b))
            self.assertFalse(mesh.has_line_of_sight(b, a))

    def test_clear_ray_beside_and_above_wall(self):
        self.assertTrue(self.mesh.has_line_of_sight((-100, 201, 64), (100, 201, 64)))
        self.assertTrue(self.mesh.has_line_of_sight((-100, 0, 201), (100, 0, 201)))

    def test_segment_does_not_hit_wall_behind_target(self):
        self.assertTrue(self.mesh.has_line_of_sight((-100, 0, 64), (-1, 0, 64)))
        self.assertTrue(self.mesh.has_line_of_sight((1, 0, 64), (100, 0, 64)))

    def test_surface_endpoints_do_not_count_as_interior_occluders(self):
        self.assertTrue(self.mesh.has_line_of_sight((-100, 0, 64), (0, 0, 64)))
        self.assertTrue(self.mesh.has_line_of_sight((0, 0, 64), (100, 0, 64)))

    def test_coplanar_ray_through_wall_is_blocked(self):
        self.assertFalse(self.mesh.has_line_of_sight((0, -300, 64), (0, 300, 64)))
        self.assertTrue(self.mesh.has_line_of_sight((0, 250, 64), (0, 300, 64)))

    def test_floor_between_levels_blocks_vertical_ray(self):
        floor = np.array([[(-200, -200, 100), (200, -200, 100), (200, 200, 100)],
                          [(-200, -200, 100), (200, 200, 100), (-200, 200, 100)]])
        mesh = TriangleMeshLOS(floor)
        self.assertFalse(mesh.has_line_of_sight((0, 0, 64), (0, 0, 164)))
        self.assertTrue(mesh.has_line_of_sight((0, 0, 164), (100, 0, 164)))

    def test_ray_through_triangle_seam_and_edge_blocks(self):
        self.assertFalse(self.mesh.has_line_of_sight((-100, 0, 50), (100, 0, 50)))
        self.assertFalse(self.mesh.has_line_of_sight((-100, 200, 64), (100, 200, 64)))

    def test_invalid_mesh_is_rejected(self):
        for triangles in ([], np.zeros((1, 3, 3)), np.full((1, 3, 3), np.nan)):
            with self.assertRaises(ValueError):
                TriangleMeshLOS(triangles)
        with self.assertRaises(ValueError):
            self.mesh.has_line_of_sight((float('nan'), 0, 0), (1, 1, 1))

    def test_bvh_partition_matches_brute_force_on_random_segments(self):
        rng = np.random.default_rng(28)
        triangles = rng.uniform(-100, 100, (200, 3, 3))
        indexed = TriangleMeshLOS(triangles, leaf_size=4)
        for a, b in rng.uniform(-150, 150, (100, 2, 3)):
            delta = b - a
            length = np.linalg.norm(delta)
            brute = not indexed._triangles_hit(np.arange(len(triangles)), a, delta / length, length)
            self.assertEqual(indexed.has_line_of_sight(a, b), brute)


class MapAndDuelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.maps = Path(self.temp.name)
        self.manager = MapGeometryManager(str(self.maps))

    def tearDown(self):
        self.temp.cleanup()

    def save_wall(self, map_name='de_dust2', **kwargs):
        np.savez(self.maps / f'{map_name}_collision.npz', triangles=wall(**kwargs))
        self.assertTrue(self.manager.set_current_map(map_name))

    def test_missing_geometry_is_unknown_and_does_not_invent_files(self):
        self.assertFalse(self.manager.set_current_map('de_dust2'))
        result = self.manager.visibility((-100, 0, 64), (100, 0, 64))
        self.assertEqual(result.state, 'unknown')
        self.assertIsNone(result.visible)
        self.assertFalse(self.manager.use_real_geometry)
        self.assertEqual(list(self.maps.iterdir()), [])
        self.assertIsNone(duel_geometrically_possible((-100, 0, 0), (100, 0, 0), self.manager))

    def test_nav_and_legacy_blockmask_are_not_collision_meshes(self):
        (self.maps / 'de_dust2.nav').write_bytes(b'not a parsed nav')
        np.save(self.maps / 'de_dust2_blockmask.npy', np.ones((10, 10)))
        self.assertFalse(self.manager.set_current_map('de_dust2'))
        self.assertIsNone(self.manager.has_line_of_sight((-1, 0, 0), (1, 0, 0)))
        nav = AWPYNavGeometryManager(str(self.maps))
        self.assertTrue(nav.set_current_map('dust2'))
        self.assertIsNone(nav.has_line_of_sight((0, 0, 0), (1, 1, 1)))
        self.assertIsNone(nav.get_distance_along_navmesh((0, 0, 0), (1, 1, 1)))

    def test_map_switch_to_missing_asset_cannot_reuse_previous_map(self):
        self.save_wall()
        self.assertFalse(self.manager.has_line_of_sight((-100, 0, 64), (100, 0, 64)))
        self.manager.set_current_map('de_nuke')
        self.assertEqual(self.manager.current_map, 'de_nuke')
        self.assertIsNone(self.manager.has_line_of_sight((-100, 0, 64), (100, 0, 64)))

    def test_npz_indexed_mesh_and_invalid_face_indices(self):
        triangles = wall()
        np.savez(self.maps / 'de_dust2_collision.npz', vertices=triangles.reshape(-1, 3), faces=np.arange(6).reshape(-1, 3))
        self.assertTrue(self.manager.set_current_map('de_dust2'))
        self.assertFalse(self.manager.has_line_of_sight((-100, 0, 64), (100, 0, 64)))
        np.savez(self.maps / 'de_nuke_collision.npz', vertices=triangles.reshape(-1, 3), faces=[[-1, 0, 1]])
        self.assertFalse(self.manager.set_current_map('de_nuke'))
        self.assertIsNone(self.manager.has_line_of_sight((-100, 0, 64), (100, 0, 64)))

    def test_partial_cover_does_not_hide_visible_head(self):
        self.save_wall(high=50)
        self.assertTrue(duel_geometrically_possible((-100, 0, 0), (100, 0, 0), self.manager))
        self.assertFalse(duel_geometrically_possible((-100, 0, 0), (100, 0, 0), self.manager,
                                                   crouched_a=True, crouched_b=True))

    def test_full_cover_blocks_sampled_direct_engagement(self):
        self.save_wall()
        self.assertFalse(duel_geometrically_possible((-100, 0, 0), (100, 0, 0), self.manager))

    def test_no_arbitrary_minimum_or_maximum_kill_distance(self):
        self.save_wall(half_width=200)
        self.assertTrue(duel_geometrically_possible((-5, 300, 0), (5, 300, 0), self.manager))
        self.assertTrue(duel_geometrically_possible((-5000, 300, 0), (5000, 300, 0), self.manager))

    def test_invalid_positions_are_unknown(self):
        self.save_wall()
        for position in ((float('nan'), 0, 0), (0, 0), None):
            self.assertIsNone(self.manager.has_line_of_sight(position, (1, 1, 1)))
            self.assertIsNone(duel_geometrically_possible(position, (1, 1, 1), self.manager))

    def test_native_awpy_mesh_parser_without_optional_dependency(self):
        triangles = wall().astype('<f4')
        raw = (struct.pack('<4sIII', b'AWMH', 1, 6, 2) + triangles.tobytes()
               + np.arange(6, dtype='<u4').tobytes())
        (self.maps / 'de_dust2.mesh').write_bytes(raw)
        self.assertTrue(self.manager.set_current_map('de_dust2'))
        self.assertFalse(self.manager.has_line_of_sight((-100, 0, 64), (100, 0, 64)))
        (self.maps / 'de_nuke.mesh').write_bytes(raw[:-1])
        self.assertFalse(self.manager.set_current_map('de_nuke'))
        self.assertIsNone(self.manager.has_line_of_sight((-100, 0, 64), (100, 0, 64)))

    def test_native_awpy_mesh_rejects_bad_header_and_indices(self):
        path = self.maps / 'invalid.mesh'
        for raw in (b'AWMH', struct.pack('<4sIII', b'NAVM', 1, 0, 0),
                    struct.pack('<4sIII', b'AWMH', 999, 0, 0),
                    struct.pack('<4sIII', b'AWMH', 1, 1, 1) + bytes(12)
                    + np.array([0, 1, 2], dtype='<u4').tobytes()):
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                TriangleMeshLOS.from_awpy_mesh(path)

    def test_map_aliases_paths_and_custom_prefixes(self):
        self.assertEqual(normalize_map_name('dust2'), 'de_dust2')
        self.assertEqual(normalize_map_name('workshop/123/de_mirage.vpk'), 'de_mirage')
        self.assertEqual(normalize_map_name('cs_office'), 'cs_office')
        with self.assertRaises(ValueError):
            normalize_map_name('../..')


class AngleTests(unittest.TestCase):
    def test_source_pitch_yaw_and_vertical_angles(self):
        self.assertEqual(aim_angles((0, 0, 0), (0, 100, 100)), (-45, 90))
        self.assertEqual(aim_angles((0, 0, 0), (0, 0, 100)), (-90, 0))

    def test_forward_vector_uses_full_3d_and_normalization(self):
        self.assertAlmostEqual(aim_error_degrees((2, 0, 0), (0, 0, 0), (100, 0, 100)), 45)
        self.assertAlmostEqual(aim_error_degrees((1, 0, 0), (0, 0, 0), (-100, 0, 0)), 180)
        self.assertIsNone(aim_error_degrees((0, 0, 0), (0, 0, 0), (100, 0, 0)))


if __name__ == '__main__':
    unittest.main()
