from __future__ import annotations

import numpy as np

from decanting_workspace.mesh_geometry import (
    mesh_component_topology,
    mesh_is_closed,
    point_inside_closed_mesh,
)


def _cube(size=2.0):
    vertices = np.array(
        ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
         (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)),
        dtype=float,
    ) * size / 2.0
    triangles = np.array(
        ((0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
         (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
         (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)),
        dtype=np.intp,
    )
    return vertices, triangles


def test_split_uv_cube_is_one_closed_component():
    vertices, triangles = _cube()
    split_vertices = vertices[triangles].reshape(-1, 3)
    split_triangles = np.arange(len(split_vertices), dtype=np.intp).reshape(-1, 3)

    representatives, closed_faces = mesh_component_topology(split_vertices, split_triangles)

    assert representatives.dtype == np.dtype(np.intp)
    assert representatives.tolist() == [0]
    assert np.array_equal(closed_faces, split_triangles)
    assert mesh_is_closed(split_vertices, closed_faces)


def test_disconnected_cube_and_open_sheet_have_two_samples_and_only_closed_cube():
    cube_vertices, cube_triangles = _cube()
    sheet = np.array(((3, -1, 0), (5, -1, 0), (5, 1, 0), (3, 1, 0)), dtype=float)
    vertices = np.concatenate((cube_vertices, sheet))
    triangles = np.concatenate((cube_triangles, np.array(((8, 9, 10), (8, 10, 11)))))

    representatives, closed_faces = mesh_component_topology(vertices, triangles)

    assert representatives.tolist() == [0, 8]
    assert np.array_equal(closed_faces, cube_triangles)
    assert not mesh_is_closed(vertices, triangles)
    assert point_inside_closed_mesh(np.zeros(3), vertices, closed_faces)


def test_unreferenced_coincident_vertex_is_never_a_component_sample():
    cube_vertices, cube_triangles = _cube()
    vertices = np.concatenate((cube_vertices[:1], cube_vertices, [[25.0, 25.0, 25.0]]))
    triangles = cube_triangles + 1

    representatives, closed_faces = mesh_component_topology(vertices, triangles)

    assert representatives.tolist() == [1]
    assert set(representatives).issubset(set(triangles.ravel()))
    assert np.array_equal(closed_faces, triangles)


def test_nested_closed_shell_components_keep_their_cavity():
    outer_vertices, outer_faces = _cube(4.0)
    inner_vertices, inner_faces = _cube(2.0)
    vertices = np.concatenate((outer_vertices, inner_vertices))
    triangles = np.concatenate((outer_faces, inner_faces + len(outer_vertices)))

    representatives, closed_faces = mesh_component_topology(vertices, triangles)

    assert representatives.tolist() == [0, 8]
    assert np.array_equal(closed_faces, triangles)
    assert not point_inside_closed_mesh(np.zeros(3), vertices, closed_faces)
    assert point_inside_closed_mesh(np.array((1.5, 0.0, 0.0)), vertices, closed_faces)


def test_empty_triangle_surface_has_no_components():
    representatives, closed_faces = mesh_component_topology(
        np.zeros((1, 3)), np.empty((0, 3), dtype=np.intp)
    )

    assert representatives.shape == (0,)
    assert closed_faces.shape == (0, 3)
