"""Closed triangle-surface topology and parity queries shared by collision checks."""

from __future__ import annotations

import numpy as np


_CONTACT_TOLERANCE_M = 1e-9


def mesh_component_topology(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one used source vertex per surface component and closed faces.

    Faces belong to the same component when they share an edge after exact
    position welding.  Face/normal seams therefore do not split closed shells.
    Representatives always index vertices actually referenced by triangles;
    coincident unreferenced points cannot create a component or its sample.

    Closed components have exactly two incidences for each welded edge.  Their
    original triangles remain combined, so parity queries retain the cavities
    of nested shells while excluding unrelated open components.
    """

    points = np.asarray(vertices, dtype=float)
    faces = np.asarray(triangles, dtype=np.intp).reshape(-1, 3)
    face_count = len(faces)
    if not face_count:
        return np.empty(0, dtype=np.intp), np.empty((0, 3), dtype=np.intp)

    _, welded_ids = np.unique(points, axis=0, return_inverse=True)
    welded_faces = welded_ids[faces]
    edges = np.concatenate(
        (welded_faces[:, (0, 1)], welded_faces[:, (1, 2)], welded_faces[:, (2, 0)])
    )
    edges.sort(axis=1)
    _, edge_ids, edge_counts = np.unique(
        edges, axis=0, return_inverse=True, return_counts=True
    )
    edge_faces = np.tile(np.arange(face_count, dtype=np.intp), 3)
    edge_order = np.argsort(edge_ids, kind="stable")
    sorted_edges = edge_ids[edge_order]
    sorted_faces = edge_faces[edge_order]
    shared = sorted_edges[1:] == sorted_edges[:-1]

    parents = list(range(face_count))
    ranks = [0] * face_count

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for first, second in zip(sorted_faces[:-1][shared], sorted_faces[1:][shared]):
        first_root = find(int(first))
        second_root = find(int(second))
        if first_root == second_root:
            continue
        if ranks[first_root] < ranks[second_root]:
            first_root, second_root = second_root, first_root
        parents[second_root] = first_root
        if ranks[first_root] == ranks[second_root]:
            ranks[first_root] += 1

    roots = np.fromiter((find(index) for index in range(face_count)), dtype=np.intp)
    components, component_ids = np.unique(roots, return_inverse=True)
    representatives = np.full(len(components), len(points), dtype=np.intp)
    np.minimum.at(representatives, component_ids, np.min(faces, axis=1))
    representatives.sort()

    face_closed = np.all(edge_counts[edge_ids].reshape(3, face_count) == 2, axis=0)
    component_closed = np.ones(len(components), dtype=bool)
    component_closed[component_ids[~face_closed]] = False
    closed_faces = faces[component_closed[component_ids]]
    return representatives, closed_faces


def mesh_is_closed(vertices: np.ndarray, triangles: np.ndarray) -> bool:
    """Return whether every welded triangle edge belongs to two faces.

    USD may duplicate a position at a face/normal seam.  Weld equal positions
    for topology inspection without changing the actual collision geometry.
    """

    _, vertex_ids = np.unique(vertices, axis=0, return_inverse=True)
    faces = vertex_ids[triangles]
    edges = np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]))
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(len(counts) and np.all(counts == 2))


def origin_inside_closed_mesh(vertices: np.ndarray, triangles: np.ndarray) -> bool:
    """Test the origin inside a known closed surface using ray parity.

    Triangle orientation is irrelevant, and nested shells preserve cavities.
    The caller must establish closure with :func:`mesh_is_closed` first.
    """

    faces = vertices[triangles]
    first_edges = faces[:, 1] - faces[:, 0]
    second_edges = faces[:, 2] - faces[:, 0]
    inside_votes = 0
    # Three non-axis-aligned rays avoid depending on a single ray hitting an
    # edge or vertex.  Duplicate hits along triangulation seams count once.
    for direction in (
        (1.0, 0.371390676, 0.52917721),
        (0.217, 1.0, 0.713),
        (0.619, 0.283, 1.0),
    ):
        ray = np.asarray(direction, dtype=float)
        ray /= np.linalg.norm(ray)
        h = np.cross(ray, second_edges)
        determinant = np.einsum("ij,ij->i", first_edges, h)
        usable = np.abs(determinant) > np.finfo(float).eps
        if not np.any(usable):
            continue
        inverse = 1.0 / determinant[usable]
        relative = -faces[usable, 0]
        u = inverse * np.einsum("ij,ij->i", relative, h[usable])
        q = np.cross(relative, first_edges[usable])
        v = inverse * (q @ ray)
        distance = inverse * np.einsum("ij,ij->i", second_edges[usable], q)
        barycentric_tolerance = 1e-12
        hits = np.sort(distance[
            (u >= -barycentric_tolerance)
            & (v >= -barycentric_tolerance)
            & (u + v <= 1.0 + barycentric_tolerance)
            & (distance > _CONTACT_TOLERANCE_M)
        ])
        unique_count = 0 if not len(hits) else 1 + int(
            np.count_nonzero(np.diff(hits) > _CONTACT_TOLERANCE_M)
        )
        inside_votes += unique_count % 2
    return inside_votes >= 2


def point_inside_closed_mesh(
    point: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> bool:
    """Test a world point against an already verified closed triangle surface.

    Closure is deliberately checked by the caller so repeated geometry queries
    can reuse that result.  Broad bounds only reject points; ray parity decides
    containment and preserves empty regions between parts or nested shells.
    """

    relative = np.asarray(vertices, dtype=float) - np.asarray(point, dtype=float)
    if np.any(np.min(relative, axis=0) >= 0.0) or np.any(
        np.max(relative, axis=0) <= 0.0
    ):
        return False
    return origin_inside_closed_mesh(relative, np.asarray(triangles, dtype=np.intp))
