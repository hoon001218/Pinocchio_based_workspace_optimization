"""Simplified solid union with each overlapping volume assigned once.

The original boxes stay in LayoutScene. Every preview clips them afresh, so
moving an object away from an overlap restores its complete original solid.
Convex clipping also preserves authored inclined surfaces.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np

from .layout_models import LayoutBox, LayoutFragment, LayoutScene
from .transforms import quaternion_matrix


_EPS = 1e-9
_VOLUME_EPS = 1e-12
Polyhedron = tuple[np.ndarray, ...]


def build_layout_geometry(scene: LayoutScene) -> tuple[LayoutFragment, ...]:
    """Keep the first object's ownership, subtracting it from later solids."""

    fragments: list[LayoutFragment] = []
    occupied: list[tuple[Polyhedron, tuple[tuple[np.ndarray, float], ...], np.ndarray, np.ndarray]] = []
    for box in scene.boxes:
        original = _box_polyhedron(box)
        vertices = np.concatenate(original)
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        pieces = [original]
        for _, planes, other_lower, other_upper in occupied:
            if not _bounds_overlap(lower, upper, other_lower, other_upper):
                continue
            remaining: list[Polyhedron] = []
            for piece in pieces:
                points = np.concatenate(piece)
                if not _bounds_overlap(points.min(axis=0), points.max(axis=0), other_lower, other_upper):
                    remaining.append(piece)
                    continue
                remaining.extend(_subtract_convex(piece, planes))
            pieces = remaining
            if not pieces:
                break
        for index, piece in enumerate(pieces):
            fragments.append(_fragment(box, piece, index))
        occupied.append((original, _box_planes(box), lower, upper))
    return tuple(fragments)


def fragment_volume_m3(fragment: LayoutFragment) -> float:
    vertices = np.asarray(fragment.vertices_m)
    centre = vertices.mean(axis=0)
    triangles = vertices[np.asarray(fragment.triangles)] - centre
    return abs(float(np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum())) / 6.


def _bounds_overlap(first_lower, first_upper, second_lower, second_upper) -> bool:
    return bool(np.all(np.minimum(first_upper, second_upper) - np.maximum(first_lower, second_lower) > _EPS))


def _box_polyhedron(box: LayoutBox) -> Polyhedron:
    local = np.array(((-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
                      (-1,-1,1), (1,-1,1), (1,1,1), (-1,1,1)), dtype=float)
    rotation = quaternion_matrix((0., 0., 0.), box.rotation_xyzw)[:3, :3]
    vertices = (local * np.asarray(box.size_m) / 2.) @ rotation.T + box.center_m
    faces = ((3,2,1,0), (4,5,6,7), (0,4,7,3), (1,2,6,5), (0,1,5,4), (2,3,7,6))
    return tuple(vertices[list(face)] for face in faces)


def _box_planes(box: LayoutBox) -> tuple[tuple[np.ndarray, float], ...]:
    rotation = quaternion_matrix((0., 0., 0.), box.rotation_xyzw)[:3, :3]
    centre = np.asarray(box.center_m)
    planes = []
    for axis, sign in itertools.product(range(3), (-1., 1.)):
        normal = sign * rotation[:, axis]
        planes.append((normal, float(normal @ centre + box.size_m[axis] / 2.)))
    return tuple(planes)


def _subtract_convex(polyhedron: Polyhedron, planes: Sequence[tuple[np.ndarray, float]]) -> list[Polyhedron]:
    intersection = polyhedron
    for normal, distance in planes:
        intersection = _clip(intersection, normal, distance)
        if intersection is None:
            return [polyhedron]
    if _volume(intersection) <= _VOLUME_EPS:
        return [polyhedron]

    remainder = polyhedron
    pieces = []
    for normal, distance in planes:
        outside = _clip(remainder, -normal, -distance)
        if outside is not None and _volume(outside) > _VOLUME_EPS:
            pieces.append(outside)
        remainder = _clip(remainder, normal, distance)
        if remainder is None:
            break
    return pieces


def _clip(polyhedron: Polyhedron, normal: np.ndarray, distance: float) -> Polyhedron | None:
    all_points = np.concatenate(polyhedron)
    signed = all_points @ normal - distance
    if np.max(signed) <= _EPS:
        return polyhedron
    if np.min(signed) > _EPS:
        return None
    faces = []
    cap_points = []
    for face in polyhedron:
        clipped = []
        for first, second in zip(face, np.roll(face, -1, axis=0)):
            first_distance = float(first @ normal - distance)
            second_distance = float(second @ normal - distance)
            first_inside, second_inside = first_distance <= _EPS, second_distance <= _EPS
            if first_inside:
                clipped.append(first)
            if first_inside != second_inside:
                fraction = first_distance / (first_distance - second_distance)
                point = first + fraction * (second - first)
                clipped.append(point)
                cap_points.append(point)
        points = _polygon_points(clipped)
        if len(points) >= 3:
            faces.append(points)
            cap_points.extend(point for point in points if abs(float(point @ normal - distance)) <= _EPS * 4.)
    cap = _unique_points(cap_points)
    if len(cap) >= 3:
        centre = cap.mean(axis=0)
        unit = normal / np.linalg.norm(normal)
        reference = np.array((1., 0., 0.)) if abs(unit[0]) < .9 else np.array((0., 1., 0.))
        axis_x = np.cross(unit, reference)
        axis_x /= np.linalg.norm(axis_x)
        axis_y = np.cross(unit, axis_x)
        angles = np.arctan2((cap - centre) @ axis_y, (cap - centre) @ axis_x)
        faces.append(cap[np.argsort(angles)])
    if not faces:
        return None
    result = tuple(faces)
    return result if _volume(result) > _VOLUME_EPS else None


def _unique_points(points) -> np.ndarray:
    found = {}
    for point in points:
        found.setdefault(tuple(np.rint(np.asarray(point) / _EPS).astype(np.int64)), point)
    return np.asarray(list(found.values()), dtype=float).reshape(-1, 3)


def _polygon_points(points) -> np.ndarray:
    ordered = []
    for point in points:
        if not ordered or np.linalg.norm(point - ordered[-1]) > _EPS:
            ordered.append(point)
    if len(ordered) > 1 and np.linalg.norm(ordered[0] - ordered[-1]) <= _EPS:
        ordered.pop()
    return np.asarray(ordered, dtype=float).reshape(-1, 3)


def _volume(polyhedron: Polyhedron) -> float:
    centre = np.concatenate(polyhedron).mean(axis=0)
    volume = 0.
    for face in polyhedron:
        local = face - centre
        for index in range(1, len(face) - 1):
            volume += float(local[0] @ np.cross(local[index], local[index + 1])) / 6.
    return abs(volume)


def _fragment(box: LayoutBox, polyhedron: Polyhedron, index: int) -> LayoutFragment:
    points = []
    point_indices = {}
    triangles = []
    for face in polyhedron:
        indices = []
        for point in face:
            key = tuple(np.rint(point / _EPS).astype(np.int64))
            if key not in point_indices:
                point_indices[key] = len(points)
                points.append(tuple(float(value) for value in point))
            indices.append(point_indices[key])
        for offset in range(1, len(indices) - 1):
            if np.linalg.norm(np.cross(face[offset] - face[0], face[offset + 1] - face[0])) > _EPS ** 2:
                triangles.append((indices[0], indices[offset], indices[offset + 1]))
    return LayoutFragment(f"{box.name}/part_{index}", box.node_path, box.name,
                          tuple(points), tuple(triangles), box.role)
