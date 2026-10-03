"""Translate a USD layout hierarchy without changing its authored rotations."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np

from .layout_models import LayoutScene
from .transforms import quaternion_matrix


def apply_layout_offsets(
    scene: LayoutScene,
    offsets: Mapping[str, Sequence[float]],
) -> LayoutScene:
    """Return one candidate layout with offsets in each node's parent axes.

    Node poses and attached geometry are authored in World coordinates. An
    offset for a child is therefore rotated by its parent's original World
    rotation, then added to the displacement inherited from that parent.
    Attachments receive their owning node's total displacement once. All
    rotations, dimensions, source paths and robot joint values are preserved.

    Pass the original scene for every independent candidate; neither it nor
    ``offsets`` is mutated. A returned scene can also be used as a new baseline.
    """

    if not isinstance(offsets, Mapping):
        raise ValueError("layout offsets must map node paths to XYZ offsets")
    nodes = {node.prim_path: node for node in scene.nodes}
    if len(nodes) != len(scene.nodes):
        raise ValueError("layout node paths must be unique")
    unknown = set(offsets) - set(nodes)
    if unknown:
        raise ValueError(f"layout offsets contain unknown nodes: {sorted(unknown, key=str)}")

    vectors: dict[str, np.ndarray] = {}
    for path, values in offsets.items():
        try:
            vector = np.asarray(values, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"layout offset for {path} must contain three finite values") from exc
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"layout offset for {path} must contain three finite values")
        vectors[path] = vector

    displacements: dict[str, np.ndarray] = {}
    visiting: set[str] = set()

    def displacement(path: str) -> np.ndarray:
        if path in displacements:
            return displacements[path]
        if path in visiting:
            raise ValueError(f"layout hierarchy contains a cycle at {path}")
        visiting.add(path)
        node = nodes[path]
        offset = vectors.get(path, np.zeros(3))
        if node.parent_path:
            if node.parent_path not in nodes:
                raise ValueError(f"layout node {path} has unknown parent {node.parent_path}")
            parent = nodes[node.parent_path]
            rotation = quaternion_matrix((0.0, 0.0, 0.0), parent.rotation_xyzw)[:3, :3]
            result = displacement(node.parent_path) + rotation @ offset
        else:
            result = offset.copy()
        visiting.remove(path)
        displacements[path] = result
        return result

    # Validate every branch even when no offset or attachment uses it. Input
    # order need not follow hierarchy order.
    for path in nodes:
        displacement(path)

    def translated(position: Sequence[float], owner: str) -> tuple[float, float, float]:
        if owner not in displacements:
            raise ValueError(f"layout attachment has unknown node {owner}")
        original = np.asarray(position, dtype=float)
        if original.shape != (3,) or not np.all(np.isfinite(original)):
            raise ValueError(f"layout position for {owner} must contain three finite values")
        result = original + displacements[owner]
        if not np.all(np.isfinite(result)):
            raise ValueError(f"layout position for {owner} exceeds finite coordinates")
        return tuple(float(value) for value in result)

    return replace(
        scene,
        nodes=tuple(replace(node, translation_m=translated(node.translation_m, node.prim_path))
                    for node in scene.nodes),
        boxes=tuple(replace(box, center_m=translated(box.center_m, box.node_path))
                    for box in scene.boxes),
        robots=tuple(replace(robot, translation_m=translated(robot.translation_m, robot.node_path))
                     for robot in scene.robots),
        frames=tuple(replace(frame, translation_m=translated(frame.translation_m, frame.node_path))
                     for frame in scene.frames),
    )
