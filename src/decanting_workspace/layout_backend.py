"""Translation-only previews of a USD layout, without process evaluation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import Callable, Mapping


class LayoutPreviewBackend:
    """Rebuild every preview from the immutable imported layout."""

    def __init__(
        self,
        scene: object,
        viewer: object,
        *,
        offset_applier: Callable[..., object] | None = None,
        geometry_builder: Callable[..., tuple[object, ...]] | None = None,
        output: str | Path | None = None,
    ) -> None:
        if offset_applier is None:
            from .layout import apply_layout_offsets

            offset_applier = apply_layout_offsets
        if geometry_builder is None:
            from .layout_geometry import build_layout_geometry

            geometry_builder = build_layout_geometry
        self.original_scene = scene
        self.scene = scene
        self.viewer = viewer
        self.offset_applier = offset_applier
        self.geometry_builder = geometry_builder
        self.output = Path(output).resolve() if output is not None else None
        self.offsets: dict[str, tuple[float, float, float]] = {}
        self.fragments = tuple(self.geometry_builder(scene))
        self.viewer.render(scene, self.fragments)
        self._save_export()

    def catalog(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "mode": "layout",
            "coordinate_frame": "parent_local",
            "translation_only": True,
            "nodes": [_node_payload(node) for node in self.original_scene.nodes],
            "offsets": {name: list(value) for name, value in self.offsets.items()},
            **self._preview_payload(),
        }

    def layout(self, request: Mapping[str, object]) -> dict[str, object]:
        offsets = _validated_offsets(request, self.original_scene.nodes)
        scene = self.offset_applier(self.original_scene, offsets)
        fragments = tuple(self.geometry_builder(scene))
        self.viewer.render(scene, fragments)
        self.scene = scene
        self.fragments = fragments
        self.offsets = offsets
        self._save_export()
        return {
            "offsets": {name: list(value) for name, value in offsets.items()},
            **self._preview_payload(),
        }

    def export_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "units": "m",
            "coordinate_frame": "parent_local",
            "offsets": {name: list(value) for name, value in self.offsets.items()},
            "scene": _json_value(self.scene),
            "fragments": _json_value(self.fragments),
        }

    def _preview_payload(self) -> dict[str, object]:
        from .layout_geometry import fragment_volume_m3

        source_volume = math.fsum(math.prod(box.size_m) for box in self.scene.boxes)
        union_volume = math.fsum(fragment_volume_m3(fragment) for fragment in self.fragments)
        robot_info = getattr(self.viewer, "robot_info", None)
        robots = robot_info() if callable(robot_info) else [_robot_payload(robot) for robot in self.scene.robots]
        return {
            "source": str(self.original_scene.source_path),
            "stats": {
                "layout_nodes": len(self.scene.nodes),
                "source_boxes": len(self.scene.boxes),
                "simplified_fragments": len(self.fragments),
                "robots": len(self.scene.robots),
                "triangle_count": sum(len(fragment.triangles) for fragment in self.fragments),
                "source_volume_m3": source_volume,
                "union_volume_m3": union_volume,
                "removed_overlap_volume_m3": max(0.0, source_volume - union_volume),
            },
            "robots": robots,
            "reference_dimensions": _json_value(self.scene.reference_dimensions),
        }

    def _save_export(self) -> None:
        if self.output is not None:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.output.write_text(
                json.dumps(self.export_payload(), ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )


def _node_payload(node: object) -> dict[str, object]:
    return {
        "prim_path": node.prim_path,
        "parent_path": node.parent_path,
        "translation_m": list(node.translation_m),
        "rotation_xyzw": list(node.rotation_xyzw),
        "kind": node.kind,
    }


def _robot_payload(robot: object) -> dict[str, object]:
    return {
        "name": robot.name,
        "node_path": robot.node_path,
        "translation_m": list(robot.translation_m),
        "yaw_deg": robot.yaw_deg,
        "urdf_path": str(robot.urdf_path),
    }


def _validated_offsets(
    request: Mapping[str, object], nodes: tuple[object, ...]
) -> dict[str, tuple[float, float, float]]:
    if set(request) != {"offsets"}:
        raise ValueError("layout request must contain only 'offsets'")
    raw = request["offsets"]
    if not isinstance(raw, Mapping):
        raise ValueError("offsets must be an object keyed by USD prim path")
    known = {node.prim_path for node in nodes}
    result: dict[str, tuple[float, float, float]] = {}
    for name, values in raw.items():
        if not isinstance(name, str) or name not in known:
            raise ValueError(f"unknown layout node: {name!r}")
        if not isinstance(values, (tuple, list)) or len(values) != 3:
            raise ValueError(f"offsets[{name}] must contain XYZ values in metres")
        if any(isinstance(value, bool) for value in values):
            raise ValueError(f"offsets[{name}] must contain numeric XYZ values")
        try:
            offset = tuple(float(value) for value in values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"offsets[{name}] must contain numeric XYZ values") from exc
        if not all(math.isfinite(value) for value in offset):
            raise ValueError(f"offsets[{name}] must contain finite XYZ values")
        if any(value != 0.0 for value in offset):
            result[name] = offset
    return result


def _json_value(value: object) -> object:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
