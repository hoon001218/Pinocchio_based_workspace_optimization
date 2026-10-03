"""USD layout data, independent of SKU scenarios and optimization results."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

from .models import Quat, Vec3


def _vector(values: Sequence[float], size: int, name: str) -> tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} needs {size} finite numbers") from exc
    if len(result) != size or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} needs {size} finite numbers")
    return result


def _rotation(values: Sequence[float]) -> Quat:
    result = _vector(values, 4, "rotation_xyzw")
    if not math.isclose(sum(value * value for value in result), 1., abs_tol=1e-6):
        raise ValueError("rotation_xyzw must be a unit quaternion")
    return result


@dataclass(frozen=True)
class LayoutNode:
    """An authored group or movable unit, with its current World pose."""

    prim_path: str
    parent_path: str | None
    translation_m: Vec3
    rotation_xyzw: Quat
    kind: str = "element"

    def __post_init__(self) -> None:
        if not self.prim_path.startswith("/"):
            raise ValueError("layout node needs an absolute USD prim path")
        object.__setattr__(self, "translation_m", _vector(self.translation_m, 3, "translation_m"))
        object.__setattr__(self, "rotation_xyzw", _rotation(self.rotation_xyzw))


@dataclass(frozen=True)
class LayoutBox:
    """One uncut simplified solid; retained so overlap can be recomputed."""

    name: str
    node_path: str
    center_m: Vec3
    size_m: Vec3
    rotation_xyzw: Quat
    role: str = "obstacle"
    source_prim_path: str = ""

    def __post_init__(self) -> None:
        size = _vector(self.size_m, 3, "size_m")
        if not self.name or any(value <= 0. for value in size):
            raise ValueError("layout box needs a name and positive sizes")
        object.__setattr__(self, "center_m", _vector(self.center_m, 3, "center_m"))
        object.__setattr__(self, "size_m", size)
        object.__setattr__(self, "rotation_xyzw", _rotation(self.rotation_xyzw))


@dataclass(frozen=True)
class LayoutReferenceDimensions:
    """Object dimensions retained as reference data, without layout geometry."""

    name: str
    size_m: Vec3
    source_prim_path: str
    role: str = "tote"

    def __post_init__(self) -> None:
        size = _vector(self.size_m, 3, "size_m")
        if not self.name or any(value <= 0. for value in size):
            raise ValueError("layout reference dimensions need a name and positive sizes")
        if not self.source_prim_path.startswith("/"):
            raise ValueError("layout reference dimensions need an absolute USD prim path")
        object.__setattr__(self, "size_m", size)


@dataclass(frozen=True)
class LayoutRobot:
    """USD base-link position and configured URDF rotation/joint posture."""

    name: str
    node_path: str
    translation_m: Vec3
    yaw_deg: float
    urdf_path: Path
    nominal_q: tuple[float, ...]
    source_prim_path: str = ""

    def __post_init__(self) -> None:
        if not self.name or not math.isfinite(float(self.yaw_deg)):
            raise ValueError("layout robot needs a name and finite yaw")
        object.__setattr__(self, "translation_m", _vector(self.translation_m, 3, "translation_m"))
        object.__setattr__(self, "urdf_path", Path(self.urdf_path))
        object.__setattr__(self, "nominal_q", _vector(self.nominal_q, len(self.nominal_q), "nominal_q"))


@dataclass(frozen=True)
class LayoutFrame:
    name: str
    node_path: str
    translation_m: Vec3
    rotation_xyzw: Quat
    source_prim_path: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "translation_m", _vector(self.translation_m, 3, "translation_m"))
        object.__setattr__(self, "rotation_xyzw", _rotation(self.rotation_xyzw))


@dataclass(frozen=True)
class LayoutScene:
    source_path: Path
    nodes: tuple[LayoutNode, ...]
    boxes: tuple[LayoutBox, ...]
    robots: tuple[LayoutRobot, ...]
    frames: tuple[LayoutFrame, ...] = ()
    reference_dimensions: tuple[LayoutReferenceDimensions, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        for key in ("nodes", "boxes", "robots", "frames", "reference_dimensions"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        for values, key in ((self.nodes, "prim_path"), (self.boxes, "name"),
                            (self.robots, "name"), (self.frames, "name"),
                            (self.reference_dimensions, "name")):
            names = [getattr(value, key) for value in values]
            if len(names) != len(set(names)):
                raise ValueError(f"layout {key} values must be unique")


@dataclass(frozen=True)
class LayoutFragment:
    """Convex portion assigned to a single simplified object's owner."""

    name: str
    owner_path: str
    source_name: str
    vertices_m: tuple[Vec3, ...]
    triangles: tuple[tuple[int, int, int], ...]
    role: str = "obstacle"
