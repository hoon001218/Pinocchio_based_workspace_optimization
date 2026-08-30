"""Validated data model for a dimension-based decanting-cell scene."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


@dataclass(frozen=True)
class BasePose:
    """Robot base pose with floor-normal mounting."""

    x_m: float
    y_m: float
    z_m: float
    yaw_deg: float

    @classmethod
    def from_sequence(cls, values: Sequence[float], name: str) -> "BasePose":
        vector = _vector(values, 4, name)
        if vector[2] < 0.0:
            raise ValueError(f"{name} z must be at or above the floor")
        return cls(*vector)

    @property
    def xyz_m(self) -> Vec3:
        return self.x_m, self.y_m, self.z_m

    @property
    def yaw_rad(self) -> float:
        return math.radians(self.yaw_deg)


@dataclass(frozen=True)
class FrameSpec:
    name: str
    translation_m: Vec3
    rotation_xyzw: Quat


@dataclass(frozen=True)
class BoxPrimitive:
    """Oriented box proxy in world coordinates."""

    name: str
    role: str
    center_m: Vec3
    size_m: Vec3
    yaw_deg: float = 0.0
    collision_enabled: bool = True


@dataclass(frozen=True)
class BoxSku:
    sku: str
    label: str
    size_m: Vec3


@dataclass(frozen=True)
class PalletSpec:
    center_xy_m: Vec2
    size_m: Vec3
    lift_range_m: tuple[float, float]
    corner_margin_m: float
    corner_samples_are_alternatives: bool


@dataclass(frozen=True)
class ToteSpec:
    size_m: Vec3
    representative_frame: str
    support_surface_z_m: float
    representative_yaw_deg: float
    motion_support_box: str


@dataclass(frozen=True)
class PedestalSpec:
    size_xy_m: Vec2
    center_offset_local_xy_m: Vec2


@dataclass(frozen=True)
class ScaraWorkspaceSpec:
    link_lengths_m: Vec2
    arm_plane_height_m: float
    q1_range_deg: Vec2
    q2_range_deg: Vec2
    j3_stroke_options_m: tuple[float, ...]
    total_height_options_m: tuple[float, ...]
    q4_range_deg: Vec2
    tool_radius_m: float
    safety_clearance_m: float


@dataclass(frozen=True)
class SuctionProxySpec:
    """Provisional end-effector geometry attached to the UR20 ``tool0`` frame."""

    cylinder_length_m: float
    cylinder_radius_m: float
    pad_radius_m: float
    pad_thickness_m: float
    tcp_translation_local_m: Vec3


@dataclass(frozen=True)
class CutterProxySpec:
    """Provisional box-opening tool collision/visual box at ``tool0``."""

    center_local_m: Vec3
    size_m: Vec3
    tcp_translation_local_m: Vec3


@dataclass(frozen=True)
class RobotSpec:
    name: str
    urdf_path: Path
    nominal_base: BasePose
    nominal_q: tuple[float, ...]
    pedestal: PedestalSpec
    module_camera_mode: str = "fixed"
    module_camera_boxes_local: tuple[BoxPrimitive, ...] = ()
    workspace: ScaraWorkspaceSpec | None = None
    suction_proxy: SuctionProxySpec | None = None
    cutter_proxy: CutterProxySpec | None = None


@dataclass(frozen=True)
class InstallationRegion:
    raw_xy_min_m: Vec2
    raw_xy_max_m: Vec2
    offset_from_worktable_m: float


@dataclass(frozen=True)
class SceneSpec:
    source_path: Path
    floor_z_m: float
    installation_region: InstallationRegion
    static_boxes: tuple[BoxPrimitive, ...]
    frames: Mapping[str, FrameSpec]
    pallet: PalletSpec
    box_skus: Mapping[str, BoxSku]
    tote: ToteSpec
    robots: Mapping[str, RobotSpec]


@dataclass(frozen=True)
class SceneState:
    """Scenario values varied inside a base-candidate evaluation."""

    lift_height_m: float = 0.0
    sku: str = "123591"
    corner_samples: tuple[str, ...] = ("southwest", "southeast", "northwest", "northeast")
    tote_present: bool = True
    tote_long_axis_offset_m: float = 0.0
    sr_j3_stroke_m: float = 0.30
    clearance_m: float | None = None


def default_config_path() -> Path:
    """Locate the repository-owned nominal scene configuration."""

    return Path(__file__).resolve().parents[2] / "config" / "cell_nominal.yaml"


def load_scene_spec(path: str | Path | None = None) -> SceneSpec:
    """Load and validate a scene without creating viewer or Pinocchio objects."""

    config_path = Path(path or default_config_path()).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"scene configuration not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("scene configuration root must be a mapping")
    if raw.get("schema_version") != 1:
        raise ValueError("only scene schema_version 1 is supported")
    if raw.get("units") != "m":
        raise ValueError("scene dimensions must use metres")

    config_dir = config_path.parent
    world = _mapping(raw, "world")
    if world.get("up_axis") != "Z":
        raise ValueError("only a Z-up world is supported")
    floor_z = _finite_float(world.get("floor_z_m"), "world.floor_z_m")

    region_raw = _mapping(raw, "installation_region")
    region = InstallationRegion(
        raw_xy_min_m=_vector(region_raw.get("raw_xy_min_m"), 2, "raw_xy_min_m"),
        raw_xy_max_m=_vector(region_raw.get("raw_xy_max_m"), 2, "raw_xy_max_m"),
        offset_from_worktable_m=_positive_float(
            region_raw.get("offset_from_worktable_m"), "offset_from_worktable_m"
        ),
    )
    if any(lo >= hi for lo, hi in zip(region.raw_xy_min_m, region.raw_xy_max_m)):
        raise ValueError("installation region min values must be less than max values")

    static_boxes = tuple(
        _box_from_mapping(item, f"static_boxes[{index}]")
        for index, item in enumerate(_sequence(raw, "static_boxes"))
    )

    frames_path = (config_dir / str(raw.get("reference_frames_file"))).resolve()
    frames = _load_frames(frames_path)

    pallet_raw = _mapping(raw, "pallet")
    lift_range = _vector(
        pallet_raw.get("lift_support_height_range_m"), 2, "pallet lift range"
    )
    pallet = PalletSpec(
        center_xy_m=_vector(pallet_raw.get("center_xy_m"), 2, "pallet.center_xy_m"),
        size_m=_positive_vector(pallet_raw.get("size_m"), 3, "pallet.size_m"),
        lift_range_m=lift_range,
        corner_margin_m=_nonnegative_float(
            pallet_raw.get("corner_margin_m", 0.0), "pallet.corner_margin_m"
        ),
        corner_samples_are_alternatives=bool(
            pallet_raw.get("corner_samples_are_alternatives", True)
        ),
    )
    if lift_range[0] < floor_z or lift_range[0] > lift_range[1]:
        raise ValueError("invalid pallet lift range")

    skus_raw = _mapping(raw, "box_skus")
    skus: dict[str, BoxSku] = {}
    for sku, item in skus_raw.items():
        item_map = _expect_mapping(item, f"box_skus.{sku}")
        sku_text = str(sku)
        skus[sku_text] = BoxSku(
            sku=sku_text,
            label=str(item_map.get("label", sku_text)),
            size_m=_positive_vector(item_map.get("size_m"), 3, f"box_skus.{sku}.size_m"),
        )
    if not skus:
        raise ValueError("at least one box SKU is required")

    tote_raw = _mapping(raw, "tote")
    tote = ToteSpec(
        size_m=_positive_vector(tote_raw.get("size_m"), 3, "tote.size_m"),
        representative_frame=str(tote_raw.get("representative_frame")),
        support_surface_z_m=_finite_float(
            tote_raw.get("support_surface_z_m"), "tote.support_surface_z_m"
        ),
        representative_yaw_deg=_finite_float(
            tote_raw.get("representative_yaw_deg", 0.0),
            "tote.representative_yaw_deg",
        ),
        motion_support_box=str(tote_raw.get("motion_support_box", "")),
    )
    if tote.representative_frame not in frames:
        raise ValueError(f"unknown tote representative frame: {tote.representative_frame}")
    support_boxes = {box.name: box for box in static_boxes}
    if tote.motion_support_box not in support_boxes:
        raise ValueError(f"unknown tote motion support box: {tote.motion_support_box}")
    if support_boxes[tote.motion_support_box].role != "worktable":
        raise ValueError("tote motion support box must have role 'worktable'")

    robots_raw = _mapping(raw, "robots")
    robots: dict[str, RobotSpec] = {}
    for name, item in robots_raw.items():
        robots[str(name)] = _robot_from_mapping(str(name), item, config_dir)
    if set(robots) != {"ur20", "sr12ia"}:
        raise ValueError("robots must contain exactly ur20 and sr12ia")

    names = [box.name for box in static_boxes]
    if len(names) != len(set(names)):
        raise ValueError("static box names must be unique")

    return SceneSpec(
        source_path=config_path,
        floor_z_m=floor_z,
        installation_region=region,
        static_boxes=static_boxes,
        frames=frames,
        pallet=pallet,
        box_skus=skus,
        tote=tote,
        robots=robots,
    )


def _load_frames(path: Path) -> Mapping[str, FrameSpec]:
    if not path.is_file():
        raise FileNotFoundError(f"reference frame file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or raw.get("units") != "m":
        raise ValueError("unsupported reference-frame schema or units")
    frames: dict[str, FrameSpec] = {}
    for name, item in _mapping(raw, "frames").items():
        item_map = _expect_mapping(item, f"frames.{name}")
        quaternion = _vector(item_map.get("rotation_xyzw"), 4, f"frames.{name}.rotation")
        norm = math.sqrt(sum(value * value for value in quaternion))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError(f"frame {name} quaternion is not normalized")
        frames[str(name)] = FrameSpec(
            name=str(name),
            translation_m=_vector(
                item_map.get("translation_m"), 3, f"frames.{name}.translation"
            ),
            rotation_xyzw=quaternion,
        )
    return frames


def _robot_from_mapping(name: str, raw: Any, config_dir: Path) -> RobotSpec:
    item = _expect_mapping(raw, f"robots.{name}")
    pedestal_raw = _mapping(item, "pedestal")
    pedestal = PedestalSpec(
        size_xy_m=_positive_vector(
            pedestal_raw.get("size_xy_m"), 2, f"robots.{name}.pedestal.size_xy_m"
        ),
        center_offset_local_xy_m=_vector(
            pedestal_raw.get("center_offset_local_xy_m"),
            2,
            f"robots.{name}.pedestal.center_offset_local_xy_m",
        ),
    )
    urdf_path = (config_dir / str(item.get("urdf"))).resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"{name} URDF not found: {urdf_path}")

    if "nominal_q_deg" in item:
        nominal_q = tuple(math.radians(value) for value in _vector_any(item["nominal_q_deg"], f"robots.{name}.nominal_q_deg"))
    else:
        nominal_q = _vector_any(item.get("nominal_q", ()), f"robots.{name}.nominal_q")

    module_boxes = tuple(
        _box_from_mapping(box, f"robots.{name}.module_camera_boxes_local[{index}]")
        for index, box in enumerate(item.get("module_camera_boxes_local", ()))
    )
    camera_mode = str(item.get("module_camera_mode", "fixed"))
    if camera_mode not in {"fixed", "with_base"}:
        raise ValueError(f"unsupported {name} module_camera_mode: {camera_mode}")

    workspace = None
    if "workspace" in item:
        ws = _mapping(item, "workspace")
        workspace = ScaraWorkspaceSpec(
            link_lengths_m=_positive_vector(ws.get("link_lengths_m"), 2, "link_lengths_m"),
            arm_plane_height_m=_positive_float(
                ws.get("arm_plane_height_m"), "arm_plane_height_m"
            ),
            q1_range_deg=_vector(ws.get("q1_range_deg"), 2, "q1_range_deg"),
            q2_range_deg=_vector(ws.get("q2_range_deg"), 2, "q2_range_deg"),
            j3_stroke_options_m=tuple(
                _positive_float(value, "j3 stroke")
                for value in _sequence(ws, "j3_stroke_options_m")
            ),
            total_height_options_m=tuple(
                _positive_float(value, "total height option")
                for value in _sequence(ws, "total_height_options_m")
            ),
            q4_range_deg=_vector(ws.get("q4_range_deg"), 2, "q4_range_deg"),
            tool_radius_m=_positive_float(ws.get("tool_radius_m"), "tool_radius_m"),
            safety_clearance_m=_nonnegative_float(
                ws.get("safety_clearance_m", 0.0), "safety_clearance_m"
            ),
        )
        if len(workspace.total_height_options_m) != len(
            workspace.j3_stroke_options_m
        ):
            raise ValueError(
                "SCARA total-height options must align with J3 stroke options"
            )

    suction_proxy = None
    if "suction_proxy" in item:
        suction = _mapping(item, "suction_proxy")
        suction_proxy = SuctionProxySpec(
            cylinder_length_m=_positive_float(
                suction.get("cylinder_length_m"), "suction_proxy.cylinder_length_m"
            ),
            cylinder_radius_m=_positive_float(
                suction.get("cylinder_radius_m"), "suction_proxy.cylinder_radius_m"
            ),
            pad_radius_m=_positive_float(
                suction.get("pad_radius_m"), "suction_proxy.pad_radius_m"
            ),
            pad_thickness_m=_positive_float(
                suction.get("pad_thickness_m"), "suction_proxy.pad_thickness_m"
            ),
            tcp_translation_local_m=_vector(
                suction.get("tcp_translation_local_m"),
                3,
                "suction_proxy.tcp_translation_local_m",
            ),
        )

    cutter_proxy = None
    if "cutter_proxy" in item:
        cutter = _mapping(item, "cutter_proxy")
        cutter_proxy = CutterProxySpec(
            center_local_m=_vector(
                cutter.get("center_local_m"), 3, "cutter_proxy.center_local_m"
            ),
            size_m=_positive_vector(
                cutter.get("size_m"), 3, "cutter_proxy.size_m"
            ),
            tcp_translation_local_m=_vector(
                cutter.get("tcp_translation_local_m"),
                3,
                "cutter_proxy.tcp_translation_local_m",
            ),
        )

    return RobotSpec(
        name=name,
        urdf_path=urdf_path,
        nominal_base=BasePose.from_sequence(item.get("nominal_base"), f"robots.{name}.nominal_base"),
        nominal_q=nominal_q,
        pedestal=pedestal,
        module_camera_mode=camera_mode,
        module_camera_boxes_local=module_boxes,
        workspace=workspace,
        suction_proxy=suction_proxy,
        cutter_proxy=cutter_proxy,
    )


def _box_from_mapping(raw: Any, path: str) -> BoxPrimitive:
    item = _expect_mapping(raw, path)
    return BoxPrimitive(
        name=str(item.get("name")),
        role=str(item.get("role", "obstacle")),
        center_m=_vector(item.get("center_m"), 3, f"{path}.center_m"),
        size_m=_positive_vector(item.get("size_m"), 3, f"{path}.size_m"),
        yaw_deg=_finite_float(item.get("yaw_deg", 0.0), f"{path}.yaw_deg"),
        collision_enabled=bool(item.get("collision_enabled", True)),
    )


def _mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _expect_mapping(mapping.get(key), key)


def _expect_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _sequence(mapping: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = mapping.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a sequence")
    return value


def _vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    return tuple(_finite_float(item, name) for item in value)


def _vector_any(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return tuple(_finite_float(item, name) for item in value)


def _positive_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    vector = _vector(value, length, name)
    if any(item <= 0.0 for item in vector):
        raise ValueError(f"{name} values must be positive")
    return vector


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result
