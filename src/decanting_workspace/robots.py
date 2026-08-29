"""Pinocchio robot loading and base-candidate configuration helpers."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
from types import ModuleType
from typing import Sequence
from xml.etree import ElementTree

import numpy as np

from .models import BasePose, CutterProxySpec, SuctionProxySpec


OFFICIAL_UR20_COMMIT = "ae333289875f9ba5a9ea6649a54036efb5ccabee"
"""Universal Robots ROS 2 Description 4.3.1 commit used by this project."""

_ROBOT_DESCRIPTIONS_CACHE_ENV = "ROBOT_DESCRIPTIONS_CACHE"
_UR_DESCRIPTION_REPOSITORY = "Universal_Robots_ROS2_Description"


def _pinocchio_path(path: Path) -> Path:
    """Prefer an ASCII Windows short path when the filesystem provides one."""

    if os.name != "nt" or str(path).isascii():
        return path
    short_path = _windows_short_path(path)
    return short_path if short_path is not None else path


def _windows_short_path(path: Path) -> Path | None:
    """Return an ASCII 8.3 path using GetShortPathNameW, if available."""

    if os.name != "nt":
        return None

    import ctypes

    get_short_path = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
    get_short_path.argtypes = (
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
    )
    get_short_path.restype = ctypes.c_uint32
    required = get_short_path(str(path), None, 0)
    if required == 0:
        return None
    buffer = ctypes.create_unicode_buffer(required)
    written = get_short_path(str(path), buffer, required)
    if written == 0 or written >= required or not buffer.value.isascii():
        return None
    return Path(buffer.value)


def _urdf_xml_with_resolved_geometry_paths(path: Path) -> str:
    """Resolve relative mesh/texture filenames for string-based URDF parsing."""

    root = ElementTree.fromstring(path.read_text(encoding="utf-8"))
    for element in root.iter():
        local_tag = element.tag.rsplit("}", 1)[-1]
        if local_tag not in {"mesh", "texture"}:
            continue
        filename = element.get("filename")
        if not filename or "://" in filename or filename.startswith("$("):
            continue
        geometry_path = Path(filename)
        if geometry_path.is_absolute():
            continue
        resolved = (path.parent / geometry_path).resolve()
        element.set("filename", str(_pinocchio_path(resolved)))
    return ElementTree.tostring(root, encoding="unicode")


@contextmanager
def _robot_descriptions_cache(cache_dir: str | Path | None):
    """Temporarily direct both clone and generated-Xacro caches."""

    if cache_dir is None:
        yield
        return
    previous = os.environ.get(_ROBOT_DESCRIPTIONS_CACHE_ENV)
    os.environ[_ROBOT_DESCRIPTIONS_CACHE_ENV] = str(
        Path(cache_dir).expanduser().resolve()
    )
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_ROBOT_DESCRIPTIONS_CACHE_ENV, None)
        else:
            os.environ[_ROBOT_DESCRIPTIONS_CACHE_ENV] = previous


@contextmanager
def _git_safe_directory(repository_path: Path):
    """Scope a Git ``safe.directory`` entry to child processes in this call."""

    try:
        config_count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    except ValueError:
        config_count = 0
    key_name = f"GIT_CONFIG_KEY_{config_count}"
    value_name = f"GIT_CONFIG_VALUE_{config_count}"
    previous = {
        "GIT_CONFIG_COUNT": os.environ.get("GIT_CONFIG_COUNT"),
        key_name: os.environ.get(key_name),
        value_name: os.environ.get(value_name),
    }
    os.environ["GIT_CONFIG_COUNT"] = str(config_count + 1)
    os.environ[key_name] = "safe.directory"
    os.environ[value_name] = repository_path.resolve().as_posix()
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass
class RobotBundle:
    """Pinocchio models plus the arm-only velocity-column slice."""

    name: str
    model: object
    collision_model: object
    visual_model: object | None
    arm_velocity_slice: slice
    arm_configuration_offset: int
    native_lower_position_limits: np.ndarray
    native_upper_position_limits: np.ndarray


def load_robot_bundle(
    name: str,
    urdf_path: str | Path,
    *,
    package_dirs: Sequence[str | Path] = (),
    floating_base: bool = True,
    load_visual: bool = False,
    suction_proxy: SuctionProxySpec | None = None,
    cutter_proxy: CutterProxySpec | None = None,
) -> RobotBundle:
    """Load robot models and optionally attach the configured suction collision."""

    import pinocchio as pin

    path = Path(urdf_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"URDF not found: {path}")
    pinocchio_path = _pinocchio_path(path)
    package_paths = [
        str(_pinocchio_path(Path(item).resolve())) for item in package_dirs
    ]
    root_joint = pin.JointModelFreeFlyer() if floating_base else None

    if os.name == "nt" and not str(pinocchio_path).isascii():
        # GetShortPathNameW is attempted above. If 8.3 names are unavailable,
        # bypass urdfdom's narrow filename API and resolve relative geometry
        # filenames before parsing the XML stream.
        urdf_xml = _urdf_xml_with_resolved_geometry_paths(path)
        model = (
            pin.buildModelFromXML(urdf_xml, root_joint)
            if root_joint is not None
            else pin.buildModelFromXML(urdf_xml)
        )
        collision_model = pin.buildGeomFromUrdfString(
            model,
            urdf_xml,
            pin.GeometryType.COLLISION,
            package_dirs=package_paths,
        )
        visual_model = (
            pin.buildGeomFromUrdfString(
                model,
                urdf_xml,
                pin.GeometryType.VISUAL,
                package_dirs=package_paths,
            )
            if load_visual
            else None
        )
    else:
        model = (
            pin.buildModelFromUrdf(str(pinocchio_path), root_joint)
            if root_joint is not None
            else pin.buildModelFromUrdf(str(pinocchio_path))
        )
        collision_model = pin.buildGeomFromUrdf(
            model,
            str(pinocchio_path),
            pin.GeometryType.COLLISION,
            package_dirs=package_paths,
        )
        visual_model = (
            pin.buildGeomFromUrdf(
                model,
                str(pinocchio_path),
                pin.GeometryType.VISUAL,
                package_dirs=package_paths,
            )
            if load_visual
            else None
        )
    if visual_model is not None and visual_model.ngeoms == 0:
        raise ValueError(f"{name} URDF contains no visual geometry: {path}")
    if collision_model.ngeoms == 0:
        raise ValueError(f"{name} URDF contains no collision geometry: {path}")

    base_nv = 6 if floating_base else 0
    base_nq = 7 if floating_base else 0
    if model.nv <= base_nv:
        raise ValueError(f"{name} has no actuated joints")
    bundle = RobotBundle(
        name=name,
        model=model,
        collision_model=collision_model,
        visual_model=visual_model,
        arm_velocity_slice=slice(base_nv, model.nv),
        arm_configuration_offset=base_nq,
        native_lower_position_limits=np.asarray(model.lowerPositionLimit).copy(),
        native_upper_position_limits=np.asarray(model.upperPositionLimit).copy(),
    )
    if suction_proxy is not None:
        ensure_operational_frame(
            bundle,
            "suction_tcp",
            "tool0",
            suction_proxy.tcp_translation_local_m,
        )
        if not model.existFrame("suction_gripper"):
            attach_suction_proxy(
                bundle,
                suction_proxy.cylinder_length_m,
                suction_proxy.cylinder_radius_m,
                suction_proxy.pad_radius_m,
                suction_proxy.pad_thickness_m,
            )
    if cutter_proxy is not None:
        ensure_operational_frame(
            bundle,
            "cutter_tcp",
            "tool0",
            cutter_proxy.tcp_translation_local_m,
        )
        if not model.existFrame("cutter_proxy"):
            attach_cutter_proxy(bundle, cutter_proxy)
    return bundle


def ensure_operational_frame(
    bundle: RobotBundle,
    frame_name: str,
    parent_frame_name: str,
    translation_local_m: Sequence[float],
) -> int:
    """Add a config-defined operational frame without changing joint DOFs."""

    import pinocchio as pin

    if bundle.model.existFrame(frame_name):
        return bundle.model.getFrameId(frame_name)
    translation = np.asarray(tuple(translation_local_m), dtype=float).reshape(-1)
    if translation.size != 3 or not np.all(np.isfinite(translation)):
        raise ValueError(f"{frame_name} translation must contain three finite values")
    parent_frame_id = bundle.model.getFrameId(parent_frame_name)
    if parent_frame_id >= bundle.model.nframes:
        raise ValueError(f"{bundle.name} has no {parent_frame_name} frame")
    parent_frame = bundle.model.frames[parent_frame_id]
    placement = parent_frame.placement * pin.SE3(np.eye(3), translation)
    frame = pin.Frame(
        frame_name,
        parent_frame.parentJoint,
        parent_frame_id,
        placement,
        pin.FrameType.OP_FRAME,
    )
    return bundle.model.addFrame(frame, False)


def configure_sr_j3_stroke(bundle: RobotBundle, stroke_m: float) -> None:
    """Restrict an SR-12iA model to the selected 300/450 mm J3 option."""

    if not math.isfinite(stroke_m) or stroke_m not in (0.3, 0.45):
        raise ValueError(f"unsupported SR-12iA J3 stroke: {stroke_m}")
    joint_id = bundle.model.getJointId("joint3")
    if joint_id >= bundle.model.njoints:
        raise ValueError(f"{bundle.name} has no joint3")
    joint = bundle.model.joints[joint_id]
    if joint.nq != 1:
        raise ValueError(f"{bundle.name} joint3 is not a one-DOF joint")
    index = joint.idx_q
    available_upper = float(bundle.native_upper_position_limits[index])
    if available_upper + 1e-12 < stroke_m:
        raise ValueError(
            f"{bundle.name} J3 supports only {available_upper:.3f} m, "
            f"not the requested {stroke_m:.3f} m"
        )
    bundle.model.lowerPositionLimit[index] = 0.0
    bundle.model.upperPositionLimit[index] = stroke_m


def attach_suction_proxy(
    bundle: RobotBundle,
    length: float,
    radius: float,
    pad_radius: float,
    pad_thickness: float,
) -> tuple[int, int]:
    """Attach a two-cylinder suction collision proxy along ``tool0`` +Z.

    Dimensions are in metres. The stem starts at ``tool0`` and the pad starts
    at the stem tip. The bundle's collision model is modified in place; visual
    geometry is intentionally unchanged.
    """

    import coal
    import pinocchio as pin

    dimensions = {
        "length": length,
        "radius": radius,
        "pad_radius": pad_radius,
        "pad_thickness": pad_thickness,
    }
    for label, value in dimensions.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be a finite positive value, got {value}")

    tool_frame_id = bundle.model.getFrameId("tool0")
    if tool_frame_id >= bundle.model.nframes:
        raise ValueError(f"{bundle.name} has no tool0 frame")

    stem_name = "suction_proxy_stem"
    pad_name = "suction_proxy_pad"
    for geometry_name in (stem_name, pad_name):
        if bundle.collision_model.existGeometryName(geometry_name):
            raise ValueError(
                f"{bundle.name} collision model already contains {geometry_name}"
            )

    tool_frame = bundle.model.frames[tool_frame_id]
    parent_joint = tool_frame.parentJoint
    stem_local = pin.SE3(np.eye(3), np.array((0.0, 0.0, length / 2.0)))
    pad_local = pin.SE3(
        np.eye(3),
        np.array((0.0, 0.0, length + pad_thickness / 2.0)),
    )
    stem = pin.GeometryObject(
        stem_name,
        parent_joint,
        tool_frame_id,
        tool_frame.placement * stem_local,
        coal.Cylinder(radius, length),
    )
    pad = pin.GeometryObject(
        pad_name,
        parent_joint,
        tool_frame_id,
        tool_frame.placement * pad_local,
        coal.Cylinder(pad_radius, pad_thickness),
    )
    stem_id = bundle.collision_model.addGeometryObject(stem)
    pad_id = bundle.collision_model.addGeometryObject(pad)
    return stem_id, pad_id


def attach_cutter_proxy(
    bundle: RobotBundle,
    proxy: CutterProxySpec,
) -> int:
    """Attach the provisional cutter collision box to ``tool0`` in place."""

    import coal
    import pinocchio as pin

    if any(not math.isfinite(value) for value in proxy.center_local_m):
        raise ValueError("cutter proxy center must contain finite values")
    if any(not math.isfinite(value) or value <= 0.0 for value in proxy.size_m):
        raise ValueError("cutter proxy size must contain finite positive values")
    frame_id = bundle.model.getFrameId("tool0")
    if frame_id >= bundle.model.nframes:
        raise ValueError(f"{bundle.name} has no tool0 frame")
    geometry_name = "cutter_proxy_box"
    if bundle.collision_model.existGeometryName(geometry_name):
        raise ValueError(
            f"{bundle.name} collision model already contains {geometry_name}"
        )
    frame = bundle.model.frames[frame_id]
    local = pin.SE3(np.eye(3), np.asarray(proxy.center_local_m, dtype=float))
    geometry = pin.GeometryObject(
        geometry_name,
        frame.parentJoint,
        frame_id,
        frame.placement * local,
        coal.Box(*proxy.size_m),
    )
    return bundle.collision_model.addGeometryObject(geometry)


def floating_configuration(
    bundle: RobotBundle,
    base: BasePose,
    arm_q: Sequence[float],
) -> np.ndarray:
    """Compose Pinocchio free-flyer xyz/xyzw and robot joint values."""

    arm = np.asarray(tuple(arm_q), dtype=float).reshape(-1)
    expected = bundle.model.nq - bundle.arm_configuration_offset
    if arm.size != expected:
        raise ValueError(
            f"{bundle.name} expects {expected} arm positions, got {arm.size}"
        )
    q = np.zeros(bundle.model.nq, dtype=float)
    if bundle.arm_configuration_offset == 7:
        half_yaw = 0.5 * base.yaw_rad
        q[:7] = (
            base.x_m,
            base.y_m,
            base.z_m,
            0.0,
            0.0,
            math.sin(half_yaw),
            math.cos(half_yaw),
        )
    elif bundle.arm_configuration_offset != 0:
        raise ValueError("unsupported floating-base configuration layout")
    q[bundle.arm_configuration_offset :] = arm
    return q


def resolve_official_ur20(
    *, cache_dir: str | Path | None = None
) -> tuple[Path, tuple[Path, ...]]:
    """Resolve the official UR20 4.3.1 Xacro at its immutable commit.

    ``cache_dir`` controls both the upstream clone and generated URDF cache.
    By default, the repository-local ignored ``.cache/robot_descriptions`` is
    used. It is scoped to this call, so an existing
    ``ROBOT_DESCRIPTIONS_CACHE`` value is restored afterwards.
    """

    try:
        from robot_descriptions._cache import clone_to_cache
        from robot_descriptions._package_dirs import get_package_dirs
        from robot_descriptions._xacro import get_urdf_path
    except ImportError as exc:
        raise RuntimeError(
            "official UR20 requested; install robot_descriptions==3.1.0 and xacrodoc"
        ) from exc

    resolved_cache = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else Path(__file__).resolve().parents[2] / ".cache" / "robot_descriptions"
    )
    resolved_cache.mkdir(parents=True, exist_ok=True)
    cached_repository = (
        resolved_cache
        / f"ur_description-{OFFICIAL_UR20_COMMIT}"
        / "ur_description"
    )
    with _git_safe_directory(cached_repository), _robot_descriptions_cache(
        resolved_cache
    ):
        try:
            repository_path = Path(
                clone_to_cache(
                    _UR_DESCRIPTION_REPOSITORY,
                    commit=OFFICIAL_UR20_COMMIT,
                )
            ).resolve()
        except Exception as exc:
            raise RuntimeError(
                "official UR20 cache/download failed; check network access, "
                f"Git permissions, and cache directory {resolved_cache}"
            ) from exc
        description = ModuleType("robot_descriptions.ur20_description")
        description.REPOSITORY_PATH = str(repository_path)
        description.PACKAGE_PATH = str(repository_path)
        description.XACRO_PATH = str(repository_path / "urdf" / "ur.urdf.xacro")
        description.XACRO_ARGS = {"ur_type": "ur20", "name": "ur20"}

        try:
            urdf_path = Path(
                get_urdf_path(
                    description,
                    xacro_args={"ur_type": "ur20", "name": "ur20"},
                )
            ).resolve()
            resolved_package_dirs = tuple(
                dict.fromkeys(
                    Path(item).resolve() for item in get_package_dirs(description)
                )
            )
        except ImportError as exc:
            raise RuntimeError(
                "official UR20 Xacro requires robot_descriptions and xacrodoc"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"official UR20 Xacro generation failed in {resolved_cache}"
            ) from exc

    if not urdf_path.is_file():
        raise FileNotFoundError(f"generated official UR20 URDF not found: {urdf_path}")
    return urdf_path, resolved_package_dirs
