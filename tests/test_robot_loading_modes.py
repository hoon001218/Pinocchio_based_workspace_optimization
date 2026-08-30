from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from types import ModuleType
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from decanting_workspace import load_scene_spec
from decanting_workspace.models import CutterProxySpec
from decanting_workspace.playback_backend import _validate_sr_display_contract
from decanting_workspace.robots import (
    OFFICIAL_UR20_COMMIT,
    attach_cutter_proxy,
    attach_suction_proxy,
    configure_sr_j3_stroke,
    default_vendored_ur20_root,
    load_robot_bundle,
    resolve_official_ur20,
    validate_vendored_ur20_asset,
)


@pytest.fixture(scope="module")
def ur20_urdf() -> Path:
    return load_scene_spec().robots["ur20"].urdf_path


def test_visual_loading_is_opt_in(ur20_urdf: Path) -> None:
    collision_only = load_robot_bundle("ur20", ur20_urdf)
    assert collision_only.collision_model.ngeoms > 0
    assert collision_only.visual_model is None

    with_visuals = load_robot_bundle("ur20", ur20_urdf, load_visual=True)
    assert with_visuals.collision_model.ngeoms > 0
    assert with_visuals.visual_model is not None
    assert with_visuals.visual_model.ngeoms > 0


def test_suction_proxy_is_attached_to_tool0_positive_z(ur20_urdf: Path) -> None:
    bundle = load_robot_bundle("ur20", ur20_urdf)
    initial_count = bundle.collision_model.ngeoms
    tool_frame_id = bundle.model.getFrameId("tool0")
    tool_frame = bundle.model.frames[tool_frame_id]

    stem_id, pad_id = attach_suction_proxy(bundle, 0.20, 0.03, 0.08, 0.02)

    assert (stem_id, pad_id) == (initial_count, initial_count + 1)
    assert bundle.visual_model is None
    stem = bundle.collision_model.geometryObjects[stem_id]
    pad = bundle.collision_model.geometryObjects[pad_id]
    assert stem.name == "suction_proxy_stem"
    assert pad.name == "suction_proxy_pad"
    assert stem.parentFrame == tool_frame_id
    assert pad.parentFrame == tool_frame_id
    assert stem.geometry.radius == pytest.approx(0.03)
    assert stem.geometry.halfLength == pytest.approx(0.10)
    assert pad.geometry.radius == pytest.approx(0.08)
    assert pad.geometry.halfLength == pytest.approx(0.01)
    assert np.allclose(
        stem.placement.translation,
        tool_frame.placement.act(np.array((0.0, 0.0, 0.10))),
    )
    assert np.allclose(
        pad.placement.translation,
        tool_frame.placement.act(np.array((0.0, 0.0, 0.21))),
    )

    with pytest.raises(ValueError, match="already contains"):
        attach_suction_proxy(bundle, 0.20, 0.03, 0.08, 0.02)


def test_cutter_proxy_is_attached_to_tool0(ur20_urdf: Path) -> None:
    bundle = load_robot_bundle("test_robot", ur20_urdf)
    proxy = CutterProxySpec(
        center_local_m=(0.12, 0.0, -0.055),
        size_m=(0.25, 0.09, 0.11),
        tcp_translation_local_m=(0.22, 0.0, -0.11),
    )
    tool_frame_id = bundle.model.getFrameId("tool0")
    tool_frame = bundle.model.frames[tool_frame_id]

    geometry_id = attach_cutter_proxy(bundle, proxy)
    cutter = bundle.collision_model.geometryObjects[geometry_id]

    assert cutter.name == "cutter_proxy_box"
    assert cutter.parentFrame == tool_frame_id
    assert np.allclose(cutter.geometry.halfSide, np.asarray(proxy.size_m) / 2.0)
    assert np.allclose(
        cutter.placement.translation,
        tool_frame.placement.act(np.asarray(proxy.center_local_m)),
    )
    with pytest.raises(ValueError, match="already contains"):
        attach_cutter_proxy(bundle, proxy)


def test_only_configured_scene_tool_tcp_frames_are_added() -> None:
    spec = load_scene_spec()
    ur = load_robot_bundle(
        "ur20",
        spec.robots["ur20"].urdf_path,
        suction_proxy=spec.robots["ur20"].suction_proxy,
    )
    sr = load_robot_bundle("sr12ia", spec.robots["sr12ia"].urdf_path)

    tool_id = ur.model.getFrameId("tool0")
    tcp_id = ur.model.getFrameId("suction_tcp")
    assert tcp_id < ur.model.nframes
    expected = ur.model.frames[tool_id].placement.act(np.array((0.0, 0.0, 0.224)))
    assert np.allclose(ur.model.frames[tcp_id].placement.translation, expected)
    assert not sr.model.existFrame("cutter_tcp")
    assert not sr.model.existFrame("cutter_proxy")


def test_sr_j3_limit_is_restricted_by_selected_option() -> None:
    spec = load_scene_spec()
    sr = load_robot_bundle("sr12ia", spec.robots["sr12ia"].urdf_path)
    joint = sr.model.joints[sr.model.getJointId("joint3")]

    configure_sr_j3_stroke(sr, 0.3)

    assert sr.model.upperPositionLimit[joint.idx_q] == pytest.approx(0.3)
    configure_sr_j3_stroke(sr, 0.45)
    assert sr.model.upperPositionLimit[joint.idx_q] == pytest.approx(0.45)


def test_display_only_sr_requires_the_same_joint_and_tool0_contract() -> None:
    spec = load_scene_spec()
    evaluation = load_robot_bundle(
        "sr_evaluation",
        spec.robots["sr12ia"].urdf_path,
    )
    compatible_visual = load_robot_bundle(
        "sr_visual",
        spec.robots["sr12ia"].urdf_path,
    )
    incompatible_visual = load_robot_bundle(
        "ur_visual",
        spec.robots["ur20"].urdf_path,
    )

    _validate_sr_display_contract(evaluation, compatible_visual)
    with pytest.raises(ValueError, match="joint contract differs"):
        _validate_sr_display_contract(evaluation, incompatible_visual)


def test_default_official_resolver_uses_portable_vendored_bundle() -> None:
    urdf_path, package_dirs = resolve_official_ur20()
    asset_root = default_vendored_ur20_root().resolve()

    assert urdf_path == asset_root / "ur20.urdf"
    assert package_dirs == (asset_root,)
    assert validate_vendored_ur20_asset() == urdf_path

    root = ET.parse(urdf_path).getroot()
    mesh_filenames = {
        element.attrib["filename"] for element in root.findall(".//mesh")
    }
    assert len(mesh_filenames) == 14
    assert all("://" not in filename for filename in mesh_filenames)
    assert all(not Path(filename).is_absolute() for filename in mesh_filenames)
    assert all((asset_root / filename).is_file() for filename in mesh_filenames)

    bundle = load_robot_bundle(
        "ur20",
        urdf_path,
        package_dirs=package_dirs,
        load_visual=True,
    )
    assert bundle.model.nq == 13
    assert bundle.collision_model.ngeoms == 7
    assert bundle.visual_model is not None
    assert bundle.visual_model.ngeoms == 7


def test_vendored_official_resolver_rejects_asset_tampering(tmp_path: Path) -> None:
    copied_root = tmp_path / "official"
    shutil.copytree(default_vendored_ur20_root(), copied_root)
    (copied_root / "LICENSE-BSD-3-Clause.txt").write_text(
        "tampered\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_vendored_ur20_asset(copied_root)


def test_official_resolver_pins_commit_and_scopes_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "clones" / "ur_description"
    xacro_path = repository / "urdf" / "ur.urdf.xacro"
    xacro_path.parent.mkdir(parents=True)
    xacro_path.write_text("<robot/>", encoding="utf-8")
    generated_urdf = tmp_path / "generated" / "ur20.urdf"
    generated_urdf.parent.mkdir()
    generated_urdf.write_text("<robot name='ur20'/>", encoding="utf-8")
    requested_cache = tmp_path / "caller-cache"
    calls: dict[str, object] = {}
    original_git_config_count = os.environ.get("GIT_CONFIG_COUNT")
    initial_git_config_count = int(original_git_config_count or "0")

    package = ModuleType("robot_descriptions")
    package.__path__ = []  # type: ignore[attr-defined]
    cache_module = ModuleType("robot_descriptions._cache")
    package_dirs_module = ModuleType("robot_descriptions._package_dirs")
    xacro_module = ModuleType("robot_descriptions._xacro")

    def fake_clone_to_cache(name: str, commit: str | None = None) -> str:
        calls["repository"] = name
        calls["commit"] = commit
        calls["clone_cache"] = os.environ.get("ROBOT_DESCRIPTIONS_CACHE")
        active_count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
        calls["git_config_count"] = active_count
        calls["git_safe_key"] = os.environ.get(
            f"GIT_CONFIG_KEY_{active_count - 1}"
        )
        calls["git_safe_value"] = os.environ.get(
            f"GIT_CONFIG_VALUE_{active_count - 1}"
        )
        return str(repository)

    def fake_get_urdf_path(
        description: ModuleType,
        xacro_args: dict[str, str] | None = None,
    ) -> str:
        calls["xacro_path"] = description.XACRO_PATH
        calls["xacro_args"] = xacro_args
        calls["xacro_cache"] = os.environ.get("ROBOT_DESCRIPTIONS_CACHE")
        return str(generated_urdf)

    def fake_get_package_dirs(description: ModuleType) -> list[str]:
        calls["package_path"] = description.PACKAGE_PATH
        return [description.PACKAGE_PATH, str(repository.parent), description.PACKAGE_PATH]

    cache_module.clone_to_cache = fake_clone_to_cache  # type: ignore[attr-defined]
    package_dirs_module.get_package_dirs = fake_get_package_dirs  # type: ignore[attr-defined]
    xacro_module.get_urdf_path = fake_get_urdf_path  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "robot_descriptions", package)
    monkeypatch.setitem(sys.modules, "robot_descriptions._cache", cache_module)
    monkeypatch.setitem(
        sys.modules,
        "robot_descriptions._package_dirs",
        package_dirs_module,
    )
    monkeypatch.setitem(sys.modules, "robot_descriptions._xacro", xacro_module)
    monkeypatch.setenv("ROBOT_DESCRIPTIONS_CACHE", "preexisting-cache")

    urdf_path, package_dirs = resolve_official_ur20(cache_dir=requested_cache)

    expected_cache = str(requested_cache.resolve())
    assert OFFICIAL_UR20_COMMIT == "ae333289875f9ba5a9ea6649a54036efb5ccabee"
    assert calls["repository"] == "Universal_Robots_ROS2_Description"
    assert calls["commit"] == OFFICIAL_UR20_COMMIT
    assert calls["clone_cache"] == expected_cache
    assert calls["git_config_count"] == initial_git_config_count + 1
    assert calls["git_safe_key"] == "safe.directory"
    assert calls["git_safe_value"] == (
        requested_cache
        / f"ur_description-{OFFICIAL_UR20_COMMIT}"
        / "ur_description"
    ).resolve().as_posix()
    assert calls["xacro_cache"] == expected_cache
    assert calls["xacro_path"] == str(xacro_path.resolve())
    assert calls["xacro_args"] == {"ur_type": "ur20", "name": "ur20"}
    assert calls["package_path"] == str(repository.resolve())
    assert urdf_path == generated_urdf.resolve()
    assert package_dirs == (repository.resolve(), repository.parent.resolve())
    assert os.environ["ROBOT_DESCRIPTIONS_CACHE"] == "preexisting-cache"
    assert os.environ.get("GIT_CONFIG_COUNT") == original_git_config_count
