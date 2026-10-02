from __future__ import annotations

from dataclasses import replace

import pytest
import yaml

from decanting_workspace.models import load_scene_spec
from decanting_workspace.precompute import scene_spec_fingerprint


def _write(path, raw):
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")


def test_profile_inherits_nominal_configuration_across_directories(tmp_path):
    original = load_scene_spec()
    profile = tmp_path / "profile.yaml"
    _write(profile, {"base_config": str(original.source_path)})
    derived = load_scene_spec(profile)
    assert derived == replace(original, source_path=profile.resolve())
    assert scene_spec_fingerprint(derived) == scene_spec_fingerprint(original)


def test_nested_profile_preserves_inherited_paths_and_partial_robot_override(tmp_path):
    original = load_scene_spec()
    nested = tmp_path / "nested"
    nested.mkdir()
    base = tmp_path / "base.yaml"
    _write(base, {"base_config": str(original.source_path), "robots": {"ur20": {"nominal_base": [0,0,1,0]}}})
    profile = nested / "profile.yaml"
    _write(profile, {"base_config": "../base.yaml", "tote": {"representative_yaw_deg": 0}})
    derived = load_scene_spec(profile)
    assert derived.robots["ur20"].urdf_path == original.robots["ur20"].urdf_path
    assert derived.robots["ur20"].nominal_base.xyz_m == (0.,0.,1.)
    assert derived.frames == original.frames
    assert derived.tote.representative_yaw_deg == 0.


def test_profile_cycles_and_missing_bases_fail_clearly(tmp_path):
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    _write(first, {"base_config": "second.yaml"})
    _write(second, {"base_config": "first.yaml"})
    with pytest.raises(ValueError, match="cycle"):
        load_scene_spec(first)
    _write(first, {"base_config": "absent.yaml"})
    with pytest.raises(FileNotFoundError, match="scene configuration not found"):
        load_scene_spec(first)


def test_inherited_usd_anchors_to_file_declaring_it(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    original = load_scene_spec()
    nested = tmp_path / "nested"
    nested.mkdir()
    stage = Usd.Stage.CreateNew(str(tmp_path / "environment.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.Cube.Define(stage, "/World/Cube")
    stage.GetRootLayer().Save()
    _write(tmp_path / "base.yaml", {"base_config": str(original.source_path), "usd_scene": {"file": "environment.usda"}})
    _write(nested / "profile.yaml", {"base_config": "../base.yaml"})
    assert len(load_scene_spec(nested / "profile.yaml").static_meshes) == 1
