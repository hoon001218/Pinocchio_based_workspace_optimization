from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pxr")
from pxr import Gf, Sdf, Usd, UsdGeom

from decanting_workspace.layout_geometry import (
    _box_planes, _box_polyhedron, _clip, _volume,
    build_layout_geometry, fragment_volume_m3,
)
from decanting_workspace.models import load_scene_spec
from decanting_workspace.usd_layout import DEFAULT_GROUPS, DEFAULT_ROBOT_BASES

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("align_a25_conveyors", ROOT / "scripts/align_a25_conveyors.py")
alignment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(alignment)


def _cube(stage, path, center, size):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.)
    cube.AddTranslateOp().Set(center)
    cube.AddScaleOp().Set(size)
    return cube


def _fixture(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    for group in DEFAULT_GROUPS:
        UsdGeom.Xform.Define(stage, group)
    moving = UsdGeom.Xform.Define(stage, alignment.MOVING)
    moving.AddTranslateOp().Set((0., 0., 0.))
    moving.AddOrientOp().Set(Gf.Quatf(1.))
    moving.AddScaleOp().Set((1., 1., 1.))
    _cube(stage, alignment.HIGH_SURFACE, (0., 0., 2.2), (.8, .6, .04))
    _cube(stage, alignment.MOVING + "/Body", (0., 0., 1.), (.8, .6, 2.))
    UsdGeom.Xform.Define(stage, alignment.ANCHOR)
    _cube(stage, alignment.ANCHOR + "/Geometry/Roller1", (.6, 0., 2.3), (.6, 1., .05))
    for name in load_scene_spec().robots:
        base = UsdGeom.Xform.Define(stage, DEFAULT_ROBOT_BASES[name])
        base.AddTranslateOp().Set((3., 4., .8))
    frame = UsdGeom.Xform.Define(stage, "/World/ToteGroup/ToteLoadFrame")
    frame.AddTranslateOp().Set((5., 6., .7))
    _cube(stage, "/World/ToteGroup/Unrelated", (5., 6., .6), (1., 1., .1))
    stage.GetRootLayer().Save()
    return stage


def _xform_defaults(layer):
    result = {}
    def visit(path):
        obj = layer.GetObjectAtPath(path)
        if isinstance(obj, Sdf.AttributeSpec) and obj.name.startswith("xformOp"):
            result[str(path)] = obj.default
    layer.Traverse(Sdf.Path.absoluteRootPath, visit)
    return result


def _intersection(polyhedron, planes):
    for normal, distance in planes:
        polyhedron = _clip(polyhedron, normal, distance)
        if polyhedron is None:
            return 0.
    return _volume(polyhedron)


def test_alignment_moves_whole_a25_only_and_second_run_does_not_change_source(tmp_path):
    source, backup, report = (tmp_path / name for name in ("scene.usda", "before.usda", "report.json"))
    stage = _fixture(source)
    original_bytes = source.read_bytes()
    old = _xform_defaults(stage.GetRootLayer())
    result = alignment.align_source(source, backup=backup, report=report)
    assert result["changed"]
    assert result["world_delta_m"] == pytest.approx((0., 0., .105))
    assert backup.read_bytes() == original_bytes
    new = _xform_defaults(stage.GetRootLayer())
    changed = {path for path in old if old[path] != new[path]}
    assert changed == {alignment.MOVING + ".xformOp:translate"}
    # Body is deliberately excluded from simplified geometry, but the full USD
    # assembly still translates, including its detailed supporting structure.
    body = stage.GetPrimAtPath(alignment.MOVING + "/Body")
    assert UsdGeom.XformCache().GetLocalToWorldTransform(body).ExtractTranslation()[2] == pytest.approx(1.105)
    aligned_bytes = source.read_bytes()
    second = alignment.align_source(source, backup=backup, report=report)
    assert not second["changed"]
    assert source.read_bytes() == aligned_bytes
    assert backup.read_bytes() == original_bytes
    assert second["baseline_source_sha256"] == second["updated_source_sha256"]
    assert json.loads(report.read_text())["world_delta_m"] == [0., 0., 0.]


@pytest.fixture(scope="module")
def collected():
    if not all(path.is_file() for path in (alignment.DEFAULT_SOURCE, alignment.DEFAULT_BACKUP, alignment.DEFAULT_REPORT)):
        pytest.skip("collected USD alignment fixture is absent")
    return alignment._read_layout(alignment.DEFAULT_SOURCE)


def test_collected_high_endpoint_matches_fixed_top_conveyor(collected):
    plan = alignment.plan_alignment(collected)
    assert plan["world_delta_m"] == [0., 0., 0.]
    assert plan["a25_high_top_z_m"] == pytest.approx(plan["anchor_top_z_m"], abs=1e-6)
    assert min(plan["xy_overlap_m"]) > .1


def test_collected_source_preserves_every_other_authored_transform(collected):
    assert alignment.DEFAULT_BACKUP.is_file()
    old = _xform_defaults(Sdf.Layer.FindOrOpen(str(alignment.DEFAULT_BACKUP)))
    new = _xform_defaults(Sdf.Layer.FindOrOpen(str(alignment.DEFAULT_SOURCE)))
    assert old.keys() == new.keys()
    differences = {path for path in old if old[path] != new[path]}
    assert differences == {alignment.MOVING + ".xformOp:translate"}
    movement = np.asarray(new[next(iter(differences))]) - np.asarray(old[next(iter(differences))])
    assert movement[:2] == pytest.approx((0., 0.), abs=1e-12)
    assert movement[2] == pytest.approx(.0286629873966467, abs=1e-6)
    report = json.loads(alignment.DEFAULT_REPORT.read_text(encoding="utf-8"))
    assert report["backup_sha256"] == hashlib.sha256(alignment.DEFAULT_BACKUP.read_bytes()).hexdigest()
    assert report["updated_source_sha256"] == hashlib.sha256(alignment.DEFAULT_SOURCE.read_bytes()).hexdigest()
    assert report["preserved"]["robots"] == len(collected.robots) == 2
    assert report["preserved"]["frames"] == len(collected.frames)


def test_collected_connected_union_keeps_volume_with_one_owner(collected):
    boxes = tuple(b for b in collected.boxes if b.name in (alignment.HIGH_SURFACE, alignment.ANCHOR_SURFACE))
    assert len(boxes) == 2
    first, second = boxes
    intersection = _intersection(_box_polyhedron(second), _box_planes(first))
    assert intersection > 1e-5
    scene = replace(collected, boxes=boxes)
    fragments = build_layout_geometry(scene)
    expected = sum(float(np.prod(b.size_m)) for b in boxes) - intersection
    assert sum(fragment_volume_m3(f) for f in fragments) == pytest.approx(expected, abs=1e-8)
    first_volume = sum(fragment_volume_m3(f) for f in fragments if f.source_name == first.name)
    second_volume = sum(fragment_volume_m3(f) for f in fragments if f.source_name == second.name)
    assert first_volume == pytest.approx(float(np.prod(first.size_m)), abs=1e-8)
    assert second_volume == pytest.approx(float(np.prod(second.size_m)) - intersection, abs=1e-8)
    for first_fragment, second_fragment in itertools.combinations(fragments, 2):
        if first_fragment.source_name == second_fragment.source_name:
            continue
        # First-owned source must retain all overlap; no later fragment may
        # occupy any positive volume inside it.
        later = second_fragment if second_fragment.source_name != first.name else first_fragment
        vertices = np.asarray(later.vertices_m)
        polyhedron = tuple(vertices[list(triangle)] for triangle in later.triangles)
        assert _intersection(polyhedron, _box_planes(first)) < 1e-10
