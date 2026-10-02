from __future__ import annotations

import numpy as np
import os
import pytest

pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics

from decanting_workspace.models import MeshObstacle, load_scene_spec
from decanting_workspace.precompute import scene_spec_fingerprint
from decanting_workspace.usd_scene import _usd_open_path, load_usd_environment


def _stage(path, *, units=1., up="Z"):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, units)
    UsdGeom.SetStageUpAxis(stage, up)
    UsdGeom.Xform.Define(stage, "/World")
    return stage


def _mesh(stage, path, *, points=None, counts=None, indices=None):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points or [(0,0,0),(2,0,0),(2,1,0),(1,1,0),(1,2,0),(0,2,0)])
    mesh.CreateFaceVertexCountsAttr(counts or [6])
    mesh.CreateFaceVertexIndicesAttr(indices or [0,1,2,3,4,5])
    mesh.CreateSubdivisionSchemeAttr("none")
    return mesh


def _import(path, **options):
    return load_usd_environment(load_scene_spec(), options, usd_path=path)


def test_concave_mesh_keeps_boundary_and_world_transform(tmp_path):
    path = tmp_path / "environment.usda"
    stage = _stage(path, units=.01, up="Y")
    root = UsdGeom.Xform.Define(stage, "/World/Parent")
    root.AddTranslateOp().Set((100,200,300))
    root.AddScaleOp().Set((2,3,4))
    _mesh(stage, "/World/Parent/Concave")
    stage.GetRootLayer().Save()
    spec = _import(path)
    mesh, = spec.static_meshes
    assert len(mesh.triangles) == 4
    np.testing.assert_allclose(mesh.vertices_m[0], (1., -3., 2.))
    np.testing.assert_allclose(mesh.vertices_m[2], (1.04, -3., 2.03))
    vertices = np.asarray(mesh.vertices_m)
    area = sum(np.linalg.norm(np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])) / 2.
               for a,b,c in mesh.triangles)
    assert area == pytest.approx(3. * .02 * .03)
    assert all(not box.collision_enabled for box in spec.static_boxes)
    assert spec.frames == load_scene_spec().frames
    assert spec.robots == load_scene_spec().robots


def test_loading_via_config_and_new_objects_change_fingerprint(tmp_path):
    import yaml

    source_spec = load_scene_spec()
    config = yaml.safe_load(source_spec.source_path.read_text(encoding="utf-8"))
    config["reference_frames_file"] = str(source_spec.source_path.parent / "reference_frames.json")
    for name, robot in source_spec.robots.items():
        config["robots"][name]["urdf"] = str(robot.urdf_path)
    config["usd_scene"] = {"file": "environment.usda"}
    config_path = tmp_path / "cell.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    path = tmp_path / "environment.usda"
    stage = _stage(path)
    _mesh(stage, "/World/Concave")
    stage.GetRootLayer().Save()
    original = load_scene_spec(config_path)
    original_hash = scene_spec_fingerprint(original)
    UsdGeom.Cube.Define(stage, "/World/Added").CreateSizeAttr(.5)
    stage.GetRootLayer().Save()
    updated = load_scene_spec(config_path)
    assert len(updated.static_meshes) == 2
    assert scene_spec_fingerprint(updated) != original_hash


def test_mesh_topology_holes_and_handedness(tmp_path):
    path = tmp_path / "holes.usda"
    stage = _stage(path)
    mesh = _mesh(stage, "/World/Mesh", points=[(0,0,0),(1,0,0),(0,1,0),(0,0,1)],
                 counts=[3,3], indices=[0,1,2,0,2,3])
    mesh.CreateHoleIndicesAttr([1])
    mesh.CreateOrientationAttr("leftHanded")
    stage.GetRootLayer().Save()
    imported, = _import(path).static_meshes
    assert imported.triangles == ((0,2,1),)


def test_box_binding_uses_subtree_meshes_without_box_collision(tmp_path):
    path = tmp_path / "support.usda"
    stage = _stage(path)
    root = UsdGeom.Xform.Define(stage, "/World/Table")
    root.AddTranslateOp().Set((2,3,.5))
    root.AddRotateZOp().Set(30.)
    root.AddScaleOp().Set((2,1,.1))
    UsdGeom.Cube.Define(stage, "/World/Table/Surface").CreateSizeAttr(1.)
    stage.GetRootLayer().Save()
    spec = _import(path, box_prims={"worktable_current_a": "/World/Table"})
    support = next(box for box in spec.static_boxes if box.name == "worktable_current_a")
    assert support.center_m == pytest.approx((2,3,.5))
    assert support.size_m == pytest.approx((2,1,.1))
    assert support.yaw_deg == pytest.approx(30.)
    assert not support.collision_enabled
    assert spec.static_meshes[0].name == "worktable_current_a"
    assert spec.tote.support_surface_z_m == pytest.approx(.55)


def test_references_payloads_and_native_instances(tmp_path):
    asset = tmp_path / "asset.usda"
    asset_stage = _stage(asset)
    root = UsdGeom.Xform.Define(asset_stage, "/World/Shape")
    root.AddTranslateOp().Set((0,1,0))
    _mesh(asset_stage, "/World/Shape/Mesh", points=[(0,0,0),(1,0,0),(0,0,1)],
          counts=[3], indices=[0,1,2])
    asset_stage.GetRootLayer().Save()
    path = tmp_path / "instances.usda"
    stage = _stage(path)
    for index in range(2):
        prim = UsdGeom.Xform.Define(stage, f"/World/Instance{index}").GetPrim()
        prim.GetReferences().AddReference("asset.usda", "/World/Shape")
        prim.SetInstanceable(True)
        UsdGeom.Xformable(prim).AddTranslateOp(opSuffix="instance").Set((index * 3,0,0))
    payload = UsdGeom.Xform.Define(stage, "/World/Payload").GetPrim()
    payload.GetPayloads().AddPayload("asset.usda", "/World/Shape")
    stage.GetRootLayer().Save()
    spec = _import(path)
    assert len(spec.static_meshes) == 3
    by_name = {mesh.name: mesh for mesh in spec.static_meshes}
    assert by_name["usd:/World/Instance1/Mesh"].vertices_m[0] == pytest.approx((3,1,0))


def test_point_instances_are_expanded_and_masks_respected(tmp_path):
    path = tmp_path / "point_instances.usda"
    stage = _stage(path)
    instancer = UsdGeom.PointInstancer.Define(stage, "/World/Instances")
    cube = UsdGeom.Cube.Define(stage, "/World/Instances/Prototypes/Cube")
    cube.CreateSizeAttr(1.)
    cube.AddTranslateOp().Set((0,0,1))
    instancer.CreatePrototypesRel().SetTargets([cube.GetPath()])
    instancer.CreateProtoIndicesAttr([0,0,0])
    instancer.CreatePositionsAttr([(0,0,0),(3,0,0),(6,0,0)])
    instancer.CreateInvisibleIdsAttr([2])
    stage.GetRootLayer().Save()
    spec = _import(path)
    assert len(spec.static_meshes) == 2
    assert np.asarray(spec.static_meshes[1].vertices_m).mean(axis=0) == pytest.approx((3,0,1))


def test_selection_visibility_and_articulated_robots(tmp_path):
    path = tmp_path / "selection.usda"
    stage = _stage(path)
    _mesh(stage, "/World/Environment/Mesh")
    hidden = _mesh(stage, "/World/Hidden")
    hidden.CreateVisibilityAttr("invisible")
    robot = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot)
    _mesh(stage, "/World/Robot/Link")
    _mesh(stage, "/World/Excluded")
    stage.GetRootLayer().Save()
    spec = _import(path, exclude_prim_paths=["/World/Excluded"])
    assert [mesh.name for mesh in spec.static_meshes] == ["usd:/World/Environment/Mesh"]


@pytest.mark.parametrize("shape", ["Sphere", "Cylinder", "Cone", "Capsule", "Cube", "Plane"])
def test_analytic_primitives_are_tessellated(tmp_path, shape):
    path = tmp_path / f"{shape}.usda"
    stage = _stage(path)
    schema = getattr(UsdGeom, shape).Define(stage, "/World/Shape")
    schema.AddTranslateOp().Set((1,2,3))
    stage.GetRootLayer().Save()
    mesh, = _import(path).static_meshes
    assert len(mesh.triangles) > 0
    assert np.all(np.isfinite(mesh.vertices_m))


def test_unresolved_payload_is_error_unless_excluded(tmp_path):
    path = tmp_path / "broken.usda"
    stage = _stage(path)
    _mesh(stage, "/World/Environment")
    missing = UsdGeom.Xform.Define(stage, "/World/Missing").GetPrim()
    missing.GetPayloads().AddPayload("absent.usda")
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="unresolved composition dependency"):
        _import(path)
    assert len(_import(path, exclude_prim_paths=["/World/Missing"]).static_meshes) == 1


def _record_open_stages(monkeypatch):
    opened = []
    original_open = Usd.Stage.Open
    def recording_open(*args, **kwargs):
        stage = original_open(*args, **kwargs)
        opened.append(stage)
        return stage
    monkeypatch.setattr(Usd.Stage, "Open", recording_open)
    return opened


@pytest.mark.parametrize("selection", [
    {"exclude_prim_paths": ["/World/Missing"]},
    {"include_prim_paths": ["/World/Environment"]},
])
def test_unselected_broken_payload_stays_unloaded_without_warnings(
    tmp_path, monkeypatch, capfd, selection,
):
    path = tmp_path / "selective.usda"
    stage = _stage(path)
    UsdGeom.Cube.Define(stage, "/World/Environment/Box")
    missing = UsdGeom.Xform.Define(stage, "/World/Missing").GetPrim()
    missing.GetPayloads().AddPayload("absent.usda")
    stage.GetRootLayer().Save()
    capfd.readouterr()  # Authoring a payload on a loaded stage may warn.
    opened = _record_open_stages(monkeypatch)

    spec = _import(path, **selection)

    assert [mesh.name for mesh in spec.static_meshes] == ["usd:/World/Environment/Box"]
    assert not opened[-1].GetLoadRules().IsLoaded("/World/Missing")
    assert not opened[-1].GetCompositionErrors()
    assert "absent.usda" not in capfd.readouterr().err


def test_selected_missing_payload_remains_strict_error(tmp_path, monkeypatch):
    path = tmp_path / "selected_missing.usda"
    stage = _stage(path)
    missing = UsdGeom.Xform.Define(stage, "/World/Missing").GetPrim()
    missing.GetPayloads().AddPayload("absent.usda")
    stage.GetRootLayer().Save()
    opened = _record_open_stages(monkeypatch)

    with pytest.raises(ValueError, match="unresolved composition dependency"):
        _import(path, include_prim_paths=["/World/Missing"])

    assert opened[-1].GetLoadRules().IsLoaded("/World/Missing")
    assert opened[-1].GetCompositionErrors()


@pytest.mark.parametrize("include_root", ["/World/Selected", "/World/Selected/Box"])
def test_selected_valid_payload_and_child_selection_load_only_required_descendants(
    tmp_path, monkeypatch, capfd, include_root,
):
    asset = tmp_path / "payload.usda"
    asset_stage = _stage(asset)
    root = UsdGeom.Xform.Define(asset_stage, "/World/Shape")
    root.AddTranslateOp().Set((2, 0, 0))
    UsdGeom.Cube.Define(asset_stage, "/World/Shape/Box")
    # A sibling payload is unrelated when the include root is inside the
    # parent payload. Explicit exclusion also covers a whole-parent include.
    broken = UsdGeom.Xform.Define(asset_stage, "/World/Shape/Broken").GetPrim()
    broken.GetPayloads().AddPayload("absent.usda")
    asset_stage.GetRootLayer().Save()
    path = tmp_path / "selected.usda"
    stage = _stage(path)
    payload = UsdGeom.Xform.Define(stage, "/World/Selected").GetPrim()
    payload.GetPayloads().AddPayload("payload.usda", "/World/Shape")
    stage.GetRootLayer().Save()
    capfd.readouterr()
    opened = _record_open_stages(monkeypatch)

    spec = _import(
        path, include_prim_paths=[include_root],
        exclude_prim_paths=["/World/Selected/Broken"],
    )

    imported, = spec.static_meshes
    assert imported.name == "usd:/World/Selected/Box"
    assert np.asarray(imported.vertices_m).mean(axis=0) == pytest.approx((2, 0, 0))
    assert opened[-1].GetLoadRules().IsLoaded("/World/Selected/Box")
    assert not opened[-1].GetLoadRules().IsLoaded("/World/Selected/Broken")
    assert not opened[-1].GetCompositionErrors()
    assert "absent.usda" not in capfd.readouterr().err


def test_authored_articulation_payload_is_excluded_before_loading(tmp_path, monkeypatch, capfd):
    path = tmp_path / "robot_payload.usda"
    stage = _stage(path)
    UsdGeom.Cube.Define(stage, "/World/Environment")
    robot = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(robot)
    robot.GetPayloads().AddPayload("absent_robot.usda")
    stage.GetRootLayer().Save()
    capfd.readouterr()
    opened = _record_open_stages(monkeypatch)

    spec = _import(path)

    assert len(spec.static_meshes) == 1
    assert not opened[-1].GetLoadRules().IsLoaded("/World/Robot")
    assert not opened[-1].GetCompositionErrors()
    assert "absent_robot.usda" not in capfd.readouterr().err


@pytest.mark.parametrize("broken_prototype", [False, True])
def test_selected_instancer_loads_external_prototype_dependency_strictly(
    tmp_path, monkeypatch, capfd, broken_prototype,
):
    asset = tmp_path / "prototype.usda"
    asset_stage = _stage(asset)
    root = UsdGeom.Xform.Define(asset_stage, "/World/Shape")
    root.AddTranslateOp().Set((2, 0, 0))
    UsdGeom.Cube.Define(asset_stage, "/World/Shape/Box")
    asset_stage.GetRootLayer().Save()
    path = tmp_path / "external_prototypes.usda"
    stage = _stage(path)
    instancer = UsdGeom.PointInstancer.Define(stage, "/World/Instances")
    prototype = UsdGeom.Xform.Define(stage, "/World/Prototypes/Shape").GetPrim()
    prototype.GetPayloads().AddPayload(
        "absent_prototype.usda" if broken_prototype else "prototype.usda", "/World/Shape",
    )
    instancer.CreatePrototypesRel().SetTargets([prototype.GetPath()])
    instancer.CreateProtoIndicesAttr([0])
    instancer.CreatePositionsAttr([(3, 0, 0)])
    unrelated = UsdGeom.Xform.Define(stage, "/World/Unrelated").GetPrim()
    unrelated.GetPayloads().AddPayload("absent_unrelated.usda")
    stage.GetRootLayer().Save()
    capfd.readouterr()
    opened = _record_open_stages(monkeypatch)

    if broken_prototype:
        with pytest.raises(ValueError, match="unresolved composition dependency"):
            _import(path, include_prim_paths=["/World/Instances"])
    else:
        imported, = _import(path, include_prim_paths=["/World/Instances"]).static_meshes
        assert imported.name == "usd:/World/Instances/instance_0/Box"
        assert np.asarray(imported.vertices_m).mean(axis=0) == pytest.approx((5, 0, 0))
        assert not opened[-1].GetCompositionErrors()
    assert opened[-1].GetLoadRules().IsLoaded("/World/Prototypes/Shape")
    assert not opened[-1].GetLoadRules().IsLoaded("/World/Unrelated")
    assert "absent_unrelated.usda" not in capfd.readouterr().err


def test_invalid_topology_and_unsupported_geometry_fail_clearly(tmp_path):
    path = tmp_path / "invalid.usda"
    stage = _stage(path)
    mesh = _mesh(stage, "/World/Mesh", counts=[3], indices=[0,1,99])
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="invalid USD mesh topology"):
        _import(path)
    stage.RemovePrim(mesh.GetPath())
    UsdGeom.Points.Define(stage, "/World/Points").CreatePointsAttr([(0,0,0)])
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="unsupported USD geometry Points"):
        _import(path)


def test_numeric_time_reads_animated_snapshot(tmp_path):
    path = tmp_path / "animated.usda"
    stage = _stage(path)
    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    translate = cube.AddTranslateOp()
    translate.Set((0,0,0), 0.)
    translate.Set((10,0,0), 10.)
    stage.GetRootLayer().Save()
    mesh, = _import(path, time_code=5.).static_meshes
    assert np.asarray(mesh.vertices_m).mean(axis=0) == pytest.approx((5,0,0))


@pytest.mark.parametrize("collapsed_transform", [False, True])
def test_empty_degenerate_surface_is_rejected(tmp_path, collapsed_transform):
    path = tmp_path / "degenerate.usda"
    stage = _stage(path)
    points = [(0,0,0),(1,0,0),(0,1,0)] if collapsed_transform else [(0,0,0),(1,0,0),(2,0,0)]
    mesh = _mesh(stage, "/World/Mesh", points=points, counts=[3], indices=[0,1,2])
    if collapsed_transform:
        mesh.AddScaleOp().Set((0,0,0))
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="no non-degenerate faces"):
        _import(path)


def test_skinned_geometry_requires_baked_surface(tmp_path):
    from pxr import Sdf

    path = tmp_path / "skin.usda"
    stage = _stage(path)
    mesh = _mesh(stage, "/World/Mesh")
    mesh.GetPrim().CreateAttribute("primvars:skel:jointIndices", Sdf.ValueTypeNames.IntArray).Set([0])
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match="must be baked"):
        _import(path)


def test_y_up_task_support_bounds_are_converted_to_z_up(tmp_path):
    path = tmp_path / "y_up_table.usda"
    stage = _stage(path, up="Y")
    cube = UsdGeom.Cube.Define(stage, "/World/Table")
    cube.CreateSizeAttr(1.)
    cube.AddTranslateOp().Set((2,.5,-3))
    cube.AddScaleOp().Set((2,.1,1))
    stage.GetRootLayer().Save()
    spec = _import(path, box_prims={"worktable_current_a": "/World/Table"})
    support = next(box for box in spec.static_boxes if box.name == "worktable_current_a")
    assert support.center_m == pytest.approx((2,3,.5))
    assert support.size_m == pytest.approx((2,1,.1))
    assert spec.tote.support_surface_z_m == pytest.approx(.55)


def test_tilted_geometry_is_valid_obstacle_but_not_task_support(tmp_path):
    path = tmp_path / "tilted.usda"
    stage = _stage(path)
    cube = UsdGeom.Cube.Define(stage, "/World/Tilted")
    cube.AddRotateXOp().Set(30.)
    stage.GetRootLayer().Save()
    assert len(_import(path).static_meshes) == 1
    with pytest.raises(ValueError, match="must remain horizontal"):
        _import(path, box_prims={"worktable_current_a": "/World/Tilted"})


def test_mesh_data_cannot_change_under_identity_cached_collision_geometry():
    vertices = [[0,0,0], [1,0,0], [0,1,0]]
    triangles = [[np.int64(0), np.int64(1), np.int64(2)]]
    mesh = MeshObstacle("surface", "obstacle", vertices, triangles)
    vertices[0][0] = 99
    triangles[0][0] = 2
    assert mesh.vertices_m[0] == (0.,0.,0.)
    assert mesh.triangles == ((0,1,2),)
    with pytest.raises(ValueError, match="integer indices"):
        MeshObstacle("invalid", "obstacle", mesh.vertices_m, ((0.,1,2),))


def test_support_bounds_select_rollers_without_removing_tall_frame_geometry(tmp_path):
    path = tmp_path / "roller_conveyor.usda"
    stage = _stage(path)
    for name, center, scale in (
        ("RollerLeft", (-1, 0, .6), (1, 1, .1)),
        ("RollerRight", (1, 0, .6), (1, 1, .1)),
        ("Frame", (0, 0, 2), (5, 2, .2)),
    ):
        cube = UsdGeom.Cube.Define(stage, "/World/Conveyor/Geometry/" + name)
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(center)
        cube.AddScaleOp().Set(scale)
    stage.GetRootLayer().Save()

    spec = _import(
        path,
        box_prims={"conveyor_level_1": "/World/Conveyor"},
        box_bounds_prims={"conveyor_level_1": "/World/Conveyor/*/Roller*"},
    )

    support = next(box for box in spec.static_boxes if box.name == "conveyor_level_1")
    assert support.center_m == pytest.approx((0, 0, .6))
    assert support.size_m == pytest.approx((3, 1, .1))
    mesh, = spec.static_meshes
    assert len(mesh.triangles) == 36
    assert max(point[2] for point in mesh.vertices_m) == pytest.approx(2.1)


@pytest.mark.parametrize("selection,match", [
    ({"unknown_box": "/World/Conveyor"}, "needs a box_prims binding"),
    ({"conveyor_level_1": []}, "absolute prim paths/patterns"),
    ({"conveyor_level_1": "relative/path"}, "absolute prim paths/patterns"),
    ({"conveyor_level_1": "/World/Outside/*"}, "contains no imported geometry"),
])
def test_support_bounds_fail_instead_of_falling_back_to_wrong_geometry(tmp_path, selection, match):
    path = tmp_path / "invalid_support_bounds.usda"
    stage = _stage(path)
    UsdGeom.Cube.Define(stage, "/World/Conveyor")
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match=match):
        _import(path, box_prims={"conveyor_level_1": "/World/Conveyor"}, box_bounds_prims=selection)


@pytest.mark.skipif(os.name != "nt", reason="NTFS/MAX_PATH regression is Windows-specific")
def test_ascii_collected_path_uses_ntfs_alias_for_long_relative_dependencies(tmp_path, monkeypatch):
    from pathlib import Path
    from pxr import Sdf

    collected = tmp_path / "collected_scene_directory_with_a_long_name"
    collected.mkdir()
    source = collected / "scene.usda"
    source_stage = _stage(source)
    source_stage.GetRootLayer().Save()
    short_source = _usd_open_path(source)
    if len(short_source) >= len(str(source)):
        pytest.skip("NTFS 8.3 aliases are disabled on this filesystem")
    # Build actual dependency paths beyond MAX_PATH. Author the fixtures via
    # existing directory aliases so they exist without modifying originals.
    padding = max(32, 270 - len(str(collected / "shape.usda")))
    asset_dir = collected / ("payload_" + "x" * padding)
    Path("\\\\?\\" + str(asset_dir)).mkdir(parents=True)
    short_asset_dir = Path(_usd_open_path(asset_dir))
    material_dir = short_asset_dir / "materials"
    material_dir.mkdir()
    material_path = material_dir / "conveyors.material.usda"
    material_layer = Sdf.Layer.CreateNew(str(material_path))
    material = Sdf.CreatePrimInLayer(material_layer, "/World/Looks/Steel")
    material.specifier = Sdf.SpecifierDef
    material.typeName = "Material"
    material_layer.Save()
    asset_path = short_asset_dir / "shape.usda"
    asset_stage = _stage(asset_path)
    root = UsdGeom.Xform.Define(asset_stage, "/World/Shape")
    root.AddTranslateOp().Set((2, 3, 4))
    UsdGeom.Cube.Define(asset_stage, "/World/Shape/Box")
    asset_stage.GetRootLayer().subLayerPaths.append("materials/conveyors.material.usda")
    asset_stage.GetRootLayer().Save()
    payload = UsdGeom.Xform.Define(source_stage, "/World/Payload").GetPrim()
    payload.GetPayloads().AddPayload(
        (asset_dir / "shape.usda").relative_to(collected).as_posix(), "/World/Shape",
    )
    source_stage.GetRootLayer().Save()
    original_source = source.read_bytes()
    original_asset = asset_path.read_bytes()
    opened = _record_open_stages(monkeypatch)

    imported, = _import(source).static_meshes

    assert len(str(asset_dir / "materials" / "conveyors.material.usda")) > 260
    assert imported.name == "usd:/World/Payload/Box"
    assert np.asarray(imported.vertices_m).mean(axis=0) == pytest.approx((2, 3, 4))
    assert not opened[-1].GetCompositionErrors()
    assert any("conveyors.material.usda" in layer.identifier for layer in opened[-1].GetUsedLayers())
    assert source.read_bytes() == original_source
    assert asset_path.read_bytes() == original_asset
