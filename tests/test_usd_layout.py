from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('pxr')
from pxr import Usd, UsdGeom

from decanting_workspace.models import load_scene_spec
from decanting_workspace.transforms import quaternion_matrix
from decanting_workspace.usd_layout import DEFAULT_GROUPS, load_usd_layout


def _stage(path, *, units=1., up='Z', groups=DEFAULT_GROUPS):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, units)
    UsdGeom.SetStageUpAxis(stage, up)
    for path in groups:
        UsdGeom.Xform.Define(stage, path)
    return stage


def _cube(stage, path, center=(0., 0., 0.), size=(1., 1., 1.)):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.)
    cube.AddTranslateOp().Set(center)
    cube.AddScaleOp().Set(size)
    return cube


def _mesh_box(stage, path, center=(0., 0., 0.), size=(1., 1., 1.), rotation=None):
    points = np.array([(x, y, z) for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    points *= size
    if rotation is not None:
        points = points @ rotation.T
    points += center
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points.tolist())
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([0,1,3,2,4,6,7,5,0,4,5,1,2,3,7,6,0,2,6,4,1,5,7,3])
    return mesh, points


def _robot_roots(stage):
    ur = UsdGeom.Xform.Define(stage, '/World/LoadGroup/RB/ur20')
    ur.AddTranslateOp().Set((2., 3., .8))
    _cube(stage, '/World/LoadGroup/RB/ur20/base_link/visual', size=(4., 4., 4.))
    UsdGeom.Xform.Define(stage, '/World/LoadGroup/RB/ur20/base_link')
    sr = UsdGeom.Xform.Define(stage, '/World/UncaseGroup/RB/sr12ia')
    sr.AddTranslateOp().Set((-2., 4., .6))
    base = UsdGeom.Xform.Define(stage, '/World/UncaseGroup/RB/sr12ia/robot_base/base_link')
    base.AddTranslateOp().Set((.25, 0., .05))
    _cube(stage, '/World/UncaseGroup/RB/sr12ia/robot_base/base_link/visual', size=(4., 4., 4.))


def test_layout_uses_surfaces_and_robot_base_link_positions_without_sample_skus(tmp_path):
    path = tmp_path / 'layout.usda'
    stage = _stage(path)
    _robot_roots(stage)
    _mesh_box(stage, '/World/ConveyorGroup/ConveyorBelt_A05_Unit/Geometry/Roller1',
              center=(0., 0., 1.), size=(.7, .1, .06))
    _mesh_box(stage, '/World/ConveyorGroup/ConveyorBelt_A05_Unit/Geometry/Roller2',
              center=(0., 1., 1.), size=(.7, .1, .06))
    _mesh_box(stage, '/World/ConveyorGroup/ConveyorBelt_A05_Unit/Geometry/BodyAndLegs',
              center=(0., 0., .5), size=(2., 2., 1.))
    _cube(stage, '/World/ConveyorGroup/Conv1', size=(20., 20., 20.))
    _cube(stage, '/World/PalletGroup/pallet', size=(1.2, 1., .15))
    _cube(stage, '/World/SwedishStyleJelly', size=(20., 20., 20.))
    _cube(stage, '/World/PalletGroup/SKU_example', size=(30., 30., 30.))
    sample = UsdGeom.Xform.Define(stage, '/World/ToteGroup/FlatBox_missing').GetPrim()
    sample.GetPayloads().AddPayload('absent.usda')
    frame = UsdGeom.Xform.Define(stage, '/World/ToteGroup/ToteLoadFrame')
    frame.AddTranslateOp().Set((1., 2., 3.))
    stage.GetRootLayer().Save()
    original = load_scene_spec()
    scene = load_usd_layout(path, original.robots)
    assert len(scene.boxes) == 2
    surface = next(box for box in scene.boxes if box.role == 'conveyor')
    assert surface.size_m == pytest.approx((.7, 1.1, .06))
    assert surface.center_m == pytest.approx((0., .5, 1.))
    assert surface.node_path == '/World/ConveyorGroup/ConveyorBelt_A05_Unit'
    ur, sr = scene.robots
    assert ur.translation_m == pytest.approx((2., 3., .8))
    assert sr.translation_m == pytest.approx((-1.75, 4., .65))
    assert ur.yaw_deg == original.robots['ur20'].nominal_base.yaw_deg
    assert sr.yaw_deg == original.robots['sr12ia'].nominal_base.yaw_deg
    assert ur.urdf_path == original.robots['ur20'].urdf_path
    assert scene.frames[0].translation_m == pytest.approx((1., 2., 3.))
    assert not any('SKU' in node.prim_path or 'FlatBox' in node.prim_path for node in scene.nodes)


def test_nested_scaled_y_up_geometry_is_converted_to_world_metres(tmp_path):
    path = tmp_path / 'scaled.usda'
    group = '/World/ToteGroup'
    stage = _stage(path, units=.01, up='Y', groups=[group])
    root = stage.GetPrimAtPath(group)
    UsdGeom.Xformable(root).AddTranslateOp().Set((100., 200., 300.))
    _cube(stage, group+'/Table', center=(20., 0., 0.), size=(200., 10., 100.))
    stage.GetRootLayer().Save()
    scene = load_usd_layout(path, {}, {'groups': [group]})
    box, = scene.boxes
    assert box.center_m == pytest.approx((1.2, -3., 2.))
    rotation = quaternion_matrix((0.,0.,0.), box.rotation_xyzw)[:3,:3]
    points = np.array([(x,y,z) for x in (-1.,1.) for y in (-.05,.05) for z in (-.5,.5)])
    world = points @ rotation.T + box.center_m
    assert world.min(axis=0) == pytest.approx((.2, -3.5, 1.95))
    assert world.max(axis=0) == pytest.approx((2.2, -2.5, 2.05))
    node = next(node for node in scene.nodes if node.prim_path == group+'/Table')
    assert node.parent_path == group


def test_intrinsically_inclined_belt_keeps_thin_oriented_bounds(tmp_path):
    path = tmp_path / 'ramp.usda'
    group = '/World/ConveyorGroup'
    stage = _stage(path, groups=[group])
    angle = np.deg2rad(30.)
    rotation = np.array(((np.cos(angle),0.,np.sin(angle)),(0.,1.,0.),(-np.sin(angle),0.,np.cos(angle))))
    _, points = _mesh_box(stage, group+'/ConveyorBelt_A25/BeltRamp/SM_A25_Belt',
                         center=(0.,0.,1.), size=(2.,1.,.03), rotation=rotation)
    stage.GetRootLayer().Save()
    box, = load_usd_layout(path, {}, {'groups': [group]}).boxes
    assert sorted(box.size_m) == pytest.approx([.03,1.,2.], abs=1e-6)
    matrix = quaternion_matrix(box.center_m, box.rotation_xyzw)
    local = (points - matrix[:3,3]) @ matrix[:3,:3]
    assert np.all(np.abs(local) <= np.asarray(box.size_m)/2. + 1e-6)
    assert abs(matrix[2,0]) == pytest.approx(.5, abs=1e-6)


def test_camera_helpers_and_duplicate_collision_shapes_are_ignored(tmp_path):
    path = tmp_path / 'helpers.usda'
    group = '/World/ToteGroup'
    stage = _stage(path, groups=[group])
    _cube(stage, group+'/Camera/rsd455/Visual/Case', size=(.12,.03,.03))
    _cube(stage, group+'/Camera/rsd455/Visual/Glass', center=(.04,0.,0.), size=(.04,.02,.02))
    _cube(stage, group+'/Camera/rsd455/OmniverseKitViewportCameraMesh/CameraMode', size=(2.,2.,2.))
    _cube(stage, group+'/small_KLT/Visuals/Box', size=(.6,.4,.3))
    _cube(stage, group+'/small_KLT/Collision/Cube', size=(60.,40.,30.))
    stage.GetRootLayer().Save()
    scene = load_usd_layout(path, {}, {'groups': [group]})
    assert len(scene.boxes) == 1
    assert scene.boxes[0].role == 'camera'
    assert scene.boxes[0].size_m == pytest.approx((.12,.03,.03))
    reference, = scene.reference_dimensions
    assert reference.size_m == pytest.approx((.6,.4,.3))
    assert reference.source_prim_path == group+'/small_KLT'
    assert not any('/small_KLT' in node.prim_path for node in scene.nodes)


def test_missing_required_dependency_still_fails(tmp_path):
    path = tmp_path / 'broken.usda'
    group = '/World/ToteGroup'
    stage = _stage(path, groups=[group])
    required = UsdGeom.Xform.Define(stage, group+'/Table').GetPrim()
    required.GetPayloads().AddPayload('absent.usda')
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match='unresolved composition dependency'):
        load_usd_layout(path, {}, {'groups': [group]})


def test_native_instances_and_changed_mesh_geometry_are_reloaded(tmp_path):
    asset_path = tmp_path / 'asset.usda'
    asset = _stage(asset_path, groups=['/Unit'])
    _mesh_box(asset, '/Unit/Shape', size=(2.,1.,.4))
    asset.GetRootLayer().Save()
    path = tmp_path / 'references.usda'
    group = '/World/ToteGroup'
    stage = _stage(path, groups=[group])
    for index in range(2):
        root = UsdGeom.Xform.Define(stage, group+f'/Instance{index}')
        root.GetPrim().GetReferences().AddReference('asset.usda', '/Unit')
        root.GetPrim().SetInstanceable(True)
        root.AddTranslateOp(opSuffix='instance').Set((index*3.,0.,0.))
    stage.GetRootLayer().Save()
    original = load_usd_layout(path, {}, {'groups': [group]})
    assert len(original.boxes) == 2
    assert [box.center_m[0] for box in original.boxes] == pytest.approx([0.,3.])
    _mesh_box(stage, group+'/AddedTable', size=(1.,2.,.3))
    stage.GetRootLayer().Save()
    updated = load_usd_layout(path, {}, {'groups': [group]})
    assert len(updated.boxes) == 3
    assert any(node.prim_path == group+'/AddedTable' for node in updated.nodes)


def test_floor_proxy_extends_below_surface_and_overlap_is_allowed(tmp_path):
    path = tmp_path / 'overlap.usda'
    group = '/World/GroundPlane'
    stage = _stage(path, groups=[group])
    mesh, _ = _mesh_box(stage, group+'/Floor', size=(4.,4.,0.))
    mesh.AddRotateXOp().Set(180.)
    _cube(stage, group+'/WallA', center=(0.,0.,1.), size=(.1,4.,2.))
    _cube(stage, group+'/WallB', center=(0.,0.,1.), size=(.1,4.,2.))
    stage.GetRootLayer().Save()
    scene = load_usd_layout(path, {}, {'groups': [group]})
    floor = next(box for box in scene.boxes if box.name.endswith('/Floor'))
    assert floor.center_m[2] == pytest.approx(-.01)
    assert len(scene.boxes) == 3


def test_user_sku_role_attribute_excludes_named_product(tmp_path):
    path = tmp_path / 'roles.usda'
    group = '/World/PalletGroup'
    stage = _stage(path, groups=[group])
    _cube(stage, group+'/pallet', size=(1.,1.,.15))
    sample = _cube(stage, group+'/CustomProduct', size=(10.,10.,10.))
    _cube(stage, group+'/SKU_added', size=(12.,12.,12.))
    _cube(stage, group+'/IgnoredFixtureTable', size=(14.,14.,14.))
    from pxr import Sdf
    sample.GetPrim().CreateAttribute('userProperties:layoutRole', Sdf.ValueTypeNames.String).Set('sku')
    stage.GetRootLayer().Save()
    scene = load_usd_layout(path, {}, {'groups': [group], 'ignore_name_patterns': ['IgnoredFixture*']})
    assert len(scene.boxes) == 1
    assert not any('CustomProduct' in node.prim_path for node in scene.nodes)


def test_supply_totes_are_dimensions_only_without_layout_geometry(tmp_path):
    path = tmp_path / 'supply.usda'
    group = '/World/ConveyorGroup'
    stage = _stage(path, groups=[group])
    for index in range(2):
        prefix = group+f'/conv_middle/tote_supply/small_KLT_{index:02}'
        _cube(stage, prefix+'/Visuals/Box', center=(0., index, 1.), size=(.6,.4,.3))
        _cube(stage, prefix+'/Collision/Cube', size=(60.,40.,30.))
    stage.GetRootLayer().Save()
    scene = load_usd_layout(path, {}, {'groups': [group]})
    assert scene.boxes == ()
    assert len(scene.reference_dimensions) == 2
    assert all(reference.role == 'tote' and reference.size_m == pytest.approx((.6,.4,.3))
               for reference in scene.reference_dimensions)
    assert [reference.source_prim_path for reference in scene.reference_dimensions] == [
        group+f'/conv_middle/tote_supply/small_KLT_{index:02}' for index in range(2)
    ]
    assert not any('/tote_supply' in node.prim_path for node in scene.nodes)


def test_duplicate_frame_names_have_stable_path_ids_and_correct_attachments(tmp_path):
    path = tmp_path / 'frames.usda'
    groups = ['/World/ToteGroup', '/World/UncaseGroup']
    stage = _stage(path, groups=groups)
    for index, group in enumerate(groups):
        UsdGeom.Xformable(stage.GetPrimAtPath(group)).AddTranslateOp().Set((index*3.,0.,0.))
        _cube(stage, group+'/Table')
        frame = UsdGeom.Xform.Define(stage, group+'/WorkFrame')
        frame.AddTranslateOp().Set((1.,2.,3.))
    stage.GetRootLayer().Save()
    scene = load_usd_layout(path, {}, {'groups': groups})
    frames = {frame.name: frame for frame in scene.frames}
    assert set(frames) == {group+'/WorkFrame' for group in groups}
    for index, group in enumerate(groups):
        frame = frames[group+'/WorkFrame']
        assert frame.node_path == frame.source_prim_path == group+'/WorkFrame'
        assert frame.translation_m == pytest.approx((index*3.+1.,2.,3.))


def test_real_collected_layout_has_no_conveyor_legs_or_robot_usd_meshes():
    path = Path('USD/Collected_scene_layout/scene_layout.usd')
    if not path.is_file():
        pytest.skip('collected layout fixture is absent')
    nominal = load_scene_spec()
    scene = load_usd_layout(path, nominal.robots)
    assert len(scene.boxes) == 39
    assert len(scene.frames) == 5
    rollers = [box for box in scene.boxes if box.name.endswith('/roller_surface')]
    assert len(rollers) == 12
    assert all(box.size_m[2] < .06 for box in rollers)
    assert len([box for box in scene.boxes if box.role == 'conveyor']) == 20
    assert not any(box.role == 'tote' for box in scene.boxes)
    assert len(scene.reference_dimensions) == 5
    assert all(reference.role == 'tote' for reference in scene.reference_dimensions)
    assert not any('/ur20/' in box.name or '/sr12ia/' in box.name for box in scene.boxes)
    assert all(not any(sample in box.name for sample in ('FlatBox', 'SwedishStyleJelly', 'SKU'))
               for box in scene.boxes)
    sr = next(robot for robot in scene.robots if robot.name == 'sr12ia')
    assert sr.translation_m == pytest.approx((-1.658127802,-8.682972058,.626673053), abs=1e-8)
    assert sr.yaw_deg == nominal.robots['sr12ia'].nominal_base.yaw_deg
    assert sr.translation_m != nominal.robots['sr12ia'].nominal_base.xyz_m
    assert all(max(box.size_m) < 3.2 for box in scene.boxes if box.role != 'ground')


def test_named_tote_role_keeps_dimensions_and_skips_its_frames(tmp_path):
    from pxr import Sdf
    path = tmp_path / 'tote_role.usda'
    group = '/World/ToteGroup'
    stage = _stage(path, units=.01, groups=[group])
    _cube(stage, group+'/Table', size=(100.,100.,10.))
    tote = UsdGeom.Xform.Define(stage, group+'/CustomContainer')
    tote.AddRotateZOp().Set(35.)
    tote.AddTranslateOp().Set((500.,300.,100.))
    tote.GetPrim().CreateAttribute('userProperties:layoutRole', Sdf.ValueTypeNames.String).Set('tote')
    _cube(stage, group+'/CustomContainer/Visuals/Body', size=(60.,40.,30.))
    _cube(stage, group+'/CustomContainer/Collision/Cube', size=(6000.,4000.,3000.))
    UsdGeom.Xform.Define(stage, group+'/CustomContainer/GraspFrame')
    UsdGeom.Xform.Define(stage, group+'/ToteLoadFrame')
    stage.GetRootLayer().Save()
    scene = load_usd_layout(path, {}, {'groups': [group]})
    assert len(scene.boxes) == 1
    reference, = scene.reference_dimensions
    assert reference.size_m == pytest.approx((.6,.4,.3), abs=1e-7)
    assert reference.source_prim_path == group+'/CustomContainer'
    assert not any('/CustomContainer' in node.prim_path for node in scene.nodes)
    assert [frame.name for frame in scene.frames] == ['ToteLoadFrame']


def test_layout_import_does_not_emit_empty_prim_path_diagnostics(tmp_path, capfd):
    path = tmp_path / 'valid_paths.usda'
    group = '/World/ToteGroup'
    stage = _stage(path, groups=[group])
    _cube(stage, group+'/Table', size=(1.,1.,.1))
    _cube(stage, group+'/small_KLT/Visuals/Body', size=(.6,.4,.3))
    stage.GetRootLayer().Save()
    scene = load_usd_layout(path, {}, {'groups': [group]})
    assert len(scene.boxes) == len(scene.reference_dimensions) == 1
    assert 'Ill-formed SdfPath' not in capfd.readouterr().err
