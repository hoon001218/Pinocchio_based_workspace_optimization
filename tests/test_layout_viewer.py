from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import decanting_workspace.layout_viewer as viewer_module
from decanting_workspace.layout import apply_layout_offsets
from decanting_workspace.layout_models import LayoutFragment, LayoutFrame, LayoutNode, LayoutRobot, LayoutScene
from decanting_workspace.layout_viewer import LayoutViewer
from decanting_workspace.models import load_scene_spec
from decanting_workspace.robots import floating_configuration, load_robot_bundle
from decanting_workspace.viewer import CellViewer, LoadedRobot


class _Node:
    def __init__(self, root, path):
        self.root = root
        self.path = path
        self.objects = []
        self.transforms = []
        self.properties = []

    def delete(self):
        self.root.deleted.append(self.path)

    def set_object(self, value, *args):
        self.objects.append((value, args))

    def set_transform(self, value):
        self.transforms.append(np.asarray(value).copy())

    def set_property(self, name, value):
        self.properties.append((name, value))


class _Viewer:
    def __init__(self):
        self.nodes = {}
        self.deleted = []
        self.opened = 0

    def __getitem__(self, path):
        return self.nodes.setdefault(path, _Node(self, path))

    def url(self):
        return "http://localhost/layout"

    def open(self):
        self.opened += 1

    def static_html(self):
        return "<html>layout view</html>"


class _RobotVisualizer:
    def __init__(self):
        self.configurations = []

    def display(self, value):
        self.configurations.append(np.asarray(value).copy())


@pytest.fixture
def scene():
    identity = (0., 0., 0., 1.)
    return LayoutScene(
        Path("layout.usd"),
        nodes=(LayoutNode("/World/LoadGroup", None, (0., 0., 0.), identity, "group"),),
        boxes=(),
        robots=(LayoutRobot("ur20", "/World/LoadGroup", (1., 2., 0.5), 90.,
                            Path("configured.urdf"), (0.1, 0.2), "/World/LoadGroup/RB"),),
        frames=(LayoutFrame("load", "/World/LoadGroup", (1., 2., 1.), identity),),
    )


@pytest.fixture
def fragment():
    return LayoutFragment("part", "/World/LoadGroup", "original_box",
                          ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), ((0, 1, 2),), "conveyor")


@pytest.fixture
def recorded_robots(monkeypatch):
    calls = []

    def load(name, path, **kwargs):
        calls.append((name, Path(path), kwargs))
        return SimpleNamespace(name=name, model=SimpleNamespace(nq=9), arm_configuration_offset=7)

    def render(self, bundle, base, arm_q, **kwargs):
        visualizer = _RobotVisualizer()
        visualizer.display(floating_configuration(bundle, base, arm_q))
        self._robots.append(LoadedRobot(bundle, visualizer))

    monkeypatch.setattr(viewer_module, "load_robot_bundle", load)
    monkeypatch.setattr(CellViewer, "_render_robot", render)
    return calls


def test_only_fragments_and_frames_are_rendered_without_generated_process_geometry(scene, fragment, recorded_robots):
    native = _Viewer()
    viewer = LayoutViewer(native, ur20_source="configured")
    viewer.render(scene, (fragment,))

    environment = [path for path, node in native.nodes.items()
                   if path.startswith("decanting/environment/") and node.objects]
    assert environment == ["decanting/environment/conveyor/part"]
    assert not any("floor" in path or "guides" in path or "SKU" in path or "playback" in path
                   for path in native.nodes)
    assert viewer.fragment_metadata == {"part": {"owner_path": "/World/LoadGroup", "source_name": "original_box"}}
    assert all(f"decanting/frames/load/{axis}" in native.nodes for axis in "xyz")
    mesh = native.nodes[environment[0]].objects[0][0]
    np.testing.assert_allclose(mesh.vertices, np.array(fragment.vertices_m))
    np.testing.assert_array_equal(mesh.faces, np.array(fragment.triangles))
    np.testing.assert_array_equal(native.nodes[environment[0]].transforms[-1], np.eye(4))


def test_position_updates_reuse_robot_bundle_and_change_only_free_flyer_xyz(scene, fragment, recorded_robots):
    native = _Viewer()
    viewer = LayoutViewer(native, ur20_source="configured", show_frames=False)
    viewer.render(scene, (fragment,))
    loaded = viewer.loaded_robot("ur20")
    original_q = loaded.visualizer.configurations[0].copy()
    moved = apply_layout_offsets(scene, {"/World/LoadGroup": (0.5, -0.2, 0.3)})
    moved_fragment = replace(fragment, vertices_m=tuple(tuple(np.array(vertex) + (0.5, -0.2, 0.3))
                                                      for vertex in fragment.vertices_m))
    viewer.render(moved, (moved_fragment,))

    assert len(recorded_robots) == 1
    assert viewer.loaded_robot("ur20") is loaded
    displayed_q = loaded.visualizer.configurations[-1]
    np.testing.assert_allclose(displayed_q[:3], (1.5, 1.8, 0.8))
    np.testing.assert_array_equal(displayed_q[3:], original_q[3:])
    assert native.deleted.count("decanting/robots/ur20") == 1
    assert native.deleted.count("decanting/environment") == 2
    assert not any(path.startswith("decanting/frames/") for path in native.nodes)
    info = viewer.robot_info()[0]
    assert info["translation_m"] == [1.5, 1.8, 0.8]
    assert info["yaw_deg"] == 90.
    assert info["source"] == "configured"
    info["translation_m"][0] = 999
    assert viewer.robot_info()[0]["translation_m"][0] == 1.5


def test_official_ur20_resolves_once_and_model_reload_happens_only_for_changed_urdf(scene, fragment, recorded_robots, monkeypatch):
    official = Path("official.urdf")
    resolutions = []
    monkeypatch.setattr(viewer_module, "resolve_official_ur20",
                        lambda: (resolutions.append(True) or official, (Path("meshes"),)))
    viewer = LayoutViewer(_Viewer())
    viewer.render(scene, (fragment,))
    viewer.render(scene, (fragment,))
    assert len(resolutions) == 1
    assert len(recorded_robots) == 1
    assert recorded_robots[0][1] == official
    assert recorded_robots[0][2]["package_dirs"] == (Path("meshes"),)
    assert viewer.robot_info()[0]["source"] == "official"

    configured = LayoutViewer(_Viewer(), ur20_source="configured")
    configured.render(scene, ())
    changed = replace(scene, robots=(replace(scene.robots[0], urdf_path=Path("replacement.urdf")),))
    configured.render(changed, ())
    assert recorded_robots[-1][1] == Path("replacement.urdf")
    assert len(configured._robots) == 1


def test_sr_uses_validated_prepared_asset_without_download(scene, recorded_robots, monkeypatch, tmp_path):
    prepared = tmp_path / "sr_mesh.urdf"
    prepared.write_text("prepared", encoding="utf-8")
    validations = []
    monkeypatch.setattr(viewer_module, "default_sr12ia_output_urdf", lambda: prepared)
    monkeypatch.setattr(viewer_module, "validate_prepared_sr12ia_asset", lambda path: validations.append(path))
    robot = replace(scene.robots[0], name="sr12ia")
    viewer = LayoutViewer(_Viewer(), ur20_source="configured")
    viewer.render(replace(scene, robots=(robot,)), ())
    viewer.render(replace(scene, robots=(robot,)), ())
    assert validations == [prepared]
    assert len(recorded_robots) == 1
    assert recorded_robots[0][1] == prepared
    assert viewer.robot_info()[0]["source"] == "prepared"


@pytest.mark.parametrize("invalid", [False, True])
def test_sr_uses_configured_urdf_when_prepared_asset_missing_or_invalid(scene, recorded_robots, monkeypatch, tmp_path, invalid):
    prepared = tmp_path / "sr_mesh.urdf"
    if invalid:
        prepared.write_text("stale", encoding="utf-8")
    monkeypatch.setattr(viewer_module, "default_sr12ia_output_urdf", lambda: prepared)

    def invalid_asset(path):
        raise ValueError("stale prepared model")

    monkeypatch.setattr(viewer_module, "validate_prepared_sr12ia_asset", invalid_asset)
    robot = replace(scene.robots[0], name="sr12ia")
    viewer = LayoutViewer(_Viewer(), ur20_source="configured")
    viewer.render(replace(scene, robots=(robot,)), ())
    assert recorded_robots[0][1] == robot.urdf_path
    assert viewer.robot_info()[0]["source"] == "configured"
    if invalid:
        assert viewer.robot_info()[0]["visual_fallback_reason"] == "stale prepared model"


def test_explicit_sr_override_is_not_treated_as_a_prepared_asset(scene, recorded_robots, monkeypatch):
    monkeypatch.setattr(viewer_module, "default_sr12ia_output_urdf", lambda: pytest.fail("no auto asset lookup"))
    robot = replace(scene.robots[0], name="sr12ia")
    viewer = LayoutViewer(_Viewer(), sr12ia_visual_urdf="explicit.urdf")
    viewer.render(replace(scene, robots=(robot,)), ())
    assert recorded_robots[0][1] == Path("explicit.urdf")
    assert viewer.robot_info()[0]["source"] == "explicit"


def test_removed_robots_are_deleted_and_static_export_url_open_are_available(scene, recorded_robots, tmp_path):
    native = _Viewer()
    viewer = LayoutViewer(native, ur20_source="configured")
    viewer.render(scene, ())
    viewer.render(replace(scene, robots=()), ())
    assert viewer.robot_info() == []
    assert len(viewer._robots) == 0
    with pytest.raises(KeyError):
        viewer.loaded_robot("ur20")
    assert viewer.url() == "http://localhost/layout"
    viewer.open()
    assert native.opened == 1
    output = viewer.save_static_html(tmp_path / "layout.html")
    assert output.read_text(encoding="utf-8") == "<html>layout view</html>"


def test_duplicate_fragment_names_are_rejected_before_replacing_view(scene, fragment, recorded_robots):
    native = _Viewer()
    viewer = LayoutViewer(native, ur20_source="configured")
    with pytest.raises(ValueError, match="fragment names must be unique"):
        viewer.render(scene, (fragment, fragment))
    assert native.deleted == []


def test_real_pinocchio_configuration_tracks_imported_position_without_model_rebuild(monkeypatch):
    spec = load_scene_spec()
    configured = spec.robots["ur20"]
    bundle = load_robot_bundle("ur20", configured.urdf_path, load_visual=True)
    model_calls = []
    monkeypatch.setattr(viewer_module, "load_robot_bundle", lambda *args, **kwargs: model_calls.append(True) or bundle)

    def render(self, bundle, base, q, **kwargs):
        visualizer = _RobotVisualizer()
        visualizer.display(floating_configuration(bundle, base, q))
        self._robots.append(LoadedRobot(bundle, visualizer))

    monkeypatch.setattr(CellViewer, "_render_robot", render)
    node = LayoutNode("/World/LoadGroup", None, (0., 0., 0.), (0., 0., 0., 1.))
    robot = LayoutRobot("ur20", node.prim_path, (3., -4., 0.75), configured.nominal_base.yaw_deg,
                        configured.urdf_path, configured.nominal_q)
    baseline = LayoutScene(Path("example.usd"), (node,), (), (robot,))
    viewer = LayoutViewer(_Viewer(), ur20_source="configured")
    viewer.render(baseline, ())
    viewer.render(apply_layout_offsets(baseline, {node.prim_path: (1., 2., 0.25)}), ())
    import pinocchio as pin

    displayed = viewer.loaded_robot("ur20").visualizer.configurations[-1]
    data = bundle.model.createData()
    pin.forwardKinematics(bundle.model, data, displayed)
    np.testing.assert_allclose(data.oMi[1].translation, (4., -2., 1.))
    np.testing.assert_allclose(displayed[7:], configured.nominal_q)
    assert model_calls == [True]


def test_initial_camera_frames_work_cell_without_fitting_room_walls_or_ground_plane(scene, recorded_robots):
    native = _Viewer()
    viewer = LayoutViewer(native, ur20_source="configured")
    floor = LayoutFragment("floor", "/World/GroundPlane", "floor",
                           ((-25., -25., 0.), (25., -25., 0.), (25., 25., 0.), (-25., 25., 0.)),
                           ((0, 1, 2), (0, 2, 3)), "ground")
    wall = LayoutFragment("wall", "/World/GroundPlane", "wall",
                          ((4., -10., 0.), (4., -5., 0.), (4., -5., 4.), (4., -10., 4.)),
                          ((0, 1, 2), (0, 2, 3)), "ground")
    equipment = LayoutFragment("equipment", "/World/LoadGroup", "equipment",
                               ((-2., -8., 0.), (-1., -8., 0.), (-1., -7., 1.)), ((0, 1, 2),))
    scene = replace(scene, robots=(), frames=())
    viewer.render(scene, (floor, wall, equipment))

    np.testing.assert_array_equal(native.nodes["/Cameras/default"].transforms[0], np.eye(4))
    np.testing.assert_array_equal(native.nodes["/Cameras/default/rotated"].transforms[0], np.eye(4))
    camera = native.nodes["/Cameras/default/rotated/<object>"]
    properties = dict(camera.properties)
    target = np.asarray(properties["layout_target"])
    np.testing.assert_allclose(target, (-1.5, -7.5, .5))
    offset = np.asarray(properties["position"]) - target
    assert offset[0] < 0. and offset[1] > 0. and offset[2] > 0.
    radius = np.linalg.norm((1., 1., 1.)) / 2.
    assert np.linalg.norm(offset) > radius / np.sin(np.radians(properties["fov"] / 2.))
    assert "decanting/environment/ground/floor" in native.nodes
    assert "decanting/environment/ground/wall" in native.nodes


def test_layout_updates_preserve_user_camera_pose(scene, fragment, recorded_robots):
    native = _Viewer()
    viewer = LayoutViewer(native, ur20_source="configured")
    viewer.render(scene, (fragment,))
    camera_target = native.nodes["/Cameras/default"]
    camera = native.nodes["/Cameras/default/rotated/<object>"]
    original_transform_count = len(camera_target.transforms)
    original_property_count = len(camera.properties)
    viewer.render(apply_layout_offsets(scene, {"/World/LoadGroup": (2., -4., 1.)}), (fragment,))
    assert len(camera_target.transforms) == original_transform_count == 1
    assert len(camera.properties) == original_property_count
