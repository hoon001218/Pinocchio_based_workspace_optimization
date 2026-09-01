from __future__ import annotations

import math

import numpy as np
import pytest

import decanting_workspace.meshcat_playback as playback_module
from decanting_workspace import load_scene_spec
from decanting_workspace.kinematics import arm_frame_jacobian, frame_pose
from decanting_workspace.meshcat_playback import UR20MeshcatPlayback
from decanting_workspace.robots import floating_configuration, load_robot_bundle
from decanting_workspace.viewer import CellViewer


class _RecordingNode:
    def __init__(self) -> None:
        self.object_calls: list[tuple[object, tuple[object, ...]]] = []
        self.transforms: list[np.ndarray] = []
        self.properties: list[tuple[str, object]] = []

    def set_object(self, value: object, *args: object) -> None:
        self.object_calls.append((value, args))

    def set_transform(self, value: np.ndarray) -> None:
        self.transforms.append(np.asarray(value, dtype=float).copy())

    def set_property(self, name: str, value: object) -> None:
        self.properties.append((name, value))


class _RecordingViewer:
    def __init__(self) -> None:
        self.nodes: dict[str, _RecordingNode] = {}

    def __getitem__(self, path: str) -> _RecordingNode:
        return self.nodes.setdefault(path, _RecordingNode())


class _RecordingRobotVisualizer:
    def __init__(self) -> None:
        self.configurations: list[np.ndarray] = []

    def display(self, q: np.ndarray) -> None:
        self.configurations.append(np.asarray(q, dtype=float).copy())


@pytest.fixture(scope="module")
def ur20_playback_assets():
    spec = load_scene_spec()
    robot_spec = spec.robots["ur20"]
    bundle = load_robot_bundle(
        "ur20",
        robot_spec.urdf_path,
        suction_proxy=robot_spec.suction_proxy,
    )
    q = floating_configuration(
        bundle,
        robot_spec.nominal_base,
        robot_spec.nominal_q,
    )
    return bundle, q


def _playback(assets, *, scale: float = 0.15):
    bundle, _ = assets
    viewer = _RecordingViewer()
    robot = _RecordingRobotVisualizer()
    playback = UR20MeshcatPlayback(
        viewer,
        robot,
        bundle,
        ellipsoid_scale=scale,
    )
    return playback, viewer, robot


def test_step_displays_q_and_world_lwa_translational_ellipsoid(
    ur20_playback_assets,
):
    bundle, q = ur20_playback_assets
    scale = 0.12
    playback, viewer, robot = _playback(ur20_playback_assets, scale=scale)

    result = playback.show_step(q, step_name="pallet_box_pick")

    path = "decanting/playback/ur20/manipulability"
    node = viewer.nodes[path]
    transform = node.transforms[-1]
    jacobian = arm_frame_jacobian(
        bundle,
        q,
        "suction_tcp",
        reference_frame="LOCAL_WORLD_ALIGNED",
    )[:3, :]
    expected_singular_values = np.linalg.svd(jacobian, compute_uv=False)
    expected_tcp = frame_pose(bundle, q, "suction_tcp")[:3, 3]

    assert len(robot.configurations) == 1
    np.testing.assert_array_equal(robot.configurations[0], q)
    assert result.step_name == "pallet_box_pick"
    assert result.robot_displayed
    assert result.ellipsoid_visible
    assert result.hidden_reason is None
    np.testing.assert_allclose(result.singular_values, expected_singular_values)
    np.testing.assert_allclose(result.tcp_position_m, expected_tcp)
    np.testing.assert_allclose(transform[:3, 3], expected_tcp)
    # A = U diag(scale*sigma), hence A A^T = scale^2 Jv Jv^T.  This checks
    # both the world-aligned U axes and the singular-value axis lengths without
    # depending on arbitrary SVD column signs.
    np.testing.assert_allclose(
        transform[:3, :3] @ transform[:3, :3].T,
        scale**2 * jacobian @ jacobian.T,
        atol=1e-12,
    )
    assert np.linalg.det(transform[:3, :3]) > 0.0
    assert len(node.object_calls) == 1
    assert node.object_calls[-1][1][0].wireframe is False
    assert node.properties[-1] == ("visible", True)


def test_multiple_steps_update_robot_and_reuse_sphere_geometry(
    ur20_playback_assets,
):
    bundle, q = ur20_playback_assets
    playback, viewer, robot = _playback(ur20_playback_assets)
    second = np.array(q, copy=True)
    second[bundle.arm_configuration_offset] += math.radians(5.0)

    first_result = playback.show_step(q, step_name="step_1")
    second_result = playback.show_step(second, step_name="step_2")

    node = viewer.nodes[playback.ellipsoid_path]
    assert first_result.ellipsoid_visible
    assert second_result.ellipsoid_visible
    assert [item.step_name for item in (first_result, second_result)] == [
        "step_1",
        "step_2",
    ]
    assert len(robot.configurations) == 2
    np.testing.assert_array_equal(robot.configurations[-1], second)
    assert len(node.object_calls) == 1
    assert len(node.transforms) == 2
    assert not np.allclose(node.transforms[0], node.transforms[1])


def test_cached_step_switches_wireframe_material_and_reuses_matching_style(
    ur20_playback_assets,
):
    _, q = ur20_playback_assets
    playback, viewer, _ = _playback(ur20_playback_assets)
    render = {
        "tcp_position_m": (0.1, 0.2, 0.3),
        "axis_lengths": (3.0, 2.0, 1.0),
        "axis_directions_world": np.eye(3),
    }

    playback.show_cached_step(q, **render, ellipsoid_wireframe=False)
    node = viewer.nodes[playback.ellipsoid_path]
    assert len(node.object_calls) == 1
    assert node.object_calls[-1][1][0].wireframe is False

    playback.show_cached_step(q, **render, ellipsoid_wireframe=False)
    assert len(node.object_calls) == 1

    playback.show_cached_step(q, **render, ellipsoid_wireframe=True)
    assert len(node.object_calls) == 2
    assert node.object_calls[-1][1][0].wireframe is True

    playback.show_cached_step(q, **render, ellipsoid_wireframe=True)
    assert len(node.object_calls) == 2

    playback.show_cached_step(q, **render, ellipsoid_wireframe=False)
    assert len(node.object_calls) == 3
    assert node.object_calls[-1][1][0].wireframe is False


def test_display_callback_receives_every_displayed_configuration(
    ur20_playback_assets,
):
    bundle, q = ur20_playback_assets
    viewer = _RecordingViewer()
    robot = _RecordingRobotVisualizer()
    displayed = []
    playback = UR20MeshcatPlayback(
        viewer,
        robot,
        bundle,
        on_configuration_displayed=lambda value: displayed.append(value.copy()),
    )
    second = np.array(q, copy=True)
    second[bundle.arm_configuration_offset] += math.radians(5.0)

    playback.show_step(q)
    playback.show_cached_step(
        second,
        tcp_position_m=None,
        axis_lengths=None,
        axis_directions_world=None,
        visible=False,
    )
    playback.show_step(None, successful=False)

    assert len(robot.configurations) == 2
    assert len(displayed) == 2
    np.testing.assert_array_equal(displayed[0], q)
    np.testing.assert_array_equal(displayed[1], second)


def test_provisional_suction_proxy_follows_tool0_configuration(
    ur20_playback_assets,
):
    bundle, q = ur20_playback_assets
    spec = load_scene_spec()
    proxy = spec.robots["ur20"].suction_proxy
    assert proxy is not None
    recorder = _RecordingViewer()
    viewer = object.__new__(CellViewer)
    viewer.viewer = recorder
    viewer._robots = []
    viewer._ur20_suction_attachment = None

    viewer._render_suction_proxy(
        bundle,
        q,
        proxy,
        root_name="decanting/robots/ur20/provisional_suction",
    )
    second = np.array(q, copy=True)
    second[bundle.arm_configuration_offset] += math.radians(7.0)
    assert viewer.update_ur20_suction_proxy(second)

    stem_node = recorder.nodes[
        "decanting/robots/ur20/provisional_suction/cylinder"
    ]
    pad_node = recorder.nodes[
        "decanting/robots/ur20/provisional_suction/pad"
    ]
    assert len(stem_node.object_calls) == 1
    assert len(pad_node.object_calls) == 1
    assert len(stem_node.transforms) == 2
    assert len(pad_node.transforms) == 2
    assert not np.allclose(stem_node.transforms[0], stem_node.transforms[1])

    tool_pose = frame_pose(bundle, second, "tool0")
    expected_stem = tool_pose[:3, 3] + tool_pose[:3, :3] @ np.array(
        (0.0, 0.0, proxy.cylinder_length_m / 2.0)
    )
    expected_pad = tool_pose[:3, 3] + tool_pose[:3, :3] @ np.array(
        (
            0.0,
            0.0,
            proxy.cylinder_length_m + proxy.pad_thickness_m / 2.0,
        )
    )
    np.testing.assert_allclose(stem_node.transforms[-1][:3, 3], expected_stem)
    np.testing.assert_allclose(pad_node.transforms[-1][:3, 3], expected_pad)


def test_cached_step_uses_cached_axes_without_recomputing_pinocchio(
    ur20_playback_assets,
    monkeypatch,
):
    bundle, q = ur20_playback_assets
    playback, viewer, robot = _playback(ur20_playback_assets, scale=0.15)
    jacobian = arm_frame_jacobian(bundle, q, "suction_tcp")[:3, :]
    directions, singular_values, _ = np.linalg.svd(jacobian)
    tcp = frame_pose(bundle, q, "suction_tcp")[:3, 3]
    monkeypatch.setattr(
        playback_module,
        "arm_frame_jacobian",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("recomputed")),
    )
    monkeypatch.setattr(
        playback_module,
        "frame_pose",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("recomputed")),
    )

    result = playback.show_cached_step(
        q,
        tcp_position_m=tcp,
        axis_lengths=singular_values,
        axis_directions_world=directions.T,
        step_name="cached",
        ellipsoid_scale=0.2,
    )

    transform = viewer.nodes[playback.ellipsoid_path].transforms[-1]
    assert result.ellipsoid_visible
    np.testing.assert_allclose(transform[:3, 3], tcp)
    np.testing.assert_allclose(
        transform[:3, :3] @ transform[:3, :3].T,
        0.2**2 * jacobian @ jacobian.T,
        atol=1e-12,
    )
    assert len(robot.configurations) == 1


def test_cached_step_can_display_robot_with_ellipsoid_disabled(
    ur20_playback_assets,
):
    _, q = ur20_playback_assets
    playback, viewer, robot = _playback(ur20_playback_assets)

    result = playback.show_cached_step(
        q,
        tcp_position_m=None,
        axis_lengths=None,
        axis_directions_world=None,
        visible=False,
    )

    assert result.robot_displayed
    assert not result.ellipsoid_visible
    assert result.hidden_reason == "ellipsoid_disabled"
    assert len(robot.configurations) == 1
    assert viewer.nodes[playback.ellipsoid_path].properties[-1] == (
        "visible",
        False,
    )


def test_failed_step_hides_stale_ellipsoid_and_does_not_move_robot(
    ur20_playback_assets,
):
    _, q = ur20_playback_assets
    playback, viewer, robot = _playback(ur20_playback_assets)
    playback.show_step(q, step_name="success")
    display_count = len(robot.configurations)

    result = playback.show_step(
        None,
        step_name="failed_ik",
        successful=False,
    )

    node = viewer.nodes[playback.ellipsoid_path]
    assert not result.robot_displayed
    assert not result.ellipsoid_visible
    assert result.hidden_reason == "step_failed"
    assert len(robot.configurations) == display_count
    assert node.properties[-1] == ("visible", False)


def test_singular_step_displays_robot_but_hides_degenerate_ellipsoid(
    ur20_playback_assets,
    monkeypatch,
):
    _, q = ur20_playback_assets
    playback, viewer, robot = _playback(ur20_playback_assets)
    playback.show_step(q, step_name="regular")
    transform_count = len(viewer.nodes[playback.ellipsoid_path].transforms)

    singular_jacobian = np.zeros((6, 6))
    singular_jacobian[0, 0] = 1.0
    singular_jacobian[1, 1] = 0.5
    monkeypatch.setattr(
        playback_module,
        "arm_frame_jacobian",
        lambda *args, **kwargs: singular_jacobian,
    )
    result = playback.show_step(q, step_name="singular")

    node = viewer.nodes[playback.ellipsoid_path]
    assert result.robot_displayed
    assert not result.ellipsoid_visible
    assert result.hidden_reason == "translational_singularity"
    assert result.singular_values == pytest.approx((1.0, 0.5, 0.0))
    assert len(robot.configurations) == 2
    assert len(node.transforms) == transform_count
    assert node.properties[-1] == ("visible", False)


@pytest.mark.parametrize(
    ("q_factory", "message"),
    (
        (lambda q: q[:-1], "finite values"),
        (lambda q: np.full_like(q, np.nan), "finite values"),
        (lambda q: "not-a-configuration", "numeric configuration"),
    ),
)
def test_invalid_configuration_hides_ellipsoid_before_raising(
    ur20_playback_assets,
    q_factory,
    message,
):
    _, q = ur20_playback_assets
    playback, viewer, robot = _playback(ur20_playback_assets)
    playback.show_step(q)

    with pytest.raises(ValueError, match=message):
        playback.show_step(q_factory(q))

    assert len(robot.configurations) == 1
    assert viewer.nodes[playback.ellipsoid_path].properties[-1] == (
        "visible",
        False,
    )


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("ellipsoid_scale", 0.0),
        ("ellipsoid_scale", math.inf),
        ("singular_tolerance", -1.0),
        ("opacity", 0.0),
        ("ellipsoid_wireframe", 1),
    ),
)
def test_invalid_render_settings_are_rejected(
    ur20_playback_assets,
    keyword,
    value,
):
    bundle, _ = ur20_playback_assets
    kwargs = {keyword: value}

    with pytest.raises(ValueError):
        UR20MeshcatPlayback(
            _RecordingViewer(),
            _RecordingRobotVisualizer(),
            bundle,
            **kwargs,
        )
