from __future__ import annotations

import numpy as np
import pytest

from decanting_workspace import load_scene_spec
from decanting_workspace.kinematics import (
    IKOptions,
    arm_frame_jacobian,
    compute_manipulability,
    frame_pose,
    solve_frame_ik,
)
from decanting_workspace.robots import floating_configuration, load_robot_bundle


@pytest.fixture(scope="module")
def ur20_assets():
    spec = load_scene_spec()
    robot_spec = spec.robots["ur20"]
    bundle = load_robot_bundle(
        "ur20",
        robot_spec.urdf_path,
        suction_proxy=robot_spec.suction_proxy,
    )
    return robot_spec, bundle


def test_fk_roundtrip_ik_keeps_floating_base_exactly_fixed(ur20_assets):
    robot_spec, bundle = ur20_assets
    known_arm = np.asarray(robot_spec.nominal_q, dtype=float)
    known_q = floating_configuration(bundle, robot_spec.nominal_base, known_arm)
    target = frame_pose(bundle, known_q, "suction_tcp")
    seed = known_arm + np.deg2rad((2.0, -2.0, 2.0, -2.0, 2.0, -2.0))

    result = solve_frame_ik(
        bundle,
        robot_spec.nominal_base,
        target,
        seeds_arm=(seed,),
    )

    assert result.status == "success"
    assert result.success
    assert result.position_error_m <= 1e-6
    assert result.orientation_error_rad <= 1e-6
    assert np.array_equal(result.q[:7], known_q[:7])
    np.testing.assert_allclose(
        frame_pose(bundle, result.q, "suction_tcp"),
        target,
        atol=2e-6,
    )


def test_unreachable_pose_returns_not_found_with_residuals(ur20_assets):
    robot_spec, bundle = ur20_assets
    seed = np.asarray(robot_spec.nominal_q, dtype=float)
    known_q = floating_configuration(bundle, robot_spec.nominal_base, seed)
    target = frame_pose(bundle, known_q, "suction_tcp")
    target[:3, 3] += (10.0, 10.0, 10.0)

    result = solve_frame_ik(
        bundle,
        robot_spec.nominal_base,
        target,
        seeds_arm=(seed,),
        options=IKOptions(max_iterations=20),
    )

    assert result.status == "not_found"
    assert not result.success
    assert result.position_error_m > 1.0
    assert result.orientation_error_rad >= 0.0
    assert result.log_residual_norm > 1.0
    assert np.array_equal(result.q[:7], known_q[:7])


def test_accept_configuration_can_reject_a_converged_solution(ur20_assets):
    robot_spec, bundle = ur20_assets
    seed = np.asarray(robot_spec.nominal_q, dtype=float)
    q = floating_configuration(bundle, robot_spec.nominal_base, seed)
    target = frame_pose(bundle, q, "suction_tcp")
    calls: list[np.ndarray] = []

    def reject(configuration: np.ndarray) -> bool:
        calls.append(configuration)
        return False

    result = solve_frame_ik(
        bundle,
        robot_spec.nominal_base,
        target,
        seeds_arm=(seed,),
        accept_configuration=reject,
    )

    assert result.status == "not_found"
    assert result.rejected_by_acceptor >= 1
    assert calls
    assert all(np.array_equal(item[:7], q[:7]) for item in calls)


def test_arm_only_lwa_jacobian_matches_finite_difference(ur20_assets):
    import pinocchio as pin

    robot_spec, bundle = ur20_assets
    q = floating_configuration(
        bundle,
        robot_spec.nominal_base,
        robot_spec.nominal_q,
    )
    analytical = arm_frame_jacobian(bundle, q, "suction_tcp")
    numerical = np.zeros_like(analytical)
    step = 1e-7

    for column, tangent_column in enumerate(
        range(bundle.arm_velocity_slice.start, bundle.arm_velocity_slice.stop)
    ):
        tangent = np.zeros(bundle.model.nv)
        tangent[tangent_column] = step
        q_plus = pin.integrate(bundle.model, q, tangent)
        q_minus = pin.integrate(bundle.model, q, -tangent)
        plus = frame_pose(bundle, q_plus, "suction_tcp")
        minus = frame_pose(bundle, q_minus, "suction_tcp")
        numerical[:3, column] = (plus[:3, 3] - minus[:3, 3]) / (2.0 * step)
        numerical[3:, column] = np.asarray(
            pin.log3(plus[:3, :3] @ minus[:3, :3].T),
            dtype=float,
        ).reshape(3) / (2.0 * step)

    assert analytical.shape == (6, 6)
    np.testing.assert_allclose(analytical, numerical, atol=1e-5)


def test_manipulability_matches_svd_of_arm_jacobian(ur20_assets):
    robot_spec, bundle = ur20_assets
    q = floating_configuration(
        bundle,
        robot_spec.nominal_base,
        robot_spec.nominal_q,
    )
    length = 0.5
    metrics = compute_manipulability(
        bundle,
        q,
        frame_name="suction_tcp",
        characteristic_length_m=length,
    )
    expected_translation = np.linalg.svd(metrics.jacobian[:3], compute_uv=False)
    expected_normalized = np.linalg.svd(
        np.vstack((metrics.jacobian[:3], length * metrics.jacobian[3:])),
        compute_uv=False,
    )

    np.testing.assert_allclose(
        metrics.translational_singular_values,
        expected_translation,
    )
    np.testing.assert_allclose(metrics.normalized_singular_values, expected_normalized)
    assert metrics.translational_volume == pytest.approx(
        float(np.prod(expected_translation))
    )
    assert metrics.normalized_volume == pytest.approx(float(np.prod(expected_normalized)))
    assert metrics.translational_min_singular_value == pytest.approx(
        expected_translation[-1]
    )
    assert metrics.normalized_min_singular_value == pytest.approx(
        expected_normalized[-1]
    )
    assert metrics.characteristic_length_m == pytest.approx(length)

    with pytest.raises(ValueError, match="characteristic_length_m"):
        compute_manipulability(bundle, q, characteristic_length_m=0.0)
