"""Arm-only Pinocchio inverse kinematics and manipulability helpers.

The project loads robots with a floating root so one model can be reused for
many base candidates.  The routines in this module deliberately keep that
root fixed: only :attr:`RobotBundle.arm_velocity_slice` participates in IK or
Jacobian metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from .models import BasePose
from .robots import RobotBundle, floating_configuration


IKStatus = Literal["success", "not_found"]
ConfigurationAcceptor = Callable[[np.ndarray], bool]


@dataclass(frozen=True)
class IKOptions:
    """Numerical tolerances and iteration limits for damped least-squares IK."""

    position_tolerance_m: float = 1e-6
    orientation_tolerance_rad: float = 1e-6
    max_iterations: int = 500
    damping: float = 1e-6
    max_backtracking_steps: int = 14

    def __post_init__(self) -> None:
        finite_positive = {
            "position_tolerance_m": self.position_tolerance_m,
            "orientation_tolerance_rad": self.orientation_tolerance_rad,
            "damping": self.damping,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.max_backtracking_steps <= 0:
            raise ValueError("max_backtracking_steps must be positive")


@dataclass(frozen=True)
class IKResult:
    """Best result from a deterministic multi-start IK search.

    ``q`` is always populated.  For ``status == 'not_found'`` it is the best
    finite configuration visited, which is useful for diagnostics but must not
    be consumed as a feasible task configuration.
    """

    status: IKStatus
    q: np.ndarray
    arm_q: np.ndarray
    iterations: int
    seed_index: int
    attempted_seeds: int
    position_error_m: float
    orientation_error_rad: float
    log_residual_norm: float
    rejected_by_acceptor: int = 0

    @property
    def success(self) -> bool:
        """Whether both pose tolerances and the optional acceptor were met."""

        return self.status == "success"


@dataclass(frozen=True)
class ManipulabilityMetrics:
    """SVD representation of translational and normalized 6-D ellipsoids."""

    jacobian: np.ndarray
    translational_singular_values: np.ndarray
    translational_volume: float
    translational_min_singular_value: float
    translational_condition_number: float
    normalized_singular_values: np.ndarray
    normalized_volume: float
    normalized_min_singular_value: float
    normalized_condition_number: float
    characteristic_length_m: float


def frame_pose(
    bundle: RobotBundle,
    q: Sequence[float] | np.ndarray,
    frame_name: str,
) -> np.ndarray:
    """Return ``World_T_frame`` as a detached 4 x 4 matrix."""

    import pinocchio as pin

    configuration = _configuration(bundle, q)
    frame_id = _frame_id(bundle, frame_name)
    data = bundle.model.createData()
    pin.framesForwardKinematics(bundle.model, data, configuration)
    placement = data.oMf[frame_id]
    result = np.eye(4, dtype=float)
    result[:3, :3] = np.asarray(placement.rotation, dtype=float)
    result[:3, 3] = np.asarray(placement.translation, dtype=float).reshape(3)
    return result


def arm_frame_jacobian(
    bundle: RobotBundle,
    q: Sequence[float] | np.ndarray,
    frame_name: str,
    reference_frame: str = "LOCAL_WORLD_ALIGNED",
) -> np.ndarray:
    """Return a 6 x ``N_arm`` frame Jacobian with root columns removed."""

    import pinocchio as pin

    configuration = _configuration(bundle, q)
    frame_id = _frame_id(bundle, frame_name)
    reference = _reference_frame(pin, reference_frame)
    data = bundle.model.createData()
    pin.computeJointJacobians(bundle.model, data, configuration)
    pin.updateFramePlacements(bundle.model, data)
    full = np.asarray(
        pin.getFrameJacobian(bundle.model, data, frame_id, reference),
        dtype=float,
    )
    result = np.array(full[:, bundle.arm_velocity_slice], dtype=float, copy=True)
    expected_columns = _arm_velocity_count(bundle)
    if result.shape != (6, expected_columns) or not np.all(np.isfinite(result)):
        raise RuntimeError(
            f"invalid arm Jacobian for {bundle.name}: {result.shape}"
        )
    return result


def solve_frame_ik(
    bundle: RobotBundle,
    base: BasePose,
    target_world_T_frame: Sequence[Sequence[float]] | np.ndarray,
    *,
    frame_name: str = "suction_tcp",
    seeds_arm: Sequence[Sequence[float] | np.ndarray] = (),
    options: IKOptions | None = None,
    accept_configuration: ConfigurationAcceptor | None = None,
) -> IKResult:
    """Find a fixed-base arm configuration for a target frame pose.

    Caller-provided seeds are attempted in order.  A deterministic joint-limit
    centre seed and a clipped zero seed are appended when they are not already
    present.  The callback, when supplied, is evaluated only for numerically
    converged configurations; it can therefore reject configurations that fail
    a collision or application-specific check without coupling this module to
    a particular collision-world representation.
    """

    import pinocchio as pin

    settings = options or IKOptions()
    target_matrix = _pose_matrix(target_world_T_frame)
    target = pin.SE3(target_matrix[:3, :3], target_matrix[:3, 3])
    frame_id = _frame_id(bundle, frame_name)
    seeds = _ik_seeds(bundle, seeds_arm)
    attempted = len(seeds)

    best: _Attempt | None = None
    rejected = 0
    for seed_index, arm_seed in enumerate(seeds):
        attempt = _solve_one_seed(
            bundle,
            base,
            target,
            frame_id,
            arm_seed,
            settings,
        )
        if best is None or attempt.log_residual_norm < best.log_residual_norm:
            best = attempt
            best_seed_index = seed_index

        if not attempt.converged:
            continue
        accepted = (
            True
            if accept_configuration is None
            else bool(accept_configuration(np.array(attempt.q, copy=True)))
        )
        if not accepted:
            rejected += 1
            continue
        return _result(
            "success",
            bundle,
            attempt,
            seed_index=seed_index,
            attempted_seeds=attempted,
            rejected_by_acceptor=rejected,
        )

    if best is None:  # Defensive; _ik_seeds always returns at least one seed.
        raise RuntimeError("IK seed generation produced no configurations")
    return _result(
        "not_found",
        bundle,
        best,
        seed_index=best_seed_index,
        attempted_seeds=attempted,
        rejected_by_acceptor=rejected,
    )


def compute_manipulability(
    bundle: RobotBundle,
    q: Sequence[float] | np.ndarray,
    *,
    frame_name: str = "suction_tcp",
    characteristic_length_m: float = 0.5,
) -> ManipulabilityMetrics:
    """Compute arm-only Jacobian ellipsoid metrics at one configuration.

    The translational metric uses the first three Pinocchio motion rows.  The
    6-D metric scales angular velocity rows by ``characteristic_length_m`` so
    linear and angular components share metre-based units before SVD.
    """

    length = float(characteristic_length_m)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("characteristic_length_m must be finite and positive")

    jacobian = arm_frame_jacobian(
        bundle,
        q,
        frame_name,
        reference_frame="LOCAL_WORLD_ALIGNED",
    )
    translational = jacobian[:3, :]
    normalized = np.vstack((translational, length * jacobian[3:, :]))
    translation_singular_values = np.linalg.svd(
        translational,
        compute_uv=False,
    )
    normalized_singular_values = np.linalg.svd(normalized, compute_uv=False)
    translation_volume, translation_min, translation_condition = _svd_metrics(
        translation_singular_values,
        translational.shape,
    )
    normalized_volume, normalized_min, normalized_condition = _svd_metrics(
        normalized_singular_values,
        normalized.shape,
    )
    return ManipulabilityMetrics(
        jacobian=np.array(jacobian, copy=True),
        translational_singular_values=np.array(
            translation_singular_values,
            copy=True,
        ),
        translational_volume=translation_volume,
        translational_min_singular_value=translation_min,
        translational_condition_number=translation_condition,
        normalized_singular_values=np.array(normalized_singular_values, copy=True),
        normalized_volume=normalized_volume,
        normalized_min_singular_value=normalized_min,
        normalized_condition_number=normalized_condition,
        characteristic_length_m=length,
    )


@dataclass(frozen=True)
class _Attempt:
    q: np.ndarray
    iterations: int
    position_error_m: float
    orientation_error_rad: float
    log_residual_norm: float
    converged: bool


def _solve_one_seed(
    bundle: RobotBundle,
    base: BasePose,
    target: object,
    frame_id: int,
    arm_seed: np.ndarray,
    options: IKOptions,
) -> _Attempt:
    import pinocchio as pin

    model = bundle.model
    q = floating_configuration(bundle, base, arm_seed)
    fixed_root = np.array(q[: bundle.arm_configuration_offset], copy=True)
    data = model.createData()
    trial_data = model.createData()
    active_damping = float(options.damping)
    last = _evaluate_error(model, data, q, frame_id, target)

    for iteration in range(options.max_iterations + 1):
        if iteration > 0:
            last = _evaluate_error(model, data, q, frame_id, target)
        if (
            last.position_error_m <= options.position_tolerance_m
            and last.orientation_error_rad <= options.orientation_tolerance_rad
        ):
            return _Attempt(
                q=np.array(q, copy=True),
                iterations=iteration,
                position_error_m=last.position_error_m,
                orientation_error_rad=last.orientation_error_rad,
                log_residual_norm=last.log_residual_norm,
                converged=True,
            )
        if iteration == options.max_iterations:
            break

        full_jacobian = np.asarray(
            pin.getFrameJacobian(
                model,
                data,
                frame_id,
                pin.ReferenceFrame.LOCAL,
            ),
            dtype=float,
        )
        arm_jacobian = full_jacobian[:, bundle.arm_velocity_slice]
        task_jacobian = -np.asarray(
            pin.Jlog6(last.error_transform.inverse()),
            dtype=float,
        ) @ arm_jacobian
        normal = task_jacobian @ task_jacobian.T
        try:
            arm_velocity = -task_jacobian.T @ np.linalg.solve(
                normal + active_damping * np.eye(6),
                last.error_vector,
            )
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(arm_velocity)):
            break

        accepted_step = False
        step_scale = 1.0
        for _ in range(options.max_backtracking_steps):
            tangent = np.zeros(model.nv, dtype=float)
            tangent[bundle.arm_velocity_slice] = step_scale * arm_velocity
            candidate = np.asarray(pin.integrate(model, q, tangent), dtype=float)
            _restore_fixed_root(candidate, fixed_root)
            candidate = _clip_arm_limits(bundle, candidate)
            candidate_error = _evaluate_error(
                model,
                trial_data,
                candidate,
                frame_id,
                target,
            )
            if candidate_error.log_residual_norm < last.log_residual_norm:
                q = candidate
                last = candidate_error
                accepted_step = True
                active_damping = max(options.damping, active_damping * 0.5)
                break
            step_scale *= 0.5

        if not accepted_step:
            active_damping = min(active_damping * 10.0, 1e6)
            if active_damping >= 1e6:
                break

    final = _evaluate_error(model, data, q, frame_id, target)
    return _Attempt(
        q=np.array(q, copy=True),
        iterations=min(iteration, options.max_iterations),
        position_error_m=final.position_error_m,
        orientation_error_rad=final.orientation_error_rad,
        log_residual_norm=final.log_residual_norm,
        converged=False,
    )


@dataclass(frozen=True)
class _ErrorState:
    error_transform: object
    error_vector: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    log_residual_norm: float


def _evaluate_error(
    model: object,
    data: object,
    q: np.ndarray,
    frame_id: int,
    target: object,
) -> _ErrorState:
    import pinocchio as pin

    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    current = pin.SE3(data.oMf[frame_id])
    error_transform = current.actInv(target)
    error_vector = np.asarray(
        pin.log6(error_transform).vector,
        dtype=float,
    ).reshape(6)
    position_error = float(
        np.linalg.norm(
            np.asarray(target.translation, dtype=float).reshape(3)
            - np.asarray(current.translation, dtype=float).reshape(3)
        )
    )
    orientation_error = float(
        np.linalg.norm(
            np.asarray(
                pin.log3(current.rotation.T @ target.rotation),
                dtype=float,
            ).reshape(3)
        )
    )
    return _ErrorState(
        error_transform=error_transform,
        error_vector=error_vector,
        position_error_m=position_error,
        orientation_error_rad=orientation_error,
        log_residual_norm=float(np.linalg.norm(error_vector)),
    )


def _result(
    status: IKStatus,
    bundle: RobotBundle,
    attempt: _Attempt,
    *,
    seed_index: int,
    attempted_seeds: int,
    rejected_by_acceptor: int,
) -> IKResult:
    q = np.array(attempt.q, dtype=float, copy=True)
    arm_q = np.array(q[bundle.arm_configuration_offset :], copy=True)
    return IKResult(
        status=status,
        q=q,
        arm_q=arm_q,
        iterations=attempt.iterations,
        seed_index=seed_index,
        attempted_seeds=attempted_seeds,
        position_error_m=attempt.position_error_m,
        orientation_error_rad=attempt.orientation_error_rad,
        log_residual_norm=attempt.log_residual_norm,
        rejected_by_acceptor=rejected_by_acceptor,
    )


def _ik_seeds(
    bundle: RobotBundle,
    seeds_arm: Sequence[Sequence[float] | np.ndarray],
) -> tuple[np.ndarray, ...]:
    offset = bundle.arm_configuration_offset
    lower = np.asarray(bundle.model.lowerPositionLimit, dtype=float)[offset:]
    upper = np.asarray(bundle.model.upperPositionLimit, dtype=float)[offset:]
    count = lower.size
    candidates: list[np.ndarray] = []
    for index, seed in enumerate(seeds_arm):
        vector = _vector(seed, count, f"seeds_arm[{index}]")
        _check_arm_limits(vector, lower, upper, f"seeds_arm[{index}]")
        candidates.append(vector)

    centre = np.zeros(count, dtype=float)
    both_finite = np.isfinite(lower) & np.isfinite(upper)
    centre[both_finite] = 0.5 * (lower[both_finite] + upper[both_finite])
    centre = np.maximum(centre, np.where(np.isfinite(lower), lower, centre))
    centre = np.minimum(centre, np.where(np.isfinite(upper), upper, centre))
    candidates.append(centre)

    zero = np.zeros(count, dtype=float)
    zero = np.maximum(zero, np.where(np.isfinite(lower), lower, zero))
    zero = np.minimum(zero, np.where(np.isfinite(upper), upper, zero))
    candidates.append(zero)

    unique: list[np.ndarray] = []
    for candidate in candidates:
        if not any(np.allclose(candidate, prior, atol=1e-12, rtol=0.0) for prior in unique):
            unique.append(np.array(candidate, copy=True))
    return tuple(unique)


def _configuration(
    bundle: RobotBundle,
    q: Sequence[float] | np.ndarray,
) -> np.ndarray:
    import pinocchio as pin

    configuration = _vector(q, bundle.model.nq, "q")
    lower = np.asarray(bundle.model.lowerPositionLimit, dtype=float)
    upper = np.asarray(bundle.model.upperPositionLimit, dtype=float)
    tolerance = 1e-10
    if np.any(configuration < lower - tolerance) or np.any(configuration > upper + tolerance):
        raise ValueError("q violates robot joint position limits")
    if hasattr(pin, "isNormalized") and not pin.isNormalized(
        bundle.model,
        configuration,
        1e-8,
    ):
        raise ValueError("q is not a normalized Pinocchio configuration")
    return configuration


def _clip_arm_limits(bundle: RobotBundle, q: np.ndarray) -> np.ndarray:
    result = np.array(q, dtype=float, copy=True)
    offset = bundle.arm_configuration_offset
    lower = np.asarray(bundle.model.lowerPositionLimit, dtype=float)[offset:]
    upper = np.asarray(bundle.model.upperPositionLimit, dtype=float)[offset:]
    arm = result[offset:]
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    arm[finite_lower] = np.maximum(arm[finite_lower], lower[finite_lower])
    arm[finite_upper] = np.minimum(arm[finite_upper], upper[finite_upper])
    result[offset:] = arm
    return result


def _restore_fixed_root(q: np.ndarray, fixed_root: np.ndarray) -> None:
    if fixed_root.size:
        q[: fixed_root.size] = fixed_root


def _pose_matrix(value: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("target_world_T_frame must be a finite 4 x 4 matrix")
    if not np.allclose(matrix[3, :], (0.0, 0.0, 0.0, 1.0), atol=1e-10):
        raise ValueError("target_world_T_frame has an invalid homogeneous row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8) or not np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=1e-8,
    ):
        raise ValueError("target_world_T_frame rotation must be in SO(3)")
    return np.array(matrix, copy=True)


def _frame_id(bundle: RobotBundle, frame_name: str) -> int:
    try:
        frame_id = int(bundle.model.getFrameId(frame_name))
    except Exception as exc:
        raise ValueError(f"unknown {bundle.name} frame: {frame_name!r}") from exc
    if frame_id >= bundle.model.nframes:
        raise ValueError(f"unknown {bundle.name} frame: {frame_name!r}")
    return frame_id


def _reference_frame(pin: object, value: str) -> object:
    choices = {
        "LOCAL": pin.ReferenceFrame.LOCAL,
        "WORLD": pin.ReferenceFrame.WORLD,
        "LOCAL_WORLD_ALIGNED": pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    }
    try:
        return choices[value]
    except KeyError as exc:
        raise ValueError(
            "reference_frame must be LOCAL, WORLD, or LOCAL_WORLD_ALIGNED"
        ) from exc


def _arm_velocity_count(bundle: RobotBundle) -> int:
    start, stop, step = bundle.arm_velocity_slice.indices(bundle.model.nv)
    return len(range(start, stop, step))


def _check_arm_limits(
    arm: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    name: str,
) -> None:
    tolerance = 1e-10
    if np.any(arm < lower - tolerance) or np.any(arm > upper + tolerance):
        raise ValueError(f"{name} violates robot joint position limits")


def _vector(value: Sequence[float] | np.ndarray, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.size != length or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {length} finite values")
    return np.array(result, copy=True)


def _svd_metrics(
    singular_values: np.ndarray,
    matrix_shape: tuple[int, int],
) -> tuple[float, float, float]:
    values = np.asarray(singular_values, dtype=float).reshape(-1)
    minimum = float(values[-1]) if values.size else 0.0
    volume = float(np.prod(values, dtype=float))
    if not values.size:
        return volume, minimum, math.inf
    threshold = (
        np.finfo(float).eps
        * max(matrix_shape)
        * max(float(values[0]), 1.0)
    )
    condition = math.inf if minimum <= threshold else float(values[0] / minimum)
    return volume, minimum, condition
