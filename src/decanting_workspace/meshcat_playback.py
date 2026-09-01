"""Small MeshCat adapter for UR20 task-step playback.

The optimizer does not depend on this module.  It accepts already-computed
full Pinocchio configurations and updates an existing robot visualizer plus a
translational manipulability ellipsoid.  Keeping construction of the
``MeshcatVisualizer`` outside makes the helper usable with the cell viewer and
straightforward to test with recording fakes.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

import numpy as np

from .kinematics import arm_frame_jacobian, frame_pose
from .robots import RobotBundle


@dataclass(frozen=True)
class PlaybackRenderResult:
    """Outcome of displaying one task configuration."""

    step_name: str
    robot_displayed: bool
    ellipsoid_visible: bool
    singular_values: tuple[float, ...] = ()
    tcp_position_m: tuple[float, float, float] | None = None
    hidden_reason: str | None = None


class UR20MeshcatPlayback:
    """Display UR20 configurations and their translational ellipsoids.

    ``robot_visualizer`` is an initialized Pinocchio ``MeshcatVisualizer`` (or
    any object exposing ``display(q)``).  The ellipsoid is a unit MeshCat
    sphere transformed by ``U @ diag(scale * sigma)`` where ``U`` and
    ``sigma`` come from an SVD of the arm-only LOCAL_WORLD_ALIGNED
    translational Jacobian.  It is therefore centred at the World TCP and its
    principal axes are expressed directly in World coordinates.
    """

    def __init__(
        self,
        viewer: object,
        robot_visualizer: object,
        bundle: RobotBundle,
        *,
        frame_name: str = "suction_tcp",
        ellipsoid_path: str = "decanting/playback/ur20/manipulability",
        ellipsoid_scale: float = 0.15,
        singular_tolerance: float = 1.0e-8,
        color: int = 0x35B779,
        opacity: float = 0.34,
        ellipsoid_wireframe: bool = False,
        on_configuration_displayed: Callable[[np.ndarray], object] | None = None,
    ) -> None:
        self.viewer = viewer
        self.robot_visualizer = robot_visualizer
        self.bundle = bundle
        self.frame_name = str(frame_name)
        self.ellipsoid_path = str(ellipsoid_path)
        self.ellipsoid_scale = _positive_finite(
            ellipsoid_scale,
            "ellipsoid_scale",
        )
        self.singular_tolerance = _positive_finite(
            singular_tolerance,
            "singular_tolerance",
        )
        if not isinstance(color, int) or not 0 <= color <= 0xFFFFFF:
            raise ValueError("color must be an RGB integer")
        if not math.isfinite(opacity) or not 0.0 < opacity <= 1.0:
            raise ValueError("opacity must be finite and in (0, 1]")
        if not self.frame_name:
            raise ValueError("frame_name must not be empty")
        if not self.ellipsoid_path:
            raise ValueError("ellipsoid_path must not be empty")
        self.color = color
        self.opacity = float(opacity)
        self.ellipsoid_wireframe = _boolean(
            ellipsoid_wireframe,
            "ellipsoid_wireframe",
        )
        if on_configuration_displayed is not None and not callable(
            on_configuration_displayed
        ):
            raise TypeError("on_configuration_displayed must be callable")
        self.on_configuration_displayed = on_configuration_displayed
        self._object_initialized = False
        self._object_wireframe: bool | None = None

    def display_configuration(
        self,
        q: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Display one configuration and update any external tool attachment."""

        configuration = self._configuration(q)
        detached = np.array(configuration, copy=True)
        self.robot_visualizer.display(detached)
        if self.on_configuration_displayed is not None:
            self.on_configuration_displayed(np.array(configuration, copy=True))
        return configuration

    def show_step(
        self,
        q: Sequence[float] | np.ndarray | None,
        *,
        step_name: str = "",
        successful: bool = True,
    ) -> PlaybackRenderResult:
        """Display one successful full configuration.

        Failed IK/task steps pass ``successful=False`` (and may pass ``q=None``).
        In that case neither stale robot motion nor a stale ellipsoid is
        presented.  A valid but translationally singular configuration still
        displays the robot while hiding only the degenerate ellipsoid.
        """

        name = str(step_name)
        if not successful or q is None:
            self.hide_ellipsoid()
            return PlaybackRenderResult(
                step_name=name,
                robot_displayed=False,
                ellipsoid_visible=False,
                hidden_reason="step_failed",
            )

        configuration = self.display_configuration(q)

        jacobian = arm_frame_jacobian(
            self.bundle,
            configuration,
            self.frame_name,
            reference_frame="LOCAL_WORLD_ALIGNED",
        )[:3, :]
        try:
            world_axes, singular_values, _ = np.linalg.svd(
                jacobian,
                full_matrices=True,
            )
        except np.linalg.LinAlgError:
            self.hide_ellipsoid()
            return PlaybackRenderResult(
                step_name=name,
                robot_displayed=True,
                ellipsoid_visible=False,
                hidden_reason="svd_failed",
            )

        singular_values = np.asarray(singular_values, dtype=float).reshape(-1)
        if (
            singular_values.size != 3
            or not np.all(np.isfinite(singular_values))
            or singular_values[0] <= 0.0
            or singular_values[-1]
            <= self.singular_tolerance * max(1.0, float(singular_values[0]))
        ):
            self.hide_ellipsoid()
            return PlaybackRenderResult(
                step_name=name,
                robot_displayed=True,
                ellipsoid_visible=False,
                singular_values=tuple(float(value) for value in singular_values),
                hidden_reason="translational_singularity",
            )

        tcp_pose = frame_pose(self.bundle, configuration, self.frame_name)
        tcp_position = np.asarray(tcp_pose[:3, 3], dtype=float).reshape(3)
        visible = self._show_ellipsoid(
            tcp_position,
            world_axes,
            singular_values,
            self.ellipsoid_scale,
        )
        if not visible:
            return PlaybackRenderResult(
                step_name=name,
                robot_displayed=True,
                ellipsoid_visible=False,
                singular_values=tuple(float(value) for value in singular_values),
                hidden_reason="nonfinite_ellipsoid",
            )
        return PlaybackRenderResult(
            step_name=name,
            robot_displayed=True,
            ellipsoid_visible=True,
            singular_values=tuple(float(value) for value in singular_values),
            tcp_position_m=tuple(float(value) for value in tcp_position),
        )

    def show_cached_step(
        self,
        q: Sequence[float] | np.ndarray | None,
        *,
        tcp_position_m: Sequence[float] | np.ndarray | None,
        axis_lengths: Sequence[float] | np.ndarray | None,
        axis_directions_world: Sequence[Sequence[float]] | np.ndarray | None,
        step_name: str = "",
        successful: bool = True,
        visible: bool = True,
        ellipsoid_scale: float | None = None,
        ellipsoid_wireframe: bool | None = None,
    ) -> PlaybackRenderResult:
        """Display a configuration using precomputed Pinocchio SVD data.

        ``axis_directions_world`` contains three principal-axis vectors (one
        vector per row), matching the replay-cache representation.  This path
        deliberately performs no FK, Jacobian, or SVD calculation.
        """

        name = str(step_name)
        if not successful or q is None:
            self.hide_ellipsoid()
            return PlaybackRenderResult(
                step_name=name,
                robot_displayed=False,
                ellipsoid_visible=False,
                hidden_reason="step_failed",
            )
        configuration = self.display_configuration(q)
        if not visible:
            self.hide_ellipsoid()
            return PlaybackRenderResult(
                step_name=name,
                robot_displayed=True,
                ellipsoid_visible=False,
                hidden_reason="ellipsoid_disabled",
            )
        if tcp_position_m is None or axis_lengths is None or axis_directions_world is None:
            self.hide_ellipsoid()
            return PlaybackRenderResult(
                step_name=name,
                robot_displayed=True,
                ellipsoid_visible=False,
                hidden_reason="cached_ellipsoid_missing",
            )

        position = np.asarray(tcp_position_m, dtype=float).reshape(-1)
        lengths = np.asarray(axis_lengths, dtype=float).reshape(-1)
        direction_rows = np.asarray(axis_directions_world, dtype=float)
        if (
            position.size != 3
            or lengths.size != 3
            or direction_rows.shape != (3, 3)
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(lengths))
            or not np.all(np.isfinite(direction_rows))
        ):
            self.hide_ellipsoid()
            raise ValueError("cached ellipsoid must contain finite 3-D axes and position")
        largest = float(np.max(lengths))
        if largest <= 0.0 or float(np.min(lengths)) <= self.singular_tolerance * max(1.0, largest):
            self.hide_ellipsoid()
            return PlaybackRenderResult(
                step_name=name,
                robot_displayed=True,
                ellipsoid_visible=False,
                singular_values=tuple(float(value) for value in lengths),
                tcp_position_m=tuple(float(value) for value in position),
                hidden_reason="translational_singularity",
            )
        orientation = direction_rows.T
        if not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-7):
            self.hide_ellipsoid()
            raise ValueError("cached ellipsoid directions must be orthonormal")
        scale = (
            self.ellipsoid_scale
            if ellipsoid_scale is None
            else _positive_finite(ellipsoid_scale, "ellipsoid_scale")
        )
        wireframe = (
            self.ellipsoid_wireframe
            if ellipsoid_wireframe is None
            else _boolean(ellipsoid_wireframe, "ellipsoid_wireframe")
        )
        if not self._show_ellipsoid(
            position,
            orientation,
            lengths,
            scale,
            wireframe,
        ):
            return PlaybackRenderResult(
                step_name=name,
                robot_displayed=True,
                ellipsoid_visible=False,
                singular_values=tuple(float(value) for value in lengths),
                tcp_position_m=tuple(float(value) for value in position),
                hidden_reason="nonfinite_ellipsoid",
            )
        return PlaybackRenderResult(
            step_name=name,
            robot_displayed=True,
            ellipsoid_visible=True,
            singular_values=tuple(float(value) for value in lengths),
            tcp_position_m=tuple(float(value) for value in position),
        )

    def hide_ellipsoid(self) -> None:
        """Hide any previous ellipsoid without deleting its geometry."""

        self.viewer[self.ellipsoid_path].set_property("visible", False)

    def _ensure_ellipsoid_object(self, node: object, *, wireframe: bool) -> None:
        if self._object_initialized and self._object_wireframe == wireframe:
            return
        import meshcat.geometry as geometry

        node.set_object(
            geometry.Sphere(1.0),
            geometry.MeshLambertMaterial(
                color=self.color,
                opacity=self.opacity,
                transparent=self.opacity < 1.0,
                wireframe=wireframe,
            ),
        )
        self._object_initialized = True
        self._object_wireframe = wireframe

    def _configuration(
        self,
        q: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        try:
            configuration = np.asarray(q, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            self.hide_ellipsoid()
            raise ValueError("q must be a numeric configuration") from exc
        if (
            configuration.size != self.bundle.model.nq
            or not np.all(np.isfinite(configuration))
        ):
            self.hide_ellipsoid()
            raise ValueError(
                f"q must contain {self.bundle.model.nq} finite values"
            )
        return configuration

    def _show_ellipsoid(
        self,
        tcp_position: np.ndarray,
        orientation: np.ndarray,
        singular_values: np.ndarray,
        scale: float,
        wireframe: bool | None = None,
    ) -> bool:
        world_axes = np.asarray(orientation, dtype=float).reshape(3, 3).copy()
        # SVD axis signs are arbitrary.  Select a proper right-handed frame so
        # the affine sphere transform does not contain an unnecessary mirror.
        if float(np.linalg.det(world_axes)) < 0.0:
            world_axes[:, -1] *= -1.0
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = world_axes @ np.diag(scale * singular_values)
        transform[:3, 3] = tcp_position
        if not np.all(np.isfinite(transform)):
            self.hide_ellipsoid()
            return False
        node = self.viewer[self.ellipsoid_path]
        style = self.ellipsoid_wireframe if wireframe is None else wireframe
        self._ensure_ellipsoid_object(node, wireframe=style)
        node.set_transform(transform)
        node.set_property("visible", True)
        return True


def _positive_finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value
