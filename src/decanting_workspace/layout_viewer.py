"""Meshcat display of simplified USD layouts and URDF robot models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .asset_prep import default_sr12ia_output_urdf, validate_prepared_sr12ia_asset
from .collision import _robot_collision_bounds_base_local
from .layout_models import LayoutFragment, LayoutRobot, LayoutScene
from .models import BasePose, MeshObstacle
from .robots import floating_configuration, load_robot_bundle, resolve_official_ur20
from .viewer import CellViewer


class LayoutViewer(CellViewer):
    """Render supplied environment fragments while reusing loaded URDF meshes.

    USD supplies robot positions only. Robot orientation and joint values come
    from the layout's configured URDF contract. No SKU, generated floor,
    installation rectangle, task workspace or process object is added here.
    """

    def __init__(
        self,
        viewer: object | None = None,
        *,
        ur20_source: str = "official",
        sr12ia_visual_urdf: str | Path | None = None,
        show_frames: bool = True,
    ) -> None:
        if ur20_source not in {"official", "configured"}:
            raise ValueError("ur20_source must be 'official' or 'configured'")
        super().__init__(viewer)
        self.ur20_source = ur20_source
        self.sr12ia_visual_urdf = Path(sr12ia_visual_urdf) if sr12ia_visual_urdf else None
        self.show_frames = bool(show_frames)
        self._source_keys: dict[str, tuple[Path, tuple[Path, ...]]] = {}
        self._robot_metadata: dict[str, dict[str, object]] = {}
        self._official_ur20: tuple[Path, tuple[Path, ...]] | None = None
        self._prepared_sr12ia: Path | None = None
        self._checked_prepared_sr12ia = False
        self._sr12ia_fallback_reason: str | None = None
        self._camera_initialized = False
        self.fragment_metadata: dict[str, dict[str, str]] = {}

    def render(self, scene: LayoutScene, fragments: Sequence[LayoutFragment]) -> None:
        """Replace environment fragments and move existing robot visualizers."""

        fragment_names = [fragment.name for fragment in fragments]
        if len(fragment_names) != len(set(fragment_names)):
            raise ValueError("layout fragment names must be unique")
        # Validate all environment geometry before replacing the existing view.
        meshes = tuple(MeshObstacle(
            name=fragment.name,
            role=fragment.role,
            vertices_m=fragment.vertices_m,
            triangles=fragment.triangles,
            source_prim_path=fragment.owner_path,
        ) for fragment in fragments)

        self._set_background()
        self.viewer["decanting/environment"].delete()
        for mesh in meshes:
            self._render_mesh(mesh)
        self.fragment_metadata = {
            fragment.name: {"owner_path": fragment.owner_path, "source_name": fragment.source_name}
            for fragment in fragments
        }

        self.viewer["decanting/frames"].delete()
        if self.show_frames:
            for frame in scene.frames:
                self._render_frame(frame.name, frame.translation_m, frame.rotation_xyzw)

        present = {robot.name for robot in scene.robots}
        for name in set(self._source_keys) - present:
            self.viewer[f"decanting/robots/{name}"].delete()
            self._source_keys.pop(name)
            self._robot_metadata.pop(name, None)
            self._robots[:] = [loaded for loaded in self._robots if loaded.bundle.name != name]

        for robot in scene.robots:
            urdf_path, package_dirs, source = self._robot_source(robot)
            key = (urdf_path.resolve(), tuple(Path(path).resolve() for path in package_dirs))
            base = BasePose(*robot.translation_m, robot.yaw_deg)
            if self._source_keys.get(robot.name) != key:
                # A deliberately changed URDF replaces only that robot. Normal
                # position updates retain both its model and Meshcat geometry.
                self.viewer[f"decanting/robots/{robot.name}"].delete()
                self._robots[:] = [loaded for loaded in self._robots if loaded.bundle.name != robot.name]
                bundle = load_robot_bundle(
                    robot.name, urdf_path, package_dirs=package_dirs, load_visual=True,
                )
                self._render_robot(
                    bundle, base, robot.nominal_q,
                    root_name=f"decanting/robots/{robot.name}",
                    show_collisions=False,
                )
                self._source_keys[robot.name] = key
            else:
                loaded = self.loaded_robot(robot.name)
                configuration = floating_configuration(loaded.bundle, base, robot.nominal_q)
                loaded.visualizer.display(configuration)

            loaded = self.loaded_robot(robot.name)
            self._robot_metadata[robot.name] = {
                "name": robot.name,
                "source": source,
                "urdf_path": str(key[0]),
                "configured_urdf_path": str(robot.urdf_path.resolve()),
                "translation_m": list(robot.translation_m),
                "yaw_deg": robot.yaw_deg,
                "nominal_q": list(robot.nominal_q),
                "model_nq": int(loaded.bundle.model.nq),
                "source_prim_path": robot.source_prim_path,
            }
            if robot.name == "sr12ia" and self._sr12ia_fallback_reason:
                self._robot_metadata[robot.name]["visual_fallback_reason"] = self._sr12ia_fallback_reason

        if not self._camera_initialized:
            self._fit_initial_camera(scene, fragments)

    def robot_info(self) -> list[dict[str, object]]:
        """Return selected URDFs and the base poses actually displayed."""

        # Copy nested lists so UI serialization or callers cannot alter state.
        return [{key: list(value) if isinstance(value, list) else value
                 for key, value in metadata.items()}
                for _, metadata in sorted(self._robot_metadata.items())]

    def _fit_initial_camera(self, scene: LayoutScene, fragments: Sequence[LayoutFragment]) -> None:
        """Frame the work cell once, preserving later user camera changes."""

        points = []
        for fragment in fragments:
            vertices = np.asarray(fragment.vertices_m, dtype=float)
            # The floor and room walls stay visible, but their room-sized
            # bounds must not move the orbit focus away from the work cell.
            if fragment.role == "ground":
                continue
            points.extend((vertices.min(axis=0), vertices.max(axis=0)))
        for robot in scene.robots:
            loaded = self.loaded_robot(robot.name)
            collision_model = getattr(loaded.bundle, "collision_model", None)
            if collision_model is not None and collision_model.ngeoms:
                q = floating_configuration(
                    loaded.bundle, BasePose(*robot.translation_m, robot.yaw_deg), robot.nominal_q,
                )
                points.extend(_robot_collision_bounds_base_local(loaded.bundle, q))
            else:
                points.append(np.asarray(robot.translation_m))
        points.extend(np.asarray(frame.translation_m) for frame in scene.frames)
        if not points:
            points = [np.asarray(node.translation_m) for node in scene.nodes if node.kind != "ground"]
        if not points:
            return

        coordinates = np.asarray(points)
        lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
        target = (lower + upper) / 2.
        radius = max(0.5, float(np.linalg.norm(upper - lower)) / 2.)
        field_of_view = 50.
        distance = 1.35 * radius / math.sin(math.radians(field_of_view / 2.))
        # OrbitControls requires camera.position, target and lookAt to share
        # World axes. The layout frontend also removes MeshCat's scene-root
        # axis conversion and initializes controls with a Z-up camera.
        world_direction = np.array((-0.65, 0.9, 1.1))
        world_offset = world_direction / np.linalg.norm(world_direction) * distance
        self.viewer["/Cameras/default"].set_transform(np.eye(4))
        self.viewer["/Cameras/default/rotated"].set_transform(np.eye(4))
        camera = self.viewer["/Cameras/default/rotated/<object>"]
        camera.set_property("position", (target + world_offset).tolist())
        camera.set_property("fov", field_of_view)
        camera.set_property("far", max(100., distance + 3. * radius))
        camera.set_property("layout_target", target.tolist())
        self._camera_initialized = True

    def _robot_source(self, robot: LayoutRobot) -> tuple[Path, tuple[Path, ...], str]:
        if robot.name == "ur20" and self.ur20_source == "official":
            if self._official_ur20 is None:
                urdf, packages = resolve_official_ur20()
                self._official_ur20 = (Path(urdf), tuple(Path(path) for path in packages))
            return (*self._official_ur20, "official")
        if robot.name == "sr12ia":
            if self.sr12ia_visual_urdf is not None:
                return self.sr12ia_visual_urdf, (), "explicit"
            if not self._checked_prepared_sr12ia:
                self._checked_prepared_sr12ia = True
                prepared = default_sr12ia_output_urdf()
                if prepared.is_file():
                    try:
                        validate_prepared_sr12ia_asset(prepared)
                    except (FileNotFoundError, OSError, ValueError) as exc:
                        self._sr12ia_fallback_reason = str(exc)
                    else:
                        self._prepared_sr12ia = prepared
            if self._prepared_sr12ia is not None:
                return self._prepared_sr12ia, (), "prepared"
        return robot.urdf_path, (), "configured"
