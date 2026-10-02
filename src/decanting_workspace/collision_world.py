"""Phase-specific collision worlds backed by Pinocchio and Coal.

The URDF geometry models used by this project deliberately do not carry an
SRDF.  Pinocchio therefore loads their collision geometry with *no collision
pairs*.  This module makes the pairs explicit instead of relying on
``computeCollisions`` silently checking an empty set.

The class is intentionally independent from the decanting task generator.  A
task phase supplies fixed yaw boxes, optional boxes rigidly attached to an
operational frame, and narrowly scoped allowed contacts.  This keeps the
moving box/tote state out of the immutable nominal scene snapshot.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from threading import Lock
from typing import Iterable, Sequence
import weakref

import numpy as np

from .models import BoxPrimitive, MeshObstacle
from .mesh_geometry import mesh_component_topology, origin_inside_closed_mesh
from .robots import RobotBundle
from .transforms import yaw_matrix


@dataclass(frozen=True)
class _MeshCollisionEntry:
    mesh_ref: weakref.ReferenceType[MeshObstacle]
    geometry: object
    vertices: np.ndarray
    triangles: np.ndarray
    is_closed: bool
    representatives: np.ndarray


@dataclass(frozen=True)
class _ShapeContainmentEntry:
    shape: object
    points: np.ndarray
    vertices: np.ndarray
    closed_triangles: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


_SHAPE_CONTAINMENT_CACHE_MAXSIZE = 128
# Every mesh in a living scene can reuse its BVH, even when the scene has many
# more than 128 prims. Weak owner references release old scenes automatically
# without hashing vertex tuples or letting recycled identities match. Geometry
# is immutable; placements and query results belong to individual checkers.
_MESH_COLLISION_CACHE: dict[int, _MeshCollisionEntry] = {}
_SHAPE_CONTAINMENT_CACHE: OrderedDict[int, _ShapeContainmentEntry] = OrderedDict()
_MESH_COLLISION_CACHE_LOCK = Lock()


@dataclass(frozen=True)
class AttachedBox:
    """A box rigidly attached to a Pinocchio operational frame.

    ``frame_T_box`` uses standard homogeneous-transform semantics.  It may
    contain a general rotation; carried cartons are not restricted to yaw-only
    geometry.
    """

    name: str
    size_m: tuple[float, float, float]
    frame_name: str
    frame_T_box: np.ndarray

    def __post_init__(self) -> None:
        transform = np.asarray(self.frame_T_box, dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("frame_T_box must be a finite 4x4 transform")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.size_m):
            raise ValueError("attached-box dimensions must be finite and positive")


@dataclass(frozen=True)
class CollisionPhase:
    """Collision inputs that are valid for one process phase.

    Allowed contacts use logical or geometry names.  For example,
    ``("suction_pad", "pallet_box")`` permits only that pair; the carton is
    still tested against every other robot link.
    """

    obstacles: tuple[BoxPrimitive, ...]
    attached_boxes: tuple[AttachedBox, ...] = ()
    allowed_contacts: frozenset[tuple[str, str]] = frozenset()
    floor_z_m: float | None = 0.0
    meshes: tuple[MeshObstacle, ...] = ()


@dataclass(frozen=True)
class CollisionContact:
    first: str
    second: str


@dataclass(frozen=True)
class CollisionReport:
    in_collision: bool
    contacts: tuple[CollisionContact, ...]
    minimum_distance_m: float
    nearest_pair: tuple[str, str] | None


class PinocchioCollisionChecker:
    """Build and query a collision model for one task phase.

    The checker owns Pinocchio ``Data`` and ``GeometryData`` objects and is not
    thread safe.  Construct a separate checker per concurrent candidate
    evaluation.
    """

    def __init__(
        self,
        bundle: RobotBundle,
        phase: CollisionPhase,
        *,
        own_pedestal_name: str = "ur20_pedestal",
        floor_extent_m: float = 30.0,
        include_self_collision: bool = True,
    ) -> None:
        import coal
        import pinocchio as pin

        if not math.isfinite(floor_extent_m) or floor_extent_m <= 0.0:
            raise ValueError("floor_extent_m must be finite and positive")
        self.bundle = bundle
        self.model = bundle.model
        self.geometry_model = bundle.collision_model.copy()
        self._robot_geometry_count = self.geometry_model.ngeoms
        self._logical_names: dict[int, str] = {
            index: geometry.name
            for index, geometry in enumerate(self.geometry_model.geometryObjects)
        }
        self._environment_meshes: dict[int, _MeshCollisionEntry] = {}
        self._containment_shapes: dict[int, _ShapeContainmentEntry] = {}

        if include_self_collision:
            self._add_provisional_self_pairs(pin)

        obstacle_ids: list[int] = []
        for obstacle in phase.obstacles:
            if not obstacle.collision_enabled:
                continue
            obstacle_ids.append(self._add_world_box(pin, coal, obstacle))

        for mesh in phase.meshes:
            if mesh.collision_enabled:
                obstacle_ids.append(self._add_world_mesh(pin, mesh))

        if phase.floor_z_m is not None:
            floor_z = float(phase.floor_z_m)
            if not math.isfinite(floor_z):
                raise ValueError("floor_z_m must be finite or None")
            floor = BoxPrimitive(
                name="floor",
                role="floor",
                center_m=(0.0, 0.0, floor_z - 0.01),
                size_m=(*_floor_size_for_phase(phase, floor_extent_m), 0.02),
            )
            obstacle_ids.append(self._add_world_box(pin, coal, floor))

        attached_ids = [
            self._add_attached_box(pin, coal, attached)
            for attached in phase.attached_boxes
        ]

        allowed = {
            _canonical_pair(first, second)
            for first, second in phase.allowed_contacts
        }
        self._add_external_pairs(
            pin,
            obstacle_ids=obstacle_ids,
            attached_ids=attached_ids,
            allowed=allowed,
            own_pedestal_name=own_pedestal_name,
        )

        self.data = self.model.createData()
        self.geometry_data = pin.GeometryData(self.geometry_model)

    @property
    def collision_pair_count(self) -> int:
        return len(self.geometry_model.collisionPairs)

    def check(self, q: Sequence[float]) -> CollisionReport:
        """Return all colliding named pairs and the phase minimum distance."""

        import pinocchio as pin

        configuration = np.asarray(q, dtype=float).reshape(-1)
        if configuration.size != self.model.nq or not np.all(np.isfinite(configuration)):
            raise ValueError(f"q must contain {self.model.nq} finite values")

        pin.computeCollisions(
            self.model,
            self.data,
            self.geometry_model,
            self.geometry_data,
            configuration,
            False,
        )
        contacts: list[CollisionContact] = []
        contained_pairs: set[tuple[int, int]] = set()
        for pair, result in zip(
            self.geometry_model.collisionPairs,
            self.geometry_data.collisionResults,
        ):
            in_collision = result.isCollision()
            if not in_collision and self._pair_is_contained(int(pair.first), int(pair.second)):
                in_collision = True
                contained_pairs.add((int(pair.first), int(pair.second)))
            if in_collision:
                contacts.append(
                    CollisionContact(
                        self._logical_names[int(pair.first)],
                        self._logical_names[int(pair.second)],
                    )
                )

        minimum_distance = math.inf
        nearest_pair: tuple[str, str] | None = None
        if self.geometry_model.collisionPairs:
            pin.computeDistances(
                self.model,
                self.data,
                self.geometry_model,
                self.geometry_data,
                configuration,
            )
            for pair, result in zip(
                self.geometry_model.collisionPairs,
                self.geometry_data.distanceResults,
            ):
                distance = float(result.min_distance)
                if (int(pair.first), int(pair.second)) in contained_pairs:
                    distance = min(distance, 0.0)
                if distance < minimum_distance:
                    minimum_distance = distance
                    nearest_pair = (
                        self._logical_names[int(pair.first)],
                        self._logical_names[int(pair.second)],
                    )

        return CollisionReport(
            in_collision=bool(contacts),
            contacts=tuple(contacts),
            minimum_distance_m=minimum_distance,
            nearest_pair=nearest_pair,
        )

    def is_collision_free(self, q: Sequence[float]) -> bool:
        """Small callback adapter accepted by the Pinocchio IK solver."""

        return not self.check(q).in_collision

    def _pair_is_contained(self, first: int, second: int) -> bool:
        """Restore solid interiors that a triangle-surface BVH cannot test."""

        for environment_id, candidate_id in ((first, second), (second, first)):
            entry = self._environment_meshes.get(environment_id)
            if entry is None:
                continue
            candidate = self._containment_shapes.get(candidate_id)
            if candidate is None:
                shape = self.geometry_model.geometryObjects[candidate_id].geometry
                candidate = _shape_containment_entry(shape)
                self._containment_shapes[candidate_id] = candidate
            placement = self.geometry_data.oMg[candidate_id]
            if entry.is_closed:
                points = candidate.points @ placement.rotation.T + placement.translation
                bounds = entry.geometry.aabb_local
                if _points_inside_surface(
                    points, entry.vertices, entry.triangles, bounds.min_, bounds.max_,
                ):
                    return True
            # BVH-vs-BVH also misses the reverse case: a small environment
            # component entirely inside a closed robot collision mesh.
            if len(candidate.closed_triangles):
                points = (entry.representatives - placement.translation) @ placement.rotation
                if _points_inside_surface(
                    points, candidate.vertices, candidate.closed_triangles,
                    candidate.lower, candidate.upper,
                ):
                    return True
        return False

    def _add_provisional_self_pairs(self, pin: object) -> None:
        geometries = self.geometry_model.geometryObjects
        for first in range(self._robot_geometry_count):
            first_joint = int(geometries[first].parentJoint)
            for second in range(first + 1, self._robot_geometry_count):
                second_joint = int(geometries[second].parentJoint)
                if _semantic_self_pair_is_disabled(
                    geometries[first].name,
                    geometries[second].name,
                ):
                    continue
                if _same_or_directly_adjacent_joint(
                    self.model, first_joint, second_joint
                ):
                    continue
                self.geometry_model.addCollisionPair(pin.CollisionPair(first, second))

    def _add_world_box(self, pin: object, coal: object, box: BoxPrimitive) -> int:
        name = _unique_geometry_name(self.geometry_model, f"world::{box.name}")
        rotation = yaw_matrix(math.radians(box.yaw_deg))[:3, :3]
        placement = pin.SE3(rotation, np.asarray(box.center_m, dtype=float))
        geometry = pin.GeometryObject(
            name,
            0,
            0,
            placement,
            coal.Box(*box.size_m),
        )
        geometry_id = int(self.geometry_model.addGeometryObject(geometry))
        self._logical_names[geometry_id] = box.name
        return geometry_id

    def _add_world_mesh(self, pin: object, mesh: MeshObstacle) -> int:
        entry = _mesh_collision_entry(mesh)
        name = _unique_geometry_name(self.geometry_model, f"world::{mesh.name}")
        geometry = pin.GeometryObject(
            name,
            0,
            0,
            pin.SE3.Identity(),
            entry.geometry,
        )
        geometry_id = int(self.geometry_model.addGeometryObject(geometry))
        self._logical_names[geometry_id] = mesh.name
        self._environment_meshes[geometry_id] = entry
        return geometry_id

    def _add_attached_box(
        self, pin: object, coal: object, attached: AttachedBox
    ) -> int:
        frame_id = int(self.model.getFrameId(attached.frame_name))
        if frame_id >= self.model.nframes:
            raise ValueError(
                f"{self.bundle.name} has no attached-box frame {attached.frame_name}"
            )
        frame = self.model.frames[frame_id]
        transform = np.asarray(attached.frame_T_box, dtype=float)
        frame_placement = pin.SE3(transform[:3, :3], transform[:3, 3])
        placement = frame.placement * frame_placement
        name = _unique_geometry_name(
            self.geometry_model, f"attached::{attached.name}"
        )
        geometry = pin.GeometryObject(
            name,
            frame.parentJoint,
            frame_id,
            placement,
            coal.Box(*attached.size_m),
        )
        geometry_id = int(self.geometry_model.addGeometryObject(geometry))
        self._logical_names[geometry_id] = attached.name
        return geometry_id

    def _add_external_pairs(
        self,
        pin: object,
        *,
        obstacle_ids: Iterable[int],
        attached_ids: Iterable[int],
        allowed: set[tuple[str, str]],
        own_pedestal_name: str,
    ) -> None:
        robot_ids = range(self._robot_geometry_count)
        obstacle_ids = tuple(obstacle_ids)
        attached_ids = tuple(attached_ids)

        for robot_id in robot_ids:
            robot_geometry = self.geometry_model.geometryObjects[robot_id]
            robot_name = self._logical_names[robot_id]
            for obstacle_id in obstacle_ids:
                obstacle_name = self._logical_names[obstacle_id]
                # The free-flyer base is mounted on the top face of its own
                # pedestal.  Only that support contact is exempt; arm links are
                # still paired with the pedestal.
                if (
                    obstacle_name == own_pedestal_name
                    and int(robot_geometry.parentJoint) == 1
                ):
                    continue
                if _pair_is_allowed(robot_name, obstacle_name, allowed):
                    continue
                self.geometry_model.addCollisionPair(
                    pin.CollisionPair(robot_id, obstacle_id)
                )

            for attached_id in attached_ids:
                attached_name = self._logical_names[attached_id]
                if _default_tool_payload_contact(robot_name) or _pair_is_allowed(
                    robot_name, attached_name, allowed
                ):
                    continue
                self.geometry_model.addCollisionPair(
                    pin.CollisionPair(robot_id, attached_id)
                )

        for attached_id in attached_ids:
            attached_name = self._logical_names[attached_id]
            for obstacle_id in obstacle_ids:
                obstacle_name = self._logical_names[obstacle_id]
                if _pair_is_allowed(attached_name, obstacle_name, allowed):
                    continue
                self.geometry_model.addCollisionPair(
                    pin.CollisionPair(attached_id, obstacle_id)
                )


def _same_or_directly_adjacent_joint(model: object, first: int, second: int) -> bool:
    if first == second:
        return True
    if first == 0 or second == 0:
        return False
    return int(model.parents[first]) == second or int(model.parents[second]) == first


def _semantic_self_pair_is_disabled(first: str, second: str) -> bool:
    """Apply the non-adjacent UR wrist exemption from its semantic model.

    Directly adjacent joints are handled structurally below.  Universal
    Robots' semantic description additionally marks wrist links 1 and 3 as a
    pair that never needs collision checking.  Both the official and bundled
    proxy URDFs use the same link-derived geometry names.
    """

    normalized = {
        _strip_geometry_suffix(first),
        _strip_geometry_suffix(second),
    }
    return normalized == {"wrist_1_link", "wrist_3_link"}


def _strip_geometry_suffix(name: str) -> str:
    return name[:-2] if name.endswith("_0") else name


def _canonical_pair(first: str, second: str) -> tuple[str, str]:
    return tuple(sorted((str(first), str(second))))


def mesh_collision_geometry(mesh: MeshObstacle) -> object:
    """Reuse immutable World BVHs for living meshes without hashing their data."""

    return _mesh_collision_entry(mesh).geometry


def _mesh_collision_entry(mesh: MeshObstacle) -> _MeshCollisionEntry:

    import coal

    identity = id(mesh)
    with _MESH_COLLISION_CACHE_LOCK:
        cached = _MESH_COLLISION_CACHE.get(identity)
        if cached is not None and cached.mesh_ref() is mesh:
            return cached

    vertex_array = np.asarray(mesh.vertices_m, dtype=float)
    triangle_array = np.asarray(mesh.triangles, dtype=np.intp)
    vertices = coal.StdVec_Vec3s()
    for vertex in vertex_array:
        vertices.append(vertex)
    triangles = coal.StdVec_Triangle()
    for triangle in mesh.triangles:
        triangles.append(coal.Triangle(*triangle))
    geometry = coal.BVHModelOBBRSS()
    statuses = (
        geometry.beginModel(len(mesh.triangles), len(mesh.vertices_m)),
        geometry.addSubModel(vertices, triangles),
        geometry.endModel(),
    )
    if any(status != 0 for status in statuses):
        raise ValueError(f"could not build collision mesh: {mesh.name}")
    geometry.computeLocalAABB()
    representatives, closed_triangles = mesh_component_topology(vertex_array, triangle_array)
    vertex_array.setflags(write=False)
    closed_triangles.setflags(write=False)
    def release_owner(owner_ref: weakref.ReferenceType[MeshObstacle]) -> None:
        with _MESH_COLLISION_CACHE_LOCK:
            current = _MESH_COLLISION_CACHE.get(identity)
            if current is not None and current.mesh_ref is owner_ref:
                del _MESH_COLLISION_CACHE[identity]

    entry = _MeshCollisionEntry(
        weakref.ref(mesh, release_owner), geometry, vertex_array, closed_triangles,
        bool(len(closed_triangles)), vertex_array[representatives],
    )
    with _MESH_COLLISION_CACHE_LOCK:
        cached = _MESH_COLLISION_CACHE.get(identity)
        if cached is not None and cached.mesh_ref() is mesh:
            return cached
        _MESH_COLLISION_CACHE[identity] = entry
    return entry


def _shape_containment_entry(shape: object) -> _ShapeContainmentEntry:
    """Cache actual component points and solid topology for finite shapes."""

    import coal

    identity = id(shape)
    with _MESH_COLLISION_CACHE_LOCK:
        cached = _SHAPE_CONTAINMENT_CACHE.get(identity)
        if cached is not None and cached.shape is shape:
            _SHAPE_CONTAINMENT_CACHE.move_to_end(identity)
            return cached
    vertices = np.empty((0, 3))
    closed_triangles = np.empty((0, 3), dtype=np.intp)
    if isinstance(shape, coal.BVHModelBase):
        vertices = np.asarray(shape.vertices(), dtype=float).reshape(-1, 3)
        triangles = np.asarray(
            [[shape.tri_indices(index)[axis] for axis in range(3)]
             for index in range(shape.num_tris)], dtype=np.intp,
        ).reshape(-1, 3)
        representatives, closed_triangles = mesh_component_topology(vertices, triangles)
        points = vertices[representatives]
    elif isinstance(shape, coal.ConvexBase):
        # The mean of convex vertices lies inside the actual convex body even
        # when its local origin is unrelated to its geometric location.
        points = np.asarray(shape.points(), dtype=float).reshape(-1, 3).mean(axis=0)[None, :]
    elif shape.getNodeType() == coal.GEOM_TRIANGLE:
        points = np.asarray((shape.a,), dtype=float)
    elif shape.getNodeType() in (
        coal.GEOM_BOX, coal.GEOM_SPHERE, coal.GEOM_CAPSULE, coal.GEOM_CYLINDER,
        coal.GEOM_CONE, coal.GEOM_ELLIPSOID,
    ):
        points = np.zeros((1, 3))
    else:
        points = np.empty((0, 3))
    lower = vertices.min(axis=0) if len(vertices) else np.zeros(3)
    upper = vertices.max(axis=0) if len(vertices) else np.zeros(3)
    entry = _ShapeContainmentEntry(shape, points, vertices, closed_triangles, lower, upper)
    with _MESH_COLLISION_CACHE_LOCK:
        cached = _SHAPE_CONTAINMENT_CACHE.get(identity)
        if cached is not None and cached.shape is shape:
            _SHAPE_CONTAINMENT_CACHE.move_to_end(identity)
            return cached
        _SHAPE_CONTAINMENT_CACHE[identity] = entry
        while len(_SHAPE_CONTAINMENT_CACHE) > _SHAPE_CONTAINMENT_CACHE_MAXSIZE:
            _SHAPE_CONTAINMENT_CACHE.popitem(last=False)
    return entry


def _points_inside_surface(
    points: np.ndarray,
    vertices: np.ndarray,
    triangles: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    # After Coal reports no surface intersection, each connected candidate
    # component is wholly inside or outside. One real component point suffices.
    candidates = points[
        np.all(points > lower + 1e-9, axis=1)
        & np.all(points < upper - 1e-9, axis=1)
    ]
    return any(
        origin_inside_closed_mesh(vertices - point, triangles)
        for point in candidates
    )


def _floor_size_for_phase(phase: CollisionPhase, minimum_extent_m: float) -> tuple[float, float]:
    """Keep the nominal floor and extend it around imported environment geometry."""

    half = np.full(2, minimum_extent_m / 2.0, dtype=float)
    if not phase.meshes:
        return float(2.0 * half[0]), float(2.0 * half[1])
    for mesh in phase.meshes:
        if not mesh.collision_enabled:
            continue
        bounds = mesh_collision_geometry(mesh).aabb_local
        half = np.maximum(
            half,
            np.maximum(np.abs(bounds.min_[:2]), np.abs(bounds.max_[:2])) + 0.7,
        )
    for obstacle in phase.obstacles:
        if not obstacle.collision_enabled:
            continue
        rotation = np.abs(yaw_matrix(math.radians(obstacle.yaw_deg))[:2, :2])
        extent = rotation @ (np.asarray(obstacle.size_m[:2]) / 2.0)
        half = np.maximum(half, np.abs(obstacle.center_m[:2]) + extent + 0.7)
    return float(2.0 * half[0]), float(2.0 * half[1])


def _pair_is_allowed(
    first: str,
    second: str,
    allowed: set[tuple[str, str]],
) -> bool:
    pair = _canonical_pair(first, second)
    if pair in allowed:
        return True
    # URDF variants use names such as suction_pad_0 or suction_proxy_pad.
    aliases_first = _geometry_aliases(first)
    aliases_second = _geometry_aliases(second)
    return any(
        _canonical_pair(alias_first, alias_second) in allowed
        for alias_first in aliases_first
        for alias_second in aliases_second
    )


def _geometry_aliases(name: str) -> frozenset[str]:
    aliases = {name}
    lowered = name.lower()
    if "suction" in lowered and "pad" in lowered:
        aliases.add("suction_pad")
    if "suction" in lowered and ("gripper" in lowered or "stem" in lowered):
        aliases.add("suction_stem")
    return frozenset(aliases)


def _default_tool_payload_contact(robot_geometry_name: str) -> bool:
    lowered = robot_geometry_name.lower()
    return "suction" in lowered and (
        "pad" in lowered or "gripper" in lowered or "stem" in lowered
    )


def _unique_geometry_name(geometry_model: object, requested: str) -> str:
    if not geometry_model.existGeometryName(requested):
        return requested
    suffix = 2
    while geometry_model.existGeometryName(f"{requested}#{suffix}"):
        suffix += 1
    return f"{requested}#{suffix}"
