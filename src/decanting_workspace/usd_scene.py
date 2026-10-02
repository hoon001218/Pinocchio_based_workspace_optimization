"""Read a static USD environment as triangle surfaces, without Isaac Sim.

Task frames, robot kinematics and moving payloads still belong to the scene
configuration.  ``box_prims`` optionally associates a USD subtree with a named
conveyor/worktable: its bounds serve the task generator, while its triangles
serve visualization and collision.  All other selected geometry is discovered
automatically, including geometry added after the configuration was written.
"""

from __future__ import annotations

from dataclasses import replace
from fnmatch import fnmatchcase
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .models import MeshObstacle, SceneSpec


def load_usd_environment(
    spec: SceneSpec,
    options: Mapping[str, object],
    *,
    usd_path: str | Path | None = None,
) -> SceneSpec:
    """Replace configured fixed collision boxes with a local USD snapshot.

    Relative ``file`` paths are relative to the YAML file; an explicit CLI
    override is relative to the process directory.  No generated mesh files or
    USD file paths enter the cache identity: World geometry and logical prim
    identities do.
    """
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        raise RuntimeError(
            "USD scenes require usd-core; install decanting-workspace[usd-scene]"
        ) from exc

    allowed = {"file", "include_prim_paths", "exclude_prim_paths", "box_prims", "box_bounds_prims",
               "time_code", "include_invisible", "purposes", "exclude_articulations"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"unknown usd_scene options: {sorted(unknown)}")
    includes = _path_list(options.get("include_prim_paths", ["/"]), "include_prim_paths")
    excludes = _path_list(options.get("exclude_prim_paths", []), "exclude_prim_paths", empty=True)
    include_invisible = _boolean(options, "include_invisible", False)
    exclude_articulations = _boolean(options, "exclude_articulations", True)

    def selected(path: str) -> bool:
        return any(_beneath(path, root) for root in includes) and not any(
            _beneath(path, root) for root in excludes
        )

    source_value = usd_path if usd_path is not None else options.get("file")
    if source_value is None:
        raise ValueError("usd_scene.file or --usd is required")
    source = Path(str(source_value)).expanduser()
    if usd_path is None and not source.is_absolute():
        source = spec.source_path.parent / source
    source = source.resolve()
    if source.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError(f"expected a USD/USDA/USDC environment: {source}")
    open_path = _usd_open_path(source)
    if not Path(open_path).is_file():
        raise FileNotFoundError(f"USD environment not found: {source}")
    stage = Usd.Stage.Open(open_path, load=Usd.Stage.LoadNone)
    if stage is None:
        raise ValueError(f"could not open USD environment: {source}")
    if exclude_articulations:
        # Authored APIs on unloaded payload roots are already available. Do
        # this before loading descendants, so excluded robots stay unloaded.
        excludes += [str(prim.GetPath()) for prim in Usd.PrimRange.Stage(
            stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate),
        ) if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    required_roots = list(includes)

    def apply_load_rules() -> None:
        rules = Usd.StageLoadRules.LoadNone()
        for root in required_roots:
            rules.LoadWithDescendants(root)
        for root in excludes:
            rules.Unload(root)
        stage.SetLoadRules(rules)

    apply_load_rules()
    metres = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(metres) or metres <= 0:
        raise ValueError("USD metersPerUnit must be finite and positive")
    up_axis = UsdGeom.GetStageUpAxis(stage)
    if up_axis not in (UsdGeom.Tokens.y, UsdGeom.Tokens.z):
        raise ValueError(f"unsupported USD upAxis: {up_axis}")
    basis = np.eye(3) if up_axis == UsdGeom.Tokens.z else np.array(
        ((1., 0., 0.), (0., 0., -1.), (0., 1., 0.))
    )
    time_value = options.get("time_code")
    if time_value is None:
        time = Usd.TimeCode.Default()
    else:
        value = float(time_value)
        if not math.isfinite(value):
            raise ValueError("usd_scene.time_code must be finite")
        time = Usd.TimeCode(value)
    for selected_root in includes:
        if selected_root != "/" and not stage.GetPrimAtPath(selected_root):
            raise ValueError(f"USD include prim is missing: {selected_root}")
    box_paths = options.get("box_prims", {})
    if not isinstance(box_paths, Mapping):
        raise ValueError("usd_scene.box_prims must map static-box names to USD paths")
    boxes_by_name = {box.name: box for box in spec.static_boxes}
    bindings: dict[str, str] = {}
    for name, value in box_paths.items():
        if name not in boxes_by_name:
            raise ValueError(f"USD box binding has unknown static box: {name}")
        prim_path = _path_list([value], f"box_prims.{name}")[0]
        if not stage.GetPrimAtPath(prim_path):
            raise ValueError(f"USD box binding prim is missing: {prim_path}")
        if any(_beneath(prim_path, other) or _beneath(other, prim_path) for other in bindings.values()):
            raise ValueError("USD box bindings must not overlap")
        bindings[str(name)] = prim_path

    bounds_options = options.get("box_bounds_prims", {})
    if not isinstance(bounds_options, Mapping):
        raise ValueError("usd_scene.box_bounds_prims must map bound box names to prim paths/patterns")
    bounds_patterns: dict[str, list[str]] = {}
    for name, value in bounds_options.items():
        if name not in bindings:
            raise ValueError(f"USD bounds selection needs a box_prims binding: {name}")
        patterns = [value] if isinstance(value, str) else value
        if (not isinstance(patterns, (list, tuple)) or not patterns
                or any(not isinstance(pattern, str) or not pattern.startswith("/")
                       for pattern in patterns)):
            raise ValueError(f"USD bounds selection for {name} needs absolute prim paths/patterns")
        bounds_patterns[str(name)] = list(patterns)

    purpose_values = options.get("purposes", ["default", "render"])
    if not isinstance(purpose_values, (list, tuple)) or not purpose_values:
        raise ValueError("usd_scene.purposes must be a non-empty list")
    purposes = {str(value) for value in purpose_values}
    if not purposes <= {"default", "render", "proxy", "guide"}:
        raise ValueError("invalid USD geometry purpose")
    prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    if exclude_articulations:
        discovered = [str(prim.GetPath()) for prim in prims
                      if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
                      and str(prim.GetPath()) not in excludes]
        if discovered:
            excludes += discovered
            apply_load_rules()
            prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))

    # PointInstancer prototypes may live outside the selected subtree. They
    # are dependencies of the selected instances, not unrelated obstacles.
    prototype_roots = {
        str(target) for prim in prims
        if selected(str(prim.GetPath())) and prim.IsA(UsdGeom.PointInstancer)
        for target in UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets()
        if not any(_beneath(str(target), root) for root in excludes)
    }
    extra_roots = sorted(root for root in prototype_roots
                         if not any(_beneath(root, selected_root) for selected_root in includes))
    if extra_roots:
        required_roots += extra_roots
        apply_load_rules()
        prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))

    # Missing payloads/references in imported geometry must not silently turn
    # an obstacle into empty space.  Explicitly excluded subtrees are allowed.
    for error in stage.GetCompositionErrors():
        error_path = str(error.rootSite.path)
        relevant = any(_beneath(error_path, root) or _beneath(root, error_path)
                       for root in required_roots)
        if relevant and not any(_beneath(error_path, root) for root in excludes):
            raise ValueError(f"USD environment has an unresolved composition dependency: {error}")

    xforms = UsdGeom.XformCache(time)
    groups: dict[str, tuple[str, list[tuple[float, float, float]], list[tuple[int, int, int]]]] = {}
    bounds_points: dict[str, list[np.ndarray]] = {}

    def visible(prim: object) -> bool:
        imageable = UsdGeom.Imageable(prim)
        return (not imageable or (imageable.ComputePurpose() in purposes and
                (include_invisible or imageable.ComputeVisibility(time) != UsdGeom.Tokens.invisible)))

    def add_geometry(prim: object, matrix: np.ndarray, identity: str) -> None:
        if not visible(prim):
            return
        geometry = _local_geometry(prim, time, UsdGeom)
        if geometry is None:
            return
        points, triangles = geometry
        transformed = (points @ matrix[:3, :3] + matrix[3, :3]) @ basis.T * metres
        if not np.all(np.isfinite(transformed)):
            raise ValueError(f"non-finite USD transform/points: {prim.GetPath()}")
        triangles = _nondegenerate_triangles(transformed, triangles, str(prim.GetPath()))
        if np.linalg.det(matrix[:3, :3]) < 0:
            triangles = [(a, c, b) for a, b, c in triangles]
        logical_name = next((name for name, root in bindings.items() if _beneath(identity, root)),
                            "usd:" + identity)
        source_path = bindings.get(logical_name, identity)
        if logical_name in bounds_patterns and any(
            fnmatchcase(identity, pattern) or _beneath(identity, pattern)
            for pattern in bounds_patterns[logical_name]
        ):
            bounds_points.setdefault(logical_name, []).append(transformed)
        if logical_name not in groups:
            groups[logical_name] = (source_path, [], [])
        _, vertices, faces = groups[logical_name]
        offset = len(vertices)
        vertices.extend(tuple(float(x) for x in point) for point in transformed)
        faces.extend((a + offset, b + offset, c + offset) for a, b, c in triangles)

    instancers = [prim for prim in prims if prim.IsA(UsdGeom.PointInstancer)]
    prototypes = {str(target) for prim in instancers
                  for target in UsdGeom.PointInstancer(prim).GetPrototypesRel().GetTargets()}
    for prim in prims:
        prim_path = str(prim.GetPath())
        if not selected(prim_path) or any(_beneath(prim_path, root) for root in prototypes):
            continue
        if prim.IsA(UsdGeom.PointInstancer):
            if not visible(prim):
                continue
            instancer = UsdGeom.PointInstancer(prim)
            targets = instancer.GetPrototypesRel().GetTargets()
            indices = instancer.GetProtoIndicesAttr().Get(time)
            if indices is None or not targets:
                raise ValueError(f"invalid PointInstancer prototypes: {prim_path}")
            transforms = instancer.ComputeInstanceTransformsAtTime(
                time, time, UsdGeom.PointInstancer.IncludeProtoXform,
                UsdGeom.PointInstancer.IgnoreMask,
            )
            mask = list(instancer.ComputeMaskAtTime(time))
            if len(transforms) != len(indices):
                raise ValueError(f"invalid PointInstancer transforms: {prim_path}")
            for index, (prototype_index, instance_transform) in enumerate(zip(indices, transforms)):
                if mask and not mask[index]:
                    continue
                if prototype_index < 0 or prototype_index >= len(targets):
                    raise ValueError(f"invalid PointInstancer prototype index: {prim_path}")
                prototype = stage.GetPrimAtPath(targets[prototype_index])
                if not prototype:
                    raise ValueError(f"missing PointInstancer prototype: {targets[prototype_index]}")
                for child in Usd.PrimRange(prototype, Usd.TraverseInstanceProxies()):
                    if child.IsA(UsdGeom.PointInstancer):
                        raise ValueError("nested PointInstancer is unsupported; expand it to Mesh instances")
                    child_path = str(child.GetPath())
                    if any(_beneath(child_path, root) for root in excludes):
                        continue
                    relative, _ = xforms.ComputeRelativeTransform(child, prototype)
                    matrix = np.asarray(relative) @ np.asarray(instance_transform) @ np.asarray(
                        xforms.GetLocalToWorldTransform(prim)
                    )
                    identity = prim_path + f"/instance_{index}" + child_path[len(str(prototype.GetPath())):]
                    add_geometry(child, matrix, identity)
        else:
            add_geometry(prim, np.asarray(xforms.GetLocalToWorldTransform(prim)), prim_path)

    if not groups:
        raise ValueError("USD selection contains no supported visible environment geometry")
    missing = set(bindings) - set(groups)
    if missing:
        raise ValueError(f"USD box bindings contain no imported geometry: {sorted(missing)}")
    missing_bounds = set(bounds_patterns) - set(bounds_points)
    if missing_bounds:
        raise ValueError(f"USD bounds selection contains no imported geometry: {sorted(missing_bounds)}")
    meshes = tuple(MeshObstacle(
        name=name, role=boxes_by_name[name].role if name in boxes_by_name else "obstacle",
        vertices_m=tuple(vertices), triangles=tuple(triangles), source_prim_path=path,
    ) for name, (path, vertices, triangles) in sorted(groups.items()))
    reserved = {"pallet", "representative_tote", "ur20_pedestal", "sr12ia_pedestal"}
    if any(mesh.name in reserved for mesh in meshes):
        raise ValueError("USD obstacle name conflicts with a generated process object")
    proxies = []
    for box in spec.static_boxes:
        if box.name in bindings:
            mesh = next(mesh for mesh in meshes if mesh.name == box.name)
            matrix = np.asarray(xforms.GetLocalToWorldTransform(stage.GetPrimAtPath(bindings[box.name])))
            axes = basis @ matrix[:3, :3].T
            vertical_axis = 2 if up_axis == UsdGeom.Tokens.z else 1
            horizontal_axes = [index for index in range(3) if index != vertical_axis]
            if (not np.all(np.isfinite(axes)) or abs(axes[2, vertical_axis]) < 1e-12
                    or not np.allclose(axes[2, horizontal_axes], 0., atol=1e-8)
                    or not np.allclose(axes[:2, vertical_axis], 0., atol=1e-8)):
                raise ValueError(f"task support {box.name} must remain horizontal; Mesh obstacles may tilt")
            yaw = math.atan2(axes[1, 0], axes[0, 0])
            rotation = np.array(((math.cos(yaw), -math.sin(yaw), 0.),
                                 (math.sin(yaw), math.cos(yaw), 0.), (0., 0., 1.)))
            vertices = (np.concatenate(bounds_points[box.name]) if box.name in bounds_patterns
                        else np.asarray(mesh.vertices_m))
            local = vertices @ rotation
            lo, hi = local.min(axis=0), local.max(axis=0)
            size = hi - lo
            if np.any(size <= 0.):
                raise ValueError(f"task support {box.name} needs non-zero XYZ bounds")
            center = (lo + hi) / 2. @ rotation.T
            box = replace(box, center_m=tuple(float(x) for x in center),
                          size_m=tuple(float(x) for x in size), yaw_deg=math.degrees(yaw))
        proxies.append(replace(box, collision_enabled=False))
    tote = spec.tote
    if tote.motion_support_box in bindings:
        support = next(box for box in proxies if box.name == tote.motion_support_box)
        tote = replace(tote, support_surface_z_m=support.center_m[2] + support.size_m[2] / 2.)
    return replace(spec, static_boxes=tuple(proxies), static_meshes=meshes, tote=tote)


def _usd_open_path(source: Path) -> str:
    """Use the original directory's NTFS alias to keep collected references short.

    An ASCII root path can still make nested dependencies exceed MAX_PATH.
    Request its 8.3 spelling on Windows as well; copying the root layer would
    change the anchor of relative references and is deliberately avoided.
    """

    value = str(source)
    if os.name != "nt":
        return value
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_short_path = kernel32.GetShortPathNameW
    get_short_path.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
    get_short_path.restype = wintypes.DWORD
    # The lookup itself must work even if the root file exceeds MAX_PATH.
    if value.startswith("\\\\?\\"):
        lookup = value
    elif value.startswith("\\\\"):
        lookup = "\\\\?\\UNC\\" + value[2:]
    else:
        lookup = "\\\\?\\" + value
    required = get_short_path(lookup, None, 0)
    if not required:
        # Keep the normal missing-file check for an absent ASCII source.
        if value.isascii():
            return value
        raise OSError(ctypes.get_last_error(), "could not resolve USD NTFS path", value)
    buffer = ctypes.create_unicode_buffer(required)
    written = get_short_path(lookup, buffer, required)
    if not written or written >= required:
        raise OSError(ctypes.get_last_error(), "could not resolve USD NTFS path", value)
    candidate = buffer.value
    if candidate.startswith("\\\\?\\UNC\\"):
        candidate = "\\\\" + candidate[8:]
    elif candidate.startswith("\\\\?\\"):
        candidate = candidate[4:]
    if not candidate.isascii() or len(candidate) >= 260:
        raise OSError(
            "OpenUSD needs a short ASCII Windows path; enable NTFS 8.3 names "
            "or move the collected USD directory and its assets to a shorter path"
        )
    return candidate


def _beneath(path: str, root: str) -> bool:
    return root == "/" or path == root or path.startswith(root.rstrip("/") + "/")


def _path_list(value: object, name: str, *, empty: bool = False) -> list[str]:
    from pxr import Sdf

    if not isinstance(value, (list, tuple)) or (not value and not empty):
        raise ValueError(f"usd_scene.{name} must be a list of absolute prim paths")
    result = []
    for item in value:
        path = Sdf.Path(str(item))
        if not path.IsAbsolutePath() or not (path.IsPrimPath() or str(path) == "/"):
            raise ValueError(f"invalid USD prim path in {name}: {item}")
        result.append(str(path))
    return result


def _boolean(options: Mapping[str, object], key: str, default: bool) -> bool:
    value = options.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"usd_scene.{key} must be a boolean")
    return value


def _local_geometry(prim: object, time: object, geom: object):
    if prim.IsA(geom.Mesh):
        if any(prim.GetAttribute(name).HasAuthoredValueOpinion()
               for name in ("primvars:skel:jointIndices", "skel:blendShapes")
               if prim.HasAttribute(name)):
            raise ValueError(f"skinned/blend-shape USD Mesh must be baked before import: {prim.GetPath()}")
        mesh = geom.Mesh(prim)
        points = mesh.GetPointsAttr().Get(time)
        counts = mesh.GetFaceVertexCountsAttr().Get(time)
        indices = mesh.GetFaceVertexIndicesAttr().Get(time)
        if points is None or counts is None or indices is None:
            raise ValueError(f"missing USD mesh topology at selected time: {prim.GetPath()}")
        points = np.asarray(points, dtype=float).reshape((-1, 3))
        counts, indices = list(counts), list(indices)
        if not len(points) or not np.all(np.isfinite(points)) or sum(counts) != len(indices) or any(
            count < 3 for count in counts
        ) or any(index < 0 or index >= len(points) for index in indices):
            raise ValueError(f"invalid USD mesh topology: {prim.GetPath()}")
        holes = set(mesh.GetHoleIndicesAttr().Get(time) or [])
        if any(index < 0 or index >= len(counts) for index in holes):
            raise ValueError(f"invalid USD mesh hole indices: {prim.GetPath()}")
        triangles, cursor = [], 0
        for face_index, count in enumerate(counts):
            face = indices[cursor:cursor + count]
            cursor += count
            if face_index not in holes:
                triangles.extend(_triangulate(points, face, str(prim.GetPath())))
        if mesh.GetOrientationAttr().Get(time) == geom.Tokens.leftHanded:
            triangles = [(a, c, b) for a, b, c in triangles]
        return points, _nondegenerate_triangles(points, triangles, str(prim.GetPath()))
    if prim.IsA(geom.Cube):
        size = _dimension(geom.Cube(prim).GetSizeAttr(), time, prim)
        points = np.array(((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                           (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)), dtype=float) * size / 2.
        quads = ((0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7))
        return points, [(f[0], f[1], f[2]) for f in quads] + [(f[0], f[2], f[3]) for f in quads]
    if prim.IsA(geom.Plane):
        plane = geom.Plane(prim)
        width = _dimension(plane.GetWidthAttr(), time, prim)
        length = _dimension(plane.GetLengthAttr(), time, prim)
        points = np.array(((-width / 2., -length / 2., 0.),
                           (width / 2., -length / 2., 0.),
                           (width / 2., length / 2., 0.),
                           (-width / 2., length / 2., 0.)))
        axis = str(plane.GetAxisAttr().Get(time))
        if axis == "X":
            points = points[:, (2, 0, 1)]
        elif axis == "Y":
            points = points[:, (1, 2, 0)]
        elif axis != "Z":
            raise ValueError(f"unsupported USD plane axis: {axis}")
        return points, [(0, 1, 2), (0, 2, 3)]
    for schema in (geom.Sphere, geom.Cylinder, geom.Cone, geom.Capsule):
        if prim.IsA(schema):
            shape = schema(prim)
            radius = _dimension(shape.GetRadiusAttr(), time, prim)
            height = 0. if schema == geom.Sphere else _dimension(shape.GetHeightAttr(), time, prim)
            if schema == geom.Sphere:
                rings = [(radius * math.sin(t), radius * math.cos(t))
                         for t in np.linspace(0., math.pi, 17)]
            elif schema == geom.Capsule:
                rings = [(radius * math.sin(t), height / 2. + radius * math.cos(t))
                         for t in np.linspace(0., math.pi / 2., 9)]
                rings += [(radius * math.sin(t), -height / 2. + radius * math.cos(t))
                          for t in np.linspace(math.pi / 2., math.pi, 9)]
            elif schema == geom.Cone:
                rings = [(0., height / 2.), (radius, -height / 2.), (0., -height / 2.)]
            else:
                rings = [(0., height / 2.), (radius, height / 2.),
                         (radius, -height / 2.), (0., -height / 2.)]
            points, triangles = _revolved_surface(rings)
            if schema != geom.Sphere:
                axis = str(shape.GetAxisAttr().Get(time))
                if axis == "X":
                    points = points[:, (2, 0, 1)]
                elif axis == "Y":
                    points = points[:, (1, 2, 0)]
                elif axis != "Z":
                    raise ValueError(f"unsupported USD primitive axis: {axis}")
            return points, triangles
    if prim.IsA(geom.Gprim):
        raise ValueError(f"unsupported USD geometry {prim.GetTypeName()} at {prim.GetPath()}; convert it to Mesh")
    return None


def _dimension(attribute: object, time: object, prim: object) -> float:
    value = attribute.Get(time)
    if value is None or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"invalid USD primitive dimension: {prim.GetPath()}")
    return float(value)


def _nondegenerate_triangles(points: np.ndarray, triangles, path: str):
    faces = points[np.asarray(triangles, dtype=np.intp).reshape((-1, 3))]
    areas = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1)
    result = [tuple(int(i) for i in triangle) for triangle, area in zip(triangles, areas)
              if math.isfinite(float(area)) and area > 0.]
    if not result:
        raise ValueError(f"USD geometry has no non-degenerate faces: {path}")
    return result


def _revolved_surface(rings: Sequence[tuple[float, float]]):
    segments = 32
    vertices, levels = [], []
    for radius, height in rings:
        if abs(radius) < 1e-12:
            levels.append([len(vertices)])
            vertices.append((0., 0., height))
        else:
            levels.append(list(range(len(vertices), len(vertices) + segments)))
            vertices.extend((radius * math.cos(t), radius * math.sin(t), height)
                            for t in np.linspace(0., 2. * math.pi, segments, endpoint=False))
    triangles = []
    for top, bottom in zip(levels, levels[1:]):
        for index in range(segments):
            nxt = (index + 1) % segments
            if len(top) == 1:
                triangles.append((top[0], bottom[index], bottom[nxt]))
            elif len(bottom) == 1:
                triangles.append((top[index], bottom[0], top[nxt]))
            else:
                triangles.extend(((top[index], bottom[index], bottom[nxt]),
                                  (top[index], bottom[nxt], top[nxt])))
    return np.asarray(vertices, dtype=float), triangles


def _triangulate(points: np.ndarray, face: Sequence[int], path: str):
    """Ear clipping preserves concave polygon boundaries (a fan does not)."""
    vertices = list(face)
    if len(vertices) == 3:
        return [tuple(vertices)]
    polygon = points[vertices]
    normal = np.sum(np.cross(polygon, np.roll(polygon, -1, axis=0)), axis=0)
    if np.linalg.norm(normal) <= 1e-15:
        raise ValueError(f"degenerate USD polygon at {path}")
    projected = np.delete(polygon, int(np.argmax(np.abs(normal))), axis=1)
    area = np.sum(projected[:, 0] * np.roll(projected[:, 1], -1)
                  - projected[:, 1] * np.roll(projected[:, 0], -1))
    winding = 1. if area > 0 else -1.
    tolerance = max(float(np.ptp(projected, axis=0).max()) ** 2 * 1e-12, 1e-20)

    def cross(a, b, c):
        ab, ac = b - a, c - a
        return float(ab[0] * ac[1] - ab[1] * ac[0]) * winding

    pending, result = list(range(len(vertices))), []
    while len(pending) > 3:
        for index, current in enumerate(pending):
            previous, following = pending[index - 1], pending[(index + 1) % len(pending)]
            a, b, c = projected[[previous, current, following]]
            turn = cross(a, b, c)
            if abs(turn) <= tolerance:
                del pending[index]
                break
            if turn < 0:
                continue
            if any(cross(a, b, projected[p]) >= -tolerance and
                   cross(b, c, projected[p]) >= -tolerance and
                   cross(c, a, projected[p]) >= -tolerance
                   for p in pending if p not in (previous, current, following)):
                continue
            result.append((vertices[previous], vertices[current], vertices[following]))
            del pending[index]
            break
        else:
            raise ValueError(f"cannot triangulate USD polygon at {path}; check for self-intersections")
    if len(pending) == 3:
        result.append(tuple(vertices[index] for index in pending))
    return result
