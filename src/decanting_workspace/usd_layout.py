"""Import cell layout as small oriented bounds, independently of task SKU data.

The source is read without modifying its layers. Robot USD geometry supplies
only the mounting position; the configured URDF remains the robot model.
"""

from __future__ import annotations

from collections import Counter
from fnmatch import fnmatchcase
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from .layout_models import (
    LayoutBox, LayoutFrame, LayoutNode, LayoutReferenceDimensions, LayoutRobot, LayoutScene,
)
from .models import RobotSpec
from .usd_scene import _beneath, _path_list, _usd_open_path


DEFAULT_GROUPS = tuple('/World/' + name for name in (
    'ConveyorGroup', 'UncaseGroup', 'LoadGroup', 'PalletGroup', 'ToteGroup', 'GroundPlane',
))
DEFAULT_ROBOT_BASES = {
    'ur20': '/World/LoadGroup/RB/ur20/base_link',
    'sr12ia': '/World/UncaseGroup/RB/sr12ia/robot_base/base_link',
}
DEFAULT_IGNORED_NAMES = (
    '*sku*', 'FlatBox*', 'SwedishStyleJelly', 'SquidAndPeanut', 'HalfCoatedGlove',
    'BlackMaskLarge', 'PlateLarge',
)
LEGACY_PROXY_NAMES = {'Conv1', 'Conv2', 'Conv3', 'Conv4A', 'Conv4B', 'Conv5A', 'Conv5B'}


def load_usd_layout(
    source: str | Path,
    robot_specs: Mapping[str, RobotSpec],
    options: Mapping[str, object] | None = None,
) -> LayoutScene:
    """Read group/unit hierarchy, simplified fixed geometry and robot positions.

    Conveyor bodies often combine the legs with their side panels in one mesh.
    Their roller/belt meshes are therefore the selected volume source. All
    original intersections are retained; importing is not a placement check.
    """
    from pxr import Usd, UsdGeom

    config = dict(options or {})
    allowed = {'groups', 'exclude_prim_paths', 'ignore_name_patterns', 'robot_base_prims',
               'include_invisible', 'time_code'}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f'unknown USD layout options: {sorted(unknown)}')
    groups = _path_list(config.get('groups', list(DEFAULT_GROUPS)), 'groups')
    excludes = _path_list(config.get('exclude_prim_paths', []), 'exclude_prim_paths', empty=True)
    patterns = config.get('ignore_name_patterns', ())
    if not isinstance(patterns, (list, tuple)) or any(not isinstance(p, str) for p in patterns):
        raise ValueError('ignore_name_patterns must be a list of name patterns')
    patterns = [p.lower() for p in (*DEFAULT_IGNORED_NAMES, *patterns)]
    include_invisible = config.get('include_invisible', True)
    if not isinstance(include_invisible, bool):
        raise ValueError('include_invisible must be a boolean')
    time_value = config.get('time_code')
    if time_value is not None and not math.isfinite(float(time_value)):
        raise ValueError('time_code must be finite')
    time = Usd.TimeCode.Default() if time_value is None else Usd.TimeCode(float(time_value))
    source_path = Path(source).expanduser().resolve()
    open_path = _usd_open_path(source_path)
    if not Path(open_path).is_file():
        raise FileNotFoundError(f'USD layout not found: {source_path}')
    stage = Usd.Stage.Open(open_path, load=Usd.Stage.LoadNone)
    if stage is None:
        raise ValueError(f'could not open USD layout: {source_path}')

    def ignored(prim) -> bool:
        return (any(fnmatchcase(prim.GetName().lower(), pattern) for pattern in patterns)
                or str(prim.GetAttribute('userProperties:layoutRole').Get()).lower() == 'sku')

    # Discover sample payload roots before loading them. Root examples outside
    # the selected cell groups never become dependencies of this scene.
    def collect_exclusions():
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
            if ignored(prim) and str(prim.GetPath()) not in excludes:
                excludes.append(str(prim.GetPath()))

    def apply_rules():
        rules = Usd.StageLoadRules.LoadNone()
        for root in groups:
            rules.LoadWithDescendants(root)
        for path in excludes:
            rules.Unload(path)
        stage.SetLoadRules(rules)

    collect_exclusions()
    apply_rules()
    collect_exclusions()
    apply_rules()
    for path in groups:
        if not stage.GetPrimAtPath(path):
            raise ValueError(f'USD layout group is missing: {path}')
    for error in stage.GetCompositionErrors():
        path = str(error.rootSite.path)
        if (any(_beneath(path, root) or _beneath(root, path) for root in groups)
                and not any(_beneath(path, root) for root in excludes)):
            raise ValueError(f'USD layout has an unresolved composition dependency: {error}')
    metres = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(metres) or metres <= 0:
        raise ValueError('USD metersPerUnit must be finite and positive')
    up = UsdGeom.GetStageUpAxis(stage)
    if up not in ('Y', 'Z'):
        raise ValueError(f'unsupported USD upAxis: {up}')
    basis = np.eye(3) if up == 'Z' else np.array(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.)))
    xforms = UsdGeom.XformCache(time)
    prims = [prim for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
             if any(_beneath(str(prim.GetPath()), root) for root in groups)
             and not any(_beneath(str(prim.GetPath()), root) for root in excludes)]

    def world_matrix(prim):
        usd = np.asarray(xforms.GetLocalToWorldTransform(prim), dtype=float)
        matrix = np.eye(4)
        matrix[:3, :3] = basis @ usd[:3, :3].T
        matrix[:3, 3] = basis @ usd[3, :3] * metres
        return matrix

    base_paths = dict(DEFAULT_ROBOT_BASES)
    overrides = config.get('robot_base_prims', {})
    if not isinstance(overrides, Mapping):
        raise ValueError('robot_base_prims must map robot names to base-link USD paths')
    base_paths.update({str(name): _path_list([value], f'robot_base_prims.{name}')[0]
                       for name, value in overrides.items()})
    robot_roots = {}
    for name in robot_specs:
        if name not in base_paths:
            raise ValueError(f'USD base-link mapping is missing for robot: {name}')
        if (not any(_beneath(base_paths[name], root) for root in groups)
                or any(_beneath(base_paths[name], path) for path in excludes)):
            raise ValueError(f'USD robot base-link must belong to a selected group: {base_paths[name]}')
        base = stage.GetPrimAtPath(base_paths[name])
        if not base:
            raise ValueError(f'USD robot base-link prim is missing: {base_paths[name]}')
        # The base path can have a robot_base wrapper for FANUC.
        root = base.GetParent()
        if root.GetName() == 'robot_base':
            root = root.GetParent()
        robot_roots[name] = str(root.GetPath())

    def is_robot(path):
        return any(_beneath(path, root) for root in robot_roots.values())

    def physical_assembly(path):
        parts = path.split('/')
        for index, part in enumerate(parts):
            if part.lower().startswith(('rsd455', 'small_klt')):
                return '/'.join(parts[:index + 1])
        return None

    def tote_reference_root(path):
        # Tote meshes supply dimensions only. They never own editable nodes or
        # simplified environment solids, including totes on supply conveyors.
        parts = path.split('/')
        for index, part in enumerate(parts):
            candidate = '/'.join(parts[:index + 1])
            if not candidate:
                continue
            prim = stage.GetPrimAtPath(candidate)
            role = prim.GetAttribute('userProperties:layoutRole').Get() if prim else None
            if (part.lower().startswith('small_klt') or part.lower() in {'tote', 'tote_container'}
                    or str(role).lower() == 'tote'):
                return candidate
        return None

    def supply_tote_group(path):
        return any(part.lower() == 'tote_supply' for part in path.split('/'))

    def helper_geometry(path):
        parts = path.split('/')
        return ('OmniverseKitViewportCameraMesh' in parts
                or ('GroundPlane' not in parts and any(part.lower() in
                    {'collision', 'collisions'} for part in parts)))

    nodes = {}
    def add_node(path, kind='element'):
        if tote_reference_root(path) is not None or supply_tote_group(path):
            return
        if path in nodes:
            return
        prim = stage.GetPrimAtPath(path)
        if not prim:
            return
        parent = next((candidate for candidate in sorted(nodes, key=len, reverse=True)
                       if candidate != path and _beneath(path, candidate)), None)
        matrix = world_matrix(prim)
        nodes[path] = LayoutNode(
            prim_path=path, parent_path=parent,
            translation_m=_tuple(matrix[:3, 3]), rotation_xyzw=_quaternion(_rotation(matrix[:3, :3])),
            kind=kind,
        )

    for group in groups:
        add_node(group, 'group')
    # Major units are Xform/Scope assemblies, excluding geometry/material
    # internals. Geometry leaves also remain independently translatable.
    internal_names = {'Geometry', 'Looks', 'Visuals', 'visuals', 'Collisions', 'collisions',
                      'Belt', 'BeltRamp', 'CameraMode', 'RSD455'}
    for prim in sorted(prims, key=lambda p: str(p.GetPath()).count('/')):
        path = str(prim.GetPath())
        assembly = physical_assembly(path)
        if (is_robot(path) or prim.GetName() in LEGACY_PROXY_NAMES or helper_geometry(path)
                or (assembly is not None and path != assembly)):
            continue
        if (prim.GetTypeName() in ('Xform', 'Scope', '') and prim.GetName() not in internal_names
                and not any(tag in path for tag in ('/Looks/', '/visuals/', '/collisions/'))):
            add_node(path, 'frame' if prim.GetName().endswith('Frame') else 'element')
    for root in robot_roots.values():
        add_node(root, 'robot')

    def owner(path):
        return next(candidate for candidate in sorted(nodes, key=len, reverse=True)
                    if _beneath(path, candidate))

    frame_prims = [prim for prim in prims
                   if prim.GetName().endswith('Frame') and prim.IsA(UsdGeom.Xform)
                   and tote_reference_root(str(prim.GetPath())) is None
                   and not supply_tote_group(str(prim.GetPath()))]
    frame_names = Counter(prim.GetName() for prim in frame_prims)
    frames = []
    for prim in frame_prims:
        path = str(prim.GetPath())
        matrix = world_matrix(prim)
        frames.append(LayoutFrame(
            name=prim.GetName() if frame_names[prim.GetName()] == 1 else path,
            node_path=owner(path), translation_m=_tuple(matrix[:3, 3]),
            rotation_xyzw=_quaternion(_rotation(matrix[:3, :3])), source_prim_path=path,
        ))
    robots = []
    for name, robot in robot_specs.items():
        path = base_paths[name]
        matrix = world_matrix(stage.GetPrimAtPath(path))
        robots.append(LayoutRobot(
            name=name, node_path=robot_roots[name], translation_m=_tuple(matrix[:3, 3]),
            yaw_deg=robot.nominal_base.yaw_deg, urdf_path=robot.urdf_path,
            nominal_q=robot.nominal_q, source_prim_path=path,
        ))

    buckets = {}
    reference_buckets = {}
    for prim in prims:
        path = str(prim.GetPath())
        if is_robot(path) or prim.GetName() in LEGACY_PROXY_NAMES or helper_geometry(path):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable and not include_invisible and imageable.ComputeVisibility(time) == 'invisible':
            continue
        if not prim.IsA(UsdGeom.Boundable):
            continue
        if prim.IsA(UsdGeom.PointInstancer):
            raise ValueError(f'expand PointInstancer before simplifying a cell layout: {path}')
        tote_root = tote_reference_root(path)
        if tote_root is not None:
            points = _world_points(prim, world_matrix(prim), metres, time, UsdGeom)
            if points is None or not len(points):
                continue
            if not np.all(np.isfinite(points)):
                raise ValueError(f'USD layout points must be finite: {path}')
            if tote_root not in reference_buckets:
                reference_buckets[tote_root] = {
                    'points': [],
                    'rotation': _rotation(world_matrix(stage.GetPrimAtPath(tote_root))[:3, :3]),
                }
            reference_buckets[tote_root]['points'].append(points)
            continue
        name = prim.GetName()
        conveyor = any('conveyorbelt' in part.lower() for part in path.split('/'))
        roller = 'roller' in name.lower()
        belt = '_belt' in name.lower()
        if conveyor and not (roller or belt):
            continue
        points = _world_points(prim, world_matrix(prim), metres, time, UsdGeom)
        if points is None or not len(points):
            continue
        if not np.all(np.isfinite(points)):
            raise ValueError(f'USD layout points must be finite: {path}')
        node_path = owner(path)
        assembly = physical_assembly(path)
        if roller:
            key = node_path + '/roller_surface'
            source_prim = node_path
        elif assembly is not None:
            key, source_prim, node_path = assembly, assembly, assembly
        else:
            key, source_prim = path, path
            add_node(path, 'surface' if conveyor else 'geometry')
            node_path = path
        if key not in buckets:
            role = ('conveyor' if conveyor else 'ground' if '/GroundPlane/' in path
                    else 'camera' if assembly and 'rsd455' in assembly.lower()
                    else 'tote' if assembly and 'small_KLT' in assembly
                    else 'pallet' if '/pallet/' in path.lower() else 'obstacle')
            reference_prim = stage.GetPrimAtPath(node_path) if roller or assembly else prim
            buckets[key] = {'node': node_path, 'points': [], 'rotation': _rotation(world_matrix(reference_prim)[:3, :3]),
                            'role': role, 'source': source_prim, 'pca': belt}
        buckets[key]['points'].append(points)
    boxes = []
    for name, data in sorted(buckets.items()):
        points = np.concatenate(data['points'])
        rotation = _principal_rotation(points, data['rotation']) if data['pca'] else data['rotation']
        local = points @ rotation
        lo, hi = local.min(axis=0), local.max(axis=0)
        # Infinitely thin floor faces need a finite collision proxy extending
        # below their visible surface; ordinary thin meshes retain their size.
        size = hi - lo
        if np.any(size < 1e-8):
            if data['role'] != 'ground':
                for axis in np.flatnonzero(size < 1e-8):
                    lo[axis] -= .001
                    hi[axis] += .001
            else:
                for axis in np.flatnonzero(size < 1e-8):
                    if rotation[2, axis] >= 0:
                        lo[axis] -= .02
                    else:
                        hi[axis] += .02
        center = ((lo + hi) / 2.) @ rotation.T
        boxes.append(LayoutBox(
            name=name, node_path=data['node'], center_m=_tuple(center), size_m=_tuple(hi-lo),
            rotation_xyzw=_quaternion(rotation), role=data['role'], source_prim_path=data['source'],
        ))
    reference_dimensions = []
    for path, data in sorted(reference_buckets.items()):
        points = np.concatenate(data['points']) @ data['rotation']
        size = points.max(axis=0) - points.min(axis=0)
        reference_dimensions.append(LayoutReferenceDimensions(
            name=path, size_m=_tuple(size), source_prim_path=path, role='tote',
        ))
    if not boxes and not reference_dimensions:
        raise ValueError('USD layout contains no supported simplified environment geometry')
    return LayoutScene(source_path=source_path, nodes=tuple(nodes.values()), boxes=tuple(boxes),
                       robots=tuple(robots), frames=tuple(frames),
                       reference_dimensions=tuple(reference_dimensions))


def _tuple(values):
    return tuple(float(value) for value in values)


def _rotation(linear):
    left, _, right = np.linalg.svd(linear)
    if np.linalg.det(left @ right) < 0:
        left[:, -1] *= -1
    return left @ right


def _quaternion(rotation):
    from pxr import Gf
    quaternion = Gf.Matrix3d(*rotation.T.reshape(-1).tolist()).ExtractRotation().GetQuat()
    xyz = quaternion.GetImaginary()
    values = np.asarray((*xyz, quaternion.GetReal()), dtype=float)
    values /= np.linalg.norm(values)
    if values[3] < 0:
        values *= -1
    return _tuple(values)


def _principal_rotation(points, reference):
    """Preserve inclined belt planes encoded in vertices, not just transforms."""
    centered = points - points.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    axes = axes.T
    # Axis order/signs follow the authored reference as closely as possible.
    from itertools import permutations
    order = max(permutations(range(3)), key=lambda indices:
                sum(abs(np.dot(axes[:, index], reference[:, column]))
                    for column, index in enumerate(indices)))
    rotation = axes[:, order].copy()
    for column in range(3):
        if np.dot(rotation[:, column], reference[:, column]) < 0:
            rotation[:, column] *= -1
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    return rotation


def _world_points(prim, matrix, metres, time, geom):
    if prim.IsA(geom.Mesh):
        points = geom.Mesh(prim).GetPointsAttr().Get(time)
        if points is None:
            raise ValueError(f'USD Mesh points are missing: {prim.GetPath()}')
        local = np.asarray(points, dtype=float)
    else:
        boundable = geom.Boundable(prim)
        extent = geom.Boundable.ComputeExtentFromPlugins(boundable, time)
        if extent is None:
            extent = boundable.GetExtentAttr().Get(time)
        if extent is None:
            return None
        lo, hi = np.asarray(extent, dtype=float)
        local = np.array([(x, y, z) for x in (lo[0], hi[0])
                          for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    if local.size == 0:
        return None
    if local.ndim != 2 or local.shape[1] != 3:
        raise ValueError(f'USD points need XYZ triples: {prim.GetPath()}')
    return (local * metres) @ matrix[:3, :3].T + matrix[:3, 3]
