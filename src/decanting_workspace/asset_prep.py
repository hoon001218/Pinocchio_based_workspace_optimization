"""Prepare an authorized Isaac Sim SR-12iA mesh bundle for Pinocchio.

A caller may either supply a local ``payloads/geometries.usd`` or explicitly
accept the linked FANUC terms before downloading the pinned NVIDIA asset.  The
download is content-addressed and rejected unless its SHA-256 matches.  OpenUSD
is an optional, one-time conversion dependency; importing this module does not
import ``pxr``.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from .paths import repository_path


ISAAC_SIM_VERSION = "6.0"
CONVERTER_SCHEMA_VERSION = 3
NVIDIA_SR12IA_GEOMETRIES_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/"
    "Isaac/6.0/Isaac/Robots_Multiphysics/Fanuc/sr12ia/payloads/geometries.usd"
)
NVIDIA_SR12IA_GEOMETRIES_SHA256 = (
    "1313afdb54a8c369a37e910c63007c6295cecd3d804226580fbef85c42e09740"
)
FANUC_3D_CONTENT_SHARING_AGREEMENT_URL = (
    "https://docs.isaacsim.omniverse.nvidia.com/6.0.0/_downloads/"
    "8ea27bd6ad258f4242d675e67e8f7b58/"
    "3D%20Content%20Sharing%20Agreement_FANUC.pdf"
)

_URDF_FILENAME = "sr12ia_mesh.urdf"
_PROVENANCE_FILENAME = "sr12ia_mesh.provenance.json"
_VISUAL_RPY = "1.57079632679 0 0"
_ZERO_XYZ = "0 0 0"


@dataclass(frozen=True)
class _MeshExport:
    link: str
    role: str
    prim_name: str
    filename: str

    @property
    def prim_path(self) -> str:
        return f"/Geometries/{self.prim_name}"


_MESH_EXPORTS = (
    _MeshExport("base_link", "visual", "mesh", "base_link_visual.obj"),
    _MeshExport("base_link", "collision", "mesh_1", "base_link_collision.obj"),
    _MeshExport("J1_link", "visual", "mesh_2", "j1_link_visual.obj"),
    _MeshExport("J1_link", "collision", "mesh_3", "j1_link_collision.obj"),
    _MeshExport("J2_link", "visual", "mesh_4", "j2_link_visual.obj"),
    _MeshExport("J2_link", "collision", "mesh_5", "j2_link_collision.obj"),
    _MeshExport("J3_link", "visual", "mesh_6", "j3_link_visual.obj"),
    _MeshExport("J3_link", "collision", "mesh_7", "j3_link_collision.obj"),
)


@dataclass(frozen=True)
class _ObjMesh:
    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, ...], ...]


def default_sr12ia_output_urdf() -> Path:
    """Return the repository-local, ignored cache path for the generated URDF."""

    return repository_path(
        ".cache", "robot_assets", "sr12ia", _URDF_FILENAME
    )


def validate_prepared_sr12ia_asset(
    urdf_path: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Validate the generated cache contract and return its provenance."""

    urdf = Path(urdf_path or default_sr12ia_output_urdf()).resolve()
    if not urdf.is_file():
        raise FileNotFoundError(f"prepared SR-12iA URDF not found: {urdf}")
    provenance_path = urdf.with_name(_PROVENANCE_FILENAME)
    if not provenance_path.is_file():
        raise ValueError(f"prepared SR-12iA provenance is missing: {provenance_path}")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"invalid SR-12iA provenance: {provenance_path}") from exc
    if not isinstance(provenance, dict):
        raise ValueError("SR-12iA provenance root must be a mapping")
    if provenance.get("converter_schema_version") != CONVERTER_SCHEMA_VERSION:
        raise ValueError(
            "prepared SR-12iA cache is stale; rerun decanting-assets with the "
            "current converter"
        )
    try:
        j3_stroke_m = float(provenance.get("j3_stroke_m", -1.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("prepared SR-12iA cache has an invalid J3 stroke") from exc
    if j3_stroke_m != 0.3:
        raise ValueError("prepared SR-12iA cache has an unsupported J3 stroke")
    _validate_bundled_mesh_references(urdf)
    return provenance


def _validate_bundled_mesh_references(urdf: Path) -> None:
    """Require the generated URDF to reference only its colocated OBJ bundle."""

    try:
        root = ET.parse(urdf).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"invalid prepared SR-12iA URDF: {urdf}") from exc

    mesh_elements = root.findall(".//mesh")
    expected_names = tuple(spec.filename for spec in _MESH_EXPORTS)
    if len(mesh_elements) != len(expected_names):
        raise ValueError(
            "prepared SR-12iA URDF must contain exactly "
            f"{len(expected_names)} mesh references"
        )

    bundle_root = urdf.parent.resolve()
    referenced_names: list[str] = []
    missing: list[str] = []
    for element in mesh_elements:
        filename = element.get("filename", "").strip()
        windows_path = Path(filename)
        # ``Path.is_absolute`` follows the host OS.  These extra checks reject
        # Windows paths and URI forms even when validation runs on POSIX.
        is_windows_absolute = (
            len(filename) >= 3
            and filename[0].isalpha()
            and filename[1] == ":"
            and filename[2] in {"/", "\\"}
        ) or filename.startswith(("\\\\", "//"))
        if (
            not filename
            or "://" in filename
            or ":" in filename
            or "\\" in filename
            or windows_path.is_absolute()
            or is_windows_absolute
        ):
            raise ValueError(
                "prepared SR-12iA mesh references must be portable relative "
                f"paths: {filename!r}"
            )

        target = (bundle_root / windows_path).resolve()
        try:
            target.relative_to(bundle_root)
        except ValueError as exc:
            raise ValueError(
                "prepared SR-12iA mesh reference escapes its asset bundle: "
                f"{filename!r}"
            ) from exc
        referenced_names.append(windows_path.name)
        if not target.is_file():
            missing.append(filename)

    if sorted(referenced_names) != sorted(expected_names):
        raise ValueError(
            "prepared SR-12iA URDF mesh set does not match the converter "
            f"contract: {sorted(referenced_names)}"
        )
    if missing:
        raise ValueError(f"prepared SR-12iA cache is missing meshes: {missing}")


def download_sr12ia_geometry(
    destination: str | os.PathLike[str],
    *,
    accept_fanuc_license: bool = False,
) -> Path:
    """Download the pinned NVIDIA geometry layer after explicit license consent.

    No bytes are written unless ``accept_fanuc_license`` is true.  The completed
    file replaces ``destination`` only after its SHA-256 matches the value
    recorded for the Isaac Sim 6.0 asset.
    """

    if not accept_fanuc_license:
        raise ValueError(
            "direct SR-12iA download requires --accept-fanuc-license after "
            "reviewing " + FANUC_3D_CONTENT_SHARING_AGREEMENT_URL
        )

    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".download", dir=target.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as stream, urlopen(
            NVIDIA_SR12IA_GEOMETRIES_URL,
            timeout=60,
        ) as response:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                digest.update(chunk)
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
        actual = digest.hexdigest()
        if actual != NVIDIA_SR12IA_GEOMETRIES_SHA256:
            raise ValueError(
                "downloaded SR-12iA geometry SHA-256 mismatch: "
                f"expected {NVIDIA_SR12IA_GEOMETRIES_SHA256}, got {actual}"
            )
        os.replace(temporary, target)
        return target
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _local_source_path(input_usd: str | os.PathLike[str]) -> Path:
    raw = os.fspath(input_usd)
    lowered = raw.lower()
    if "://" in raw or lowered.startswith(("http:", "https:", "omniverse:")):
        raise ValueError(
            "input_usd must be a local geometries.usd file; network URLs are "
            "not downloaded by this converter"
        )
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SR-12iA geometry USD not found: {path}")
    if path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError(f"expected an OpenUSD file, got: {path}")
    return path


def _windows_short_path(path: Path) -> str:
    """Return an NTFS 8.3 spelling for a non-ASCII path on Windows.

    Pixar USD on Windows has failed to open this project's Korean path.  The
    source file and output directory are known to exist before this helper is
    called, so ``GetShortPathNameW`` can resolve their real 8.3 aliases.
    """

    value = str(path)
    if os.name != "nt" or value.isascii():
        return value

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_short_path = kernel32.GetShortPathNameW
    get_short_path.argtypes = (
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        wintypes.DWORD,
    )
    get_short_path.restype = wintypes.DWORD

    required = get_short_path(value, None, 0)
    if required == 0:
        error = ctypes.get_last_error()
        raise OSError(
            error,
            "GetShortPathNameW could not resolve the non-ASCII path; enable "
            "8.3 names or copy the USD to an ASCII-only local directory",
            value,
        )
    buffer = ctypes.create_unicode_buffer(required)
    written = get_short_path(value, buffer, required)
    if written == 0 or written >= required:
        error = ctypes.get_last_error()
        raise OSError(error, "GetShortPathNameW failed", value)
    return buffer.value


def _usd_open_path(source: Path, resources: ExitStack) -> str:
    """Return a path that ``Usd.Stage.Open`` can consume on Windows.

    NTFS is allowed to retain Unicode characters in an 8.3 name.  If that
    occurs, make a short-lived ASCII staging copy.  NVIDIA's
    ``payloads/geometries.usd`` is a self-contained layer, so no relative
    payloads are lost by this fallback.
    """

    candidate = _windows_short_path(source)
    if os.name != "nt" or candidate.isascii():
        return candidate

    temporary_dir = Path(
        resources.enter_context(tempfile.TemporaryDirectory(prefix="sr12ia_usd_"))
    )
    staged = temporary_dir / f"geometries{source.suffix.lower()}"
    shutil.copyfile(source, staged)
    candidate = _windows_short_path(staged)
    if not candidate.isascii():
        raise OSError(
            "could not create an ASCII path for Pixar USD; set the Windows "
            "TEMP directory to an ASCII-only location"
        )
    return candidate


def _read_usd_meshes(source: Path) -> dict[str, _ObjMesh]:
    try:
        from pxr import Gf, Usd, UsdGeom
    except ImportError as exc:
        raise RuntimeError(
            "SR-12iA asset preparation requires the optional 'usd-core' "
            "package (the pxr Python modules)"
        ) from exc

    resources = ExitStack()
    try:
        stage = Usd.Stage.Open(_usd_open_path(source, resources))
        if stage is None:
            raise ValueError(f"OpenUSD could not open the geometry layer: {source}")
        metres_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not metres_per_unit > 0.0:
            raise ValueError(f"invalid metersPerUnit in {source}: {metres_per_unit}")

        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        exported: dict[str, _ObjMesh] = {}
        for spec in _MESH_EXPORTS:
            root_prim = stage.GetPrimAtPath(spec.prim_path)
            if not root_prim.IsValid():
                raise ValueError(f"required USD prim is missing: {spec.prim_path}")

            vertices: list[tuple[float, float, float]] = []
            faces: list[tuple[int, ...]] = []
            mesh_prims = [
                prim for prim in Usd.PrimRange(root_prim) if prim.IsA(UsdGeom.Mesh)
            ]
            if not mesh_prims:
                raise ValueError(f"USD prim contains no Mesh: {spec.prim_path}")

            for mesh_prim in mesh_prims:
                mesh = UsdGeom.Mesh(mesh_prim)
                points = mesh.GetPointsAttr().Get(Usd.TimeCode.Default())
                counts = mesh.GetFaceVertexCountsAttr().Get(Usd.TimeCode.Default())
                indices = mesh.GetFaceVertexIndicesAttr().Get(Usd.TimeCode.Default())
                if points is None or counts is None or indices is None:
                    raise ValueError(f"incomplete mesh topology: {mesh_prim.GetPath()}")

                point_values = tuple(points)
                count_values = tuple(int(value) for value in counts)
                index_values = tuple(int(value) for value in indices)
                if not point_values or not count_values:
                    raise ValueError(f"empty mesh: {mesh_prim.GetPath()}")
                if any(count < 3 for count in count_values):
                    raise ValueError(
                        "mesh has a face with fewer than 3 vertices: "
                        f"{mesh_prim.GetPath()}"
                    )
                if sum(count_values) != len(index_values):
                    raise ValueError(f"inconsistent face topology: {mesh_prim.GetPath()}")
                if any(
                    index < 0 or index >= len(point_values)
                    for index in index_values
                ):
                    raise ValueError(
                        f"face index outside point array: {mesh_prim.GetPath()}"
                    )

                matrix = xform_cache.GetLocalToWorldTransform(mesh_prim)
                vertex_offset = len(vertices)
                for point in point_values:
                    transformed = matrix.Transform(
                        Gf.Vec3d(
                            float(point[0]), float(point[1]), float(point[2])
                        )
                    )
                    vertices.append(
                        (
                            float(transformed[0]) * metres_per_unit,
                            float(transformed[1]) * metres_per_unit,
                            float(transformed[2]) * metres_per_unit,
                        )
                    )

                left_handed = (
                    mesh.GetOrientationAttr().Get(Usd.TimeCode.Default())
                    == UsdGeom.Tokens.leftHanded
                )
                cursor = 0
                for count in count_values:
                    face = [
                        vertex_offset + index_values[cursor + offset]
                        for offset in range(count)
                    ]
                    if left_handed:
                        face.reverse()
                    faces.append(tuple(face))
                    cursor += count

            exported[spec.prim_name] = _ObjMesh(tuple(vertices), tuple(faces))
        return exported
    finally:
        resources.close()


def _obj_text(spec: _MeshExport, mesh: _ObjMesh) -> str:
    lines = [
        "# Generated locally from NVIDIA Isaac Sim 6.0 FANUC SR-12iA geometry",
        f"# Source prim: {spec.prim_path}",
        f"o {spec.link}_{spec.role}",
    ]
    lines.extend(
        f"v {x:.17g} {y:.17g} {z:.17g}" for x, y, z in mesh.vertices
    )
    lines.extend(
        "f " + " ".join(str(index + 1) for index in face) for face in mesh.faces
    )
    return "\n".join(lines) + "\n"


def _add_origin(parent: ET.Element, xyz: str, rpy: str = _ZERO_XYZ) -> None:
    ET.SubElement(parent, "origin", {"xyz": xyz, "rpy": rpy})


def _add_mesh_link(
    robot: ET.Element,
    link_name: str,
    visual_filename: str,
    collision_filename: str,
    visual_material: str,
) -> None:
    link = ET.SubElement(robot, "link", {"name": link_name})
    visual = ET.SubElement(link, "visual")
    _add_origin(visual, _ZERO_XYZ, _VISUAL_RPY)
    visual_geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(visual_geometry, "mesh", {"filename": visual_filename})
    ET.SubElement(visual, "material", {"name": visual_material})

    collision = ET.SubElement(link, "collision")
    _add_origin(collision, _ZERO_XYZ, _ZERO_XYZ)
    collision_geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(collision_geometry, "mesh", {"filename": collision_filename})


def _add_joint(
    robot: ET.Element,
    *,
    name: str,
    joint_type: str,
    parent: str,
    child: str,
    origin: str,
    axis: str | None = None,
    lower: str | None = None,
    upper: str | None = None,
    velocity: str | None = None,
) -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": joint_type})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})
    _add_origin(joint, origin)
    if axis is not None:
        ET.SubElement(joint, "axis", {"xyz": axis})
    if lower is not None and upper is not None and velocity is not None:
        ET.SubElement(
            joint,
            "limit",
            {
                "lower": lower,
                "upper": upper,
                "effort": "1",
                "velocity": velocity,
            },
        )


def _urdf_text() -> str:
    robot = ET.Element("robot", {"name": "fanuc_sr12ia_mesh"})
    dark = ET.SubElement(robot, "material", {"name": "fanuc_dark"})
    ET.SubElement(dark, "color", {"rgba": "0.12 0.13 0.14 1"})
    yellow = ET.SubElement(robot, "material", {"name": "fanuc_yellow"})
    ET.SubElement(yellow, "color", {"rgba": "0.96 0.77 0.05 1"})
    ET.SubElement(robot, "link", {"name": "world"})
    filenames = {
        (spec.link, spec.role): spec.filename for spec in _MESH_EXPORTS
    }
    for link_name in ("base_link", "J1_link", "J2_link", "J3_link"):
        _add_mesh_link(
            robot,
            link_name,
            filenames[(link_name, "visual")],
            filenames[(link_name, "collision")],
            "fanuc_dark" if link_name == "base_link" else "fanuc_yellow",
        )
    ET.SubElement(robot, "link", {"name": "J4_link"})
    ET.SubElement(robot, "link", {"name": "tool0"})

    _add_joint(
        robot,
        name="base_joint",
        joint_type="fixed",
        parent="world",
        child="base_link",
        origin=_ZERO_XYZ,
    )
    _add_joint(
        robot,
        name="joint1",
        joint_type="revolute",
        parent="base_link",
        child="J1_link",
        origin=_ZERO_XYZ,
        axis="0 0 1",
        lower="-2.53072741539",
        upper="2.53072741539",
        velocity="7.85398163397",
    )
    _add_joint(
        robot,
        name="joint2",
        joint_type="revolute",
        parent="J1_link",
        child="J2_link",
        origin="0.45 0 0.336",
        axis="0 0 1",
        lower="-2.53072741539",
        upper="2.53072741539",
        velocity="7.85398163397",
    )
    _add_joint(
        robot,
        name="joint3",
        joint_type="prismatic",
        parent="J2_link",
        child="J3_link",
        origin="0.45 0 0",
        axis="0 0 -1",
        lower="0",
        upper="0.3",
        velocity="3",
    )
    _add_joint(
        robot,
        name="joint4",
        joint_type="revolute",
        parent="J3_link",
        child="J4_link",
        origin=_ZERO_XYZ,
        axis="0 0 1",
        lower="-12.5663706144",
        upper="12.5663706144",
        velocity="26.1799387799",
    )
    _add_joint(
        robot,
        name="tool0_joint",
        joint_type="fixed",
        parent="J4_link",
        child="tool0",
        origin=_ZERO_XYZ,
    )

    ET.indent(robot, space="  ")
    xml = ET.tostring(robot, encoding="unicode", short_empty_elements=True)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + xml + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance_text(source: Path) -> str:
    mapping: dict[str, dict[str, dict[str, str]]] = {}
    for spec in _MESH_EXPORTS:
        mapping.setdefault(spec.link, {})[spec.role] = {
            "usd_prim": spec.prim_path,
            "obj": spec.filename,
        }
    provenance = {
        "schema_version": 1,
        "converter_schema_version": CONVERTER_SCHEMA_VERSION,
        "source_file": source.name,
        "source_sha256": _sha256(source),
        "nvidia_asset_url": NVIDIA_SR12IA_GEOMETRIES_URL,
        "nvidia_asset_sha256": NVIDIA_SR12IA_GEOMETRIES_SHA256,
        "isaac_sim_version": ISAAC_SIM_VERSION,
        "j3_stroke_m": 0.3,
        "fanuc_3d_content_sharing_agreement_url": (
            FANUC_3D_CONTENT_SHARING_AGREEMENT_URL
        ),
        "generated_urdf": _URDF_FILENAME,
        "mesh_prim_mapping": mapping,
        "derived_asset_notice": (
            "Generated locally from a user-supplied asset. Review the linked "
            "agreement before redistributing derived geometry."
        ),
    }
    return json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
        raise


def prepare_sr12ia_assets(
    input_usd: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Convert a local Isaac Sim geometry layer into OBJ meshes and a URDF.

    Args:
        input_usd: Local ``payloads/geometries.usd`` (or USDA/USDC equivalent).
        output_dir: Destination directory.  The default is the repository's
            ignored ``.cache/robot_assets/sr12ia`` directory.

    Returns:
        The long, resolved path to ``sr12ia_mesh.urdf``.

    Raises:
        RuntimeError: If the optional ``pxr`` modules are not installed.
        ValueError: If the source does not contain the eight required prims or
            valid mesh topology.  URLs are also rejected here by design.
    """

    source = _local_source_path(input_usd)
    if output_dir is None:
        destination = default_sr12ia_output_urdf().parent
    else:
        destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"output_dir is not a directory: {destination}")

    # Read and validate every prim before replacing any generated asset.
    meshes = _read_usd_meshes(source)
    provenance_text = _provenance_text(source)
    destination.mkdir(parents=True, exist_ok=True)
    io_destination = Path(_windows_short_path(destination))
    # Keep the bundle relocatable.  The production robot loader resolves these
    # filenames from the URDF directory before handing geometry to Pinocchio.
    urdf_text = _urdf_text()
    for spec in _MESH_EXPORTS:
        _atomic_write_text(
            io_destination / spec.filename,
            _obj_text(spec, meshes[spec.prim_name]),
        )
    _atomic_write_text(io_destination / _URDF_FILENAME, urdf_text)
    _atomic_write_text(io_destination / _PROVENANCE_FILENAME, provenance_text)
    return destination / _URDF_FILENAME


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a local or explicitly accepted, pinned Isaac Sim 6.0 "
            "FANUC SR-12iA geometries.usd into cached OBJ meshes and a "
            "four-axis Pinocchio URDF."
        )
    )
    parser.add_argument(
        "input_usd",
        nargs="?",
        help="local path to Fanuc/sr12ia/payloads/geometries.usd",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help=(
            "download the pinned NVIDIA Isaac Sim 6.0 geometry instead of "
            "using a local input file"
        ),
    )
    parser.add_argument(
        "--accept-fanuc-license",
        action="store_true",
        help=(
            "confirm that the FANUC 3D Content Sharing Agreement was reviewed; "
            "required with --download"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "output directory (default: repository "
            ".cache/robot_assets/sr12ia)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicit local-or-pinned asset converter interface."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.download and args.input_usd is not None:
        parser.error("input_usd and --download are mutually exclusive")
    if not args.download and args.input_usd is None:
        parser.error("provide input_usd or select --download")
    if args.download and not args.accept_fanuc_license:
        parser.error("--download requires --accept-fanuc-license")
    if args.accept_fanuc_license and not args.download:
        parser.error("--accept-fanuc-license is only valid with --download")

    if args.download:
        with tempfile.TemporaryDirectory(prefix="sr12ia_download_") as directory:
            source = download_sr12ia_geometry(
                Path(directory) / "geometries.usd",
                accept_fanuc_license=args.accept_fanuc_license,
            )
            urdf_path = prepare_sr12ia_assets(source, args.output_dir)
    else:
        urdf_path = prepare_sr12ia_assets(args.input_usd, args.output_dir)
    print(urdf_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main().
    raise SystemExit(main())
