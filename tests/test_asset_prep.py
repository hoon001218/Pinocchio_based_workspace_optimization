from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pytest

from decanting_workspace import asset_prep


def _joint(root: ET.Element, name: str) -> ET.Element:
    joint = root.find(f"./joint[@name='{name}']")
    assert joint is not None
    return joint


def _synthetic_usda() -> str:
    mesh_names = (
        "mesh",
        "mesh_1",
        "mesh_2",
        "mesh_3",
        "mesh_4",
        "mesh_5",
        "mesh_6",
        "mesh_7",
    )
    prims = []
    for index, name in enumerate(mesh_names):
        prims.append(
            f'''    def Xform "{name}"
    {{
        double3 xformOp:translate = ({index}, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]

        def Mesh "{name}"
        {{
            point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
        }}
    }}'''
        )
    return '''#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)

def Scope "Geometries"
{
''' + "\n\n".join(prims) + "\n}\n"


def _write_validator_bundle(directory: Path) -> Path:
    directory.mkdir(parents=True)
    urdf = directory / "sr12ia_mesh.urdf"
    urdf.write_text(asset_prep._urdf_text(), encoding="utf-8")
    for spec in asset_prep._MESH_EXPORTS:
        (directory / spec.filename).write_text("o test\n", encoding="utf-8")
    (directory / "sr12ia_mesh.provenance.json").write_text(
        json.dumps(
            {
                "converter_schema_version": asset_prep.CONVERTER_SCHEMA_VERSION,
                "j3_stroke_m": 0.3,
            }
        ),
        encoding="utf-8",
    )
    return urdf


def test_default_output_is_the_repository_cache_path():
    repository_root = Path(__file__).resolve().parents[1]
    assert asset_prep.default_sr12ia_output_urdf() == (
        repository_root
        / ".cache"
        / "robot_assets"
        / "sr12ia"
        / "sr12ia_mesh.urdf"
    )


def test_tracked_download_manifest_matches_pinned_constants():
    repository_root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            repository_root
            / "assets"
            / "robots"
            / "sr12ia"
            / "download_manifest.json"
        ).read_text(encoding="utf-8")
    )

    assert manifest["url"] == asset_prep.NVIDIA_SR12IA_GEOMETRIES_URL
    assert manifest["sha256"] == asset_prep.NVIDIA_SR12IA_GEOMETRIES_SHA256
    assert (
        manifest["fanuc_3d_content_sharing_agreement_url"]
        == asset_prep.FANUC_3D_CONTENT_SHARING_AGREEMENT_URL
    )


def test_remote_input_is_rejected_before_optional_pxr_import():
    with pytest.raises(ValueError, match="local geometries.usd"):
        asset_prep.prepare_sr12ia_assets(
            "https://example.invalid/Fanuc/sr12ia/payloads/geometries.usd"
        )


def test_download_requires_explicit_fanuc_license_acceptance(tmp_path, monkeypatch):
    called = False

    def unexpected_open(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("network must not be opened without consent")

    monkeypatch.setattr(asset_prep, "urlopen", unexpected_open)

    with pytest.raises(ValueError, match="--accept-fanuc-license"):
        asset_prep.download_sr12ia_geometry(tmp_path / "geometries.usd")

    assert not called
    assert not (tmp_path / "geometries.usd").exists()


def test_download_verifies_pinned_url_and_sha256(tmp_path, monkeypatch):
    payload = b"pinned synthetic USD"
    calls = []

    def fake_open(url, *, timeout):
        calls.append((url, timeout))
        return io.BytesIO(payload)

    monkeypatch.setattr(asset_prep, "urlopen", fake_open)
    monkeypatch.setattr(
        asset_prep,
        "NVIDIA_SR12IA_GEOMETRIES_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )

    result = asset_prep.download_sr12ia_geometry(
        tmp_path / "geometries.usd",
        accept_fanuc_license=True,
    )

    assert result.read_bytes() == payload
    assert calls == [(asset_prep.NVIDIA_SR12IA_GEOMETRIES_URL, 60)]


def test_download_hash_mismatch_does_not_replace_destination(tmp_path, monkeypatch):
    target = tmp_path / "geometries.usd"
    target.write_bytes(b"existing authorized source")
    monkeypatch.setattr(asset_prep, "urlopen", lambda *args, **kwargs: io.BytesIO(b"bad"))

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        asset_prep.download_sr12ia_geometry(
            target,
            accept_fanuc_license=True,
        )

    assert target.read_bytes() == b"existing authorized source"
    assert not tuple(tmp_path.glob("*.download"))


def test_download_cli_requires_explicit_acceptance(capsys):
    with pytest.raises(SystemExit) as caught:
        asset_prep.main(["--download"])

    assert caught.value.code == 2
    assert "--download requires --accept-fanuc-license" in capsys.readouterr().err


def test_download_cli_converts_verified_temporary_source(tmp_path, monkeypatch):
    calls = []

    def fake_download(destination, *, accept_fanuc_license):
        source = Path(destination)
        source.write_bytes(b"verified")
        calls.append(("download", accept_fanuc_license, source.is_file()))
        return source

    def fake_prepare(source, output_dir):
        calls.append(("prepare", Path(source).read_bytes(), Path(output_dir)))
        return Path(output_dir) / "sr12ia_mesh.urdf"

    monkeypatch.setattr(asset_prep, "download_sr12ia_geometry", fake_download)
    monkeypatch.setattr(asset_prep, "prepare_sr12ia_assets", fake_prepare)

    assert asset_prep.main(
        [
            "--download",
            "--accept-fanuc-license",
            "--output-dir",
            str(tmp_path),
        ]
    ) == 0
    assert calls == [
        ("download", True, True),
        ("prepare", b"verified", tmp_path),
    ]


def test_generated_urdf_has_the_exact_four_axis_contract():
    text = asset_prep._urdf_text()
    root = ET.fromstring(text)

    assert root.attrib["name"] == "fanuc_sr12ia_mesh"
    assert [joint.attrib["name"] for joint in root.findall("joint") if joint.attrib["type"] != "fixed"] == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
    ]
    assert len(root.findall(".//visual/geometry/mesh")) == 4
    assert len(root.findall(".//collision/geometry/mesh")) == 4
    assert {
        origin.attrib["rpy"] for origin in root.findall(".//visual/origin")
    } == {"1.57079632679 0 0"}
    assert {
        origin.attrib["rpy"] for origin in root.findall(".//collision/origin")
    } == {"0 0 0"}
    visual_materials = {
        link.attrib["name"]: link.find("visual/material").attrib["name"]
        for link in root.findall("link")
        if link.find("visual") is not None
    }
    assert visual_materials == {
        "base_link": "fanuc_dark",
        "J1_link": "fanuc_yellow",
        "J2_link": "fanuc_yellow",
        "J3_link": "fanuc_yellow",
    }

    expected_origins = {
        "joint1": "0 0 0",
        "joint2": "0.45 0 0.336",
        "joint3": "0.45 0 0",
        "joint4": "0 0 0",
        "tool0_joint": "0 0 0",
    }
    for name, xyz in expected_origins.items():
        origin = _joint(root, name).find("origin")
        assert origin is not None
        assert origin.attrib["xyz"] == xyz

    joint3 = _joint(root, "joint3")
    assert joint3.find("axis").attrib["xyz"] == "0 0 -1"
    assert joint3.find("limit").attrib["lower"] == "0"
    assert joint3.find("limit").attrib["upper"] == "0.3"

    if importlib.util.find_spec("pinocchio") is not None:
        import pinocchio as pin

        model = pin.buildModelFromXML(text)
        assert model.nq == 4
        assert model.nv == 4
        assert list(model.names[1:]) == ["joint1", "joint2", "joint3", "joint4"]
        assert model.getFrameId("tool0") < model.nframes


@pytest.mark.parametrize(
    ("filename", "message"),
    (
        ("C:/other/machine/mesh.obj", "portable relative paths"),
        ("file:///tmp/mesh.obj", "portable relative paths"),
        ("../outside.obj", "escapes its asset bundle"),
    ),
)
def test_validator_rejects_nonportable_or_out_of_bundle_mesh_references(
    tmp_path,
    filename,
    message,
):
    urdf_path = _write_validator_bundle(tmp_path / "bundle")
    tree = ET.parse(urdf_path)
    tree.getroot().find(".//mesh").set("filename", filename)
    tree.write(urdf_path, encoding="utf-8", xml_declaration=True)

    with pytest.raises(ValueError, match=message):
        asset_prep.validate_prepared_sr12ia_asset(urdf_path)


def test_validator_rejects_missing_referenced_mesh(tmp_path):
    urdf_path = _write_validator_bundle(tmp_path / "bundle")
    (urdf_path.parent / "j3_link_visual.obj").unlink()

    with pytest.raises(ValueError, match="missing meshes"):
        asset_prep.validate_prepared_sr12ia_asset(urdf_path)


def test_synthetic_usda_exports_eight_objs_urdf_and_provenance(tmp_path):
    try:
        import pxr  # noqa: F401
    except ImportError:
        pytest.skip("usd-core/pxr is an optional one-time conversion dependency")

    input_dir = tmp_path / "한글 입력"
    input_dir.mkdir()
    source = input_dir / "geometries.usda"
    source.write_text(_synthetic_usda(), encoding="utf-8")
    output_dir = tmp_path / "한글 출력"

    urdf_path = asset_prep.prepare_sr12ia_assets(source, output_dir)

    assert urdf_path == output_dir.resolve() / "sr12ia_mesh.urdf"
    assert urdf_path.is_file()
    obj_paths = sorted(output_dir.glob("*.obj"))
    assert len(obj_paths) == 8
    for obj_path in obj_paths:
        lines = obj_path.read_text(encoding="utf-8").splitlines()
        assert len([line for line in lines if line.startswith("v ")]) == 3
        assert [line for line in lines if line.startswith("f ")] == ["f 1 2 3"]

    # The Xform on /Geometries/mesh_2 is baked into its OBJ in metres.
    j1_visual = (output_dir / "j1_link_visual.obj").read_text(encoding="utf-8")
    assert "v 2 0 0" in j1_visual.splitlines()

    urdf = ET.parse(urdf_path).getroot()
    referenced_meshes = {
        element.attrib["filename"] for element in urdf.findall(".//mesh")
    }
    assert {Path(filename).name for filename in referenced_meshes} == {
        path.name for path in obj_paths
    }
    assert referenced_meshes == {
        spec.filename for spec in asset_prep._MESH_EXPORTS
    }
    assert all(not Path(filename).is_absolute() for filename in referenced_meshes)

    if importlib.util.find_spec("pinocchio") is not None:
        from decanting_workspace.robots import load_robot_bundle

        bundle = load_robot_bundle(
            "sr12ia-test",
            urdf_path,
            floating_base=False,
            load_visual=True,
        )
        assert (bundle.model.nq, bundle.model.nv) == (4, 4)
        assert bundle.visual_model is not None
        assert (bundle.collision_model.ngeoms, bundle.visual_model.ngeoms) == (4, 4)

    provenance_path = output_dir / "sr12ia_mesh.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert asset_prep.validate_prepared_sr12ia_asset(urdf_path) == provenance
    assert provenance["converter_schema_version"] == 3
    assert provenance["j3_stroke_m"] == pytest.approx(0.3)
    assert provenance["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert provenance["isaac_sim_version"] == "6.0"
    assert provenance["nvidia_asset_url"].endswith(
        "/Isaac/Robots/Fanuc/sr12ia/payloads/geometries.usd"
    )
    assert (
        provenance["nvidia_asset_sha256"]
        == asset_prep.NVIDIA_SR12IA_GEOMETRIES_SHA256
    )
    assert provenance["fanuc_3d_content_sharing_agreement_url"].endswith(
        "3D%20Content%20Sharing%20Agreement_FANUC.pdf"
    )
    assert set(provenance["mesh_prim_mapping"]) == {
        "base_link",
        "J1_link",
        "J2_link",
        "J3_link",
    }

    relocated = tmp_path / "relocated" / "sr12ia"
    relocated.parent.mkdir()
    shutil.move(str(output_dir), relocated)
    moved_urdf = relocated / "sr12ia_mesh.urdf"
    assert asset_prep.validate_prepared_sr12ia_asset(moved_urdf)[
        "source_sha256"
    ] == hashlib.sha256(source.read_bytes()).hexdigest()
