from __future__ import annotations

from pathlib import Path

from decanting_workspace.models import load_scene_spec
from decanting_workspace.setup_cli import SetupDependencies, build_parser, main


class _Cache:
    cases = (object(), object())


def _dependencies(tmp_path, calls, *, existing_sr=True):
    spec = load_scene_spec()
    urdf = spec.robots["ur20"].urdf_path
    sr = tmp_path / "sr12ia" / "sr12ia_mesh.urdf"
    if existing_sr:
        sr.parent.mkdir()
        sr.write_text("<robot name='sr'/>", encoding="utf-8")

    def validate_sr(path):
        calls.append(("validate_sr", Path(path)))
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        return {"source_sha256": "a" * 64}

    def prepare_sr(source, output_dir):
        calls.append(("prepare_sr", Path(source), Path(output_dir)))
        target = Path(output_dir) / "sr12ia_mesh.urdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("<robot name='sr'/>", encoding="utf-8")
        return target

    return SetupDependencies(
        spec_loader=lambda path: spec,
        ur20_validator=lambda: urdf,
        ur20_resolver=lambda: (urdf, ()),
        sr12ia_path=lambda: sr,
        sr12ia_validator=validate_sr,
        sr12ia_preparer=prepare_sr,
        sr12ia_downloader=lambda destination, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected download")
        ),
        cache_loader=lambda path, *, spec: _Cache(),
    )


def test_setup_parser_does_not_claim_license_acceptance_by_default():
    args = build_parser().parse_args([])

    assert not args.accept_fanuc_license
    assert args.sr12ia_usd is None


def test_setup_validates_existing_assets_and_caches(tmp_path, monkeypatch, capsys):
    calls = []
    deps = _dependencies(tmp_path, calls)
    monkeypatch.setattr(
        "decanting_workspace.setup_cli._validate_candidate_report_paths",
        lambda: calls.append(("reports",)),
    )

    assert main([], dependencies=deps) == 0
    output = capsys.readouterr().out
    assert "Setup complete" in output
    assert sum(call[0] == "validate_sr" for call in calls) == 2
    assert ("reports",) in calls


def test_setup_missing_sr_asset_requires_explicit_source_or_acceptance(
    tmp_path,
    capsys,
):
    calls = []
    deps = _dependencies(tmp_path, calls, existing_sr=False)

    assert main(["--skip-cache-validation"], dependencies=deps) == 2
    assert "--accept-fanuc-license" in capsys.readouterr().err


def test_setup_can_prepare_from_authorized_local_usd(tmp_path, capsys):
    calls = []
    deps = _dependencies(tmp_path, calls, existing_sr=False)
    source = tmp_path / "geometries.usd"
    source.write_text("#usda 1.0", encoding="utf-8")

    assert main(
        [
            "--sr12ia-usd",
            str(source),
            "--skip-cache-validation",
        ],
        dependencies=deps,
    ) == 0
    assert any(call[0] == "prepare_sr" for call in calls)
    assert "Setup complete" in capsys.readouterr().out
