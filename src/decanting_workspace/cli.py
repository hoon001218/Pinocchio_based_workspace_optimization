"""Command-line entry point for the first decanting-cell visualization."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .asset_prep import (
    default_sr12ia_output_urdf,
    validate_prepared_sr12ia_asset,
)
from .models import BasePose, SceneState, default_config_path, load_scene_spec
from .robots import resolve_official_ur20
from .scene import CORNER_SIGNS, materialize_scene
from .viewer import CellViewer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize the dimension-based decanting cell in Meshcat."
    )
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument(
        "--ur-base",
        nargs=4,
        type=float,
        metavar=("X_M", "Y_M", "Z_M", "YAW_DEG"),
        help="Override the nominal UR20 base candidate.",
    )
    parser.add_argument(
        "--sr-base",
        nargs=4,
        type=float,
        metavar=("X_M", "Y_M", "Z_M", "YAW_DEG"),
        help="Override the nominal SR-12iA base candidate.",
    )
    parser.add_argument("--lift-height-mm", type=float, default=0.0)
    parser.add_argument("--sku", default="123591")
    parser.add_argument(
        "--corner",
        choices=("all", *CORNER_SIGNS),
        default="all",
        help="Show all four alternative lowest-layer samples or one physical box.",
    )
    parser.add_argument("--sr-j3-stroke-mm", type=float, choices=(300.0, 450.0), default=300.0)
    parser.add_argument("--clearance-mm", type=float)
    parser.add_argument("--no-tote", action="store_true")
    parser.add_argument(
        "--sr-module-camera-mode",
        choices=("fixed", "with_base"),
        help="Keep the two cutting-module camera poles fixed or move them with the SR base.",
    )
    parser.add_argument(
        "--ur20-source",
        choices=("primitive", "official"),
        default="official",
        help="Use the pinned official mesh URDF (default) or the primitive fallback.",
    )
    parser.add_argument("--ur20-urdf", type=Path, help="Explicit visual URDF; overrides --ur20-source.")
    parser.add_argument(
        "--sr12ia-source",
        choices=("auto", "mesh", "approximate"),
        default="auto",
        help=(
            "Prefer a locally prepared FANUC/Isaac mesh asset, require it, or use "
            "the bundled kinematic approximation."
        ),
    )
    parser.add_argument(
        "--sr12ia-urdf",
        type=Path,
        help="Explicit SR-12iA visual URDF; overrides --sr12ia-source.",
    )
    parser.add_argument("--show-collisions", action="store_true")
    parser.add_argument("--hide-frames", action="store_true")
    parser.add_argument("--hide-installation-region", action="store_true")
    parser.add_argument(
        "--hide-sr-cutting-footprint",
        "--hide-sr-workspace",
        dest="hide_sr_cutting_footprint",
        action="store_true",
        help="Hide the cutter-TCP task footprint on the SR support cube.",
    )
    parser.add_argument(
        "--hide-sr-cutting-footprint-fill",
        "--hide-sr-envelope",
        dest="hide_sr_footprint_fill",
        action="store_true",
        help="Show only the cutting-footprint boundary (legacy alias retained).",
    )
    parser.add_argument(
        "--save-html",
        type=Path,
        help="Write a self-contained proxy-only Meshcat snapshot.",
    )
    parser.add_argument("--open", action="store_true", help="Open the live Meshcat view in a browser.")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not wait for Enter after opening the live browser view.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = load_scene_spec(args.config)
        corners = tuple(CORNER_SIGNS) if args.corner == "all" else (args.corner,)
        state = SceneState(
            lift_height_m=args.lift_height_mm / 1000.0,
            sku=str(args.sku),
            corner_samples=corners,
            tote_present=not args.no_tote,
            sr_j3_stroke_m=args.sr_j3_stroke_mm / 1000.0,
            clearance_m=None if args.clearance_mm is None else args.clearance_mm / 1000.0,
        )
        ur_base = (
            BasePose.from_sequence(args.ur_base, "--ur-base") if args.ur_base else None
        )
        sr_base = (
            BasePose.from_sequence(args.sr_base, "--sr-base") if args.sr_base else None
        )
        snapshot = materialize_scene(
            spec,
            state,
            ur_base=ur_base,
            sr_base=sr_base,
            sr_module_camera_mode=args.sr_module_camera_mode,
        )

        ur20_urdf = args.ur20_urdf
        ur20_package_dirs: tuple[Path, ...] = ()
        uses_official_ur20_mesh = (
            ur20_urdf is None and args.ur20_source == "official"
        )
        if uses_official_ur20_mesh:
            ur20_urdf, ur20_package_dirs = resolve_official_ur20()

        sr12ia_urdf = args.sr12ia_urdf
        uses_prepared_sr12ia_mesh = False
        if sr12ia_urdf is None and args.sr12ia_source in {"auto", "mesh"}:
            prepared_sr12ia = default_sr12ia_output_urdf()
            if prepared_sr12ia.is_file():
                try:
                    validate_prepared_sr12ia_asset(prepared_sr12ia)
                except ValueError as exc:
                    if args.sr12ia_source == "mesh":
                        raise ValueError(str(exc)) from exc
                    print(
                        f"WARNING: {exc}; using the corrected primitive fallback.",
                        file=sys.stderr,
                    )
                else:
                    if state.sr_j3_stroke_m > 0.300 + 1e-12:
                        if args.sr12ia_source == "mesh":
                            raise ValueError(
                                "the prepared Isaac SR-12iA mesh is the 300 mm J3 "
                                "model; it cannot represent the 450 mm option"
                            )
                        print(
                            "WARNING: the prepared Isaac SR-12iA mesh is the 300 mm "
                            "J3 model. The 450 mm option uses the corrected provisional "
                            "fallback until a matching manufacturer asset is supplied.",
                            file=sys.stderr,
                        )
                    else:
                        sr12ia_urdf = prepared_sr12ia
                        uses_prepared_sr12ia_mesh = True
            elif args.sr12ia_source == "mesh":
                raise FileNotFoundError(
                    "prepared SR-12iA mesh URDF not found; run "
                    "`decanting-assets PATH_TO_GEOMETRIES_USD` first"
                )
            else:
                print(
                    "WARNING: prepared SR-12iA manufacturer mesh was not found; "
                    "using the corrected primitive fallback. Run `decanting-assets "
                    "PATH_TO_GEOMETRIES_USD` to prepare your licensed Isaac asset.",
                    file=sys.stderr,
                )

        if args.save_html and (
            uses_official_ur20_mesh or uses_prepared_sr12ia_mesh
        ):
            raise ValueError(
                "self-contained HTML embeds the licensed robot mesh files. "
                "Use --open for the official-mesh view, or select "
                "--ur20-source primitive --sr12ia-source approximate for a "
                "shareable proxy HTML snapshot."
            )

        viewer = CellViewer()
        viewer.render(
            spec,
            snapshot,
            ur20_urdf=ur20_urdf,
            ur20_package_dirs=ur20_package_dirs,
            sr12ia_urdf=sr12ia_urdf,
            show_collisions=args.show_collisions,
            show_frames=not args.hide_frames,
            show_installation_region=not args.hide_installation_region,
            show_sr_cutting_footprint=not args.hide_sr_cutting_footprint,
            show_sr_footprint_fill=not args.hide_sr_footprint_fill,
        )

        print(f"Meshcat URL: {viewer.url()}")
        if args.save_html:
            output = viewer.save_static_html(args.save_html)
            print(f"Saved static view: {output}")
        if args.open:
            viewer.open()
            if not args.no_wait and sys.stdin.isatty():
                input("Press Enter to close the viewer... ")
        elif not args.save_html:
            print("Use --open for a live view or --save-html PATH for a snapshot.")
        return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
