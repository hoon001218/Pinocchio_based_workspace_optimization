"""Open a simplified USD environment with hierarchical XYZ layout controls."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Mapping
import webbrowser

import yaml

from .layout_backend import LayoutPreviewBackend
from .layout_ui import LayoutControlServer
from .models import _read_scene_configuration, load_scene_spec
from .paths import repository_path


@dataclass(frozen=True)
class LayoutDependencies:
    spec_loader: Callable[..., object]
    config_loader: Callable[[Path], Mapping[str, object]]
    layout_loader: Callable[..., object]
    viewer_factory: Callable[..., object]
    backend_factory: Callable[..., object]
    server_factory: Callable[..., object]
    browser_open: Callable[[str], object]


def default_layout_config_path() -> Path:
    return repository_path("config", "cell_layout.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preview a simplified USD layout and translate groups or elements in parent-local XYZ."
    )
    parser.add_argument("--config", type=Path, default=default_layout_config_path())
    parser.add_argument("--usd", type=Path, help="Override the USD file configured in usd_layout.file.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--open", action="store_true", help="Open the layout control page in a browser.")
    parser.add_argument("--output", type=Path, help="Write the current layout and every subsequent preview to JSON.")
    parser.add_argument("--ur20-source", choices=("official", "configured"), default="official")
    parser.add_argument("--sr12ia-visual-urdf", type=Path)
    parser.add_argument("--hide-frames", action="store_true")
    return parser


def main(
    argv: list[str] | None = None, *, dependencies: LayoutDependencies | None = None
) -> int:
    args = build_parser().parse_args(argv)
    deps = dependencies or _default_dependencies()
    server = None
    try:
        config = deps.config_loader(args.config.resolve())
        options = config.get("usd_layout", {})
        if not isinstance(options, Mapping):
            raise ValueError("usd_layout must be a mapping")
        options = dict(options)
        configured_source = options.pop("file", None)
        source = args.usd
        if source is None:
            if not isinstance(configured_source, str) or not configured_source:
                raise ValueError("provide --usd or configure usd_layout.file")
            source = Path(configured_source)
        spec = deps.spec_loader(args.config)
        if not 0 <= args.port <= 65535:
            raise ValueError("--port must be between 0 and 65535")
        scene = deps.layout_loader(source.resolve(), spec.robots, options)
        viewer = deps.viewer_factory(
            ur20_source=args.ur20_source,
            sr12ia_visual_urdf=args.sr12ia_visual_urdf,
            show_frames=not args.hide_frames,
        )
        backend = deps.backend_factory(scene, viewer, output=args.output)
        server = deps.server_factory((args.host, args.port), backend, viewer.url())
        print(f"USD layout: {scene.source_path}")
        print("LAYOUT CONTROL UI (parent-local XYZ translations):")
        print(server.control_url, flush=True)
        stats = backend.catalog()["stats"]
        print(
            f"Nodes: {stats['layout_nodes']}; source proxies: {stats['source_boxes']}; "
            f"simplified fragments: {stats['simplified_fragments']}; URDF robots: {stats['robots']}"
        )
        if args.output is not None:
            print(f"Layout JSON: {args.output.resolve()}")
        if args.open and deps.browser_open(server.control_url) is False:
            print(f"WARNING: open the layout URL manually: {server.control_url}", file=sys.stderr)
        print("Press Ctrl+C to stop the layout UI.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if server is not None:
            server.server_close()


def _default_dependencies() -> LayoutDependencies:
    from .layout_viewer import LayoutViewer
    from .usd_layout import load_usd_layout

    return LayoutDependencies(
        spec_loader=load_scene_spec,
        config_loader=_read_scene_configuration,
        layout_loader=load_usd_layout,
        viewer_factory=LayoutViewer,
        backend_factory=LayoutPreviewBackend,
        server_factory=LayoutControlServer,
        browser_open=webbrowser.open,
    )


if __name__ == "__main__":
    raise SystemExit(main())
