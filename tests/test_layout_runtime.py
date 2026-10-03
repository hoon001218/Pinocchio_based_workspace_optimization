from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from decanting_workspace.layout import apply_layout_offsets
from decanting_workspace.layout_backend import LayoutPreviewBackend
from decanting_workspace.layout_cli import LayoutDependencies, build_parser, main
from decanting_workspace.layout_geometry import build_layout_geometry
from decanting_workspace.layout_models import LayoutBox, LayoutNode, LayoutRobot, LayoutScene
from decanting_workspace.layout_ui import LayoutControlServer, build_layout_control_page


class _Viewer:
    def __init__(self):
        self.renders = []

    def render(self, scene, fragments):
        self.renders.append((scene, fragments))

    def url(self):
        return "http://127.0.0.1:7000/static/"


def _scene():
    return LayoutScene(
        Path("fixture.usda"),
        (
            LayoutNode("/World/LoadGroup", None, (1., 2., 0.), (0., 0., 0., 1.), "group"),
            LayoutNode("/World/LoadGroup/RB", "/World/LoadGroup", (1., 2., .8), (0., 0., 0., 1.)),
        ),
        (LayoutBox("support", "/World/LoadGroup", (1., 2., .4), (.5, .5, .8), (0., 0., 0., 1.)),),
        (LayoutRobot("ur20", "/World/LoadGroup/RB", (1., 2., .8), 90., Path("robot.urdf"), (0.,)),),
    )


def _backend(*, output=None):
    viewer = _Viewer()
    backend = LayoutPreviewBackend(
        _scene(), viewer,
        offset_applier=apply_layout_offsets,
        geometry_builder=build_layout_geometry,
        output=output,
    )
    return backend, viewer


def test_layout_preview_moves_attached_robot_and_restarts_from_original():
    backend, viewer = _backend()
    first = backend.layout({"offsets": {"/World/LoadGroup": [.5, 0, 0], "/World/LoadGroup/RB": [0, .2, 0]}})
    assert first["robots"][0]["translation_m"] == pytest.approx([1.5, 2.2, .8])
    assert first["robots"][0]["yaw_deg"] == 90.
    second = backend.layout({"offsets": {"/World/LoadGroup": [.5, 0, 0]}})
    assert second["robots"][0]["translation_m"] == pytest.approx([1.5, 2., .8])
    reset = backend.layout({"offsets": {}})
    assert reset["robots"][0]["translation_m"] == [1., 2., .8]
    assert len(viewer.renders) == 4
    assert backend.original_scene == _scene()


@pytest.mark.parametrize("payload", [
    {"offsets": {"/World/Unknown": [0, 0, 0]}},
    {"offsets": {"/World/LoadGroup": [True, 0, 0]}},
    {"offsets": {"/World/LoadGroup": [float("nan"), 0, 0]}},
    {"offsets": {"/World/LoadGroup": [0, 0]}},
    {"offsets": []},
    {"offsets": {}, "yaw": 90},
])
def test_invalid_layout_requests_do_not_change_preview(payload):
    backend, viewer = _backend()
    with pytest.raises(ValueError):
        backend.layout(payload)
    assert len(viewer.renders) == 1
    assert backend.offsets == {}
    assert backend.scene == backend.original_scene


def test_layout_export_contains_current_geometry_and_updates_output(tmp_path):
    output = tmp_path / "layout.json"
    backend, _ = _backend(output=output)
    backend.layout({"offsets": {"/World/LoadGroup": [0, 0, .1]}})
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == backend.export_payload()
    assert payload["scene"]["robots"][0]["translation_m"] == pytest.approx([1., 2., .9])
    assert payload["scene"]["robots"][0]["urdf_path"] == "robot.urdf"
    assert payload["coordinate_frame"] == "parent_local"


def test_catalog_reports_the_urdf_model_actually_loaded_by_viewer():
    backend, viewer = _backend()
    actual = {"name": "ur20", "source": "official", "urdf_path": "official.urdf", "translation_m": [1., 2., .8], "yaw_deg": 90., "model_nq": 13}
    viewer.robot_info = lambda: [actual]
    assert backend.catalog()["robots"] == [actual]


def test_layout_server_catalog_preview_reset_and_export():
    backend, _ = _backend()
    server = LayoutControlServer(("127.0.0.1", 0), backend, "http://127.0.0.1:7000/static/")
    thread = server.start_background()
    try:
        with urlopen(server.control_url + "api/catalog", timeout=3) as response:
            catalog = json.load(response)
        assert catalog["mode"] == "layout"
        assert "parameter_schema" not in catalog
        with urlopen(server.control_url + "meshcat/", timeout=3) as response:
            camera_page = response.read().decode("utf-8")
        assert "viewer.controls.mouseButtons.MIDDLE = viewer.controls.mouseButtons.RIGHT" in camera_page
        assert "viewer.connect(websocket.href)" in camera_page
        with urlopen(server.control_url + "meshcat/main.min.js", timeout=3) as response:
            assert response.headers["Content-Type"].startswith("text/javascript")
            assert response.read() == server.meshcat_script
        with pytest.raises(HTTPError) as exc:
            urlopen(server.control_url + "meshcat/ws", timeout=3)
        assert exc.value.code == 400
        request = Request(
            server.control_url + "api/layout",
            data=json.dumps({"offsets": {"/World/LoadGroup": [.25, 0, 0]}}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=3) as response:
            preview = json.load(response)
        assert preview["robots"][0]["translation_m"] == [1.25, 2., .8]
        with urlopen(server.control_url + "api/export", timeout=3) as response:
            exported = json.load(response)
        assert exported["offsets"] == preview["offsets"]
        invalid = Request(server.control_url + "api/layout", data=b'{"offsets":{"bad":[0,0,0]}}')
        with pytest.raises(HTTPError) as exc:
            urlopen(invalid, timeout=3)
        assert exc.value.code == 400
        assert backend.offsets == {"/World/LoadGroup": (.25, 0., 0.)}
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(server.control_url + "api/evaluate", data=b'{}'), timeout=3)
        assert exc.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_layout_page_has_translation_controls_without_process_inputs():
    page = build_layout_control_page('http://localhost:7000/static/?label="layout"')
    assert 'src="/meshcat/"' in page
    assert 'data-source-url="http://localhost:7000/static/?label=&quot;layout&quot;"' in page
    for control in ("node", "dx", "dy", "dz", "resetNode", "resetAll", "export",
                    "viewPerspective", "viewFront", "viewSide", "viewTop"):
        assert f'id="{control}"' in page
    assert 'fetch("/api/layout"' in page
    assert 'fetch("/api/export"' in page
    assert "[nodeSelect.value]: values" in page
    assert 'id="sku"' not in page
    assert "/api/evaluate" not in page
    assert "nominal_q" not in page


def test_layout_cli_uses_imported_robot_positions_without_case_grid(tmp_path, capsys):
    calls = []
    scene = _scene()
    viewer = _Viewer()
    backend, _ = _backend()
    server = SimpleNamespace(
        control_url="http://127.0.0.1:8767/",
        serve_forever=lambda: calls.append("serve"),
        server_close=lambda: calls.append("close"),
    )

    def importer(source, robots, options):
        calls.append(("import", source, robots, options))
        return scene

    dependencies = LayoutDependencies(
        spec_loader=lambda path: SimpleNamespace(robots={"ur20": "URDF settings"}),
        config_loader=lambda path: {"usd_layout": {"file": str(tmp_path / "configured.usd"), "groups": ["/World/LoadGroup"]}},
        layout_loader=importer,
        viewer_factory=lambda **kwargs: viewer,
        backend_factory=lambda *args, **kwargs: backend,
        server_factory=lambda *args: server,
        browser_open=lambda url: calls.append(("open", url)),
    )
    result = main(["--usd", str(tmp_path / "override.usd"), "--open"], dependencies=dependencies)
    assert result == 0
    assert calls[0][1] == tmp_path / "override.usd"
    assert calls[0][2] == {"ur20": "URDF settings"}
    assert calls[0][3] == {"groups": ["/World/LoadGroup"]}
    assert calls[-2:] == ["serve", "close"]
    assert "LAYOUT CONTROL UI" in capsys.readouterr().out
    args = build_parser().parse_args([])
    assert args.port == 8767
    assert not hasattr(args, "initial_grid")


def test_layout_cli_import_failure_does_not_start_server(capsys):
    def fail(*args):
        raise ValueError("invalid USD layout")

    dependencies = LayoutDependencies(
        spec_loader=lambda path: SimpleNamespace(robots={}),
        config_loader=lambda path: {"usd_layout": {"file": "fixture.usd"}},
        layout_loader=fail,
        viewer_factory=fail,
        backend_factory=fail,
        server_factory=fail,
        browser_open=fail,
    )
    assert main([], dependencies=dependencies) == 2
    assert "invalid USD layout" in capsys.readouterr().err


def test_camera_iframe_tunnels_initial_and_live_meshcat_commands():
    import asyncio

    import meshcat
    import umsgpack
    from tornado.httpclient import HTTPClientError, HTTPRequest
    from tornado.websocket import websocket_connect

    native = meshcat.Visualizer()
    backend, _ = _backend()
    server = LayoutControlServer(("127.0.0.1", 0), backend, native.url())
    thread = server.start_background()
    camera = native["/Cameras/default/rotated/<object>"]
    camera.set_property("position", [3., 2., 1.])

    async def check_stream():
        url = server.control_url.replace("http:", "ws:") + "meshcat/ws"
        request = HTTPRequest(url, headers={"Origin": server.control_url.rstrip("/")}, connect_timeout=3)
        connection = await websocket_connect(request)
        try:
            initial = umsgpack.unpackb(await asyncio.wait_for(connection.read_message(), 3))
            assert initial["type"] == "set_property"
            assert initial["path"] == "/Cameras/default/rotated/<object>"
            assert initial["property"] == "position"
            assert initial["value"] == [3., 2., 1.]
            camera.set_property("position", [4., 2., 1.])
            updated = umsgpack.unpackb(await asyncio.wait_for(connection.read_message(), 3))
            assert updated["value"] == [4., 2., 1.]
        finally:
            connection.close()
        with pytest.raises(HTTPClientError) as exc:
            await websocket_connect(HTTPRequest(url, headers={"Origin": "http://other.example"}, connect_timeout=3))
        assert exc.value.code == 403

    try:
        asyncio.run(check_stream())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        native.window.zmq_socket.close(linger=0)
        native.window.server_proc.terminate()
        native.window.server_proc.wait(timeout=5)
