"""Exercise installed MeshCat OrbitControls in an optional real headless browser."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
from urllib.request import Request, urlopen

import numpy as np
import pytest

from decanting_workspace.layout_backend import LayoutPreviewBackend
from decanting_workspace.layout_models import LayoutBox, LayoutNode, LayoutScene
from decanting_workspace.layout_ui import LayoutControlServer
from decanting_workspace.layout_viewer import LayoutViewer


def _chrome_binary():
    candidates = [shutil.which(name) for name in ("chromium", "chromium-browser", "google-chrome", "chrome")]
    for directory in (os.environ.get("PROGRAMFILES", r"C:\Program Files") if os.name == "nt" else None,
                      os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)") if os.name == "nt" else None):
        if directory:
            candidates.extend((str(Path(directory) / "Google/Chrome/Application/chrome.exe"),
                               str(Path(directory) / "Microsoft/Edge/Application/msedge.exe")))
    return next((candidate for candidate in candidates if candidate and Path(candidate).is_file()), None)


@pytest.fixture
def camera_browser(tmp_path):
    chrome_binary = _chrome_binary()
    if chrome_binary is None:
        pytest.skip("Camera interaction regression needs an installed Chrome/Chromium/Edge browser")
    import meshcat

    native = meshcat.Visualizer()
    identity = (0., 0., 0., 1.)
    scene = LayoutScene(Path("camera.usd"), (LayoutNode("/Cell", None, (8., -9., 3.), identity),), (
        LayoutBox("cell", "/Cell", (8., -9., 3.), (2., 3., 1.2), identity),
        LayoutBox("room_floor", "/Cell", (0., 0., -.05), (50., 50., .1), identity, "ground"),
        LayoutBox("room_wall", "/Cell", (0., -25., 2.5), (50., .1, 5.), identity, "ground"),
    ), ())
    backend = LayoutPreviewBackend(scene, LayoutViewer(native, ur20_source="configured", show_frames=False))
    server = LayoutControlServer(("127.0.0.1", 0), backend, native.url())
    thread = server.start_background()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        debug_port = reservation.getsockname()[1]
    browser = subprocess.Popen([
        chrome_binary, "--headless=new", "--disable-gpu", "--enable-unsafe-swiftshader",
        "--no-sandbox", "--disable-gpu-sandbox", "--in-process-gpu", "--no-first-run",
        "--no-default-browser-check", f"--user-data-dir={tmp_path / 'chrome-profile'}",
        f"--remote-debugging-port={debug_port}", "--remote-allow-origins=*",
        "--window-size=1200,900", "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        yield server, debug_port, browser
    finally:
        browser.terminate()
        browser.wait(timeout=10)
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
        native.window.zmq_socket.close(linger=0)
        native.window.server_proc.terminate()
        native.window.server_proc.wait(timeout=5)


class _CameraBrowser:
    def __init__(self, websocket):
        self.websocket = websocket
        self.sequence = 0
        self.events = set()

    async def command(self, method, parameters=None):
        self.sequence += 1
        await self.websocket.write_message(json.dumps({
            "id": self.sequence, "method": method, "params": parameters or {},
        }))
        while True:
            message = json.loads(await asyncio.wait_for(self.websocket.read_message(), 10))
            if "method" in message:
                self.events.add(message["method"])
            if message.get("id") == self.sequence:
                assert "error" not in message, message
                return message.get("result", {})

    async def evaluate(self, expression):
        result = await self.command("Runtime.evaluate", {
            "expression": expression, "returnByValue": True, "awaitPromise": True,
        })
        assert "exceptionDetails" not in result, result
        return result["result"].get("value")

    async def state(self):
        return await self.evaluate('''new Promise(resolve => requestAnimationFrame(() => {
            const v = document.getElementById("meshcat").contentWindow.eval("viewer");
            v.render();
            resolve({position: v.camera.position.toArray(),
                world: v.camera.getWorldPosition(v.camera.position.clone()).toArray(),
                target: v.controls.target.toArray(), up: v.camera.up.toArray(),
                projected: v.controls.target.clone().project(v.camera).toArray(),
                distance: v.controls.getDistance(), minDistance: v.controls.minDistance,
                parent: v.camera.parent.matrixWorld.toArray(), buttons: v.controls.mouseButtons});
        }))''')

    async def drag(self, x, y, dx, dy, button="left", buttons=1):
        await self.command("Input.dispatchMouseEvent", {
            "type": "mousePressed", "x": x, "y": y, "button": button, "buttons": buttons, "clickCount": 1,
        })
        await self.command("Input.dispatchMouseEvent", {
            "type": "mouseMoved", "x": x + dx, "y": y + dy, "button": button, "buttons": buttons,
        })
        await self.command("Input.dispatchMouseEvent", {
            "type": "mouseReleased", "x": x + dx, "y": y + dy, "button": button, "buttons": 0, "clickCount": 1,
        })

    async def wheel(self, x, y, delta):
        await self.command("Input.dispatchMouseEvent", {
            "type": "mouseWheel", "x": x, "y": y, "deltaX": 0, "deltaY": delta,
        })


def _assert_world_focus(state, target):
    np.testing.assert_allclose(state["world"], state["position"], atol=1e-9)
    np.testing.assert_allclose(state["target"], target, atol=1e-9)
    np.testing.assert_allclose(state["projected"][:2], (0., 0.), atol=1e-9)
    np.testing.assert_allclose(state["up"], (0., 0., 1.), atol=1e-9)
    np.testing.assert_allclose(state["parent"], np.eye(4).T.ravel(), atol=1e-9)
    assert np.linalg.norm(np.asarray(state["world"]) - target) == pytest.approx(state["distance"], abs=1e-9)


def test_camera_world_focus_orbit_pan_zoom_and_presets(camera_browser):
    async def exercise():
        from tornado.websocket import websocket_connect

        server, debug_port, process = camera_browser
        target = None
        for _ in range(100):
            try:
                request = Request(f"http://127.0.0.1:{debug_port}/json/new?about:blank", method="PUT")
                with urlopen(request, timeout=1) as response:
                    target = json.load(response)
                break
            except OSError:
                assert process.poll() is None, "Headless browser exited before initialization"
                await asyncio.sleep(.1)
        assert target, "Headless browser debugging endpoint did not start"
        websocket = await websocket_connect(target["webSocketDebuggerUrl"], connect_timeout=5)
        page = _CameraBrowser(websocket)
        try:
            await page.command("Page.enable")
            await page.command("Page.navigate", {"url": server.control_url})
            while "Page.loadEventFired" not in page.events:
                message = json.loads(await asyncio.wait_for(websocket.read_message(), 10))
                if "method" in message:
                    page.events.add(message["method"])
            await page.evaluate('''new Promise((resolve, reject) => {
                const deadline = Date.now() + 10000;
                const timer = setInterval(() => {
                    if (document.getElementById("meshcat")?.contentWindow?.setLayoutCameraView?.("perspective")) {
                        clearInterval(timer); resolve(true);
                    } else if (Date.now() > deadline) {
                        clearInterval(timer); reject(new Error("World camera did not initialize"));
                    }
                }, 50);
            })''')
            rect = await page.evaluate('document.getElementById("meshcat").getBoundingClientRect().toJSON()')
            x, y = rect["x"] + rect["width"] * .55, rect["y"] + rect["height"] * .6
            initial = await page.state()
            focus = np.array((8., -9., 3.))
            _assert_world_focus(initial, focus)
            assert initial["buttons"]["MIDDLE"] == initial["buttons"]["RIGHT"]
            assert initial["buttons"]["LEFT"] != initial["buttons"]["RIGHT"]

            # A horizontal orbit must rotate around World Z, not the old cached Y-up axis.
            await page.drag(x, y, 85, 0)
            horizontal = await page.state()
            _assert_world_focus(horizontal, focus)
            assert horizontal["position"] != initial["position"]
            assert horizontal["world"][2] == pytest.approx(initial["world"][2], abs=1e-8)
            assert horizontal["distance"] == pytest.approx(initial["distance"], abs=1e-8)
            await page.drag(x, y, 0, 50)
            vertical = await page.state()
            _assert_world_focus(vertical, focus)
            assert vertical["world"][2] != pytest.approx(horizontal["world"][2], abs=1e-8)
            assert vertical["distance"] == pytest.approx(initial["distance"], abs=1e-8)

            ray = (np.asarray(vertical["world"]) - focus) / vertical["distance"]
            previous = vertical
            for delta in (-100.,) * 4 + (100.,) * 4:
                await page.wheel(x, y, delta)
                current = await page.state()
                _assert_world_focus(current, focus)
                assert (current["distance"] < previous["distance"]) == (delta < 0)
                np.testing.assert_allclose((np.asarray(current["world"]) - focus) / current["distance"], ray, atol=1e-8)
                previous = current

            # Pan moves camera and orbit focus together, keeping the focus projected at the centre.
            for button, buttons in (("middle", 4), ("right", 2)):
                before = await page.state()
                await page.drag(x, y, 45, 25, button, buttons)
                after = await page.state()
                assert after["target"] != before["target"]
                _assert_world_focus(after, np.asarray(after["target"]))
                assert after["distance"] == pytest.approx(before["distance"], abs=1e-8)
                np.testing.assert_allclose(np.asarray(after["world"]) - before["world"],
                                           np.asarray(after["target"]) - before["target"], atol=1e-8)

            # Repeated close zoom reaches a positive radius; the next orbit never crosses or jumps past the focus.
            for _ in range(180):
                await page.wheel(x, y, -100.)
            close = await page.state()
            close_focus = np.asarray(close["target"])
            _assert_world_focus(close, close_focus)
            assert close["distance"] == pytest.approx(close["minDistance"], abs=1e-8)
            await page.drag(x, y, 35, 20)
            close_orbit = await page.state()
            _assert_world_focus(close_orbit, close_focus)
            assert close_orbit["distance"] == pytest.approx(close["distance"], abs=1e-8)
            assert np.linalg.norm(np.asarray(close_orbit["world"]) - close["world"]) <= 2 * close["distance"] + 1e-9

            for button, axis, sign in (("viewFront", 1, 1), ("viewSide", 0, -1), ("viewTop", 2, 1)):
                await page.evaluate(f'document.getElementById("{button}").click(); true')
                preset = await page.state()
                _assert_world_focus(preset, close_focus)
                offset = np.asarray(preset["world"]) - close_focus
                assert offset[axis] * sign > 0
                assert all(abs(offset[index]) < 1e-6 for index in range(3) if index != axis)

            await page.evaluate('document.getElementById("viewPerspective").click(); true')
            reset = await page.state()
            _assert_world_focus(reset, focus)
            np.testing.assert_allclose(reset["world"], initial["world"], atol=1e-8)
            await page.evaluate('''fetch("/api/layout", {method: "POST", headers: {"Content-Type":"application/json"},
                body: JSON.stringify({offsets: {"/Cell": [.25, 0, 0]}})}).then(response => response.json())''')
            unchanged = await page.state()
            _assert_world_focus(unchanged, focus)
            np.testing.assert_allclose(unchanged["world"], reset["world"], atol=1e-8)
        finally:
            websocket.close()
    asyncio.run(exercise())
