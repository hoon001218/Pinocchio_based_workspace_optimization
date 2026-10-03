"""Local controls for hierarchical USD layout translations."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import json
import os
from pathlib import Path
import select
import socket
import ssl
from threading import Lock, Thread
from urllib.parse import urlsplit


class LayoutControlServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = os.name != "nt"

    def __init__(self, address, backend, meshcat_url: str) -> None:
        self.backend = backend
        self.control_page = build_layout_control_page(meshcat_url)
        self.meshcat_url = urlsplit(meshcat_url)
        from meshcat.servers.zmqserver import VIEWER_ROOT

        self.meshcat_script = (Path(VIEWER_ROOT) / "main.min.js").read_bytes()
        self.preview_lock = Lock()
        super().__init__(address, _LayoutRequestHandler)

    def server_bind(self) -> None:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    @property
    def control_url(self) -> str:
        host, port = self.server_address[:2]
        visible_host = "127.0.0.1" if host in {"", "0.0.0.0", "::"} else host
        return f"http://{visible_host}:{port}/"

    def start_background(self) -> Thread:
        thread = Thread(target=self.serve_forever, name="layout-control", daemon=True)
        thread.start()
        return thread


class _LayoutRequestHandler(BaseHTTPRequestHandler):
    server: LayoutControlServer

    def do_GET(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        if path == "/":
            self._send(HTTPStatus.OK, self.server.control_page.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/meshcat/":
            self._send(HTTPStatus.OK, _MESHCAT_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/meshcat/main.min.js":
            self._send(HTTPStatus.OK, self.server.meshcat_script, "text/javascript; charset=utf-8")
        elif path == "/meshcat/ws":
            self._meshcat_websocket()
        elif path == "/api/catalog":
            with self.server.preview_lock:
                payload = self.server.backend.catalog()
            self._json(HTTPStatus.OK, payload)
        elif path == "/api/export":
            with self.server.preview_lock:
                payload = self.server.backend.export_payload()
            self._json(HTTPStatus.OK, payload)
        elif path == "/healthz":
            self._json(HTTPStatus.OK, {"status": "ok"})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path != "/api/layout":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 64 * 1024:
                raise ValueError("layout request body must be between 1 and 65536 bytes")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("layout request root must be an object")
            with self.server.preview_lock:
                result = self.server.backend.layout(payload)
            self._json(HTTPStatus.OK, result)
        except (KeyError, TypeError, ValueError, UnicodeError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid_layout", "message": str(exc)})
        except Exception as exc:  # pragma: no cover - defensive HTTP boundary
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "layout_failed", "message": str(exc)})

    def _meshcat_websocket(self) -> None:
        """Tunnel only this server's MeshCat stream for the same-origin iframe."""

        origin = self.headers.get("Origin")
        if origin and urlsplit(origin).netloc.lower() != self.headers.get("Host", "").lower():
            self._json(HTTPStatus.FORBIDDEN, {"error": "cross_origin_websocket"})
            return
        if (self.headers.get("Upgrade", "").lower() != "websocket"
                or "upgrade" not in [part.strip() for part in self.headers.get("Connection", "").lower().split(",")]
                or not self.headers.get("Sec-WebSocket-Key")):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "websocket_upgrade_required"})
            return
        upstream = None
        upgraded = False
        self.close_connection = True
        try:
            endpoint = self.server.meshcat_url
            port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
            upstream = socket.create_connection((endpoint.hostname, port), timeout=10)
            if endpoint.scheme == "https":
                upstream = ssl.create_default_context().wrap_socket(upstream, server_hostname=endpoint.hostname)
            # MeshCat's Tornado handler checks the origin including its port.
            # Browser access to this tunnel is checked above; the upstream sees
            # its own fixed origin instead of the layout control server's port.
            headers = [
                "GET / HTTP/1.1",
                f"Host: {endpoint.netloc}",
                "Upgrade: websocket",
                "Connection: Upgrade",
                f"Origin: {endpoint.scheme}://{endpoint.netloc}",
                f"Sec-WebSocket-Key: {self.headers['Sec-WebSocket-Key']}",
                f"Sec-WebSocket-Version: {self.headers.get('Sec-WebSocket-Version', '13')}",
            ]
            upstream.sendall(("\r\n".join(headers) + "\r\n\r\n").encode("ascii"))
            response = bytearray()
            while b"\r\n\r\n" not in response:
                block = upstream.recv(4096)
                if not block or len(response) + len(block) > 64 * 1024:
                    raise ValueError("invalid MeshCat websocket handshake")
                response.extend(block)
            header = bytes(response).split(b"\r\n\r\n", 1)[0]
            if header.split(b"\r\n", 1)[0].split()[1] != b"101":
                raise ValueError("MeshCat websocket upgrade was rejected")
            self.connection.sendall(response)
            upgraded = True
            upstream.settimeout(None)
            self.connection.settimeout(None)
            while True:
                readable, _, _ = select.select((self.connection, upstream), (), (), 1)
                for source in readable:
                    block = source.recv(64 * 1024)
                    if not block:
                        return
                    destination = upstream if source is self.connection else self.connection
                    destination.sendall(block)
        except (OSError, ValueError, IndexError):
            if not upgraded:
                self._json(HTTPStatus.BAD_GATEWAY, {"error": "meshcat_connection_failed"})
        finally:
            if upstream is not None:
                upstream.close()

    def _json(self, status, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send(self, status, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


def build_layout_control_page(meshcat_url: str) -> str:
    parsed = urlsplit(meshcat_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("MeshCat URL must be an absolute http or https URL")
    return _PAGE.replace("__MESHCAT_URL__", html.escape(meshcat_url, quote=True))


_PAGE = r'''<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>USD layout</title>
<style>
:root { color-scheme: dark; font-family: system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; color: #e6eaf0; background: #161b22; }
main { display: grid; grid-template-columns: 370px 1fr; height: 100vh; }
aside { overflow-y: auto; padding: 22px; border-right: 1px solid #39414d; }
h1 { margin: 0 0 12px; font-size: 23px; }
h2 { margin: 22px 0 8px; font-size: 16px; }
#toteDimensions { padding: 0; list-style: none; font-size: 13px; line-height: 1.8; }
p { font-size: 13px; line-height: 1.6; color: #aeb9c8; }
label { display: block; margin-top: 16px; font-size: 13px; }
select, input, button { font: inherit; color: inherit; background: #232b36; border: 1px solid #4a586a; border-radius: 5px; padding: 9px; }
select { width: 100%; margin-top: 7px; }
.axis { display: grid; grid-template-columns: 48px 1fr 24px; align-items: center; gap: 8px; }
.axis input { min-width: 0; width: 100%; }
.buttons { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
button { cursor: pointer; }
button.primary { background: #265b85; border-color: #3d84b9; }
pre, output { display: block; white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; line-height: 1.65; }
#status { padding: 12px; background: #202733; border-radius: 5px; margin-top: 18px; }
#status[data-level="failed"] { color: #ffaaaa; }
iframe { width: 100%; height: 100%; border: 0; }
@media (max-width: 760px) { main { grid-template-columns: 1fr; grid-template-rows: auto minmax(400px,1fr); height: auto; min-height: 100vh; } aside { border-right: 0; } iframe { min-height: 500px; } }
</style>
</head>
<body><main>
<aside>
<h1>USD layout</h1>
<p>USD 환경을 단순화한 형상과 URDF 로봇을 표시합니다. 예시 SKU와 토트는 환경 형상으로 생성하지 않습니다.</p>
<label>그룹 / 요소<select id="node"></select></label>
<p id="nodeDescription"></p>
<p>원본 자세에 대한 이동량입니다. 축은 부모의 로컬 좌표이며 회전은 유지합니다.</p>
<label class="axis">ΔX <input id="dx" type="number" step="0.01" value="0"> m</label>
<label class="axis">ΔY <input id="dy" type="number" step="0.01" value="0"> m</label>
<label class="axis">ΔZ <input id="dz" type="number" step="0.01" value="0"> m</label>
<div class="buttons"><button class="primary" id="apply">이동 적용</button><button id="resetNode">요소 초기화</button><button id="resetAll">전체 초기화</button><button id="export">JSON 저장</button></div>
<p>왼쪽 드래그: 회전 · 가운데/오른쪽 드래그: 이동 · 휠: 줌</p>
<div class="buttons" aria-label="카메라 보기"><button id="viewPerspective" data-camera-view="perspective">처음 보기</button><button id="viewFront" data-camera-view="front">정면</button><button id="viewSide" data-camera-view="side">측면</button><button id="viewTop" data-camera-view="top">상단</button></div>
<output id="status" role="status">레이아웃을 불러오는 중…</output>
<pre id="stats"></pre><pre id="robots"></pre>
<section id="toteReferences" hidden aria-labelledby="toteReferenceHeading"><h2 id="toteReferenceHeading">토트 참고 치수</h2><p>USD에서 읽은 외형 치수입니다.</p><ul id="toteDimensions"></ul></section>
</aside>
<iframe id="meshcat" src="/meshcat/" data-source-url="__MESHCAT_URL__" title="Simplified USD layout"></iframe>
</main>
<script>
"use strict";
const nodeSelect = document.getElementById("node");
const axes = ["dx", "dy", "dz"].map(id => document.getElementById(id));
const status = document.getElementById("status");
let nodes = [];
let offsets = {};
let busy = false;
function showStatus(message, level = "success") { status.textContent = message; status.dataset.level = level; }
function showReferenceDimensions(references) {
  const groups = new Map();
  for (const reference of references.filter(item => item.role === "tote")) {
    const dimensions = reference.size_m.map(value => Number((Number(value) * 1000).toFixed(1)));
    const key = dimensions.join(",");
    if (!groups.has(key)) groups.set(key, {dimensions, sources: []});
    groups.get(key).sources.push(reference.source_prim_path);
  }
  const list = document.getElementById("toteDimensions");
  list.replaceChildren();
  for (const group of groups.values()) {
    const item = document.createElement("li");
    const count = group.sources.length > 1 ? ` · ${group.sources.length}개` : "";
    item.textContent = `${group.dimensions.map(value => value.toLocaleString("ko-KR")).join(" × ")} mm${count}`;
    item.title = group.sources.join("\n");
    list.append(item);
  }
  document.getElementById("toteReferences").hidden = groups.size === 0;
}
function showPreview(preview) {
  showReferenceDimensions(preview.reference_dimensions ?? []);
  const s = preview.stats;
  document.getElementById("stats").textContent = `요소 ${s.layout_nodes} · 원본 프록시 ${s.source_boxes}\n겹침 처리 후 형상 ${s.simplified_fragments} · 로봇 ${s.robots}\n중복 제거 부피 ${Number(s.removed_overlap_volume_m3).toFixed(4)} m³`;
  document.getElementById("robots").textContent = preview.robots.map(r => `${r.name}\n  XYZ ${r.translation_m.map(v => Number(v).toFixed(3)).join(", ")} m\n  yaw ${Number(r.yaw_deg).toFixed(1)}°`).join("\n\n");
}
function showNode() {
  const node = nodes.find(item => item.prim_path === nodeSelect.value);
  if (!node) return;
  const value = offsets[node.prim_path] ?? [0, 0, 0];
  axes.forEach((input, index) => { input.value = String(value[index]); });
  document.getElementById("nodeDescription").textContent = `부모: ${node.parent_path || "World"}\n원본 World XYZ: ${node.translation_m.map(v => Number(v).toFixed(3)).join(", ")} m`;
}
async function update(nextOffsets) {
  if (busy) return;
  busy = true;
  document.querySelectorAll("button").forEach(button => { button.disabled = true; });
  showStatus("레이아웃을 갱신하는 중…");
  try {
    const response = await fetch("/api/layout", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({offsets: nextOffsets})});
    const preview = await response.json();
    if (!response.ok) throw new Error(preview.message || preview.error);
    offsets = preview.offsets;
    showNode(); showPreview(preview); showStatus("이동을 적용했습니다.");
  } catch (error) { showStatus(error.message, "failed"); }
  finally { busy = false; document.querySelectorAll("button").forEach(button => { button.disabled = false; }); }
}
document.querySelectorAll("[data-camera-view]").forEach(button => {
  button.addEventListener("click", () => {
    const setView = document.getElementById("meshcat").contentWindow?.setLayoutCameraView;
    if (typeof setView !== "function" || !setView(button.dataset.cameraView)) {
      showStatus("3D 뷰를 불러오는 중입니다.", "failed");
    }
  });
});
document.getElementById("apply").addEventListener("click", () => {
  const values = axes.map(input => Number(input.value));
  if (axes.some(input => input.value.trim() === "") || values.some(value => !Number.isFinite(value))) { showStatus("XYZ에 유한한 숫자를 입력하세요.", "failed"); return; }
  void update({...offsets, [nodeSelect.value]: values});
});
document.getElementById("resetNode").addEventListener("click", () => { const next = {...offsets}; delete next[nodeSelect.value]; void update(next); });
document.getElementById("resetAll").addEventListener("click", () => { void update({}); });
nodeSelect.addEventListener("change", showNode);
document.getElementById("export").addEventListener("click", async () => {
  try {
    const response = await fetch("/api/export");
    if (!response.ok) throw new Error("JSON 내보내기에 실패했습니다.");
    const data = await response.json();
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: "application/json"}));
    const link = document.createElement("a"); link.href = url; link.download = "layout.json"; link.click(); URL.revokeObjectURL(url);
  } catch (error) { showStatus(error.message, "failed"); }
});
fetch("/api/catalog").then(response => { if (!response.ok) throw new Error("레이아웃을 불러오지 못했습니다."); return response.json(); }).then(catalog => {
  nodes = catalog.nodes; offsets = catalog.offsets;
  for (const node of nodes) { const option = document.createElement("option"); option.value = node.prim_path; option.textContent = node.prim_path; nodeSelect.append(option); }
  showNode(); showPreview(catalog); showStatus("그룹 또는 요소를 선택해 XYZ 이동량을 적용하세요.");
}).catch(error => { showStatus(error.message, "failed"); });
</script></body></html>'''


_MESHCAT_PAGE = r'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>USD layout 3D view</title>
<style>body { margin: 0; } #meshcat-pane { width: 100vw; height: 100vh; overflow: hidden; }</style>
</head><body><div id="meshcat-pane"></div><script src="/meshcat/main.min.js"></script><script>
"use strict";
const viewer = new MeshCat.Viewer(document.getElementById("meshcat-pane"));
// Layout geometry and camera poses use World Z-up. MeshCat's default Scene
// rotates these coordinates for its Y-up camera; remove that conversion here.
viewer.scene.quaternion.identity();
let initialCamera = null;
function configureCameraControls() {
  // Use the installed OrbitControls mapping, without depending on enum numbers.
  viewer.controls.mouseButtons.MIDDLE = viewer.controls.mouseButtons.RIGHT;
  viewer.controls.screenSpacePanning = true;
  viewer.controls.minDistance = Math.max(0.02, viewer.camera.near * 2);
}
function setWorldCameraTarget(target) {
  const camera = viewer.camera;
  // OrbitControls caches the up-axis conversion in its constructor. Recreate
  // the controls after setting Z-up, rather than changing up on live controls.
  if (camera.up.z !== 1) {
    viewer.controls.dispose();
    camera.up.set(0, 0, 1);
    viewer.set_camera(camera);
  }
  configureCameraControls();
  viewer.controls.target.set(...target);
  viewer.controls.update();
  viewer.controls.saveState();
  initialCamera = {position: camera.position.clone(), target: viewer.controls.target.clone()};
  viewer.set_dirty();
}
configureCameraControls();
const handleCommand = viewer.handle_command.bind(viewer);
viewer.handle_command = command => {
  handleCommand(command);
  configureCameraControls();
  if (command.type === "set_property" && command.property === "layout_target"
      && command.path === "/Cameras/default/rotated/<object>") {
    setWorldCameraTarget(command.value);
  }
};
window.setLayoutCameraView = name => {
  if (!initialCamera) return false;
  const camera = viewer.camera;
  const controls = viewer.controls;
  if (name === "perspective") {
    camera.position.copy(initialCamera.position);
    controls.target.copy(initialCamera.target);
  } else {
    const direction = {front: [0, 1, 0], side: [-1, 0, 0], top: [0, 0.000001, 1]}[name];
    if (!direction) return false;
    const distance = Math.max(controls.minDistance, camera.position.distanceTo(controls.target));
    const offset = camera.position.clone().set(...direction).normalize().multiplyScalar(distance);
    camera.position.copy(controls.target).add(offset);
  }
  controls.update();
  viewer.set_dirty();
  return true;
};
const websocket = new URL("/meshcat/ws", window.location.href);
websocket.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
viewer.connect(websocket.href);
</script></body></html>'''
