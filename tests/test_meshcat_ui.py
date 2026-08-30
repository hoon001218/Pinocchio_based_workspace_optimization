from __future__ import annotations

import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from decanting_workspace.meshcat_ui import (
    MeshcatControlServer,
    build_control_page,
)


class _Backend:
    def __init__(self) -> None:
        self.requests = []

    def catalog(self):
        return {"cases": [{"id": "case-1", "key": {}, "steps": []}]}

    def select(self, request):
        self.requests.append(dict(request))
        if request.get("case_id") != "case-1":
            raise KeyError("unknown case")
        return {"step_status": "success", "status_text": "cached"}


def _json(url: str, *, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urlopen(request, timeout=2.0) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def test_page_contains_parameter_step_and_meshcat_controls():
    page = build_control_page("http://127.0.0.1:7000/static/")

    assert '<iframe id="meshcat" src="http://127.0.0.1:7000/static/"' in page
    assert 'data-slider-key="ur_x"' in page
    assert 'data-input-key="ur_x"' in page
    assert 'data-slider-key="sr_z"' in page
    assert 'data-input-key="sr_z"' in page
    assert 'data-slider-key="lift"' in page
    assert 'data-input-key="lift"' in page
    assert 'data-key="sku"' in page
    assert 'id="step"' in page
    assert 'id="ellipsoidScale"' in page
    assert 'id="cacheSummary"' in page
    assert "data.default_case_id" in page
    assert 'fetch("/api/catalog")' in page
    assert 'fetch("/api/select"' in page
    assert "checkSelect.value=String(step.checks.length-1)" in page
    assert "is not precomputed" in page


@pytest.mark.parametrize("url", ("file:///tmp/a", "not-a-url", "ftp://localhost/x"))
def test_page_rejects_non_http_meshcat_urls(url):
    with pytest.raises(ValueError, match="absolute http"):
        build_control_page(url)


def test_local_server_serves_catalog_and_serializes_selections():
    backend = _Backend()
    server = MeshcatControlServer(
        ("127.0.0.1", 0),
        backend,
        "http://127.0.0.1:7000/static/",
    )
    thread = server.start_background()
    try:
        status, health = _json(server.control_url + "healthz")
        assert status == 200 and health == {"status": "ok"}

        status, catalog = _json(server.control_url + "api/catalog")
        assert status == 200
        assert catalog["meshcat_url"] == "http://127.0.0.1:7000/static/"
        assert catalog["cases"][0]["id"] == "case-1"

        status, selected = _json(
            server.control_url + "api/select",
            payload={"case_id": "case-1", "step_index": 1},
        )
        assert status == 200
        assert selected["status_text"] == "cached"
        assert backend.requests == [{"case_id": "case-1", "step_index": 1}]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def test_local_server_returns_structured_bad_selection():
    server = MeshcatControlServer(
        ("127.0.0.1", 0),
        _Backend(),
        "http://127.0.0.1:7000/static/",
    )
    thread = server.start_background()
    try:
        with pytest.raises(HTTPError) as caught:
            _json(
                server.control_url + "api/select",
                payload={"case_id": "missing"},
            )
        assert caught.value.code == 400
        payload = json.loads(caught.value.read().decode("utf-8"))
        assert payload["error"] == "invalid_selection"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
