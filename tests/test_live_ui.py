from __future__ import annotations

import re

import pytest

from decanting_workspace.live_ui import build_live_control_page


def test_page_contains_live_parameter_and_meshcat_controls():
    page = build_live_control_page("http://127.0.0.1:7000/static/")

    assert '<iframe id="meshcat" src="http://127.0.0.1:7000/static/"' in page
    assert 'id="parameterControls"' in page
    assert 'id="profile"' in page
    assert 'id="profileDescription"' in page
    assert 'id="autoEvaluate" type="checkbox" checked' in page
    assert 'id="evaluate"' in page
    assert 'id="busy"' in page
    assert 'id="elapsed"' in page
    assert 'id="evaluationSummary" class="result-box"' in page
    assert 'id="status" class="result-box" aria-live="polite"' in page
    assert "LIVE CONTROL UI · PARAMETERS + RESULTS" in page
    assert 'id="step"' in page
    assert 'id="check"' in page
    assert 'id="sample"' in page
    assert 'id="ellipsoidScale"' in page
    assert 'id="showEllipsoid"' in page


def test_page_consumes_flat_catalog_schema_and_builds_both_parameter_kinds():
    page = build_live_control_page("http://localhost:7000/static/")

    assert 'fetch("/api/catalog"' in page
    assert "catalog.initial_parameters" in page
    assert "catalog.parameter_schema" in page
    assert "catalog.profiles" in page
    assert "catalog.default_profile" in page
    assert 'definition.kind === "number"' in page
    assert 'definition.kind === "choice"' in page
    assert "slider.dataset.parameterSlider = key" in page
    assert "number.dataset.parameterInput = key" in page
    assert "select.dataset.parameterChoice = key" in page
    assert "for (const input of [slider, number])" in page
    assert "for (const input of (slider, number))" not in page
    assert 'slider.addEventListener("input"' in page
    assert 'number.addEventListener("input"' in page


def test_page_posts_exact_live_evaluation_contract_and_replays_nested_selection():
    page = build_live_control_page("http://localhost:7000/static/")

    assert "const parameters = readParameters();" in page
    assert "const profile = selectedProfile();" in page
    assert "const ellipsoid_scale = Number(ellipsoidScale.value);" in page
    assert "const show_ellipsoid = showEllipsoid.checked;" in page
    assert "return {parameters, profile, ellipsoid_scale, show_ellipsoid};" in page
    assert 'postJson("/api/evaluate", payload)' in page
    assert "result.case" in page
    assert "result.selection" in page
    assert "result.elapsed_seconds" in page
    assert "result.profile" in page
    assert "showEvaluationSummary(result);" in page
    assert "Cell feasible:" in page
    assert "Installation valid:" in page
    assert "Hard task checks:" in page
    assert "SR workspace:" in page
    assert "activeCase.id" in page
    assert 'postJson("/api/select", payload)' in page
    assert re.search(r"case_id:\s*activeCase\.id", page)
    assert re.search(r"step_index:\s*stepSelect\.disabled", page)
    assert re.search(r"check_index:\s*checkSelect\.disabled", page)
    assert re.search(r"sample_index:\s*sampleSelect\.disabled", page)


def test_page_debounces_and_keeps_only_latest_queued_evaluation():
    page = build_live_control_page("http://localhost:7000/static/")

    assert "const EVALUATION_DEBOUNCE_MS = 500;" in page
    assert "clearTimeout(evaluationTimer);" in page
    assert "}, EVALUATION_DEBOUNCE_MS);" in page
    assert "if (evaluationInFlight)" in page
    assert "evaluationRerunRequested = true;" in page
    assert "if (evaluationRerunRequested)" in page
    assert "evaluationRerunRequested = false;" in page
    assert re.search(
        r"if \(evaluationRerunRequested\).*?clearTimeout\(evaluationTimer\);"
        r".*?void evaluateLatest\(\);",
        page,
        re.DOTALL,
    )
    assert 'slider.addEventListener("change", scheduleParameterEvaluation)' in page
    assert 'number.addEventListener("change", scheduleParameterEvaluation)' in page
    assert 'select.addEventListener("change", scheduleParameterEvaluation)' in page
    assert 'profileSelect.addEventListener("change", () =>' in page
    assert "showProfileDescription();" in page
    assert "entry.description" in page
    assert 'evaluateButton.addEventListener("click"' in page
    assert re.search(
        r'catch \(error\) \{\s*setStatus\("failed", `입력 오류:.*?setBusy\(false\);',
        page,
        re.DOTALL,
    )
    assert "selectionVersion += 1;" in page
    assert "version === selectionVersion" in page
    assert "payload.case_id === activeCase?.id" in page


@pytest.mark.parametrize("url", ("file:///tmp/a", "not-a-url", "ftp://localhost/x"))
def test_page_reuses_meshcat_http_url_validation(url):
    with pytest.raises(ValueError, match="absolute http"):
        build_live_control_page(url)


def test_page_html_escapes_the_validated_meshcat_url():
    page = build_live_control_page(
        'http://127.0.0.1:7000/static/?view=cell&label="live"'
    )

    assert (
        'src="http://127.0.0.1:7000/static/?view=cell&amp;label=&quot;live&quot;"'
        in page
    )
