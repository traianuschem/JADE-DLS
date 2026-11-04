"""Tests for ade_dls.engine.pipeline (Qt-free analysis-pipeline core) and the Qt wrapper
ade_dls.gui.core.pipeline.TransparentPipeline (signals only)."""
import json
import os

import pandas as pd
import pytest

from ade_dls.engine.pipeline import AnalysisStep, Pipeline


def _double(x, factor=2):
    return x * factor


@pytest.fixture
def pipe():
    return Pipeline()


# ----------------------------------------------------------------------------- AnalysisStep
def test_analysis_step_execute_passes_args_params_and_stores_result():
    step = AnalysisStep("double", _double, {"factor": 3})
    assert step.execute(5) == 15 and step.result == 15
    assert step.function_name == "_double" and "def _double" in step.code


def test_analysis_step_without_function_does_nothing():
    step = AnalysisStep("manual")
    assert step.execute() is None and step.result is None


def test_analysis_step_to_dict_is_json_serializable():
    d = AnalysisStep("double", _double, {"factor": 3}).to_dict()
    assert set(d) == {"name", "function", "params", "timestamp", "metadata", "step_type"}
    assert json.loads(json.dumps(d))["params"] == {"factor": 3}


def test_analysis_step_generate_code_variants():
    assert "result_my_step = _double(factor=3)" in AnalysisStep("my step", _double, {"factor": 3}).generate_code()
    assert "x = 1" in AnalysisStep("c", custom_code="x = 1", step_type="custom").generate_code()
    assert "Manual step" in AnalysisStep("m").generate_code()


# ----------------------------------------------------------------------------- Pipeline
def test_add_step_records_and_notifies(pipe):
    got = []
    pipe.subscribe("step_added", got.append)
    step = pipe.add_step("double", _double, {"factor": 4})
    assert pipe.steps == [step]
    assert len(got) == 1 and got[0]["name"] == "double" and got[0]["params"] == {"factor": 4}


def test_unknown_event_is_rejected(pipe):
    with pytest.raises(ValueError):
        pipe.subscribe("nope", print)


def test_execute_all_runs_steps_in_order_and_emits_results(pipe):
    done = []
    pipe.subscribe("analysis_completed", done.append)
    pipe.add_step("a", lambda: 1)
    pipe.add_step("b", lambda: 2)
    pipe.execute_all()
    assert pipe.results == {"a": 1, "b": 2}
    assert done == [{"a": 1, "b": 2}]


def test_execute_step_index_out_of_range(pipe):
    with pytest.raises(IndexError):
        pipe.execute_step(0)


def test_filter_step_params_and_noise_params_only_when_active(pipe):
    s = pipe.add_filter_step("countrate", ["a.asc", "b.asc"], 10, 8,
                             noise_params={"baseline_correction": False, "intercept_correction": False})
    assert s.step_type == "filter" and s.params["excluded_files"] == ["a.asc", "b.asc"]
    assert "noise_params" not in s.params
    s2 = pipe.add_filter_step("correlation", [], 8, 8,
                              noise_params={"baseline_correction": True, "baseline_pct": 5,
                                            "ignored": 1})
    assert s2.params["noise_params"] == {"baseline_correction": True, "baseline_pct": 5}
    assert "No files excluded" in s2.custom_code


@pytest.mark.parametrize("method,laplace", [("Method C", False), ("NNLS", True), ("Regularized", True)])
def test_post_refinement_step_generates_compilable_code(pipe, method, laplace):
    s = pipe.add_post_refinement_step(method, q_range=(1e-4, 5e-4), excluded_files=["x.asc"],
                                      is_laplace=laplace, extra_params={"fit_through_origin": True,
                                                                        "method": "ignored"})
    assert s.name == f"{method} Post-Refinement" and s.step_type == "refinement"
    assert s.params["method"] == method                    # explicit keys win over extra_params
    assert s.params["fit_through_origin"] is True
    assert s.params["q_range"] == (1e-4, 5e-4) and s.params["excluded_files"] == ["x.asc"]
    compile(s.generate_code(), f"<{method}>", "exec")


def test_generated_script_is_valid_python(pipe, tmp_path):
    pipe.add_filter_step("countrate", ["a.asc"], 4, 3)
    pipe.add_post_refinement_step("NNLS", is_laplace=True)
    code = pipe.get_current_code()
    compile(code, "<script>", "exec")
    assert "Step 1: Filter Countrate" in code and "Step 2: NNLS Post-Refinement" in code
    out = tmp_path / "s.py"
    pipe.export_to_script(str(out))
    assert out.read_text() == code or "Step 2" in out.read_text()


def test_parameter_history_and_summary(pipe):
    pipe.add_step("double", _double, {"factor": 4, "x": 1})
    h = pipe.get_parameter_history()
    assert list(h["parameter"]) == ["factor", "x"] and list(h["value"]) == [4, 1]
    assert isinstance(h, pd.DataFrame)
    pipe.data = {"data_folder": "/d", "num_files": 3}
    s = pipe.get_summary()
    assert (s["total_steps"], s["data_folder"], s["num_files"], s["steps"]) == (1, "/d", 3, ["double"])


def test_clear_removes_steps_and_results(pipe):
    pipe.add_step("a", lambda: 1)
    pipe.execute_all()
    pipe.clear()
    assert pipe.steps == [] and pipe.results == {}


def test_session_roundtrip_restores_metadata_and_data(pipe, tmp_path):
    pipe.data = {"data_folder": "/d", "num_files": 3}
    pipe.add_step("a", _double, {"factor": 2})
    f = tmp_path / "session.json"
    pipe.save_session(str(f))
    saved = json.loads(f.read_text())
    assert saved["steps"][0]["name"] == "a" and saved["data"]["num_files"] == 3
    other = Pipeline()
    other.load_session(str(f))
    assert other.data == pipe.data and other.metadata["description"] == pipe.metadata["description"]
    assert other.steps == []        # documented limitation: functions are not restored


def test_export_to_pdf_writes_text_report_with_steps(pipe, tmp_path):
    pipe.data = {"data_folder": "/d", "num_files": 3}
    pipe.add_step("double", _double, {"factor": 4})
    pipe.export_to_pdf(str(tmp_path / "r.pdf"))
    txt = (tmp_path / "r.txt").read_text()
    assert "Number of Files: 3" in txt and "1. double" in txt and "{'factor': 4}" in txt


def test_export_to_notebook_contains_script_cells(pipe, tmp_path):
    nbformat = pytest.importorskip("nbformat")
    pipe.add_step("double", _double, {"factor": 4})
    f = tmp_path / "n.ipynb"
    pipe.export_to_notebook(str(f))
    nb = nbformat.read(str(f), as_version=4)
    assert any("_double" in c.source for c in nb.cells)


# ----------------------------------------------------------------------------- Qt wrapper
def test_qt_wrapper_exposes_events_as_signals_and_keeps_listeners():
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from ade_dls.gui.core.pipeline import AnalysisStep as GuiStep, TransparentPipeline

    assert GuiStep is AnalysisStep                      # re-exported, same class
    p = TransparentPipeline()
    sig, plain, done = [], [], []
    p.step_added.connect(sig.append)
    p.analysis_completed.connect(done.append)
    p.subscribe("step_added", plain.append)
    p.add_step("a", lambda: 7)
    p.execute_all()
    assert len(sig) == 1 and sig[0]["name"] == "a" and len(plain) == 1
    assert done == [{"a": 7}]
    assert isinstance(p, Pipeline)
