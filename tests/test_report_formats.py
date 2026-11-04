"""Tests for ade_dls.engine.export.report_formats (Qt-free report helpers) and the report panel's
use of them. Expected strings are written out by hand from the documented formats."""
import os
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ade_dls.engine.export import report_formats as rf

_APP = None
HEADER_TABLE =[[("A", True), ("Bee", True)], [("1", False), ("22222", False)]]


# ----------------------------------------------------------------------------- HTML tables
def test_parse_html_tables_extracts_cells_headers_and_entities():
    html = ("<p>x</p><table><tr><th>A</th><th>B &amp; C</th></tr>"
            "<tr><td>1</td><td>&#955;</td></tr></table><table><tr><td>z</td></tr></table>")
    assert rf.parse_html_tables(html) == [
        [[("A", True), ("B & C", True)], [("1", False), ("\u03bb", False)]],
        [[("z", False)]],
    ]


def test_parse_html_tables_without_table_or_with_empty_rows():
    assert rf.parse_html_tables("<p>no table</p>") == []
    assert rf.parse_html_tables("<table><tr></tr></table>") == []


def test_table_to_tsv():
    assert rf.table_to_tsv(HEADER_TABLE) == "A\tBee\n1\t22222"
    assert rf.table_to_tsv([]) == ""


def test_table_to_markdown_with_header_row():
    assert rf.table_to_markdown(HEADER_TABLE) == (
        "| A   | Bee   |\n"
        "| --- | ----- |\n"
        "| 1   | 22222 |")


def test_table_to_markdown_without_header_has_no_separator_and_pads_ragged_rows():
    table = [[("x", False)], [("y", False), ("zz", False)]]
    assert rf.table_to_markdown(table) == (
        "| x   |     |\n"
        "| y   | zz  |")


def test_table_to_markdown_empty_table_is_empty_string():
    assert rf.table_to_markdown([]) == ""


def test_table_to_latex_header_and_body():
    assert rf.table_to_latex(HEADER_TABLE) == (
        "\\begin{tblr}{width=\\linewidth, colspec={XX}}\n"
        "  \\SetRow{font=\\bfseries} A & Bee \\\\\n"
        "  \\hline\n"
        "  1 & 22222 \\\\\n"
        "\\end{tblr}")
    assert rf.table_to_latex([]) == ""


def test_table_to_latex_escapes_cells():
    out = rf.table_to_latex([[("50%", False), ("a_b", False)]])
    assert "  50\\% & a\\_b \\\\" in out


# ----------------------------------------------------------------------------- LaTeX escaping
@pytest.mark.parametrize("raw,expected", [
    ("50% of a_b", r"50\% of a\_b"),
    ("D [m\u00b2/s]", "D [m$^{2}$/s]"),
    ("a\\b", r"a\textbackslash{}b"),
    ("{x}", r"\{x\}"),
    ("\u0393 \u00b1 5", r"$\Gamma$ $\pm$ 5"),
    ("a&b#c~d^e", r"a\&b\#c\textasciitilde{}d\textasciicircum{}e"),
    ("plain", "plain"),
])
def test_tex_escape(raw, expected):
    assert rf.tex_escape(raw) == expected


def test_tex_escape_unicode_substitutions_are_not_escaped_again():
    """The $ of the inserted math mode must survive (substitution happens after escaping)."""
    assert rf.tex_escape("\u03c4\u00b2").count("$") == 4


def test_tex_preamble_is_a_complete_document_head():
    pre = rf.tex_preamble()
    assert pre.startswith("\\documentclass[a4paper, 11pt]{scrartcl}")
    assert "\\usepackage{tabularray}" in pre and "\\title{DLS Analysis Report}" in pre
    assert "\\begin{document}" not in pre


# ----------------------------------------------------------------------------- step parameters
def test_safe_param_str_collapses_long_sequences_only():
    assert rf.safe_param_str(np.arange(10)) == "[10 values]"
    assert rf.safe_param_str([1, 2, 3]) == "[1, 2, 3]"
    assert rf.safe_param_str("a" * 20) == "a" * 20
    assert rf.safe_param_str(0.5) == "0.5"


def test_format_step_params_empty_and_generic():
    assert rf.format_step_params("X", {}) == ""
    assert rf.format_step_params("Method C", {"q_range": (1, 2), "x": np.arange(10)}) == (
        "q_range=(1, 2); x=[10 values]")


def test_format_step_params_regularized_shows_reproducibility_parameters_only():
    params = {
        "decay_times": np.logspace(-8, 1, 200), "alpha": 0.2, "prominence": 0.005, "distance": 1,
        "normalize": True, "sparsity_penalty": 0.0, "enforce_unimodality": False,
        "use_clustering": True, "clustering_strategy": "silhouette_refined",
        "distance_threshold": 2.0, "peak_method": "maximum", "fit_beta": True,
    }
    assert rf.format_step_params("Regularized NNLS Analysis", params) == (
        "tau: 1.00e-08\u20261.00e+01 s (200 pts, log); alpha=0.2; prominence=0.005; distance=1; "
        "normalize=True; use_clustering=True; clustering_strategy=silhouette_refined; "
        "distance_threshold=2.0; peak_method=maximum")


def test_format_step_params_regularized_shows_non_default_penalty_and_short_tau():
    out = rf.format_step_params("Regularized NNLS Analysis",
                                {"decay_times": [1e-3, 1e-2], "sparsity_penalty": 0.1,
                                 "enforce_unimodality": True})
    assert out == "tau: [0.001, 0.01]; sparsity_penalty=0.1; enforce_unimodality=True"


@pytest.mark.parametrize("name", ["Filter Correlations", "Filter Countrates"])
def test_format_step_params_filter_steps(name):
    params = {"original_count": 6, "remaining_count": 4, "excluded_files": ["a", "b"],
              "noise_params": {"baseline_correction": True, "baseline_pct": 5,
                               "intercept_correction": True, "intercept_pct": 2}}
    assert rf.format_step_params(name, params) == (
        "original=6; remaining=4; excluded=2; "
        "noise reduction: baseline correction (5%), intercept correction (2%)")
    assert rf.format_step_params(name, {"excluded_files": []}) == "original=?; remaining=?"


# ----------------------------------------------------------------------------- metadata fields
def test_collect_metadata_fields_without_pipeline():
    assert rf.collect_metadata_fields(None) == [("Status", "No pipeline data available")]


def test_collect_metadata_fields_from_basedata():
    base = pd.DataFrame({"filename": ["a", "b"], "angle [\u00b0]": [60.0, 30.0],
                         "temperature [K]": [298.0, 300.0], "viscosity [cp]": [0.89, 0.91]})
    pipe = SimpleNamespace(metadata={"created": datetime(2026, 1, 2, 3, 4, 5), "version": "3.0"},
                           data={"basedata": base})
    assert rf.collect_metadata_fields(pipe) == [
        ("Analysis started", "2026-01-02 03:04:05"),
        ("JADE-DLS version", "3.0"),
        ("Angles", "30\u00b0, 60\u00b0"),
        ("Temperature", "299.00 \u00b1 1.414 K"),
        ("Viscosity", "0.9000 cP"),
        ("Files", "2"),
    ]


def test_collect_metadata_fields_files_only_and_empty():
    pipe = SimpleNamespace(metadata={}, data={"files": ["x.asc", "y.asc", "z.asc"]})
    assert rf.collect_metadata_fields(pipe) == [("JADE-DLS version", "N/A"), ("Files loaded", "3")]
    assert rf.collect_metadata_fields(SimpleNamespace(metadata={}, data={})) == [
        ("JADE-DLS version", "N/A")]


# ----------------------------------------------------------------------------- report panel (Qt)
@pytest.fixture(scope="module")
def panel_module():
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    global _APP
    _APP = QApplication.instance() or QApplication([])     # module-level: widgets crash without a live app
    from ade_dls.gui.widgets import report_panel
    return report_panel


def test_report_panel_uses_the_engine_functions(panel_module):
    assert panel_module._table_to_markdown is rf.table_to_markdown
    assert panel_module._table_to_latex is rf.table_to_latex
    assert panel_module._tex_escape is rf.tex_escape
    assert panel_module._parse_html_tables is rf.parse_html_tables
    assert panel_module.ReportPanel._tex_preamble() == rf.tex_preamble()


def test_preprocessing_block_formats_steps_through_engine(panel_module):
    step = SimpleNamespace(name="Filter Countrates", timestamp=datetime(2026, 1, 1, 12, 0, 5),
                           params={"original_count": 6, "remaining_count": 5})
    block = panel_module.PreprocessingBlock({}, SimpleNamespace(steps=[step]))
    assert block.to_markdown() == (
        "## Preprocessing & Analysis Steps\n\n"
        "| # | Time | Step | Key Parameters |\n"
        "|---|------|------|----------------|\n"
        "| 1 | 12:00:05 | Filter Countrates | original=6; remaining=5 |")
    assert block.to_text() == (
        "=== Preprocessing & Analysis Steps ===\n"
        "1. [12:00:05] Filter Countrates\n"
        "   Parameters: original=6; remaining=5")


def test_metadata_block_renders_fields_from_engine(panel_module):
    block = panel_module.MetadataBlock({}, None)
    assert block.to_text() == "=== Experiment Metadata ===\nStatus: No pipeline data available"
