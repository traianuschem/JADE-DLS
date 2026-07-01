"""
Inspector Panel Widget
Shows provenance record, parameters, and documentation for transparency.
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QTabWidget, QTextEdit,
                             QTableWidget, QTableWidgetItem, QLabel,
                             QPushButton, QToolButton, QMenu, QHBoxLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QSyntaxHighlighter, QTextCharFormat, QColor
import re

from .report_panel import ReportPanel
from .provenance_panel import ProvenancePanel


class PythonHighlighter(QSyntaxHighlighter):
    """Syntax highlighter for Python code (used in hidden Code tab)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlighting_rules = []

        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#0000FF"))
        keyword_format.setFontWeight(QFont.Bold)
        for word in ['def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif',
                     'for', 'while', 'try', 'except', 'with', 'as', 'lambda', 'yield',
                     'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is']:
            self.highlighting_rules.append(
                (re.compile(f'\\b{word}\\b'), keyword_format)
            )

        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#008000"))
        self.highlighting_rules.append((re.compile(r'"[^"\\]*(\\.[^"\\]*)*"'), string_format))
        self.highlighting_rules.append((re.compile(r"'[^'\\]*(\\.[^'\\]*)*'"), string_format))

        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#808080"))
        comment_format.setFontItalic(True)
        self.highlighting_rules.append((re.compile(r'#[^\n]*'), comment_format))

        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#FF8C00"))
        self.highlighting_rules.append((re.compile(r'\b[0-9]+\.?[0-9]*\b'), number_format))

    def highlightBlock(self, text):
        for pattern, fmt in self.highlighting_rules:
            for match in pattern.finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), fmt)


class InspectorPanel(QWidget):
    """
    Right panel with four tabs:
      0. 📄 Report   — modular block-based report builder
      1. 🔍 Provenance — live PROV-JSON provenance record (replaces old Code tab)
      2. ⚙️ Parameters — parameter history table
      3. 📖 Docs      — inline method documentation

    The old Python-code view is still available via File > Export as Python Script /
    Jupyter Notebook in the menu bar.
    """

    def __init__(self, pipeline, version: str = "unknown", parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self._version = version
        self.init_ui()

        # Connect pipeline signals
        self.pipeline.step_added.connect(self._on_step_added)

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Inspector")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        self.tabs = QTabWidget()

        # Tab 0: Report
        self.report_panel = ReportPanel(self.pipeline, parent=self)
        self.tabs.addTab(self.report_panel, "📄 Report")

        # Tab 1: Provenance (replaces old Code tab)
        self.provenance_panel = ProvenancePanel(version=self._version, parent=self)
        self.tabs.addTab(self.provenance_panel, "🔍 Provenance")

        # Wire provenance panel into report panel so "Add Block → Provenance" works
        self.report_panel.set_provenance_panel(self.provenance_panel)

        # Tab 2: Parameters
        self.params_tab = self._create_params_tab()
        self.tabs.addTab(self.params_tab, "⚙️ Parameters")

        # Tab 3: Documentation
        self.docs_tab = self._create_docs_tab()
        self.tabs.addTab(self.docs_tab, "📖 Docs")

        layout.addWidget(self.tabs)

        # Bottom toolbar
        btn_row = QHBoxLayout()

        export_btn = QToolButton()
        export_btn.setText("💾 Export Report")
        export_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        export_btn.setPopupMode(QToolButton.InstantPopup)
        export_menu = QMenu(self)
        export_menu.addAction("Export as TXT…").triggered.connect(
            self.report_panel._export_txt
        )
        export_menu.addAction("Export as Markdown…").triggered.connect(
            self.report_panel._export_markdown
        )
        export_menu.addSeparator()
        export_menu.addAction("Export as PDF (Portrait)…").triggered.connect(
            lambda: self.report_panel._export_pdf(landscape=False)
        )
        export_menu.addAction("Export as PDF (Landscape)…").triggered.connect(
            lambda: self.report_panel._export_pdf(landscape=True)
        )
        export_menu.addSeparator()
        export_menu.addAction("Export as LaTeX package…").triggered.connect(
            self.report_panel._export_tex
        )
        export_btn.setMenu(export_menu)
        btn_row.addWidget(export_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    def _on_step_added(self, step_dict: dict) -> None:
        """Forward new pipeline steps to the provenance panel and update params."""
        self.provenance_panel.on_step_added(step_dict)
        self.update_params_table()

    # Keep legacy name so main_window.py's existing call doesn't break
    def update_code_view(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Parameters tab
    # ------------------------------------------------------------------

    def _create_params_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Current analysis parameters:"))

        self.params_table = QTableWidget()
        self.params_table.setColumnCount(3)
        self.params_table.setHorizontalHeaderLabels(["Step", "Parameter", "Value"])
        layout.addWidget(self.params_table)

        self.update_params_table()
        widget.setLayout(layout)
        return widget

    def update_params_table(self):
        param_history = self.pipeline.get_parameter_history()
        if not param_history.empty:
            self.params_table.setRowCount(len(param_history))
            for i, row in param_history.iterrows():
                self.params_table.setItem(i, 0, QTableWidgetItem(str(row['step'])))
                self.params_table.setItem(i, 1, QTableWidgetItem(str(row['parameter'])))
                self.params_table.setItem(i, 2, QTableWidgetItem(str(row['value'])))
            self.params_table.resizeColumnsToContents()
        else:
            self.params_table.setRowCount(3)
            for i, (step, param, value) in enumerate([
                ("General", "Temperature", "297.94 K"),
                ("General", "Viscosity", "0.894 cP"),
                ("Cumulant", "Fit Range", "1e-9 to 10 s"),
            ]):
                self.params_table.setItem(i, 0, QTableWidgetItem(step))
                self.params_table.setItem(i, 1, QTableWidgetItem(param))
                self.params_table.setItem(i, 2, QTableWidgetItem(value))
            self.params_table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Documentation tab
    # ------------------------------------------------------------------

    def _create_docs_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.docs_text = QTextEdit()
        self.docs_text.setReadOnly(True)
        self.docs_text.setHtml(self.get_default_docs())
        layout.addWidget(self.docs_text)
        widget.setLayout(layout)
        return widget

    def update_for_step(self, step_name):
        docs_map = {
            'load_data': self.get_load_data_docs(),
            'preprocess': self.get_preprocess_docs(),
            'cumulant_a': self.get_cumulant_a_docs(),
            'cumulant_b': self.get_cumulant_b_docs(),
            'cumulant_c': self.get_cumulant_c_docs(),
            'cumulant_d': self.get_cumulant_d_docs(),
            'method_d': self.get_cumulant_d_docs(),
            'nnls': self.get_nnls_docs(),
            'regularized': self.get_regularized_docs(),
        }
        self.docs_text.setHtml(docs_map.get(step_name, self.get_default_docs()))

    def get_default_docs(self):
        return """
<h3>JADE-DLS Analysis Methods</h3>

<h4>Cumulant Methods</h4>
<p>Cumulant analysis fits the autocorrelation function to extract decay rates and polydispersity:</p>
<ul>
<li><b>Method A:</b> Uses cumulant fit data from ALV software (1st, 2nd, 3rd order)</li>
<li><b>Method B:</b> Simple linear fit of ln[g(τ)^0.5] vs τ</li>
<li><b>Method C:</b> Iterative non-linear least squares fit</li>
<li><b>Method D:</b> Multi-exponential (Dirac-delta) decomposition into populations</li>
</ul>

<h4>Inverse Laplace Methods</h4>
<p>These methods solve the inverse problem to obtain size distributions:</p>
<ul>
<li><b>NNLS:</b> Non-Negative Least Squares without constraints</li>
<li><b>Regularized:</b> NNLS with Tikhonov-Phillips regularization</li>
</ul>

<h4>Key Equations</h4>
<p>Diffusion coefficient: Γ = D × q²</p>
<p>Hydrodynamic radius: R<sub>h</sub> = k<sub>B</sub>T / (6πηD)</p>
<p>Polydispersity Index: PDI = μ₂/Γ²</p>

<h4>Provenance</h4>
<p>The <b>🔍 Provenance</b> tab records every step in a FAIR-compliant JSON
record (PROV-inspired schema).  Export it alongside your report so that any
output can be traced back to its exact input files and analysis parameters.</p>
"""

    def get_load_data_docs(self):
        return """
<h3>Load Data</h3>
<p>Load .asc files from ALV DLS instrument.</p>
<p><b>Expected format:</b> ALV-5000/E correlator .asc files</p>
<p><b>Files are filtered to exclude:</b> *averaged.asc files</p>
<p><b>Provenance:</b> Input files are SHA-256 hashed and registered
in the provenance record for integrity verification.</p>
"""

    def get_preprocess_docs(self):
        return """
<h3>Preprocessing</h3>
<p>Extract metadata and correlation data from files.</p>
<p><b>Extracted data includes:</b></p>
<ul>
<li>Angle, Temperature, Wavelength</li>
<li>Refractive index, Viscosity</li>
<li>Count rates, Correlation functions</li>
</ul>
"""

    def get_cumulant_a_docs(self):
        return """
<h3>Cumulant Method A</h3>
<p>Uses pre-calculated cumulant fit data from ALV software.</p>
<p><b>Provides:</b> 1st, 2nd, and 3rd order cumulant fits</p>
<p><b>Best for:</b> Quick analysis when ALV software fits are reliable</p>
"""

    def get_cumulant_b_docs(self):
        return """
<h3>Cumulant Method B</h3>
<p>Linear fit method: ln[g(τ)^0.5] vs τ</p>
<p><b>Fit function:</b> 0.5·ln(a) - b·τ + 0.5·c·τ²</p>
<p><b>Best for:</b> Simple, fast analysis of monodisperse samples</p>
<p><b>Note:</b> Keep fit limits narrow (0 to 0.2 ms typical)</p>
"""

    def get_cumulant_c_docs(self):
        return """
<h3>Cumulant Method C</h3>
<p>Iterative non-linear least squares fit up to 4th cumulant.</p>
<p><b>Optimization methods:</b> Levenberg-Marquardt, TRF, Dogbox</p>
<p><b>Best for:</b> Accurate analysis of moderately polydisperse samples</p>
<p><b>Features:</b> Adaptive initial parameter guessing</p>
"""

    def get_cumulant_d_docs(self):
        return """
<h3>Cumulant Method D</h3>
<p>Multi-exponential decomposition: fits g²(τ) to a sum of Dirac-delta
relaxation modes, g₁(τ) = (1/n)·Σᵢ exp(-Γᵢτ).</p>
<p><b>Algorithm:</b> Iteratively increases the number of modes and selects the
optimal order via AIC / convergence checks; fitted decay rates are then grouped
into populations.</p>
<p><b>Provides:</b> per-population Γ, D, Rh plus distribution moments
(PDI, skewness, kurtosis)</p>
<p><b>Best for:</b> Bimodal samples or when PDI &gt; 0.2 (e.g. monomer + aggregate)</p>
"""

    def get_nnls_docs(self):
        return """
<h3>NNLS Analysis</h3>
<p>Non-Negative Least Squares inverse Laplace transform.</p>
<p><b>Provides:</b> Size distribution without regularization</p>
<p><b>Best for:</b> Initial exploration of polydisperse samples</p>
<p><b>Note:</b> May be noisy without regularization</p>
"""

    def get_regularized_docs(self):
        return """
<h3>Regularized Fit</h3>
<p>NNLS with Tikhonov-Phillips regularization.</p>
<p><b>Key parameter:</b> α (alpha) controls regularization strength</p>
<p><b>Best for:</b> Polydisperse samples, smooth distributions</p>
<p><b>Tip:</b> Use alpha analyzer to find optimal α value</p>
"""

    # ------------------------------------------------------------------
    # Compatibility stubs for main_window.py
    # ------------------------------------------------------------------

    def set_code_view_visible(self, visible: bool) -> None:
        """Show/hide the Provenance tab (was: Code tab)."""
        idx = self.tabs.indexOf(self.provenance_panel)
        if visible and idx == -1:
            self.tabs.insertTab(1, self.provenance_panel, "🔍 Provenance")
        elif not visible and idx != -1:
            self.tabs.removeTab(idx)

    def set_params_view_visible(self, visible: bool) -> None:
        idx = self.tabs.indexOf(self.params_tab)
        if visible and idx == -1:
            self.tabs.insertTab(2, self.params_tab, "⚙️ Parameters")
        elif not visible and idx != -1:
            self.tabs.removeTab(idx)
