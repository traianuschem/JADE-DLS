"""
Workflow Panel Widget
Categorized tool buttons for analysis workflow
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QPushButton,
                             QLabel, QScrollArea, QSizePolicy, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


CATEGORIES = [
    ("Loading & Preprocessing", [
        ("Load Data",                "load_data",   "Load .asc data files"),
        ("Preprocess & Filter",      "preprocess",  "Filter & process correlations"),
    ]),
    ("Monomodal Analysis", [
        ("A - ALV cumulant data",    "cumulant_a",  "Extract cumulant fit from ALV output"),
        ("B - Linear cumulant fit",  "cumulant_b",  "Linear fit of ln(g\u00b2-1)"),
        ("C - Iterative non-linear", "cumulant_c",  "Iterative non-linear cumulant fit"),
    ]),
    ("Multimodal Analysis", [
        ("D - Multi-exponential",    "cumulant_d",  "Multi-exponential decomposition"),
        ("NNLS",                     "nnls",        "Inverse Laplace NNLS"),
        ("Regularized",              "regularized", "Tikhonov-Phillips regularization"),
    ]),
]


class WorkflowPanel(QWidget):
    """
    Left panel with categorized tool buttons.

    Signals:
        step_selected: Emitted when a button is clicked
        run_analysis:  Emitted when a button is clicked — triggers dialog in main_window
    """

    step_selected = pyqtSignal(str)
    run_analysis  = pyqtSignal(str)

    def __init__(self, pipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self._buttons = {}   # step_id -> QPushButton
        self._init_ui()

    def _init_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(4, 4, 4, 4)
        outer_layout.setSpacing(4)

        # Title
        title = QLabel("Analysis Workflow")
        title.setStyleSheet("font-size: 13pt; font-weight: bold;")
        outer_layout.addWidget(title)

        # Scroll area so the panel stays usable when the window is small
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(2, 2, 2, 2)
        container_layout.setSpacing(4)

        for category_name, tools in CATEGORIES:
            # Category header label
            cat_label = QLabel(category_name)
            cat_font = QFont()
            cat_font.setBold(True)
            cat_label.setFont(cat_font)
            cat_label.setStyleSheet(
                "color: palette(mid);"
                "border-bottom: 1px solid palette(mid);"
                "padding-bottom: 2px;"
                "margin-top: 8px;"
            )
            container_layout.addWidget(cat_label)

            # Tool buttons
            for label, step_id, tooltip in tools:
                btn = QPushButton(label)
                btn.setToolTip(tooltip)
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                btn.setMinimumHeight(28)
                btn.setStyleSheet("text-align: left; padding-left: 8px;")
                btn.clicked.connect(lambda checked, sid=step_id: self._on_button_clicked(sid))
                container_layout.addWidget(btn)
                self._buttons[step_id] = btn

        container_layout.addStretch()
        scroll.setWidget(container)
        outer_layout.addWidget(scroll)

    def _on_button_clicked(self, step_id):
        self.step_selected.emit(step_id)
        self.run_analysis.emit(step_id)

    # --- Public API (compatible with existing main_window.py usage) ---

    def activate_step(self, step_id):
        """Highlight a step button with a blue border."""
        btn = self._buttons.get(step_id)
        if btn:
            btn.setStyleSheet(
                "text-align: left; padding-left: 8px;"
                "border: 2px solid #2196F3;"
            )

    def mark_step_complete(self, step_id):
        """Mark a step as complete — green background and ✓ prefix."""
        btn = self._buttons.get(step_id)
        if btn is None:
            return

        text = btn.text()
        if not text.startswith("\u2713 "):
            btn.setText("\u2713 " + text)

        from PyQt5.QtWidgets import QApplication
        is_dark = QApplication.palette().base().color().lightness() < 128

        if is_dark:
            btn.setStyleSheet(
                "text-align: left; padding-left: 8px;"
                "background-color: #1B5E20; color: #C8E6C9;"
            )
        else:
            btn.setStyleSheet(
                "text-align: left; padding-left: 8px;"
                "background-color: #E8F5E9; color: #1B5E20;"
            )
