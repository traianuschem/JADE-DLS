"""
Provenance Panel Widget
Live JSON view of the JADE-DLS data provenance record.
Replaces the Python-code panel that was originally planned.
"""

import re
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import (
    QColor, QFont, QSyntaxHighlighter, QTextCharFormat,
)
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QTextEdit, QToolButton, QVBoxLayout, QWidget,
)

from ..core.provenance import ProvenanceRecord


# ---------------------------------------------------------------------------
# JSON syntax highlighter
# ---------------------------------------------------------------------------

class _JsonHighlighter(QSyntaxHighlighter):
    """Minimal syntax highlighter for pretty-printed JSON."""

    def __init__(self, document, dark_mode: bool = False):
        super().__init__(document)

        def fmt(color_light, color_dark=None, bold=False, italic=False):
            f = QTextCharFormat()
            c = QColor(color_dark if (dark_mode and color_dark) else color_light)
            f.setForeground(c)
            if bold:
                f.setFontWeight(QFont.Bold)
            if italic:
                f.setFontItalic(True)
            return f

        self._rules = [
            # Object keys  ("key":)
            (re.compile(r'"([^"\\]|\\.)*"\s*(?=:)'), fmt("#0057ae", "#88b4e7", bold=True)),
            # String values
            (re.compile(r':\s*"([^"\\]|\\.)*"'), fmt("#008000", "#6aab73")),
            # Standalone strings (not after colon — e.g. array items)
            (re.compile(r'(?<!:)\s"([^"\\]|\\.)*"'), fmt("#008000", "#6aab73")),
            # Numbers
            (re.compile(r'(?<!["\w])-?\d+(\.\d+)?([eE][+-]?\d+)?(?!["\w])'), fmt("#c65900", "#d4874f")),
            # Booleans & null
            (re.compile(r'\b(true|false|null)\b'), fmt("#9400d3", "#cc99ff", bold=True)),
            # Schema / record_id lines (top-level keys starting with $)
            (re.compile(r'"\\$[^"]*"'), fmt("#888888", "#aaaaaa", italic=True)),
        ]

    def highlightBlock(self, text: str) -> None:
        for pattern, fmt in self._rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


# ---------------------------------------------------------------------------
# ProvenancePanel
# ---------------------------------------------------------------------------

class ProvenancePanel(QWidget):
    """
    Inspector panel tab that shows the live provenance record as formatted JSON.

    Workflow:
    1. Call ``initialize_input(data_folder, files, basedata_df)`` when data is loaded.
    2. Connect ``pipeline.step_added`` signal to ``on_step_added``.
    3. Call ``register_output(type, label, filepath)`` after each export.
    4. The JSON view auto-refreshes on every change.
    """

    def __init__(self, version: str = "unknown", parent=None):
        super().__init__(parent)
        self._version = version
        self._record = ProvenanceRecord(version=version)
        self._init_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _init_ui(self) -> None:
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Record ID display ---
        id_row = QHBoxLayout()
        id_lbl = QLabel("Record ID:")
        id_lbl.setStyleSheet("font-weight: bold;")
        id_row.addWidget(id_lbl)
        self._id_label = QLabel(self._record.record_id)
        self._id_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._id_label.setStyleSheet("font-family: monospace; font-size: 9pt;")
        id_row.addWidget(self._id_label)
        id_row.addStretch()
        layout.addLayout(id_row)

        # --- Toolbar ---
        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)

        refresh_btn = QPushButton("↺ Refresh")
        refresh_btn.setFixedWidth(90)
        refresh_btn.setToolTip("Rebuild provenance JSON from current session state")
        refresh_btn.clicked.connect(self._refresh_display)
        toolbar.addWidget(refresh_btn)

        copy_btn = QPushButton("📋 Copy JSON")
        copy_btn.setFixedWidth(100)
        copy_btn.clicked.connect(self._copy_json)
        toolbar.addWidget(copy_btn)

        export_btn = QPushButton("💾 Export JSON…")
        export_btn.setFixedWidth(120)
        export_btn.clicked.connect(self._export_json)
        toolbar.addWidget(export_btn)

        prov_btn = QPushButton("💾 Export PROV-JSON…")
        prov_btn.setFixedWidth(150)
        prov_btn.setToolTip("W3C PROV-JSON — kompatibel mit prov-Bibliothek und PROV-Toolbox")
        prov_btn.clicked.connect(self._export_prov_json)
        toolbar.addWidget(prov_btn)

        toolbar.addStretch()

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray; font-size: 8pt;")
        toolbar.addWidget(self._status_label)

        layout.addLayout(toolbar)

        # --- JSON text view ---
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setFont(QFont("Courier New", 9))
        self._text.setLineWrapMode(QTextEdit.NoWrap)

        dark = QApplication.palette().base().color().lightness() < 128
        self._highlighter = _JsonHighlighter(self._text.document(), dark_mode=dark)

        layout.addWidget(self._text)
        self.setLayout(layout)

        self._refresh_display()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize_input(self, data_folder: str, files: list,
                         basedata_df=None) -> None:
        """
        Reset the provenance record for a new session and register input files.

        *files* is a list of absolute file paths.
        *basedata_df* is an optional DataFrame with per-file metadata columns
        (angle, temperature, viscosity) — used to enrich entity metadata.
        """
        self._record = ProvenanceRecord(version=self._version)
        self._id_label.setText(self._record.record_id)
        self._record.set_input_folder(data_folder)

        # Build a lookup: filename → metadata row
        meta_lookup: dict = {}
        if basedata_df is not None and not basedata_df.empty:
            for _, row in basedata_df.iterrows():
                fname = row.get("filename", "")
                if fname:
                    meta: dict = {}
                    for col, key in [
                        ("angle", "angle_deg"),
                        ("temperature [K]", "temperature_K"),
                        ("viscosity [cp]", "viscosity_mPas"),
                    ]:
                        if col in row and row[col] is not None:
                            try:
                                meta[key] = float(row[col])
                            except (TypeError, ValueError):
                                pass
                    if meta:
                        meta_lookup[fname] = meta

        for fp in files:
            p = Path(fp)
            metadata = meta_lookup.get(p.name) or meta_lookup.get(p.stem) or None
            self._record.add_input_entity(fp, metadata=metadata)

        self._status_label.setText(
            f"Initialised  ·  {len(files)} input file(s)"
        )
        self._refresh_display()

    def on_step_added(self, step_dict: dict) -> None:
        """Slot connected to ``pipeline.step_added`` — appends an activity."""
        self._record.add_activity_from_step(step_dict)

        # Propagate excluded files from filter and refinement steps into
        # the top-level input.excluded_files catalog so they are visible
        # alongside the input entities rather than buried in parameters.
        step_type = step_dict.get("step_type", "")
        if step_type in ("filter", "refinement"):
            for fname in step_dict.get("params", {}).get("excluded_files", []):
                if fname:
                    self._record.mark_file_excluded(fname)

        self._refresh_display()

    def register_output(
        self, output_type: str, label: str, filepath: str = None,
        extra_fields: dict = None,
    ) -> str:
        """Register an exported artifact in the output catalog."""
        out_id = self._record.add_output(output_type, label, filepath,
                                         extra_fields=extra_fields)
        self._refresh_display()
        return out_id

    def get_record_id(self) -> str:
        return self._record.record_id

    def get_record(self) -> ProvenanceRecord:
        return self._record

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh_display(self) -> None:
        json_text = self._record.to_json()
        # Block signals to avoid cursor jump
        self._text.blockSignals(True)
        self._text.setPlainText(json_text)
        self._text.blockSignals(False)

    def _copy_json(self) -> None:
        QApplication.clipboard().setText(self._record.to_json())
        self._status_label.setText("Copied to clipboard")

    def _export_json(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"DLS_Provenance_{ts}.json"
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export Provenance JSON",
            default_name,
            "JSON files (*.json)",
        )
        if not filepath:
            return
        self._export_json_to(filepath)

    def _export_prov_json(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"DLS_Provenance_PROV_{ts}.json"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export W3C PROV-JSON", default_name, "JSON files (*.json)"
        )
        if not filepath:
            return
        try:
            self._record.export_prov_json_to_file(filepath)
            self._refresh_display()
            self._status_label.setText(f"PROV-JSON exportiert: {Path(filepath).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_json_to(self, filepath: str) -> None:
        """Write the provenance JSON to *filepath* and update the display."""
        try:
            self._record.export_to_file(filepath)
            self._refresh_display()
            self._status_label.setText(f"Exported: {Path(filepath).name}")
            QMessageBox.information(
                self,
                "Provenance Exported",
                f"Provenance record saved to:\n{filepath}\n\n"
                f"Record ID: {self._record.record_id}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
