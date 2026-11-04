"""
Report Panel Widget
Modular block-based report builder for JADE-DLS analysis results.
Blocks can be added, removed, and reordered. The panel supports export
to plain text, Markdown, and PDF.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QToolButton, QMenu, QAction, QScrollArea, QTextEdit, QTableWidget,
    QTableWidgetItem, QFormLayout, QSizePolicy, QFileDialog, QMessageBox,
    QHeaderView, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QImage, QPixmap


def _is_dark_mode() -> bool:
    """Detect current dark/light mode."""
    return QApplication.palette().base().color().lightness() < 128


def _header_bg_color() -> str:
    """Return a subtle background color for block headers."""
    if _is_dark_mode():
        return "#2d2d2d"
    return "#e8e8e8"


# ---------------------------------------------------------------------------
# HTML table parser helpers
# ---------------------------------------------------------------------------

from ade_dls.engine.export.report_formats import (
    parse_html_tables as _parse_html_tables,
    table_to_tsv as _table_to_tsv,
    table_to_markdown as _table_to_markdown,
    table_to_latex as _table_to_latex,
    tex_escape as _tex_escape,
    collect_metadata_fields as _collect_metadata_fields,
    format_step_params as _format_step_params,
    safe_param_str as _safe_param_str,
    tex_preamble as _tex_preamble,
)

# ---------------------------------------------------------------------------
# Base block
# ---------------------------------------------------------------------------

class ReportBlock(QFrame):
    """
    Abstract base class for all report blocks.

    Provides:
    - Collapsible header with title, ↑/↓ reorder buttons, and × remove button
    - Content area (subclasses fill self.content_widget)
    - Abstract rendering interface: to_text(), to_markdown(), to_html()
    """

    removed = pyqtSignal(object)   # emits self so the panel can find it in its list

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._collapsed = False
        self._init_frame()

    def _init_frame(self):
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Header ---
        self._header = QFrame()
        self._header.setStyleSheet(
            f"background-color: {_header_bg_color()}; border-radius: 3px;"
        )
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(6, 4, 6, 4)

        self._toggle_btn = QToolButton()
        self._toggle_btn.setText("▼")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setFixedSize(20, 20)
        self._toggle_btn.setStyleSheet("border: none; background: transparent;")
        self._toggle_btn.toggled.connect(self._on_toggle)
        header_layout.addWidget(self._toggle_btn)

        self._title_label = QLabel(self._title)
        font = QFont()
        font.setBold(True)
        self._title_label.setFont(font)
        header_layout.addWidget(self._title_label)
        header_layout.addStretch()

        self._up_btn = QToolButton()
        self._up_btn.setText("↑")
        self._up_btn.setFixedSize(22, 22)
        self._up_btn.setStyleSheet("border: none; background: transparent;")
        self._up_btn.setToolTip("Move up")
        header_layout.addWidget(self._up_btn)

        self._down_btn = QToolButton()
        self._down_btn.setText("↓")
        self._down_btn.setFixedSize(22, 22)
        self._down_btn.setStyleSheet("border: none; background: transparent;")
        self._down_btn.setToolTip("Move down")
        header_layout.addWidget(self._down_btn)

        self._remove_btn = QToolButton()
        self._remove_btn.setText("×")
        self._remove_btn.setFixedSize(22, 22)
        self._remove_btn.setStyleSheet(
            "border: none; background: transparent; color: #c0392b; font-weight: bold;"
        )
        self._remove_btn.setToolTip("Remove block")
        self._remove_btn.clicked.connect(lambda: self.removed.emit(self))
        header_layout.addWidget(self._remove_btn)

        self._header.setLayout(header_layout)
        outer.addWidget(self._header)

        # --- Content ---
        self.content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(6, 6, 6, 6)
        self._fill_content(content_layout)
        self.content_widget.setLayout(content_layout)
        outer.addWidget(self.content_widget)

        self.setLayout(outer)

    def _fill_content(self, layout: QVBoxLayout):
        """Override in subclasses to populate the content area."""
        pass

    def _on_toggle(self, checked: bool):
        self._collapsed = checked
        self.content_widget.setVisible(not checked)
        self._toggle_btn.setText("▶" if checked else "▼")

    # ------------------------------------------------------------------
    # Rendering interface (subclasses must override)
    # ------------------------------------------------------------------

    def to_text(self) -> str:
        raise NotImplementedError

    def to_markdown(self) -> str:
        raise NotImplementedError

    def to_html(self) -> str:
        raise NotImplementedError

    def to_latex(self, **kwargs) -> str:
        raise NotImplementedError

    # Future hook (not yet implemented)
    # def to_notebook_cell(self) -> dict: raise NotImplementedError


# ---------------------------------------------------------------------------
# Metadata block
# ---------------------------------------------------------------------------

class MetadataBlock(ReportBlock):
    """Block showing experiment metadata (from pipeline.data and pipeline.metadata)."""

    def __init__(self, payload: dict, pipeline=None):
        self._pipeline = pipeline
        self._fields = self._collect_fields()
        super().__init__("Experiment Metadata")

    def _collect_fields(self) -> list:
        """Build list of (label, value) pairs from available pipeline data."""
        return _collect_metadata_fields(self._pipeline)

    def _fill_content(self, layout: QVBoxLayout):
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        for label, value in self._fields:
            form.addRow(QLabel(f"<b>{label}:</b>"), QLabel(str(value)))
        container = QWidget()
        container.setLayout(form)
        layout.addWidget(container)

    def to_text(self) -> str:
        lines = ["=== Experiment Metadata ==="]
        for label, value in self._fields:
            lines.append(f"{label}: {value}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = ["## Experiment Metadata", ""]
        for label, value in self._fields:
            lines.append(f"**{label}:** {value}  ")
        return "\n".join(lines)

    def to_html(self) -> str:
        rows = "".join(
            f"<tr><td><b>{label}</b></td><td>{value}</td></tr>"
            for label, value in self._fields
        )
        return (
            "<div style='margin-bottom:12px'>"
            "<h3>Experiment Metadata</h3>"
            f"<table cellspacing='4'>{rows}</table>"
            "</div>"
        )

    def to_latex(self, **kwargs) -> str:
        rows = "\n".join(
            f"  \\textbf{{{_tex_escape(label)}}} & {_tex_escape(str(value))} \\\\"
            for label, value in self._fields
        )
        return (
            "\\section*{Experiment Metadata}\n"
            "\\begin{tblr}{width=\\linewidth, colspec={lX}}\n"
            f"{rows}\n"
            "\\end{tblr}"
        )


# ---------------------------------------------------------------------------
# Preprocessing block
# ---------------------------------------------------------------------------

class PreprocessingBlock(ReportBlock):
    """Block listing all pipeline steps (preprocessing and analysis)."""

    def __init__(self, payload: dict, pipeline=None):
        self._pipeline = pipeline
        self._steps = self._collect_steps()
        super().__init__("Preprocessing & Analysis Steps")

    _safe_param_str = staticmethod(_safe_param_str)
    _format_params = staticmethod(_format_step_params)

    def _collect_steps(self) -> list:
        """Return list of (step_name, timestamp, params_summary) tuples."""
        if self._pipeline is None:
            return []
        result = []
        for step in getattr(self._pipeline, 'steps', []):
            name = getattr(step, 'name', 'Unknown')
            ts = getattr(step, 'timestamp', None)
            ts_str = ts.strftime("%H:%M:%S") if ts else ""
            params = getattr(step, 'params', {})
            params_summary = self._format_params(name, params)
            result.append((name, ts_str, params_summary))
        return result

    def _fill_content(self, layout: QVBoxLayout):
        if not self._steps:
            layout.addWidget(QLabel("No pipeline steps recorded."))
            return

        table = QTableWidget(len(self._steps), 3)
        table.setHorizontalHeaderLabels(["Step", "Time", "Key Parameters"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setMaximumHeight(150)

        for i, (name, ts, params) in enumerate(self._steps):
            table.setItem(i, 0, QTableWidgetItem(name))
            table.setItem(i, 1, QTableWidgetItem(ts))
            table.setItem(i, 2, QTableWidgetItem(params))

        layout.addWidget(table)

    def to_text(self) -> str:
        lines = ["=== Preprocessing & Analysis Steps ==="]
        for i, (name, ts, params) in enumerate(self._steps, 1):
            lines.append(f"{i}. [{ts}] {name}")
            if params:
                lines.append(f"   Parameters: {params}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = ["## Preprocessing & Analysis Steps", ""]
        if self._steps:
            lines.append("| # | Time | Step | Key Parameters |")
            lines.append("|---|------|------|----------------|")
            for i, (name, ts, params) in enumerate(self._steps, 1):
                lines.append(f"| {i} | {ts} | {name} | {params} |")
        else:
            lines.append("*No steps recorded.*")
        return "\n".join(lines)

    def to_html(self) -> str:
        if not self._steps:
            return (
                "<div style='margin-bottom:12px'>"
                "<h3>Preprocessing &amp; Analysis Steps</h3>"
                "<p><i>No steps recorded.</i></p></div>"
            )
        rows = "".join(
            f"<tr><td>{i}</td><td>{ts}</td><td>{name}</td><td>{params}</td></tr>"
            for i, (name, ts, params) in enumerate(self._steps, 1)
        )
        return (
            "<div style='margin-bottom:12px'>"
            "<h3>Preprocessing &amp; Analysis Steps</h3>"
            "<table border='1' cellspacing='0' cellpadding='4' style='border-collapse:collapse'>"
            "<tr><th>#</th><th>Time</th><th>Step</th><th>Key Parameters</th></tr>"
            f"{rows}</table></div>"
        )

    def to_latex(self, **kwargs) -> str:
        if not self._steps:
            return (
                "\\section*{Preprocessing \\& Analysis Steps}\n"
                "\\emph{No steps recorded.}"
            )
        header = (
            "  \\SetRow{font=\\bfseries} "
            "\\# & Time & Step & Key Parameters \\\\\n"
            "  \\hline"
        )
        rows = "\n".join(
            f"  {i} & {_tex_escape(ts)} & {_tex_escape(name)} & {_tex_escape(params)} \\\\"
            for i, (name, ts, params) in enumerate(self._steps, 1)
        )
        return (
            "\\section*{Preprocessing \\& Analysis Steps}\n"
            "\\begin{tblr}{width=\\linewidth, colspec={rllX}}\n"
            f"{header}\n"
            f"{rows}\n"
            "\\end{tblr}"
        )


# ---------------------------------------------------------------------------
# Result Summary block
# ---------------------------------------------------------------------------

class ResultSummaryBlock(ReportBlock):
    """Block showing the 8-column results table row (summary level)."""

    def __init__(self, payload: dict, pipeline=None):
        self._payload = payload
        method = payload.get("method_name", "Unknown Method")
        ts = payload.get("timestamp", "")
        title = f"Result Summary — {method}"
        if ts:
            title += f"  [{ts}]"
        super().__init__(title)

    def _fill_content(self, layout: QVBoxLayout):
        data: dict = self._payload.get("data", {})
        if not data:
            layout.addWidget(QLabel("No data in payload."))
            return

        table = QTableWidget(1, len(data))
        table.setHorizontalHeaderLabels(list(data.keys()))
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setMaximumHeight(70)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        for col, (key, value) in enumerate(data.items()):
            item = QTableWidgetItem(str(value))
            item.setTextAlignment(Qt.AlignCenter)
            table.setItem(0, col, item)

        layout.addWidget(table)

    def to_text(self) -> str:
        data = self._payload.get("data", {})
        method = self._payload.get("method_name", "Unknown")
        lines = [f"=== Result Summary — {method} ==="]
        for key, value in data.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        data = self._payload.get("data", {})
        method = self._payload.get("method_name", "Unknown")
        ts = self._payload.get("timestamp", "")
        lines = [f"## Result Summary — {method}", ""]
        if ts:
            lines.append(f"*{ts}*  ")
            lines.append("")
        if data:
            lines.append("| " + " | ".join(data.keys()) + " |")
            lines.append("| " + " | ".join("---" for _ in data) + " |")
            lines.append("| " + " | ".join(str(v) for v in data.values()) + " |")
        return "\n".join(lines)

    def to_html(self) -> str:
        data = self._payload.get("data", {})
        method = self._payload.get("method_name", "Unknown")
        ts = self._payload.get("timestamp", "")
        headers = "".join(f"<th style='padding:4px 8px'>{k}</th>" for k in data.keys())
        cells = "".join(
            f"<td style='padding:4px 8px;text-align:center'>{v}</td>"
            for v in data.values()
        )
        ts_str = f"<p style='font-size:small;color:gray'>{ts}</p>" if ts else ""
        return (
            "<div style='margin-bottom:12px'>"
            f"<h3>Result Summary — {method}</h3>"
            f"{ts_str}"
            "<table border='1' cellspacing='0' style='border-collapse:collapse'>"
            f"<tr>{headers}</tr><tr>{cells}</tr>"
            "</table></div>"
        )

    def to_latex(self, **kwargs) -> str:
        data = self._payload.get("data", {})
        method = _tex_escape(self._payload.get("method_name", "Unknown"))
        ts = self._payload.get("timestamp", "")
        ts_str = f"\n\\emph{{{_tex_escape(ts)}}}" if ts else ""
        if not data:
            return f"\\section*{{Result Summary --- {method}}}\n\\emph{{No data.}}"
        col_spec = f"width=\\linewidth, colspec={{{'X' * len(data)}}}"
        header_cells = " & ".join(f"\\textbf{{{_tex_escape(k)}}}" for k in data.keys())
        value_cells = " & ".join(_tex_escape(str(v)) for v in data.values())
        return (
            f"\\section*{{Result Summary --- {method}}}{ts_str}\n"
            f"\\begin{{tblr}}{{{col_spec}}}\n"
            f"  {header_cells} \\\\\n"
            "  \\hline\n"
            f"  {value_cells} \\\\\n"
            "\\end{tblr}"
        )


# ---------------------------------------------------------------------------
# Result Detail block
# ---------------------------------------------------------------------------

class ResultDetailBlock(ReportBlock):
    """Block showing the full HTML detail view for a result row."""

    def __init__(self, payload: dict, pipeline=None):
        self._payload = payload
        method = payload.get("method_name", "Unknown Method")
        ts = payload.get("timestamp", "")
        title = f"Result Details — {method}"
        if ts:
            title += f"  [{ts}]"
        super().__init__(title)

    def _fill_content(self, layout: QVBoxLayout):
        html = self._payload.get("html", "")
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setHtml(html if html else "<i>No detail content available.</i>")
        self._text_edit.setMinimumHeight(200)
        self._text_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._text_edit)

    def to_text(self) -> str:
        method = self._payload.get("method_name", "Unknown")
        html = self._payload.get("html", "")
        lines = [f"=== Result Details — {method} ==="]

        tables = _parse_html_tables(html)
        for i, table in enumerate(tables, 1):
            lines.append(f"\n# Table {i}")
            lines.append(_table_to_tsv(table))

        return "\n".join(lines)

    def to_markdown(self) -> str:
        method = self._payload.get("method_name", "Unknown")
        ts = self._payload.get("timestamp", "")
        html = self._payload.get("html", "")
        lines = [f"## Result Details — {method}", ""]
        if ts:
            lines.append(f"*{ts}*")
            lines.append("")

        tables = _parse_html_tables(html)
        for table in tables:
            md_table = _table_to_markdown(table)
            if md_table:
                lines.append(md_table)
                lines.append("")

        return "\n".join(lines)

    def to_html(self) -> str:
        method = self._payload.get("method_name", "Unknown")
        ts = self._payload.get("timestamp", "")
        html = self._payload.get("html", "")
        ts_str = f"<p style='font-size:small;color:gray'>{ts}</p>" if ts else ""
        return (
            "<div style='margin-bottom:12px'>"
            f"<h3>Result Details — {method}</h3>"
            f"{ts_str}"
            f"{html}"
            "</div>"
        )

    def to_latex(self, **kwargs) -> str:
        method = _tex_escape(self._payload.get("method_name", "Unknown"))
        ts = self._payload.get("timestamp", "")
        ts_str = f"\n\\emph{{{_tex_escape(ts)}}}" if ts else ""
        html = self._payload.get("html", "")
        tables = _parse_html_tables(html)
        if tables:
            table_parts = [_table_to_latex(t) for t in tables]
            tables_str = "\n\n".join(table_parts)
        else:
            tables_str = "\\emph{No detail tables available.}"
        return (
            f"\\section*{{Result Details --- {method}}}{ts_str}\n\n"
            f"{tables_str}"
        )


# ---------------------------------------------------------------------------
# Plot block
# ---------------------------------------------------------------------------

class PlotBlock(ReportBlock):
    """Block containing a matplotlib figure captured as a PNG image."""

    def __init__(self, payload: dict, pipeline=None):
        self._payload = payload
        title = payload.get("title", "Plot")
        ts = payload.get("timestamp", "")
        block_title = f"Plot — {title}"
        if ts:
            block_title += f"  [{ts}]"
        super().__init__(block_title)

    def _fill_content(self, layout: QVBoxLayout):
        image_bytes: bytes = self._payload.get("image_bytes", b"")
        if not image_bytes:
            layout.addWidget(QLabel("No image data."))
            return

        img = QImage.fromData(image_bytes, "PNG")
        if img.isNull():
            layout.addWidget(QLabel("Could not decode image."))
            return

        pixmap = QPixmap.fromImage(img)
        # Scale to max width 600px preserving aspect ratio
        if pixmap.width() > 600:
            pixmap = pixmap.scaledToWidth(600, Qt.SmoothTransformation)

        label = QLabel()
        label.setPixmap(pixmap)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

    @staticmethod
    def _safe_filename(title: str) -> str:
        """Convert a plot title to a safe filename stem (max 60 chars)."""
        import re
        safe = re.sub(r'[^\w\-]', '_', title)
        safe = re.sub(r'_+', '_', safe).strip('_')
        return safe[:60] or "plot"

    def to_text(self, figures_dir=None, fig_index: int = 1) -> str:
        """
        Plain-text representation.

        If *figures_dir* (a Path) is provided the PNG is saved there and a
        relative file reference is returned.  Otherwise a generic placeholder
        is used.
        """
        from pathlib import Path
        title = self._payload.get("title", "Plot")
        ts = self._payload.get("timestamp", "")
        ts_str = f" [{ts}]" if ts else ""
        image_bytes: bytes = self._payload.get("image_bytes", b"")

        if figures_dir is not None and image_bytes:
            figures_dir = Path(figures_dir)
            fname = f"plot_{fig_index:02d}_{self._safe_filename(title)}.png"
            (figures_dir / fname).write_bytes(image_bytes)
            rel = f"{figures_dir.name}/{fname}"
            return f"=== Plot \u2014 {title}{ts_str} ===\n[Figure: {rel}]"

        return f"=== Plot \u2014 {title}{ts_str} ===\n[Figure not representable in plain text]"

    def to_markdown(self, figures_dir=None, fig_index: int = 1) -> str:
        """
        Markdown representation.

        If *figures_dir* (a Path) is provided the PNG is saved there and a
        relative ``![alt](path)`` reference is returned.  Otherwise the image
        is embedded as a base64 data-URI (elabFTW-compatible fallback).
        """
        import base64
        from pathlib import Path
        title = self._payload.get("title", "Plot")
        ts = self._payload.get("timestamp", "")
        image_bytes: bytes = self._payload.get("image_bytes", b"")
        lines = [f"## Plot \u2014 {title}", ""]
        if ts:
            lines.append(f"*{ts}*")
            lines.append("")

        if figures_dir is not None and image_bytes:
            figures_dir = Path(figures_dir)
            fname = f"plot_{fig_index:02d}_{self._safe_filename(title)}.png"
            (figures_dir / fname).write_bytes(image_bytes)
            rel = f"{figures_dir.name}/{fname}"
            lines.append(f"![{title}]({rel})")
        elif image_bytes:
            # Fallback: inline base64 (works in elabFTW and most MD renderers)
            b64 = base64.b64encode(image_bytes).decode("ascii")
            lines.append(f"![{title}](data:image/png;base64,{b64})")
        else:
            lines.append("*[No image data]*")

        return "\n".join(lines)

    def to_html(self) -> str:
        import base64
        title = self._payload.get("title", "Plot")
        ts = self._payload.get("timestamp", "")
        image_bytes: bytes = self._payload.get("image_bytes", b"")
        ts_str = f"<p style='font-size:small;color:gray;margin:2px 0'>{ts}</p>" if ts else ""
        if image_bytes:
            b64 = base64.b64encode(image_bytes).decode("ascii")
            img_tag = (
                f"<img src='data:image/png;base64,{b64}' "
                f"style='max-width:100%; max-height:160mm; width:auto; height:auto; "
                f"display:block; margin:4px auto'/>"
            )
        else:
            img_tag = "<p><i>[No image data]</i></p>"
        # page-break-inside:avoid keeps title, timestamp and image together on one page
        return (
            "<div style='margin-bottom:8px; page-break-inside:avoid;'>"
            f"<h3 style='margin:4px 0'>{title}</h3>"
            f"{ts_str}"
            f"{img_tag}"
            "</div>"
        )

    def to_latex(self, figures_dir=None, fig_index: int = 1, **kwargs) -> str:
        from pathlib import Path
        title = self._payload.get("title", "Plot")
        ts = self._payload.get("timestamp", "")
        pdf_bytes: bytes = self._payload.get("pdf_bytes", b"")
        image_bytes: bytes = self._payload.get("image_bytes", b"")
        stem = f"plot_{fig_index:02d}_{self._safe_filename(title)}"
        escaped_title = _tex_escape(title)
        ts_comment = f"  % {ts}\n" if ts else ""

        if figures_dir is None:
            return f"% Plot '{title}' — run export via 'Export as LaTeX' to save figures"

        figures_dir = Path(figures_dir)
        if pdf_bytes:
            fig_file = f"{stem}.pdf"
            (figures_dir / fig_file).write_bytes(pdf_bytes)
        elif image_bytes:
            # Fallback: no pdf_bytes (payload from before this feature)
            fig_file = f"{stem}.png"
            (figures_dir / fig_file).write_bytes(image_bytes)
        else:
            return f"% Plot '{title}' — no image data available"

        return (
            "\\begin{figure}[htbp]\n"
            "  \\centering\n"
            f"{ts_comment}"
            f"  \\includegraphics[width=\\linewidth]{{figures/{fig_file}}}\n"
            f"  \\caption{{{escaped_title}}}\n"
            f"  \\label{{fig:{stem}}}\n"
            "\\end{figure}"
        )


# ---------------------------------------------------------------------------
# Provenance block
# ---------------------------------------------------------------------------

class ProvenanceBlock(ReportBlock):
    """
    Block embedding a JADE-DLS provenance record (JSON) into the report.

    The ``record_id`` links this report to the provenance JSON file exported
    alongside it.  The full JSON is included in TXT/Markdown exports; a
    compact header is shown in the GUI and PDF.
    """

    def __init__(self, payload: dict, pipeline=None):
        self._record_id = payload.get("record_id", "")
        self._json_str = payload.get("json_str", "{}")
        self._timestamp = payload.get("timestamp", "")
        short_id = self._record_id[:8] + "…" if len(self._record_id) > 8 else self._record_id
        super().__init__(f"Provenance Record  [{short_id}]")

    def _fill_content(self, layout: QVBoxLayout):
        from PyQt5.QtCore import Qt as _Qt

        id_label = QLabel(f"<b>Record ID:</b> <code>{self._record_id}</code>")
        id_label.setTextInteractionFlags(_Qt.TextSelectableByMouse)
        id_label.setTextFormat(_Qt.RichText)
        layout.addWidget(id_label)

        if self._timestamp:
            ts_label = QLabel(f"<i>Created: {self._timestamp}</i>")
            ts_label.setTextFormat(_Qt.RichText)
            layout.addWidget(ts_label)

        note = QLabel(
            "Full provenance JSON is included in TXT and Markdown exports. "
            "Export the provenance JSON separately via the Provenance tab."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: gray; font-size: 8pt;")
        layout.addWidget(note)

        # Compact JSON preview (first 20 lines)
        preview = QTextEdit()
        preview.setReadOnly(True)
        preview.setFont(QFont("Courier New", 8))
        preview.setMaximumHeight(160)
        lines = self._json_str.split("\n")
        preview_text = "\n".join(lines[:20])
        if len(lines) > 20:
            preview_text += f"\n... ({len(lines) - 20} more lines)"
        preview.setPlainText(preview_text)
        layout.addWidget(preview)

    def to_text(self, **kwargs) -> str:
        sep = "=" * 60
        return (
            f"{sep}\nPROVENANCE RECORD\n{sep}\n"
            f"Record ID : {self._record_id}\n"
            f"Created   : {self._timestamp}\n\n"
            f"{self._json_str}\n{sep}"
        )

    def to_markdown(self, **kwargs) -> str:
        return (
            f"## Provenance Record\n\n"
            f"**Record ID:** `{self._record_id}`  \n"
            f"*Created: {self._timestamp}*\n\n"
            f"```json\n{self._json_str}\n```"
        )

    def to_html(self, **kwargs) -> str:
        import html as _html
        escaped = _html.escape(self._json_str)
        return (
            "<div style='page-break-inside:avoid;'>"
            "<h3>Provenance Record</h3>"
            f"<p><b>Record ID:</b> <code>{self._record_id}</code></p>"
            f"<p><i>Created: {self._timestamp}</i></p>"
            "<pre style='font-family: monospace; font-size: 8pt; "
            "background: #f5f5f5; padding: 8px; overflow-x: auto; "
            "white-space: pre-wrap; word-break: break-all;'>"
            f"{escaped}</pre>"
            "</div>"
        )

    def to_latex(self, **kwargs) -> str:
        short_id = self._record_id[:8] if self._record_id else "unknown"
        return f"% Provenance record omitted from TeX export --- see provenance_{short_id}.json"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_BLOCK_REGISTRY = {
    "metadata": MetadataBlock,
    "preprocessing": PreprocessingBlock,
    "result_summary": ResultSummaryBlock,
    "result_detail": ResultDetailBlock,
    "plot": PlotBlock,
    "provenance": ProvenanceBlock,
}


def create_block(payload: dict, pipeline=None) -> ReportBlock:
    """Factory: create the appropriate ReportBlock subclass from a payload dict."""
    block_type = payload.get("block_type", "")
    cls = _BLOCK_REGISTRY.get(block_type)
    if cls is None:
        raise ValueError(f"Unknown block type: {block_type!r}")
    return cls(payload, pipeline=pipeline)


# ---------------------------------------------------------------------------
# Report Panel
# ---------------------------------------------------------------------------

class ReportPanel(QWidget):
    """
    Modular report builder panel.

    Blocks can be added from:
    - The "Add Block" toolbar menu (Metadata, Preprocessing Steps)
    - Programmatically via add_block_from_payload() (called by main_window
      when the user right-clicks a result row and sends it to the report)

    Export formats: TXT, Markdown, PDF.
    """

    def __init__(self, pipeline=None, parent=None):
        super().__init__(parent)
        self._pipeline = pipeline
        self._provenance_panel = None   # set by InspectorPanel after construction
        self._blocks: list = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Toolbar ---
        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)

        add_btn = QToolButton()
        add_btn.setText("＋ Add Block")
        add_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        add_btn.setPopupMode(QToolButton.InstantPopup)
        add_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        add_menu = QMenu(self)
        add_menu.addAction("Metadata").triggered.connect(
            lambda: self.add_block_from_payload({"block_type": "metadata"})
        )
        add_menu.addAction("Preprocessing Steps").triggered.connect(
            lambda: self.add_block_from_payload({"block_type": "preprocessing"})
        )
        add_menu.addSeparator()
        add_menu.addAction("Provenance Record").triggered.connect(
            self._add_provenance_block
        )
        add_btn.setMenu(add_menu)
        toolbar.addWidget(add_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        clear_btn.clicked.connect(self._clear_all)
        toolbar.addWidget(clear_btn)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # --- Scroll area for blocks ---
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)

        self._container = QWidget()
        self._blocks_layout = QVBoxLayout()
        self._blocks_layout.setContentsMargins(0, 0, 0, 0)
        self._blocks_layout.setSpacing(6)
        self._blocks_layout.addStretch()   # spacer at the bottom

        self._container.setLayout(self._blocks_layout)
        self._scroll.setWidget(self._container)
        layout.addWidget(self._scroll)

        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Block management
    # ------------------------------------------------------------------

    def add_block_from_payload(self, payload: dict):
        """Create a block from a payload dict and add it to the panel."""
        try:
            block = create_block(payload, pipeline=self._pipeline)
        except ValueError as exc:
            QMessageBox.warning(self, "Report", str(exc))
            return
        self.add_block(block)

    def add_block(self, block: ReportBlock):
        """Insert block before the stretch spacer. Wire up controls."""
        block.removed.connect(self.remove_block)
        block._up_btn.clicked.connect(lambda: self._move_block_up(block))
        block._down_btn.clicked.connect(lambda: self._move_block_down(block))

        # Insert before the stretch spacer (last item)
        insert_pos = self._blocks_layout.count() - 1
        self._blocks_layout.insertWidget(insert_pos, block)
        self._blocks.append(block)

        # Scroll to the new block
        self._scroll.ensureWidgetVisible(block)

    def remove_block(self, block: ReportBlock):
        """Remove a block from the panel."""
        if block in self._blocks:
            self._blocks.remove(block)
            self._blocks_layout.removeWidget(block)
            block.deleteLater()

    def _move_block_up(self, block: ReportBlock):
        idx = self._blocks.index(block)
        if idx <= 0:
            return
        # Swap in list
        self._blocks[idx], self._blocks[idx - 1] = self._blocks[idx - 1], self._blocks[idx]
        # Swap in layout (layout positions don't include the trailing spacer)
        layout_pos = self._blocks_layout.indexOf(block)
        self._blocks_layout.removeWidget(block)
        self._blocks_layout.insertWidget(layout_pos - 1, block)

    def _move_block_down(self, block: ReportBlock):
        idx = self._blocks.index(block)
        if idx >= len(self._blocks) - 1:
            return
        self._blocks[idx], self._blocks[idx + 1] = self._blocks[idx + 1], self._blocks[idx]
        layout_pos = self._blocks_layout.indexOf(block)
        self._blocks_layout.removeWidget(block)
        self._blocks_layout.insertWidget(layout_pos + 1, block)

    def _clear_all(self):
        if not self._blocks:
            return
        reply = QMessageBox.question(
            self, "Clear Report",
            "Remove all blocks from the report?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            for block in list(self._blocks):
                self.remove_block(block)

    def set_provenance_panel(self, panel) -> None:
        """Store a reference to the ProvenancePanel so we can snapshot it."""
        self._provenance_panel = panel

    def _add_provenance_block(self) -> None:
        """Add the current provenance record as a report block."""
        if self._provenance_panel is None:
            QMessageBox.information(
                self,
                "Provenance",
                "No provenance data available yet. Load data and run an analysis first.",
            )
            return
        from datetime import datetime as _dt
        record = self._provenance_panel.get_record()
        payload = {
            "block_type": "provenance",
            "record_id": record.record_id,
            "json_str": record.to_json(),
            "timestamp": record.created,
        }
        self.add_block_from_payload(payload)

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def _collect_output(self, mode: str) -> str:
        """Collect rendered output from all blocks (HTML uses base64 for images)."""
        parts = []
        for block in self._blocks:
            if mode == "text":
                parts.append(block.to_text())
            elif mode == "markdown":
                parts.append(block.to_markdown())
            elif mode == "html":
                parts.append(block.to_html())
        separator = "\n\n" if mode != "html" else "\n"
        return separator.join(parts)

    def _collect_output_with_figures(self, mode: str, figures_dir=None) -> str:
        """
        Collect rendered output.  PlotBlocks are passed *figures_dir* so they
        save their PNG there and return a relative file reference instead of
        embedding base64 data.  Other blocks are rendered normally.
        """
        parts = []
        fig_index = 1
        for block in self._blocks:
            if isinstance(block, PlotBlock) and figures_dir is not None:
                if mode == "text":
                    parts.append(block.to_text(figures_dir=figures_dir,
                                                fig_index=fig_index))
                elif mode == "markdown":
                    parts.append(block.to_markdown(figures_dir=figures_dir,
                                                    fig_index=fig_index))
                fig_index += 1
            else:
                if mode == "text":
                    parts.append(block.to_text())
                elif mode == "markdown":
                    parts.append(block.to_markdown())
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Export: TXT
    # ------------------------------------------------------------------

    def _export_txt(self):
        if not self._blocks:
            QMessageBox.information(self, "Export", "No blocks in report.")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Report as TXT", "report.txt", "Text files (*.txt)"
        )
        if not filepath:
            return
        try:
            from pathlib import Path
            base = Path(filepath)
            # Create figures sub-folder only when PlotBlocks are present
            plot_blocks = [b for b in self._blocks if isinstance(b, PlotBlock)]
            if plot_blocks:
                figures_dir = base.parent / (base.stem + "_figures")
                figures_dir.mkdir(exist_ok=True)
            else:
                figures_dir = None

            content = self._collect_output_with_figures("text", figures_dir)
            base.write_text(content, encoding="utf-8")

            msg = f"Report saved to:\n{filepath}"
            if figures_dir:
                msg += f"\nFigures saved to:\n{figures_dir}"
            QMessageBox.information(self, "Export", msg)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ------------------------------------------------------------------
    # Export: Markdown
    # ------------------------------------------------------------------

    def _export_markdown(self):
        if not self._blocks:
            QMessageBox.information(self, "Export", "No blocks in report.")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Report as Markdown", "report.md", "Markdown files (*.md)"
        )
        if not filepath:
            return
        try:
            from pathlib import Path
            base = Path(filepath)
            # Create figures sub-folder only when PlotBlocks are present
            plot_blocks = [b for b in self._blocks if isinstance(b, PlotBlock)]
            if plot_blocks:
                figures_dir = base.parent / (base.stem + "_figures")
                figures_dir.mkdir(exist_ok=True)
            else:
                figures_dir = None

            content = self._collect_output_with_figures("markdown", figures_dir)
            base.write_text(content, encoding="utf-8")

            msg = f"Report saved to:\n{filepath}"
            if figures_dir:
                msg += f"\nFigures saved to:\n{figures_dir}"
            QMessageBox.information(self, "Export", msg)
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _export_pdf(self, landscape: bool = False):
        if not self._blocks:
            QMessageBox.information(self, "Export", "No blocks in report.")
            return
        orientation_label = "Landscape" if landscape else "Portrait"
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            f"Export Report as PDF ({orientation_label})",
            f"report_{orientation_label.lower()}.pdf",
            "PDF files (*.pdf)"
        )
        if not filepath:
            return
        try:
            from PyQt5.QtPrintSupport import QPrinter
            from PyQt5.QtGui import QTextDocument

            full_html = (
                "<html><body style='font-family: Arial, sans-serif; font-size: 11pt;'>"
                + self._collect_output("html")
                + "</body></html>"
            )

            doc = QTextDocument()
            doc.setHtml(full_html)

            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(filepath)
            printer.setPageMargins(20, 20, 20, 20, QPrinter.Millimeter)
            if landscape:
                printer.setOrientation(QPrinter.Landscape)
            else:
                printer.setOrientation(QPrinter.Portrait)
            doc.print_(printer)

            QMessageBox.information(self, "Export", f"PDF saved to:\n{filepath}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ------------------------------------------------------------------
    # Export: LaTeX
    # ------------------------------------------------------------------

    _tex_preamble = staticmethod(_tex_preamble)

    def _collect_latex(self, figures_dir=None) -> str:
        """Collect LaTeX output from all blocks. PlotBlocks receive figures_dir."""
        parts = []
        fig_index = 1
        for block in self._blocks:
            if isinstance(block, PlotBlock):
                parts.append(block.to_latex(figures_dir=figures_dir,
                                             fig_index=fig_index))
                fig_index += 1
            else:
                parts.append(block.to_latex())
        return "\n\n".join(parts)

    def _export_tex(self):
        if not self._blocks:
            QMessageBox.information(self, "Export", "No blocks in report.")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Report as LaTeX", "report.tex", "TeX files (*.tex)"
        )
        if not filepath:
            return
        try:
            from pathlib import Path
            base = Path(filepath)
            tex_dir = base.parent / (base.stem + "_tex")
            figures_dir = tex_dir / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)

            body = self._collect_latex(figures_dir)
            content = (
                self._tex_preamble()
                + "\n\\begin{document}\n"
                + "\\maketitle\n\n"
                + body
                + "\n\n\\end{document}\n"
            )
            (tex_dir / "report.tex").write_text(content, encoding="utf-8")
            QMessageBox.information(self, "Export",
                                    f"LaTeX package saved to:\n{tex_dir}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))
