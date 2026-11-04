"""Report format helpers (no Qt): HTML-table parsing, Markdown/LaTeX/TSV table rendering,
LaTeX escaping, report preamble, and the text summaries used by the report blocks.

The GUI report panel (ade_dls.gui.widgets.report_panel) builds its blocks from these functions.
"""


from html.parser import HTMLParser


class TableParser(HTMLParser):
    """
    Extracts all <table> blocks from HTML as structured data.

    Result: list of tables, each table = list of rows,
    each row = list of (cell_text, is_header) tuples.
    """

    def __init__(self):
        super().__init__()
        self.tables: list = []          # final result
        self._cur_table: list = []      # rows in current table
        self._cur_row: list = []        # cells in current row
        self._cur_cell: list = []       # chars in current cell
        self._cur_is_header: bool = False
        self._in_table: int = 0         # nesting depth
        self._in_cell: bool = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._in_table += 1
            if self._in_table == 1:
                self._cur_table = []
        elif tag == "tr" and self._in_table == 1:
            self._cur_row = []
        elif tag in ("td", "th") and self._in_table == 1:
            self._cur_cell = []
            self._cur_is_header = (tag == "th")
            self._in_cell = True

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._in_cell:
            self._cur_row.append(("".join(self._cur_cell).strip(), self._cur_is_header))
            self._in_cell = False
        elif tag == "tr" and self._in_table == 1 and self._cur_row:
            self._cur_table.append(self._cur_row)
            self._cur_row = []
        elif tag == "table":
            if self._in_table == 1 and self._cur_table:
                self.tables.append(self._cur_table)
                self._cur_table = []
            self._in_table = max(0, self._in_table - 1)

    def handle_data(self, data):
        if self._in_cell:
            self._cur_cell.append(data)

    def handle_entityref(self, name):
        # Handle common HTML entities
        _entities = {"amp": "&", "lt": "<", "gt": ">", "nbsp": " ", "quot": '"'}
        if self._in_cell:
            self._cur_cell.append(_entities.get(name, ""))

    def handle_charref(self, name):
        if self._in_cell:
            try:
                if name.startswith("x"):
                    char = chr(int(name[1:], 16))
                else:
                    char = chr(int(name))
                self._cur_cell.append(char)
            except (ValueError, OverflowError):
                pass


def parse_html_tables(html: str):
    """
    Parse all <table> elements from an HTML string.

    Returns list of tables. Each table is a list of rows.
    Each row is a list of (cell_text, is_header) tuples.
    """
    parser = TableParser()
    parser.feed(html)
    return parser.tables


def table_to_tsv(table) -> str:
    """Render a parsed table as tab-separated values (TSV)."""
    lines = []
    for row in table:
        lines.append("\t".join(cell for cell, _ in row))
    return "\n".join(lines)


def table_to_markdown(table) -> str:
    """Render a parsed table as a GFM markdown table."""
    if not table:
        return ""

    # Collect all rows as plain text
    rows = [[cell for cell, _ in row] for row in table]

    # Determine column widths
    col_count = max(len(r) for r in rows)
    # Pad rows with fewer columns
    rows = [r + [""] * (col_count - len(r)) for r in rows]
    widths = [max(len(rows[i][c]) for i in range(len(rows))) for c in range(col_count)]
    widths = [max(w, 3) for w in widths]  # GFM requires at least 3 dashes

    def fmt_row(r):
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)) + " |"

    lines = []
    # First row of <th> cells is the header
    first_row_has_header = any(is_hdr for _, is_hdr in table[0])
    if first_row_has_header:
        lines.append(fmt_row(rows[0]))
        lines.append("| " + " | ".join("-" * widths[i] for i in range(col_count)) + " |")
        for r in rows[1:]:
            lines.append(fmt_row(r))
    else:
        # No header row — just output all rows, no separator
        for r in rows:
            lines.append(fmt_row(r))

    return "\n".join(lines)


# Mapping of Unicode characters that need a LaTeX equivalent.
# Applied by tex_escape() after the 10 special-character substitutions.
UNICODE_LATEX: dict = {
    # Superscript digits
    '²': '$^{2}$', '³': '$^{3}$', '¹': '$^{1}$', '⁰': '$^{0}$',
    '⁴': '$^{4}$', '⁵': '$^{5}$', '⁶': '$^{6}$',
    '⁷': '$^{7}$', '⁸': '$^{8}$', '⁹': '$^{9}$',
    # Subscript digits
    '₀': '$_{0}$', '₁': '$_{1}$', '₂': '$_{2}$', '₃': '$_{3}$',
    '₄': '$_{4}$', '₅': '$_{5}$', '₆': '$_{6}$',
    '₇': '$_{7}$', '₈': '$_{8}$', '₉': '$_{9}$',
    # Common math/unit symbols
    '±': '$\\pm$',
    'µ': '$\\mu$',       # U+00B5 MICRO SIGN
    'μ': '$\\mu$',       # U+03BC GREEK SMALL LETTER MU
    '°': '$^{\\circ}$',
    '×': '$\\times$',
    '≈': '$\\approx$',
    '≤': '$\\leq$',
    '≥': '$\\geq$',
    '≠': '$\\neq$',
    '∞': '$\\infty$',
    # Punctuation / dashes
    '…': '\\ldots{}',
    '–': '--',
    '—': '---',
    '→': '$\\rightarrow$',
    '←': '$\\leftarrow$',
    # Greek letters relevant for DLS
    'α': '$\\alpha$',    'β': '$\\beta$',   'γ': '$\\gamma$',
    'δ': '$\\delta$',    'ε': '$\\varepsilon$',
    'η': '$\\eta$',      'κ': '$\\kappa$',  'λ': '$\\lambda$',
    'ν': '$\\nu$',       'π': '$\\pi$',     'σ': '$\\sigma$',
    'τ': '$\\tau$',      'φ': '$\\phi$',    'ω': '$\\omega$',
    'Γ': '$\\Gamma$',    'Λ': '$\\Lambda$', 'Σ': '$\\Sigma$',
}


def tex_escape(s: str) -> str:
    """Escape special LaTeX characters and convert common Unicode to LaTeX equivalents."""
    # Use a placeholder so that the {} in \textbackslash{} survive the brace escaping below
    _BS = '\x00'
    s = s.replace('\\', _BS)
    s = s.replace('&',  r'\&')
    s = s.replace('%',  r'\%')
    s = s.replace('$',  r'\$')
    s = s.replace('#',  r'\#')
    s = s.replace('_',  r'\_')
    s = s.replace('{',  r'\{')
    s = s.replace('}',  r'\}')
    s = s.replace('~',  r'\textasciitilde{}')
    s = s.replace('^',  r'\textasciicircum{}')
    s = s.replace(_BS,  r'\textbackslash{}')
    for char, latex in UNICODE_LATEX.items():
        s = s.replace(char, latex)
    return s


def table_to_latex(table) -> str:
    """Render a parsed HTML table as a full-width tabularray tblr environment."""
    if not table:
        return ""
    col_count = max(len(row) for row in table)
    # X columns fill available width equally; width=\linewidth pins the table to the text column
    col_spec = f"width=\\linewidth, colspec={{{'X' * col_count}}}"
    first_row_is_header = any(is_hdr for _, is_hdr in table[0])
    lines = [f"\\begin{{tblr}}{{{col_spec}}}"]
    for row_idx, row in enumerate(table):
        cells = [cell for cell, _ in row]
        cells += [""] * (col_count - len(cells))
        escaped = [tex_escape(c) for c in cells]
        row_content = " & ".join(escaped)
        if row_idx == 0 and first_row_is_header:
            lines.append(f"  \\SetRow{{font=\\bfseries}} {row_content} \\\\")
            lines.append("  \\hline")
        else:
            lines.append(f"  {row_content} \\\\")
    lines.append("\\end{tblr}")
    return "\n".join(lines)


def collect_metadata_fields(pipeline) -> list:
    """Build list of (label, value) pairs from available pipeline data."""
    fields = []
    if pipeline is None:
        return [("Status", "No pipeline data available")]

    from datetime import datetime

    # Pipeline-level metadata
    meta = getattr(pipeline, 'metadata', {})
    if 'created' in meta:
        created = meta['created']
        if isinstance(created, datetime):
            fields.append(("Analysis started", created.strftime("%Y-%m-%d %H:%M:%S")))

    fields.append(("JADE-DLS version", meta.get('version', 'N/A')))

    # Dataset information
    data = getattr(pipeline, 'data', {})
    import pandas as pd

    basedata = data.get('basedata', pd.DataFrame())
    if not basedata.empty:
        if 'angle [°]' in basedata.columns:
            angles = sorted(basedata['angle [°]'].unique())
            fields.append(("Angles", ", ".join(f"{a:.0f}°" for a in angles)))
        if 'temperature [K]' in basedata.columns:
            mean_t = basedata['temperature [K]'].mean()
            std_t = basedata['temperature [K]'].std()
            fields.append(("Temperature", f"{mean_t:.2f} ± {std_t:.3f} K"))
        if 'viscosity [cp]' in basedata.columns:
            mean_v = basedata['viscosity [cp]'].mean()
            fields.append(("Viscosity", f"{mean_v:.4f} cP"))
        fields.append(("Files", str(len(basedata))))

    files = data.get('files', [])
    if files and basedata.empty:
        fields.append(("Files loaded", str(len(files))))

    if not fields:
        fields.append(("Status", "No data loaded yet"))

    return fields


def safe_param_str(val) -> str:
    """Convert a parameter value to a concise string (no huge arrays)."""
    if hasattr(val, '__len__') and not isinstance(val, str):
        if len(val) > 6:
            return f"[{len(val)} values]"
    return str(val)

def format_step_params(step_name: str, params: dict) -> str:
    """
    Return a human-readable, single-line parameter summary for a pipeline step.

    For 'Regularized NNLS Analysis' the decay_times array is compressed to a
    range description and only the user-configurable reproducibility parameters
    are shown.  For all other steps every parameter is included (they are
    always short).
    """
    if not params:
        return ""

    if step_name == "Regularized NNLS Analysis":
        parts = []

        # --- decay time range (compressed) ---
        taus = params.get('decay_times')
        if taus is not None and hasattr(taus, '__len__') and len(taus) > 2:
            parts.append(
                f"tau: {float(taus[0]):.2e}\u2026{float(taus[-1]):.2e} s "
                f"({len(taus)} pts, log)"
            )
        elif taus is not None:
            parts.append(f"tau: {taus}")

        # --- reproducibility-relevant parameters (in dialog order) ---
        _SHOWN = [
            'alpha', 'prominence', 'distance',
            'normalize', 'sparsity_penalty', 'enforce_unimodality',
            'use_clustering', 'clustering_strategy', 'distance_threshold',
            'peak_method',
        ]
        _SKIP_IF = {
            'sparsity_penalty': 0.0,
            'enforce_unimodality': False,
        }
        for key in _SHOWN:
            if key not in params:
                continue
            val = params[key]
            # Skip default values that add no information
            if key in _SKIP_IF and val == _SKIP_IF[key]:
                continue
            parts.append(f"{key}={val}")

        return "; ".join(parts)

    if step_name in ("Filter Correlations", "Filter Countrates"):
        parts = [
            f"original={params.get('original_count', '?')}",
            f"remaining={params.get('remaining_count', '?')}",
        ]
        excl = params.get('excluded_files', [])
        if excl:
            parts.append(f"excluded={len(excl)}")

        noise = params.get('noise_params')
        if noise:
            noise_parts = []
            if noise.get('baseline_correction'):
                noise_parts.append(f"baseline correction ({noise.get('baseline_pct', '?')}%)")
            if noise.get('intercept_correction'):
                noise_parts.append(f"intercept correction ({noise.get('intercept_pct', '?')}%)")
            if noise_parts:
                parts.append("noise reduction: " + ", ".join(noise_parts))

        return "; ".join(parts)

    # --- generic fallback for all other steps ---
    return "; ".join(f"{k}={safe_param_str(v)}"
                     for k, v in params.items())


def tex_preamble() -> str:
    return (
        "\\documentclass[a4paper, 11pt]{scrartcl}\n"
        "\\usepackage[T1]{fontenc}\n"
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage[english]{babel}\n"
        "\\usepackage[a4paper, margin=25mm]{geometry}\n"
        "\\usepackage{microtype}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{tabularray}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{siunitx}\n"
        "\\usepackage[hidelinks]{hyperref}\n"
        "\n"
        "\\title{DLS Analysis Report}\n"
        "\\author{}\n"
        "\\date{\\today}\n"
    )
