# -*- coding: utf-8 -*-
"""
plot_series_export.py
=======================
Generic extraction of per-series data tables from a matplotlib Figure, for
the "Export current plot as CSV" button in the Analysis View.

This module is the **data layer only** — pure Python/matplotlib/pandas, no
Qt. It deliberately reads from the Figure itself rather than from any
per-analyzer data structure: every plot in the app's plot registry
(``analysis_view.current_plots``) already carries a complete, labelled set
of matplotlib artists (see the plotting modules in ``ade_dls/analysis`` and
``ade_dls/gui/analysis``), so a single generic code path covers all of them
— Cumulant B/C/D fit panels, Γ-vs-q² summaries, NNLS/Regularized
distributions, clustering overviews, and so on — without per-plot-type
special casing.

Follow-up (not implemented here): for a handful of well-known plots (e.g.
Cumulant Method A) the analyzer also retains the exact numeric DataFrame
behind the plot, which could be exported with cleaner column names/units
than generic artist introspection allows. That "enrichment" layer, if ever
wanted, should live in a separate module and fall back to this one for any
plot it doesn't specifically recognise.
"""
import re
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.container import BarContainer, ErrorbarContainer

from ade_dls.gui.export.csv_export import _safe_filename


@dataclass
class PlotSeries:
    """One exportable data series from one panel (Axes) of a Figure."""
    name: str            # filename-safe slug, unique within its panel
    label: str           # original artist label ('' when matplotlib auto-generated it)
    panel_index: int
    panel_title: str
    xlabel: str
    ylabel: str
    xscale: str          # 'linear' | 'log'
    yscale: str
    kind: str             # 'line' | 'scatter' | 'errorbar' | 'bar'
    data: pd.DataFrame    # columns depend on kind, see extract_plot_series()


def _slug(label, fallback: str, max_len: int = 25) -> str:
    """Turn an artist label into a short filename-safe token.

    Strips parenthesised statistics (e.g. "Fit (R²=0.9931)" -> "Fit")
    before sanitising, so repeated exports of the same series family don't
    each get a different, stat-dependent filename.
    """
    s = re.sub(r'\s*\(.*?\)', '', str(label or ''))
    s = re.sub(r'[^0-9A-Za-z]+', '_', s).strip('_').lower()
    return (s or fallback)[:max_len]


def _is_auto_label(label) -> bool:
    """True for matplotlib's own auto-generated labels ('_child0', '_line1', ...)
    or an empty/missing label — i.e. the artist was never explicitly labelled."""
    return (not label) or str(label).startswith('_')


def _unique(name: str, taken: set) -> str:
    base, n = name, 2
    while name in taken:
        name = f"{base}_{n}"
        n += 1
    taken.add(name)
    return name


def extract_plot_series(fig, plot_name: str) -> List[PlotSeries]:
    """
    Walk every Axes (panel) of *fig* and return one :class:`PlotSeries` per
    exportable artist.

    Handling per artist type:
      - ``ax.containers`` (checked first, so their helper Line2D objects are
        not double-counted as plain lines):
          - ``ErrorbarContainer`` -> kind ``'errorbar'``, columns
            ``x, y[, y_err_lo, y_err_hi]``.
          - ``BarContainer`` (e.g. a histogram) -> kind ``'bar'``, columns
            ``bin_left, bin_right, height``.
      - ``ax.collections``: only ``PathCollection`` (scatter) with a
        non-empty offsets array -> kind ``'scatter'``, columns ``x, y``.
        ``PolyCollection`` (``fill_between`` shading) is skipped — the line
        it shades is exported anyway.
      - ``ax.get_lines()``: ordinary ``ax.plot`` lines -> kind ``'line'``,
        columns ``x, y``. ``axhline``/``axvline`` are detected (and
        skipped) via their blended axes transform; zero-length proxy
        legend lines (``ax.plot([], [], label=...)``, used e.g. for
        Guinier legend entries) are skipped; unlabelled lines (Q-Q plots,
        residual panels) get a fallback name.
    """
    out: List[PlotSeries] = []
    for k, ax in enumerate(fig.get_axes()):
        title = ax.get_title() or f"panel{k}"
        common = dict(panel_index=k, panel_title=title, xlabel=ax.get_xlabel(),
                      ylabel=ax.get_ylabel(), xscale=ax.get_xscale(), yscale=ax.get_yscale())
        taken = set()
        consumed = set()   # id() of Line2D objects already handled via a container
        is_qq = ('q-q' in title.casefold()) or ('probability' in title.casefold())
        n_unlabeled = 0

        # 1) containers first (errorbar / hist)
        for cont in ax.containers:
            if isinstance(cont, ErrorbarContainer):
                data_line, caplines, barlinecols = cont.lines
                for cl in caplines:
                    consumed.add(id(cl))
                if data_line is None:
                    continue
                consumed.add(id(data_line))
                x = np.asarray(data_line.get_xdata(), dtype=float)
                y = np.asarray(data_line.get_ydata(), dtype=float)
                df = pd.DataFrame({'x': x, 'y': y})
                if cont.has_yerr and len(barlinecols):
                    segs = np.asarray(barlinecols[-1].get_segments(), dtype=float)
                    if len(segs) == len(y):
                        df['y_err_lo'] = y - segs[:, 0, 1]
                        df['y_err_hi'] = segs[:, 1, 1] - y
                lbl = cont.get_label()
                name = _unique(_slug(lbl, 'errorbar'), taken)
                out.append(PlotSeries(name=name, label='' if _is_auto_label(lbl) else lbl,
                                      kind='errorbar', data=df, **common))
            elif isinstance(cont, BarContainer):
                rows = [(p.get_x(), p.get_x() + p.get_width(), p.get_height())
                        for p in cont.patches]
                if not rows:
                    continue
                df = pd.DataFrame(rows, columns=['bin_left', 'bin_right', 'height'])
                lbl = cont.get_label()
                name = _unique(_slug(lbl, 'histogram'), taken)
                out.append(PlotSeries(name=name, label='' if _is_auto_label(lbl) else lbl,
                                      kind='bar', data=df, **common))

        # 2) scatter (PathCollection); PolyCollection (fill_between) skipped
        n_scatter = 0
        for coll in ax.collections:
            if not isinstance(coll, PathCollection):
                continue
            offsets = np.ma.getdata(coll.get_offsets())
            if offsets is None or len(offsets) == 0:
                continue
            n_scatter += 1
            lbl = coll.get_label()
            name = _unique(_slug(lbl, f'scatter_{n_scatter}'), taken)
            out.append(PlotSeries(
                name=name, label='' if _is_auto_label(lbl) else lbl, kind='scatter',
                data=pd.DataFrame({'x': offsets[:, 0].astype(float),
                                   'y': offsets[:, 1].astype(float)}),
                **common))

        # 3) lines
        for line in ax.get_lines():
            if id(line) in consumed:
                continue
            if line.get_transform() is not ax.transData:
                continue  # axhline/axvline use a blended axes transform
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            if len(x) == 0:
                continue  # zero-length proxy legend line
            lbl = line.get_label()
            if _is_auto_label(lbl):
                n_unlabeled += 1
                markers_only = str(line.get_linestyle()) in ('None', '', ' ')
                if is_qq:
                    fallback = 'qq_points' if markers_only else 'qq_fit'
                else:
                    fallback = f'points_{n_unlabeled}' if markers_only else f'line_{n_unlabeled}'
                name, label = _unique(fallback, taken), ''
            else:
                name, label = _unique(_slug(lbl, 'line'), taken), lbl
            out.append(PlotSeries(name=name, label=label, kind='line',
                                  data=pd.DataFrame({'x': x, 'y': y}), **common))
    return out


def series_to_tables(series: List[PlotSeries], plot_name: str) -> Dict[str, pd.DataFrame]:
    """
    Map each series to a ``write_tables``-ready ``{key: DataFrame}`` dict.

    Keys look like ``<plot slug, <=40 chars>__panel<k>__<series>`` and stay
    comfortably under the 80-char cap ``_safe_filename`` applies, so no
    silent truncation/collision happens at write time. Dict insertion order
    matches *series* order.
    """
    plot_slug = _safe_filename(plot_name)[:40].rstrip('_.')
    tables: Dict[str, pd.DataFrame] = {}
    for s in series:
        key = f"{plot_slug}__panel{s.panel_index}__{s.name}"
        n = 2
        base = key
        while key in tables:
            key = f"{base}_{n}"
            n += 1
        tables[key] = s.data
    return tables


def describe_series(s: PlotSeries, plot_name: str, method_tag: str) -> dict:
    """Structured description for a provenance catalog entry / export metadata."""
    return {
        'plot': plot_name,
        'method': method_tag or None,
        'panel': s.panel_index,
        'panel_title': s.panel_title,
        'series': s.label or s.name,
        'kind': s.kind,
        'x_label': s.xlabel,
        'y_label': s.ylabel,
        'x_scale': s.xscale,
        'y_scale': s.yscale,
        'columns': list(s.data.columns),
        'n_points': int(len(s.data)),
    }
