"""
Method D Post-Fit Refinement Dialog

Two-stage refinement for multi-exponential (Method D) results:
- Stage 1 (Clustering tab): Re-configure cross-file clustering parameters,
  re-run cluster_all_gammas, update all population rows.
- Stage 2 (Population N tabs): Adjust q-range, outlier filtering, and min.
  points per population; re-run only the OLS for that mode.
- Combined tab: Adjust q-range for the gamma_mean → Rh regression.
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QWidget, QGroupBox, QFormLayout, QLabel,
                             QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
                             QPushButton, QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt
import numpy as np

from .postfit_refinement_dialog import InteractivePlotWidget


class MethodDPostFitDialog(QDialog):
    """
    Post-fit refinement dialog for Method D (multi-exponential decomposition).

    Opens tabs based on data available in the cumulant_analyzer:
    - "Clustering" tab: re-tune cross-file clustering parameters
    - "Population N" tab (one per reliable population): q-range + outlier filtering
    - "Combined" tab: q-range for the gamma_mean → Rh regression
    """

    def __init__(self, cumulant_analyzer, parent=None):
        super().__init__(parent)
        self.analyzer = cumulant_analyzer
        self.setWindowTitle("Method D – Post-Fit Refinement")
        self.setMinimumWidth(700)
        self.setMinimumHeight(720)

        self._params = None   # filled by accept()

        # Read current cluster state
        self._clustered_df = getattr(self.analyzer, 'method_d_clustered_df', None)
        self._cluster_info = getattr(self.analyzer, 'method_d_cluster_info', {})
        self._method_d_data = getattr(self.analyzer, 'method_d_data', None)

        # Collect reliable population column names
        from ade_dls.analysis.clustering import get_reliable_gamma_cols
        self._reliable_cols = get_reliable_gamma_cols(self._cluster_info) if self._cluster_info else []

        self._init_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout()

        tabs = QTabWidget()
        tabs.addTab(self._build_clustering_tab(), "Clustering")

        # One tab per reliable population
        self._pop_widgets = {}   # pop_num -> dict of spinbox refs
        for col in self._reliable_cols:
            pop_num = int(col.replace('gamma_pop', ''))
            tab, refs = self._build_population_tab(pop_num, col)
            tabs.addTab(tab, f"Population {pop_num}")
            self._pop_widgets[pop_num] = refs

        # Combined tab
        tab_combined, refs_combined = self._build_combined_tab()
        tabs.addTab(tab_combined, "Combined")
        self._combined_refs = refs_combined

        layout.addWidget(tabs)
        layout.addLayout(self._button_row())
        self.setLayout(layout)

    def _build_clustering_tab(self):
        """Clustering parameter tab, pre-populated from current cluster_info."""
        widget = QWidget()
        main_layout = QVBoxLayout()

        info = QLabel(
            "<b>Cross-File Clustering Parameters</b><br><br>"
            "Adjust the clustering settings and click <i>Apply Refinement</i> to re-run "
            "the cross-file population clustering. All population Rh values will be updated."
        )
        info.setWordWrap(True)
        main_layout.addWidget(info)

        group = QGroupBox("Clustering Settings")
        form = QFormLayout()

        self._cl_method = QComboBox()
        self._cl_method.addItems(["Hierarchical – Ward linkage", "Simple – gap-based"])
        # pre-populate from cluster_info if available
        current_method = self._cluster_info.get('method', 'hierarchical')
        self._cl_method.setCurrentIndex(0 if 'hierarchical' in current_method else 1)
        form.addRow("Clustering method:", self._cl_method)

        self._cl_n_clusters = QSpinBox()
        self._cl_n_clusters.setRange(0, 20)
        self._cl_n_clusters.setValue(0)
        self._cl_n_clusters.setSpecialValueText("Auto-detect")
        n_pops = self._cluster_info.get('n_populations', 0)
        if isinstance(n_pops, int) and n_pops > 0:
            self._cl_n_clusters.setValue(n_pops)
        form.addRow("Number of populations:", self._cl_n_clusters)

        self._cl_distance = QDoubleSpinBox()
        self._cl_distance.setRange(0.01, 2.0)
        self._cl_distance.setValue(self._cluster_info.get('distance_threshold', 0.3))
        self._cl_distance.setDecimals(2)
        self._cl_distance.setSingleStep(0.05)
        self._cl_distance.setToolTip("Lower = more populations detected")
        form.addRow("Distance threshold (log-space):", self._cl_distance)

        self._cl_abundance = QDoubleSpinBox()
        self._cl_abundance.setRange(0.0, 1.0)
        self._cl_abundance.setValue(self._cluster_info.get('min_abundance', 0.3))
        self._cl_abundance.setDecimals(2)
        self._cl_abundance.setSingleStep(0.05)
        form.addRow("Min. population abundance:", self._cl_abundance)

        self._cl_silhouette = QCheckBox("Enable silhouette-based cluster refinement")
        current_strat = self._cluster_info.get('clustering_strategy', 'simple')
        self._cl_silhouette.setChecked('silhouette' in str(current_strat))
        form.addRow("", self._cl_silhouette)

        hint = QLabel(
            "<i>Re-clustering uses the already-fitted per-file gamma values and does "
            "not require re-fitting the correlations.</i>"
        )
        hint.setWordWrap(True)
        form.addRow("", hint)

        group.setLayout(form)
        main_layout.addWidget(group)

        # ------------------------------------------------------------------
        # Clustering preview (matplotlib canvas)
        # ------------------------------------------------------------------
        refresh_btn = QPushButton("↺ Refresh Clustering Preview")
        refresh_btn.setToolTip(
            "Re-run clustering with the current parameters above and update the visualization."
        )
        refresh_btn.clicked.connect(self._refresh_clustering_preview)
        main_layout.addWidget(refresh_btn)

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure as MplFigure
        self._cl_figure = MplFigure(figsize=(9, 4))
        self._cl_canvas = FigureCanvasQTAgg(self._cl_figure)
        self._cl_toolbar = NavigationToolbar2QT(self._cl_canvas, widget)
        main_layout.addWidget(self._cl_toolbar)
        main_layout.addWidget(self._cl_canvas)

        self._cl_stats_lbl = QLabel("")
        self._cl_stats_lbl.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self._cl_stats_lbl)

        # Draw initial preview from current clustering result
        if self._clustered_df is not None and self._cluster_info:
            self._draw_clustering_on_figure(self._clustered_df, self._cluster_info)

        widget.setLayout(main_layout)
        return widget

    def _draw_clustering_on_figure(self, clustered_df, cluster_info):
        """Re-draw the clustering preview canvas in-place."""
        import numpy as np
        import matplotlib.pyplot as _plt

        self._cl_figure.clear()
        ax1 = self._cl_figure.add_subplot(1, 2, 1)
        ax2 = self._cl_figure.add_subplot(1, 2, 2)

        COLORS = _plt.cm.tab10.colors
        reliable_pops = cluster_info.get('reliable_populations', [])
        q2 = clustered_df['q^2'].values

        for i, pop_num in enumerate(reliable_pops):
            col = f'gamma_pop{pop_num}'
            if col not in clustered_df.columns:
                continue
            gamma = clustered_df[col].values
            mask = ~np.isnan(gamma) & (q2 > 0)
            if not mask.any():
                continue
            D_pm2 = (gamma[mask] / q2[mask]) * 1e12   # 10⁻¹² m²/s (display-friendly)
            log_D  = np.log10(gamma[mask] / q2[mask])
            color  = COLORS[i % len(COLORS)]
            label  = f'Population {pop_num}'
            ax1.scatter(q2[mask], D_pm2, color=color, label=label, s=30, alpha=0.75)
            ax2.hist(log_D, bins=15, color=color, alpha=0.6, label=label)

        ax1.set_xlabel('q² [nm⁻²]')
        ax1.set_ylabel('D [10⁻¹² m²/s]')
        ax1.set_title('D vs q² by Population')
        ax1.legend(fontsize=8)

        ax2.set_xlabel('log₁₀(D [m²/s])')
        ax2.set_ylabel('Count')
        ax2.set_title('D Distribution by Population')
        ax2.legend(fontsize=8)

        self._cl_figure.tight_layout()
        self._cl_canvas.draw()

        # Update stats label
        n   = cluster_info.get('n_populations', 0)
        sil = cluster_info.get('silhouette_score')
        abu = cluster_info.get('population_abundances', [])
        sil_str = f'  |  Silhouette: {sil:.2f}' if sil is not None else ''
        abu_str = ('  |  Abundance: [' + ', '.join(f'{a:.0%}' for a in abu) + ']') if abu else ''
        self._cl_stats_lbl.setText(f'{n} population(s) found{sil_str}{abu_str}')

    def _refresh_clustering_preview(self):
        """Re-run cluster_all_gammas with current dialog params and redraw preview."""
        from ade_dls.analysis.clustering import cluster_all_gammas

        method_d_fit = getattr(self.analyzer, 'method_d_fit', None)
        if method_d_fit is None or self._method_d_data is None:
            self._cl_stats_lbl.setText('No Method D fit data available for preview.')
            return

        gamma_pop_cols = sorted([c for c in method_d_fit.columns
                                 if c.startswith('gamma_') and c != 'gamma_mean'])
        if not gamma_pop_cols:
            self._cl_stats_lbl.setText('No per-file gamma columns found — cannot preview.')
            return

        method_map = {0: 'hierarchical', 1: 'simple'}
        n_val = self._cl_n_clusters.value()
        params = {
            'method':             method_map[self._cl_method.currentIndex()],
            'n_clusters':         'auto' if n_val == 0 else n_val,
            'distance_threshold': self._cl_distance.value(),
            'min_abundance':      self._cl_abundance.value(),
            'clustering_strategy': ('silhouette_refined'
                                    if self._cl_silhouette.isChecked() else 'simple'),
        }
        try:
            clustered_df, cluster_info = cluster_all_gammas(
                self._method_d_data,
                gamma_cols=gamma_pop_cols,
                q_squared_col='q^2',
                normalize_by_q2=True,
                uncertainty_flags=False,
                plot=False,
                **params,
            )
            self._draw_clustering_on_figure(clustered_df, cluster_info)
        except Exception as exc:
            self._cl_stats_lbl.setText(f'Preview failed: {exc}')

    def _build_population_tab(self, pop_num, col):
        """Build a refinement tab for one population (gamma_pop_N vs q²)."""
        widget = QWidget()
        layout = QVBoxLayout()

        # Interactive plot
        if self._clustered_df is not None and col in self._clustered_df.columns:
            plot_data = self._clustered_df[[col, 'q^2']].dropna()
            plot_widget = InteractivePlotWidget(
                data=plot_data,
                gamma_col=col,
                q_squared_col='q^2',
                method_name=f"Method D – Population {pop_num}",
                parent=self,
            )
            layout.addWidget(plot_widget)
        else:
            layout.addWidget(QLabel(f"No data available for Population {pop_num}."))

        # Controls
        ctrl_group = QGroupBox("Regression Settings")
        form = QFormLayout()

        # q range
        q_row = QHBoxLayout()
        q_min_sb = QDoubleSpinBox()
        q_min_sb.setRange(0.0, 1e9)
        q_min_sb.setDecimals(4)
        q_min_sb.setValue(0.0)
        q_min_sb.setSuffix(" nm⁻²")
        q_min_sb.setSpecialValueText("—")

        q_max_sb = QDoubleSpinBox()
        q_max_sb.setRange(0.0, 1e9)
        q_max_sb.setDecimals(4)
        q_max_sb.setValue(0.0)
        q_max_sb.setSuffix(" nm⁻²")
        q_max_sb.setSpecialValueText("—")

        # Pre-fill from actual data range
        if self._clustered_df is not None and col in self._clustered_df.columns:
            valid_q = self._clustered_df.loc[self._clustered_df[col].notna(), 'q^2'].dropna()
            if len(valid_q) > 0:
                q_min_sb.setValue(float(valid_q.min()))
                q_max_sb.setValue(float(valid_q.max()))

        q_row.addWidget(QLabel("Min:"))
        q_row.addWidget(q_min_sb)
        q_row.addWidget(QLabel("Max:"))
        q_row.addWidget(q_max_sb)
        form.addRow("q² range [nm⁻²]:", q_row)

        enable_q = QCheckBox("Restrict q² range")
        enable_q.setChecked(False)
        q_min_sb.setEnabled(False)
        q_max_sb.setEnabled(False)
        enable_q.toggled.connect(q_min_sb.setEnabled)
        enable_q.toggled.connect(q_max_sb.setEnabled)
        form.addRow("", enable_q)

        # Outlier threshold
        outlier_sb = QDoubleSpinBox()
        outlier_sb.setRange(0.0, 10.0)
        outlier_sb.setValue(0.0)
        outlier_sb.setDecimals(1)
        outlier_sb.setSingleStep(0.5)
        outlier_sb.setSpecialValueText("Disabled (0)")
        outlier_sb.setToolTip("Points with |residual| > k×σ are excluded. 0 = disabled.")
        form.addRow("Outlier threshold (k×σ):", outlier_sb)

        # Min points
        min_pts_sb = QSpinBox()
        min_pts_sb.setRange(2, 100)
        min_pts_sb.setValue(2)
        min_pts_sb.setToolTip("Minimum number of data points required for OLS regression.")
        form.addRow("Min. points for regression:", min_pts_sb)

        ctrl_group.setLayout(form)
        layout.addWidget(ctrl_group)

        widget.setLayout(layout)

        refs = {
            'enable_q': enable_q,
            'q_min': q_min_sb,
            'q_max': q_max_sb,
            'outlier_sigma': outlier_sb,
            'min_points': min_pts_sb,
        }
        return widget, refs

    def _build_combined_tab(self):
        """Build combined (gamma_mean vs q²) tab."""
        widget = QWidget()
        layout = QVBoxLayout()

        info = QLabel(
            "<b>Combined Rh (Z-Average)</b><br><br>"
            "The combined row uses the intensity-weighted mean decay rate ⟨Γ⟩ "
            "vs q² to compute a single Rh analogous to Methods A/B/C. "
            "Adjust the q² range here."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Interactive plot for gamma_mean
        if self._method_d_data is not None and 'gamma_mean' in self._method_d_data.columns:
            plot_data = self._method_d_data[['gamma_mean', 'q^2']].dropna()
            plot_widget = InteractivePlotWidget(
                data=plot_data,
                gamma_col='gamma_mean',
                q_squared_col='q^2',
                method_name="Method D – Combined (⟨Γ⟩)",
                parent=self,
            )
            layout.addWidget(plot_widget)

        ctrl_group = QGroupBox("Combined Regression Settings")
        form = QFormLayout()

        q_row = QHBoxLayout()
        q_min_sb = QDoubleSpinBox()
        q_min_sb.setRange(0.0, 1e9)
        q_min_sb.setDecimals(4)
        q_min_sb.setValue(0.0)
        q_min_sb.setSuffix(" nm⁻²")
        q_min_sb.setSpecialValueText("—")

        q_max_sb = QDoubleSpinBox()
        q_max_sb.setRange(0.0, 1e9)
        q_max_sb.setDecimals(4)
        q_max_sb.setValue(0.0)
        q_max_sb.setSuffix(" nm⁻²")
        q_max_sb.setSpecialValueText("—")

        if self._method_d_data is not None and 'q^2' in self._method_d_data.columns:
            valid_q = self._method_d_data['q^2'].dropna()
            if len(valid_q) > 0:
                q_min_sb.setValue(float(valid_q.min()))
                q_max_sb.setValue(float(valid_q.max()))

        q_row.addWidget(QLabel("Min:"))
        q_row.addWidget(q_min_sb)
        q_row.addWidget(QLabel("Max:"))
        q_row.addWidget(q_max_sb)
        form.addRow("q² range [nm⁻²]:", q_row)

        enable_q = QCheckBox("Restrict q² range")
        enable_q.setChecked(False)
        q_min_sb.setEnabled(False)
        q_max_sb.setEnabled(False)
        enable_q.toggled.connect(q_min_sb.setEnabled)
        enable_q.toggled.connect(q_max_sb.setEnabled)
        form.addRow("", enable_q)

        ctrl_group.setLayout(form)
        layout.addWidget(ctrl_group)
        layout.addStretch()
        widget.setLayout(layout)

        refs = {'enable_q': enable_q, 'q_min': q_min_sb, 'q_max': q_max_sb}
        return widget, refs

    # ------------------------------------------------------------------
    # Plot interaction (called by InteractivePlotWidget.on_mouse_release)
    # ------------------------------------------------------------------

    def update_q_range_from_plot(self, q_min, q_max, method_name):
        """Called by InteractivePlotWidget when user drags a selection on the plot."""
        if 'Population' in method_name:
            # Find which population tab's spinboxes to update
            import re
            m = re.search(r'Population (\d+)', method_name)
            if m:
                pop_num = int(m.group(1))
                if pop_num in self._pop_widgets:
                    refs = self._pop_widgets[pop_num]
                    refs['enable_q'].setChecked(True)
                    refs['q_min'].setValue(q_min)
                    refs['q_max'].setValue(q_max)
        elif 'Combined' in method_name or '⟨Γ⟩' in method_name:
            self._combined_refs['enable_q'].setChecked(True)
            self._combined_refs['q_min'].setValue(q_min)
            self._combined_refs['q_max'].setValue(q_max)

    # ------------------------------------------------------------------
    # Button row
    # ------------------------------------------------------------------

    def _button_row(self):
        row = QHBoxLayout()
        reset_btn = QPushButton("Reset All")
        reset_btn.setToolTip("Reset all settings to their current (pre-refinement) values.")
        reset_btn.clicked.connect(self._reset_all)
        row.addWidget(reset_btn)
        row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(cancel_btn)

        apply_btn = QPushButton("Apply Refinement")
        apply_btn.setDefault(True)
        apply_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 16px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        apply_btn.clicked.connect(self.accept)
        row.addWidget(apply_btn)
        return row

    def _reset_all(self):
        """Reset all spinboxes to defaults."""
        for refs in self._pop_widgets.values():
            refs['enable_q'].setChecked(False)
            refs['outlier_sigma'].setValue(0.0)
            refs['min_points'].setValue(2)
        self._combined_refs['enable_q'].setChecked(False)

    # ------------------------------------------------------------------
    # accept / get_refinement_params
    # ------------------------------------------------------------------

    def accept(self):
        """Validate and collect all refinement parameters."""
        # Validate clustering params
        n_pops = self._cl_n_clusters.value()
        dist = self._cl_distance.value()
        abundance = self._cl_abundance.value()

        if dist <= 0:
            QMessageBox.warning(self, "Invalid Parameter",
                                "Distance threshold must be greater than 0.")
            return

        # Collect clustering params
        method_map = {0: 'hierarchical', 1: 'simple'}
        clustering = {
            'method': method_map[self._cl_method.currentIndex()],
            'n_clusters': 'auto' if n_pops == 0 else n_pops,
            'distance_threshold': dist,
            'min_abundance': abundance,
            'clustering_strategy': 'silhouette_refined' if self._cl_silhouette.isChecked() else 'simple',
        }

        # Collect per-population mode params
        mode_params = {}
        for pop_num, refs in self._pop_widgets.items():
            if refs['enable_q'].isChecked():
                q_min = refs['q_min'].value()
                q_max = refs['q_max'].value()
                if q_min >= q_max:
                    QMessageBox.warning(self, "Invalid q² Range",
                                        f"Population {pop_num}: min q² must be < max q².")
                    return
                mode_params[pop_num] = {
                    'q_min': q_min,
                    'q_max': q_max,
                    'outlier_sigma': refs['outlier_sigma'].value() or None,
                    'min_points': refs['min_points'].value(),
                }
            elif refs['outlier_sigma'].value() > 0 or refs['min_points'].value() != 2:
                mode_params[pop_num] = {
                    'q_min': None,
                    'q_max': None,
                    'outlier_sigma': refs['outlier_sigma'].value() or None,
                    'min_points': refs['min_points'].value(),
                }

        # Collect combined q range
        combined_q = None
        if self._combined_refs['enable_q'].isChecked():
            q_min = self._combined_refs['q_min'].value()
            q_max = self._combined_refs['q_max'].value()
            if q_min >= q_max:
                QMessageBox.warning(self, "Invalid q² Range",
                                    "Combined tab: min q² must be < max q².")
                return
            combined_q = (q_min, q_max)

        self._params = {
            'clustering': clustering,
            'mode_params': mode_params,
            'combined_q_range': combined_q,
        }
        super().accept()

    def get_refinement_params(self):
        """Return collected refinement parameters dict."""
        return self._params
