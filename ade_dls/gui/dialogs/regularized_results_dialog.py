"""
Regularized NNLS Results Dialog
Displays Regularized NNLS analysis results with multiple peaks
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QPushButton, QTableWidget, QTableWidgetItem,
                             QTabWidget, QWidget, QHeaderView, QMessageBox,
                             QFileDialog, QTextEdit, QSpinBox, QDoubleSpinBox,
                             QFormLayout, QCheckBox, QSizePolicy, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor
import pandas as pd
import numpy as np


class RegularizedResultsDialog(QDialog):
    """
    Dialog to display Regularized NNLS analysis results

    Shows:
        - Summary table with Rh values for all peaks
        - Detailed statistics
        - Export options
    """

    def __init__(self, laplace_analyzer, parent=None):
        super().__init__(parent)
        self.laplace_analyzer = laplace_analyzer

        self.init_ui()
        self.setWindowTitle("Regularized Results")
        self.resize(900, 700)

    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Regularized NNLS Analysis Results")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Check if results exist
        if self.laplace_analyzer.regularized_final_results is None:
            error_label = QLabel("No Regularized results available.\nPlease run Regularized NNLS analysis first.")
            error_label.setStyleSheet("color: red; font-size: 12pt; padding: 20px;")
            error_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(error_label)

            close_btn = QPushButton("Close")
            close_btn.clicked.connect(self.reject)
            layout.addWidget(close_btn)

            self.setLayout(layout)
            return

        # Tabs
        tabs = QTabWidget()

        # Tab 1: Summary
        summary_tab = self.create_summary_tab()
        tabs.addTab(summary_tab, "📊 Summary")

        # Tab 2: Detailed Data
        data_tab = self.create_data_tab()
        tabs.addTab(data_tab, "📋 Detailed Data")

        # Tab 3: Statistics
        stats_tab = self.create_statistics_tab()
        tabs.addTab(stats_tab, "📈 Statistics")

        # Tab 4: SLS Analysis
        sls_tab = self.create_sls_tab()
        tabs.addTab(sls_tab, "📡 SLS Analysis")

        layout.addWidget(tabs)

        # Bottom buttons
        button_layout = QHBoxLayout()

        export_btn = QPushButton("📁 Export Results")
        export_btn.clicked.connect(self.export_results)
        button_layout.addWidget(export_btn)

        refine_btn = QPushButton("🔧 Refine Results")
        refine_btn.setToolTip("Remove outliers and recalculate")
        refine_btn.clicked.connect(self.refine_results)
        button_layout.addWidget(refine_btn)

        button_layout.addStretch()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_summary_tab(self):
        """Create summary results tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Info box
        info_group = QGroupBox("Analysis Summary")
        info_layout = QVBoxLayout()

        num_datasets = len(self.laplace_analyzer.regularized_data) if self.laplace_analyzer.regularized_data is not None else 0
        num_peaks = len(self.laplace_analyzer.regularized_final_results)
        params = self.laplace_analyzer.regularized_params or {}

        info_text = f"""
        <b>Method:</b> Regularized NNLS (Tikhonov-Phillips)<br>
        <b>Datasets analyzed:</b> {num_datasets}<br>
        <b>Peaks found:</b> {num_peaks}<br>
        <b>Alpha:</b> {params.get('alpha', 'N/A')}<br>
        <b>Distance threshold:</b> {params.get('distance_threshold', 'N/A')}
        """

        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        info_layout.addWidget(info_label)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        # Results table
        results_group = QGroupBox("Hydrodynamic Radii")
        results_layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.populate_results_table()

        results_layout.addWidget(self.results_table)
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Interpretation
        interp_group = QGroupBox("Interpretation")
        interp_layout = QVBoxLayout()

        interpretation = self.generate_interpretation()
        interp_label = QLabel(interpretation)
        interp_label.setWordWrap(True)
        interp_label.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")

        interp_layout.addWidget(interp_label)
        interp_group.setLayout(interp_layout)
        layout.addWidget(interp_group)

        widget.setLayout(layout)
        return widget

    def populate_results_table(self):
        """Populate the results table"""
        df = self.laplace_analyzer.regularized_final_results

        # Set up table
        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            'Peak', 'Rh [nm]', 'Error [nm]', 'R²', 'D [m²/s]',
            'Skewness', 'Kurtosis', 'Alpha'
        ])

        def _fmt(val, fmt):
            try:
                return format(float(val), fmt)
            except (TypeError, ValueError):
                return 'N/A'

        # Populate data
        for i, row in df.iterrows():
            # Peak number
            peak_item = QTableWidgetItem(row['Fit'])
            peak_item.setFont(QFont('Arial', 10, QFont.Bold))
            self.results_table.setItem(i, 0, peak_item)

            # Rh
            rh_item = QTableWidgetItem(_fmt(row['Rh [nm]'], '.2f'))
            rh_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 1, rh_item)

            # Error
            err_item = QTableWidgetItem(f"± {_fmt(row['Rh error [nm]'], '.2f')}")
            err_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 2, err_item)

            # R²
            r2_item = QTableWidgetItem(_fmt(row['R_squared'], '.4f'))
            r2_item.setTextAlignment(Qt.AlignCenter)

            # Color code based on R² quality
            try:
                r2_val = float(row['R_squared'])
                if r2_val > 0.99:
                    r2_item.setBackground(QColor('#90EE90'))  # Light green
                elif r2_val > 0.95:
                    r2_item.setBackground(QColor('#FFE4B5'))  # Light orange
                else:
                    r2_item.setBackground(QColor('#FFB6C1'))  # Light red
            except (TypeError, ValueError):
                pass

            self.results_table.setItem(i, 3, r2_item)

            # D
            d_item = QTableWidgetItem(_fmt(row['D [m^2/s]'], '.2e'))
            d_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 4, d_item)

            # Skewness
            skew_item = QTableWidgetItem(_fmt(row.get('Skewness', float('nan')), '.3f'))
            skew_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 5, skew_item)

            # Kurtosis
            kurt_item = QTableWidgetItem(_fmt(row.get('Kurtosis', float('nan')), '.3f'))
            kurt_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 6, kurt_item)

            # Alpha
            alpha_item = QTableWidgetItem(_fmt(row.get('Alpha', float('nan')), '.4f'))
            alpha_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(i, 7, alpha_item)

        # Adjust column widths
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def create_data_tab(self):
        """Create detailed data tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Full data table
        data_group = QGroupBox("Full Analysis Data")
        data_layout = QVBoxLayout()

        self.data_table = QTableWidget()
        self.populate_data_table()

        data_layout.addWidget(self.data_table)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        widget.setLayout(layout)
        return widget

    def populate_data_table(self):
        """Populate the detailed data table"""
        if self.laplace_analyzer.regularized_data is None:
            return

        df = self.laplace_analyzer.regularized_data

        # Select relevant columns
        display_cols = ['filename', 'angle [°]', 'q^2']

        # Add tau and gamma columns
        tau_cols = [col for col in df.columns if col.startswith('tau_')]
        gamma_cols = [col for col in df.columns if col.startswith('gamma_')]
        intensity_cols = [col for col in df.columns if col.startswith('intensity_')]
        area_cols = [col for col in df.columns if col.startswith('area_')]
        fwhm_cols = [col for col in df.columns if col.startswith('fwhm_')]

        all_cols = display_cols + tau_cols + gamma_cols + intensity_cols + area_cols + fwhm_cols
        available_cols = [col for col in all_cols if col in df.columns]

        # Set up table
        self.data_table.setRowCount(len(df))
        self.data_table.setColumnCount(len(available_cols))
        self.data_table.setHorizontalHeaderLabels(available_cols)

        # Populate data
        for i, row in df.iterrows():
            for j, col in enumerate(available_cols):
                value = row[col]

                if pd.isna(value):
                    item = QTableWidgetItem("")
                elif isinstance(value, (int, np.integer)):
                    item = QTableWidgetItem(str(value))
                elif isinstance(value, (float, np.floating)):
                    if abs(value) < 0.01 or abs(value) > 1000:
                        item = QTableWidgetItem(f"{value:.3e}")
                    else:
                        item = QTableWidgetItem(f"{value:.4f}")
                else:
                    item = QTableWidgetItem(str(value))

                item.setTextAlignment(Qt.AlignCenter)
                self.data_table.setItem(i, j, item)

        # Adjust column widths
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def create_statistics_tab(self):
        """Create statistics tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Statistics text
        stats_group = QGroupBox("Statistical Summary")
        stats_layout = QVBoxLayout()

        stats_text = QTextEdit()
        stats_text.setReadOnly(True)
        stats_text.setHtml(self.generate_statistics_html())

        stats_layout.addWidget(stats_text)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        widget.setLayout(layout)
        return widget

    def generate_statistics_html(self):
        """Generate HTML formatted statistics"""
        df = self.laplace_analyzer.regularized_final_results

        html = "<html><body style='font-family: Arial;'>"
        html += "<h2>Regularized NNLS Analysis Statistics</h2>"

        # For each peak
        for i, row in df.iterrows():
            html += f"<h3>{row['Fit']}</h3>"
            html += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"

            html += f"<tr><td><b>Rh [nm]</b></td><td>{row['Rh [nm]']:.2f} ± {row['Rh error [nm]']:.2f}</td></tr>"
            html += f"<tr><td><b>D [m²/s]</b></td><td>{row['D [m^2/s]']:.3e} ± {row['D error [m^2/s]']:.3e}</td></tr>"
            html += f"<tr><td><b>R²</b></td><td>{row['R_squared']:.6f}</td></tr>"
            html += f"<tr><td><b>Residuals Normality</b></td><td>{row.get('Residuals', 'N/A')}</td></tr>"
            html += f"<tr><td><b>Alpha</b></td><td>{row.get('Alpha', 'N/A')}</td></tr>"

            html += "</table><br>"

        # Overall statistics
        html += "<h3>Overall Statistics</h3>"
        html += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"

        html += f"<tr><td><b>Number of Peaks</b></td><td>{len(df)}</td></tr>"
        html += f"<tr><td><b>Mean R²</b></td><td>{df['R_squared'].mean():.4f}</td></tr>"
        html += f"<tr><td><b>Min R²</b></td><td>{df['R_squared'].min():.4f}</td></tr>"

        html += "</table>"

        html += "</body></html>"

        return html

    def generate_interpretation(self):
        """Generate interpretation text"""
        df = self.laplace_analyzer.regularized_final_results
        num_peaks = len(df)

        if num_peaks == 0:
            return "⚠️ No peaks found. Consider adjusting alpha and distance parameters."
        elif num_peaks == 1:
            rh = df.iloc[0]['Rh [nm]']
            r2 = df.iloc[0]['R_squared']

            interpretation = f"✅ <b>Monodisperse sample detected</b><br>"
            interpretation += f"Single population with Rh = {rh:.2f} nm<br>"

            try:
                r2_val = float(r2)
                if r2_val > 0.99:
                    interpretation += "Excellent fit quality (R² > 0.99)"
                elif r2_val > 0.95:
                    interpretation += "Good fit quality (R² > 0.95)"
                else:
                    interpretation += "⚠️ Moderate fit quality - consider refining parameters"
            except (TypeError, ValueError):
                pass

        elif num_peaks == 2:
            rh1 = df.iloc[0]['Rh [nm]']
            rh2 = df.iloc[1]['Rh [nm]']

            interpretation = f"✅ <b>Bidisperse sample detected</b><br>"
            interpretation += f"Two populations:<br>"
            interpretation += f"  • Peak 1: Rh = {rh1:.2f} nm<br>"
            interpretation += f"  • Peak 2: Rh = {rh2:.2f} nm<br>"
            try:
                interpretation += f"  • Size ratio: {rh2/rh1:.2f}"
            except ZeroDivisionError:
                pass

        else:
            interpretation = f"✅ <b>Polydisperse sample detected</b><br>"
            interpretation += f"{num_peaks} distinct populations found:<br>"

            for i, row in df.iterrows():
                interpretation += f"  • Peak {i+1}: Rh = {row['Rh [nm]']:.2f} nm<br>"

            interpretation += "<br>💡 Multiple peaks indicate a heterogeneous sample"

        return interpretation

    # ------------------------------------------------------------------
    # SLS Analysis Tab
    # ------------------------------------------------------------------

    def create_sls_tab(self):
        """Create the SLS analysis tab with configuration, Guinier plot and summary table."""
        widget = QWidget()
        outer = QVBoxLayout()

        # ── Configuration ────────────────────────────────────────────
        config_group = QGroupBox("Configuration")
        form = QFormLayout()

        # n_populations — auto-set from regularized_final_results
        n_auto = len(self.laplace_analyzer.regularized_final_results) \
            if self.laplace_analyzer.regularized_final_results is not None else 1
        self.sls_npop_spin = QSpinBox()
        self.sls_npop_spin.setRange(1, 4)
        self.sls_npop_spin.setValue(min(n_auto, 4))
        self.sls_npop_spin.setToolTip("Number of populations to analyse")
        form.addRow("Populations:", self.sls_npop_spin)

        # q² range for Guinier fit
        q2_layout = QHBoxLayout()
        self.sls_q2_min = QDoubleSpinBox()
        self.sls_q2_min.setRange(0.0, 1000.0)
        self.sls_q2_min.setDecimals(6)
        self.sls_q2_min.setValue(0.0)
        self.sls_q2_min.setToolTip("Lower q² bound for Guinier fit (0 = no limit)")
        self.sls_q2_max = QDoubleSpinBox()
        self.sls_q2_max.setRange(0.0, 1000.0)
        self.sls_q2_max.setDecimals(6)
        self.sls_q2_max.setValue(0.0)
        self.sls_q2_max.setToolTip("Upper q² bound for Guinier fit (0 = no limit)")
        q2_layout.addWidget(QLabel("min:"))
        q2_layout.addWidget(self.sls_q2_min)
        q2_layout.addWidget(QLabel("max:"))
        q2_layout.addWidget(self.sls_q2_max)
        form.addRow("q² Guinier range [nm⁻²]:", q2_layout)

        # Exponent
        self.sls_exponent_spin = QSpinBox()
        self.sls_exponent_spin.setRange(1, 10)
        self.sls_exponent_spin.setValue(6)
        self.sls_exponent_spin.setToolTip(
            "Rh exponent for number-weighting:\n"
            "6 = Rayleigh (compact spheres)\n"
            "5 = Daoud-Cotton (star polymers)"
        )
        form.addRow("Rh exponent:", self.sls_exponent_spin)

        # Number-weighting checkbox
        self.sls_use_nw_cb = QCheckBox("Apply number-weighting correction")
        self.sls_use_nw_cb.setChecked(True)
        form.addRow("", self.sls_use_nw_cb)

        config_group.setLayout(form)
        outer.addWidget(config_group)

        # ── Action buttons ────────────────────────────────────────────
        btn_layout = QHBoxLayout()

        self.sls_load_btn = QPushButton("📂 Load Intensity Data")
        self.sls_load_btn.setToolTip(
            "Select the folder containing the ALV .ASC files to read "
            "count-rate and monitor-diode data from."
        )
        self.sls_load_btn.clicked.connect(self._sls_load_intensity)
        btn_layout.addWidget(self.sls_load_btn)

        self.sls_run_btn = QPushButton("▶ Run SLS Analysis")
        self.sls_run_btn.setEnabled(False)
        self.sls_run_btn.clicked.connect(self._sls_run_analysis)
        btn_layout.addWidget(self.sls_run_btn)

        self.sls_status_label = QLabel("Load intensity data to start.")
        self.sls_status_label.setStyleSheet("color: #666; font-style: italic;")
        btn_layout.addWidget(self.sls_status_label)
        btn_layout.addStretch()

        outer.addLayout(btn_layout)

        # ── Guinier Plot ──────────────────────────────────────────────
        plot_group = QGroupBox("Guinier Plot")
        plot_layout = QVBoxLayout()

        try:
            import matplotlib
            matplotlib.use('Qt5Agg')
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavToolbar
            import matplotlib.pyplot as plt

            self.sls_figure, self.sls_ax = plt.subplots(figsize=(7, 4))
            self.sls_canvas = FigureCanvas(self.sls_figure)
            self.sls_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.sls_nav = NavToolbar(self.sls_canvas, widget)

            # placeholder text
            self.sls_ax.text(0.5, 0.5, "Run SLS Analysis to see the Guinier plot",
                             ha='center', va='center', fontsize=11, color='#aaa',
                             transform=self.sls_ax.transAxes)
            self.sls_ax.set_xticks([])
            self.sls_ax.set_yticks([])

            plot_layout.addWidget(self.sls_nav)
            plot_layout.addWidget(self.sls_canvas)
            self._sls_plot_available = True
        except Exception:
            self._sls_plot_available = False
            fallback = QLabel("Matplotlib Qt5 backend not available. "
                              "Plot will be shown in a separate window.")
            fallback.setWordWrap(True)
            plot_layout.addWidget(fallback)

        plot_group.setLayout(plot_layout)
        outer.addWidget(plot_group)

        # ── Summary Table ─────────────────────────────────────────────
        summary_group = QGroupBox("SLS Summary")
        summary_layout = QVBoxLayout()

        self.sls_summary_table = QTableWidget()
        self.sls_summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.sls_summary_table.setAlternatingRowColors(True)
        self.sls_summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        summary_layout.addWidget(self.sls_summary_table)

        export_csv_btn = QPushButton("💾 Export SLS Summary (CSV)")
        export_csv_btn.clicked.connect(self._sls_export_csv)
        summary_layout.addWidget(export_csv_btn)

        summary_group.setLayout(summary_layout)
        outer.addWidget(summary_group)

        widget.setLayout(outer)
        return widget

    def _sls_load_intensity(self):
        """Slot: open folder dialog, load intensity data from ALV files."""
        import os

        folder = QFileDialog.getExistingDirectory(
            self, "Select folder with ALV .ASC files", "",
            QFileDialog.ShowDirsOnly
        )
        if not folder:
            return

        # Collect .ASC files
        file_paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith('.asc')
        ]

        if not file_paths:
            QMessageBox.warning(self, "No Files Found",
                                f"No .ASC files found in:\n{folder}")
            return

        try:
            ok = self.laplace_analyzer.load_intensity_data(file_paths)
            if ok:
                n = len(self.laplace_analyzer.df_intensity)
                self.sls_status_label.setText(
                    f"✅ Loaded {n} intensity records from {folder}")
                self.sls_run_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "Load Failed",
                                    "Could not load intensity data. "
                                    "Check that the files contain MeanCR0/MeanCR1 fields.")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _sls_run_analysis(self):
        """Slot: run SLS analysis and update plot + table."""
        n_pop = self.sls_npop_spin.value()
        q2_min = self.sls_q2_min.value()
        q2_max = self.sls_q2_max.value()
        exponent = self.sls_exponent_spin.value()
        use_nw = self.sls_use_nw_cb.isChecked()

        q2_range = None
        if q2_min > 0 or q2_max > 0:
            q2_range = (q2_min if q2_min > 0 else 0.0,
                        q2_max if q2_max > 0 else float('inf'))

        try:
            summary_df = self.laplace_analyzer.run_sls_analysis(
                n_populations=n_pop,
                q2_range=q2_range,
                exponent=exponent,
                use_nw=use_nw,
            )
            self._sls_update_plot()
            self._sls_populate_table(summary_df)
            self.sls_status_label.setText("✅ SLS analysis complete.")
        except RuntimeError as e:
            QMessageBox.warning(self, "SLS Analysis Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", str(e))

    def _sls_update_plot(self):
        """Redraw the embedded Guinier plot."""
        from ade_dls.analysis.sls import plot_guinier

        if not self._sls_plot_available:
            # Fallback: standalone window
            plot_guinier(
                self.laplace_analyzer.guinier_results,
                total_result=self.laplace_analyzer.guinier_total,
            )
            return

        self.sls_ax.clear()
        plot_guinier(
            self.laplace_analyzer.guinier_results,
            total_result=self.laplace_analyzer.guinier_total,
            ax=self.sls_ax,
        )
        self.sls_figure.tight_layout()
        self.sls_canvas.draw()

    def _sls_populate_table(self, df):
        """Fill the SLS summary QTableWidget from the summary DataFrame."""
        if df is None or df.empty:
            return

        self.sls_summary_table.setRowCount(len(df))
        self.sls_summary_table.setColumnCount(len(df.columns))
        self.sls_summary_table.setHorizontalHeaderLabels(list(df.columns))

        def _fmt(val):
            if pd.isna(val):
                return '—'
            try:
                f = float(val)
                if abs(f) < 0.01 or abs(f) > 1e4:
                    return f'{f:.4e}'
                return f'{f:.4f}'
            except (TypeError, ValueError):
                return str(val)

        for r, (_, row) in enumerate(df.iterrows()):
            for c, col in enumerate(df.columns):
                item = QTableWidgetItem(_fmt(row[col]))
                item.setTextAlignment(Qt.AlignCenter)
                # Colour qRg_max > 1.3 warning
                if col == 'qRg_max':
                    try:
                        if float(row[col]) > 1.3:
                            item.setBackground(QColor('#FFB6C1'))
                    except (TypeError, ValueError):
                        pass
                self.sls_summary_table.setItem(r, c, item)

    def _sls_export_csv(self):
        """Export SLS summary table to CSV."""
        if self.laplace_analyzer.sls_summary is None:
            QMessageBox.warning(self, "No Data", "Run SLS Analysis first.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export SLS Summary", "", "CSV Files (*.csv)")
        if filename:
            try:
                self.laplace_analyzer.sls_summary.to_csv(filename, index=False)
                QMessageBox.information(self, "Export Successful",
                                        f"SLS summary saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def export_results(self):
        """Export results to file"""
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Regularized Results",
            "",
            "Excel Files (*.xlsx);;CSV Files (*.csv);;Text Files (*.txt)"
        )

        if filename:
            try:
                self.laplace_analyzer.export_results(filename, method='regularized')
                QMessageBox.information(self, "Export Successful",
                                       f"Results exported to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error",
                                    f"Error exporting results:\n{str(e)}")

    def refine_results(self):
        """Open refinement dialog"""
        from ade_dls.gui.dialogs.postfilter_dialog import PostFilterDialog

        dialog = PostFilterDialog(
            self.laplace_analyzer.regularized_data,
            'Regularized',
            parent=self
        )

        if dialog.exec_():
            filtered_data = dialog.get_filtered_data()
            if filtered_data is not None:
                original_index = set(self.laplace_analyzer.regularized_data.index)
                kept_index = set(filtered_data.index)
                indices_to_remove = list(original_index - kept_index)
                if indices_to_remove:
                    # Remove outliers
                    self.laplace_analyzer.remove_regularized_outliers(indices_to_remove)

                    # Recalculate with stored params
                    params = self.laplace_analyzer.regularized_params or {}
                    self.laplace_analyzer.calculate_regularized_diffusion_coefficients(
                        use_clustering=params.get('use_clustering', True),
                        distance_threshold=params.get('distance_threshold', 2.0),
                        clustering_strategy=params.get('clustering_strategy', 'silhouette_refined'),
                    )

                    # Refresh display
                    self.populate_results_table()
                    self.populate_data_table()

                    QMessageBox.information(self, "Refinement Complete",
                                           f"Removed {len(indices_to_remove)} outliers and recalculated results.")
