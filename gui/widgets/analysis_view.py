"""
Analysis View Widget
Main central panel for displaying data, plots, and results
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QTableWidget, QTableWidgetItem, QLabel, QTextEdit,
                             QGroupBox, QScrollArea, QListWidget, QListWidgetItem,
                             QPushButton, QSplitter)
from PyQt5.QtCore import Qt
import pandas as pd


class AnalysisView(QWidget):
    """
    Central panel showing data overview, plots, and results
    """

    def __init__(self, pipeline, parent=None):
        super().__init__(parent)
        self.pipeline = pipeline
        self.init_ui()

    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()

        # Create tab widget for different views
        self.tabs = QTabWidget()

        # Tab 1: Data Overview
        self.data_tab = self.create_data_overview_tab()
        self.tabs.addTab(self.data_tab, "📊 Data Overview")

        # Tab 2: Plots
        self.plot_tab = self.create_plot_tab()
        self.tabs.addTab(self.plot_tab, "📈 Plots")

        # Tab 3: Results
        self.results_tab = self.create_results_tab()
        self.tabs.addTab(self.results_tab, "📋 Results")

        # Tab 4: Comparison
        self.comparison_tab = self.create_comparison_tab()
        self.tabs.addTab(self.comparison_tab, "⚖️ Comparison")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def create_data_overview_tab(self):
        """Create data overview tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Data Overview")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Statistics group
        stats_group = QGroupBox("Dataset Statistics")
        stats_layout = QVBoxLayout()

        self.stats_label = QLabel("No data loaded")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Files table
        files_group = QGroupBox("Loaded Files")
        files_layout = QVBoxLayout()

        self.files_table = QTableWidget()
        self.files_table.setColumnCount(4)
        self.files_table.setHorizontalHeaderLabels(["Filename", "Angle (°)", "Temp (K)", "Status"])
        files_layout.addWidget(self.files_table)

        files_group.setLayout(files_layout)
        layout.addWidget(files_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_plot_tab(self):
        """Create plot tab with matplotlib canvas"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Visualization")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Navigation controls (initially hidden)
        self.nav_widget = QWidget()
        nav_layout = QVBoxLayout()

        from PyQt5.QtWidgets import QPushButton, QListWidget, QComboBox

        # Plot filter (above the list)
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        self.plot_filter_combo = QComboBox()
        self.plot_filter_combo.addItems(["All Plots", "Diffusion Analysis Plots", "Fit Plots"])
        self.plot_filter_combo.currentTextChanged.connect(self._on_plot_filter_changed)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.plot_filter_combo)
        filter_layout.addStretch()
        nav_layout.addLayout(filter_layout)

        # Plot list (above the plot area, limited height)
        self.plot_list = QListWidget()
        self.plot_list.setMaximumHeight(120)
        self.plot_list.currentRowChanged.connect(self._on_plot_selected)
        nav_layout.addWidget(self.plot_list)

        # Navigation buttons
        nav_buttons = QHBoxLayout()
        self.prev_plot_btn = QPushButton("◀ Previous")
        self.prev_plot_btn.clicked.connect(self._show_previous_plot)
        self.next_plot_btn = QPushButton("Next ▶")
        self.next_plot_btn.clicked.connect(self._show_next_plot)
        self.grid_view_btn = QPushButton("Grid View")
        self.grid_view_btn.clicked.connect(self._show_grid_view)

        nav_buttons.addWidget(self.prev_plot_btn)
        nav_buttons.addWidget(self.next_plot_btn)
        nav_buttons.addWidget(self.grid_view_btn)
        nav_buttons.addStretch()

        nav_layout.addLayout(nav_buttons)

        # Plot container
        plot_group = QGroupBox("Current Plot")
        plot_layout = QVBoxLayout()

        # Placeholder for matplotlib canvas
        self.plot_placeholder = QLabel("Plots will appear here after analysis")
        self.plot_placeholder.setAlignment(Qt.AlignCenter)
        self.plot_placeholder.setMinimumHeight(400)
        self.plot_placeholder.setStyleSheet("border: 2px dashed #ccc;")
        plot_layout.addWidget(self.plot_placeholder)

        # Import matplotlib if available
        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
            from matplotlib.figure import Figure

            self.figure = Figure(figsize=(8, 6))
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.toolbar = NavigationToolbar2QT(self.canvas, widget)

            # Enable context menu for canvas
            self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
            self.canvas.customContextMenuRequested.connect(self._show_plot_context_menu)

            plot_layout.addWidget(self.toolbar)
            plot_layout.addWidget(self.canvas)
            self.plot_placeholder.hide()
        except ImportError:
            # Matplotlib not available, use placeholder
            pass

        plot_group.setLayout(plot_layout)
        nav_layout.addWidget(plot_group)

        # Plot info label
        self.plot_info_label = QLabel("")
        self.plot_info_label.setWordWrap(True)
        nav_layout.addWidget(self.plot_info_label)

        self.nav_widget.setLayout(nav_layout)
        self.nav_widget.hide()  # Initially hidden

        layout.addWidget(self.nav_widget)

        widget.setLayout(layout)

        # Store current plots and all plots for filtering
        self.current_plots = {}
        self.all_plots = {}  # Store all plots before filtering
        self.current_plot_index = 0

        return widget

    def create_results_tab(self):
        """Create results tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Analysis Results")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Results table
        results_group = QGroupBox("Current Results")
        results_layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Method", "Rh (nm)", "Error (nm)", "R²", "PDI"
        ])
        # Enable sorting and better column sizing
        self.results_table.setSortingEnabled(True)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setAlternatingRowColors(True)
        # Enable context menu
        self.results_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._show_results_context_menu)
        results_layout.addWidget(self.results_table)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Details text
        details_group = QGroupBox("Detailed Results")
        details_layout = QVBoxLayout()

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMinimumHeight(200)
        self.details_text.setHtml("")  # Initialize as HTML
        details_layout.addWidget(self.details_text)

        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        # Post-filtering button section (initially hidden)
        self.postfilter_widget = QWidget()
        postfilter_layout = QHBoxLayout()
        postfilter_layout.addWidget(QLabel("<b>Post-Filter:</b>"))

        # Buttons for each method (created dynamically when methods are loaded)
        self.postfilter_buttons = {}

        postfilter_layout.addStretch()
        self.postfilter_widget.setLayout(postfilter_layout)
        self.postfilter_widget.hide()  # Initially hidden
        layout.addWidget(self.postfilter_widget)

        widget.setLayout(layout)
        return widget

    def create_comparison_tab(self):
        """Create comparison tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Method Comparison")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Comparison table
        comparison_group = QGroupBox("All Methods")
        comparison_layout = QVBoxLayout()

        self.comparison_table = QTableWidget()
        self.comparison_table.setColumnCount(7)
        self.comparison_table.setHorizontalHeaderLabels([
            "Method", "Rh (nm)", "Error (nm)", "R²", "PDI", "Residuals", "Recommendation"
        ])
        comparison_layout.addWidget(self.comparison_table)

        comparison_group.setLayout(comparison_layout)
        layout.addWidget(comparison_group)

        widget.setLayout(layout)
        return widget

    def update_data_overview(self):
        """Update data overview with loaded data"""
        if 'files' not in self.pipeline.data:
            return

        files = self.pipeline.data['files']
        num_files = len(files)
        basedata = self.pipeline.data.get('basedata', pd.DataFrame())

        # Calculate statistics
        if not basedata.empty:
            mean_temp = basedata['temperature [K]'].mean()
            std_temp = basedata['temperature [K]'].std()
            mean_visc = basedata['viscosity [cp]'].mean()
            angles = sorted(basedata['angle [°]'].unique())
            angle_list = ', '.join([f"{a:.0f}°" for a in angles])

            stats_text = f"""
<b>Files Loaded:</b> {num_files}<br>
<b>Unique Angles:</b> {len(angles)} ({angle_list})<br>
<b>Mean Temperature:</b> {mean_temp:.2f} ± {std_temp:.3f} K<br>
<b>Mean Viscosity:</b> {mean_visc:.4f} cP<br>
<b>Correlations:</b> {len(self.pipeline.data.get('correlations', {}))} datasets<br>
<b>Count Rates:</b> {len(self.pipeline.data.get('countrates', {}))} datasets<br>
<b>Status:</b> <span style="color: green;">Ready for analysis</span>
"""
        else:
            stats_text = f"""
<b>Files Loaded:</b> {num_files}<br>
<b>Status:</b> Data loaded, but no base data extracted
"""

        self.stats_label.setText(stats_text)

        # Update files table with actual data
        if not basedata.empty:
            self.files_table.setRowCount(len(basedata))

            for i, row in basedata.iterrows():
                self.files_table.setItem(i-1, 0, QTableWidgetItem(row['filename']))
                self.files_table.setItem(i-1, 1, QTableWidgetItem(f"{row['angle [°]']:.1f}"))
                self.files_table.setItem(i-1, 2, QTableWidgetItem(f"{row['temperature [K]']:.2f}"))
                self.files_table.setItem(i-1, 3, QTableWidgetItem("✓ Loaded"))

            self.files_table.resizeColumnsToContents()

    def display_results(self, results):
        """Display analysis results"""
        # This would be populated with actual results
        self.details_text.setText(f"Analysis completed:\n{results}")

    def show_step(self, step_name):
        """Show relevant tab for step"""
        step_to_tab = {
            'load_data': 0,
            'preprocess': 0,
            'cumulant_a': 2,
            'cumulant_b': 2,
            'cumulant_c': 2,
            'nnls': 2,
            'regularized': 2,
            'compare': 3
        }

        tab_index = step_to_tab.get(step_name, 0)
        self.tabs.setCurrentIndex(tab_index)

    def show_comparison(self):
        """Show comparison tab"""
        self.tabs.setCurrentIndex(3)

    def plot_data(self, x_data, y_data, xlabel, ylabel, title):
        """Plot data on matplotlib canvas"""
        if hasattr(self, 'figure'):
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(x_data, y_data, 'o-')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            self.canvas.draw()

    # ========== Cumulant Analysis Display Methods ==========

    def display_cumulant_results(self, method_name, results_df, plots_dict=None, fit_quality=None, switch_tab=True, regression_stats=None, analyzer=None):
        """
        Display cumulant analysis results in Results and Plots tabs

        Args:
            method_name: Name of the method (e.g., "Method B")
            results_df: DataFrame with final results
            plots_dict: Dictionary {filename: (fig, data)} or None
            fit_quality: Dictionary {filename: {'R2': float, ...}} or None
            switch_tab: Whether to switch to appropriate tab (default True)
            regression_stats: Dictionary with regression statistics (all methods)
            analyzer: CumulantAnalyzer instance for recomputation (optional)
        """
        # Store analyzer for Method C postfilter recomputation
        if analyzer is not None and 'Method C' in method_name:
            self.method_c_analyzer = analyzer

        # Update Results tab
        self._update_results_table(method_name, results_df, regression_stats)

        # Update Plots tab if plots are available
        if plots_dict:
            self._load_plots(method_name, plots_dict, fit_quality)
            if switch_tab:
                # Switch to Plots tab
                self.tabs.setCurrentIndex(1)
        elif switch_tab:
            # Switch to Results tab
            self.tabs.setCurrentIndex(2)

        # Add postfilter button for Methods B and C
        if 'Method B' in method_name or 'Method C' in method_name:
            self._add_postfilter_button(method_name, results_df, plots_dict, fit_quality)

    def show_results_tab(self):
        """Switch to Results tab"""
        self.tabs.setCurrentIndex(2)

    def _update_results_table(self, method_name, results_df, regression_stats=None):
        """Update results table with new results"""
        print(f"[ANALYSIS VIEW] _update_results_table called for {method_name}")
        print(f"  results_df type: {type(results_df)}")
        print(f"  results_df shape: {results_df.shape if not results_df.empty else 'empty'}")
        print(f"  results_df columns: {results_df.columns.tolist() if not results_df.empty else 'empty'}")
        print(f"  results_df:\n{results_df}")

        if results_df.empty:
            print(f"[ANALYSIS VIEW WARNING] results_df is empty for {method_name}!")
            return

        # Get existing row count
        current_rows = self.results_table.rowCount()
        print(f"[ANALYSIS VIEW] Current row count before adding: {current_rows}")
        print(f"[ANALYSIS VIEW] Adding {len(results_df)} rows for {method_name}")

        # Add rows for new results
        for i, row in results_df.iterrows():
            print(f"[ANALYSIS VIEW]   Adding row {i} at position {current_rows + i}")
            self.results_table.insertRow(current_rows + i)

            # Method - use the 'Fit' column for the full description
            fit_name = str(row.get('Fit', method_name))
            method_item = QTableWidgetItem(fit_name)

            # Set background color based on method type - adapt to system theme
            from PyQt5.QtGui import QColor
            from PyQt5.QtWidgets import QApplication
            palette = QApplication.palette()
            base_color = palette.base().color()
            is_dark = base_color.lightness() < 128

            if 'Method A' in method_name or '1st order' in fit_name or '2nd order' in fit_name or '3rd order' in fit_name:
                if is_dark:
                    method_item.setBackground(QColor(20, 40, 80))  # Dark blue
                else:
                    method_item.setBackground(QColor(230, 240, 255))  # Light blue
            elif 'Method B' in method_name or 'linear' in fit_name:
                if is_dark:
                    method_item.setBackground(QColor(20, 60, 20))  # Dark green
                else:
                    method_item.setBackground(QColor(240, 255, 230))  # Light green
            elif 'Method C' in method_name or 'iterative' in fit_name or 'non-linear' in fit_name:
                if is_dark:
                    method_item.setBackground(QColor(60, 40, 20))  # Dark orange
                else:
                    method_item.setBackground(QColor(255, 240, 230))  # Light orange

            self.results_table.setItem(current_rows + i, 0, method_item)

            # Rh
            rh_val = row.get('Rh [nm]', 0)
            if isinstance(rh_val, (list, pd.Series)):
                rh_val = rh_val[0] if len(rh_val) > 0 else 0
            rh_item = QTableWidgetItem(f"{rh_val:.2f}")
            rh_item.setData(0x0100, float(rh_val))  # Set sort role to numeric value
            self.results_table.setItem(current_rows + i, 1, rh_item)

            # Error
            error_val = row.get('Rh error [nm]', 0)
            if isinstance(error_val, (list, pd.Series)):
                error_val = error_val[0] if len(error_val) > 0 else 0
            error_item = QTableWidgetItem(f"{error_val:.2f}")
            error_item.setData(0x0100, float(error_val))
            self.results_table.setItem(current_rows + i, 2, error_item)

            # R²
            r2_val = row.get('R_squared', row.get('R-squared', 0))
            if isinstance(r2_val, (list, pd.Series)):
                r2_val = r2_val[0] if len(r2_val) > 0 else 0
            r2_item = QTableWidgetItem(f"{r2_val:.4f}")
            r2_item.setData(0x0100, float(r2_val))
            self.results_table.setItem(current_rows + i, 3, r2_item)

            # PDI
            pdi_val = row.get('PDI', 'N/A')
            if pdi_val != 'N/A' and not pd.isna(pdi_val):
                if isinstance(pdi_val, (list, pd.Series)):
                    pdi_val = pdi_val[0] if len(pdi_val) > 0 else 'N/A'
                if pdi_val != 'N/A':
                    pdi_str = f"{pdi_val:.4f}"
                    pdi_item = QTableWidgetItem(pdi_str)
                    pdi_item.setData(0x0100, float(pdi_val))
                else:
                    pdi_str = 'N/A'
                    pdi_item = QTableWidgetItem(pdi_str)
            else:
                pdi_str = 'N/A'
                pdi_item = QTableWidgetItem(pdi_str)
            self.results_table.setItem(current_rows + i, 4, pdi_item)

        new_row_count = self.results_table.rowCount()
        print(f"[ANALYSIS VIEW] Finished adding rows. New row count: {new_row_count}")
        self.results_table.resizeColumnsToContents()

        # Update details text with formatted output
        # Get current HTML content
        current_html = self.details_text.toHtml()

        # Detect dark mode for HTML styling
        from PyQt5.QtWidgets import QApplication
        palette = QApplication.palette()
        base_color = palette.base().color()
        is_dark = base_color.lightness() < 128

        # Set colors based on theme
        if is_dark:
            title_color = "#66B3FF"  # Lighter blue for dark mode
            header_bg = "#2D2D2D"  # Dark gray
            row_bg_1 = "#1E1E1E"  # Darker gray
            row_bg_2 = "#252525"  # Slightly lighter gray
            text_color = "#E0E0E0"  # Light text
        else:
            title_color = "#0066cc"  # Standard blue
            header_bg = "#e0e0e0"  # Light gray
            row_bg_1 = "#f9f9f9"  # Very light gray
            row_bg_2 = "#ffffff"  # White
            text_color = "#000000"  # Black text

        # Start building new method section
        new_section = f"<h3 style='color: {title_color}; margin-top: 15px;'>{method_name}</h3>"

        # Add main results table
        new_section += "<h4>Results Summary:</h4>"
        new_section += f"<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%; color: {text_color};'>"
        new_section += f"<tr style='background-color: {header_bg}; font-weight: bold;'>"

        # Add headers
        for col in results_df.columns:
            new_section += f"<th style='padding: 8px; text-align: left;'>{col}</th>"
        new_section += "</tr>"

        # Add data rows with alternating colors
        for idx, row in results_df.iterrows():
            bg_color = row_bg_1 if idx % 2 == 0 else row_bg_2
            new_section += f"<tr style='background-color: {bg_color};'>"
            for col in results_df.columns:
                val = row[col]
                if isinstance(val, (list, pd.Series)):
                    val = val[0] if len(val) > 0 else val
                if isinstance(val, float):
                    if pd.isna(val):
                        cell_text = "N/A"
                    else:
                        cell_text = f"{val:.4f}"
                else:
                    cell_text = str(val)
                new_section += f"<td style='padding: 8px;'>{cell_text}</td>"
            new_section += "</tr>"
        new_section += "</table>"

        # Add detailed fit statistics for Gamma vs q² linear regression
        if regression_stats and 'regression_results' in regression_stats:
            # Method A: Multiple regressions (1st, 2nd, 3rd order)
            new_section += "<h4>Detailed Fit Results: Γ vs q² Linear Regressions</h4>"
            new_section += f"<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%; color: {text_color};'>"
            new_section += f"<tr style='background-color: {header_bg}; font-weight: bold;'>"
            new_section += "<th>Order</th><th>Slope (D×10¹⁵ m²/s)</th><th>Std Error</th><th>R²</th><th>Quality</th>"
            new_section += "</tr>"

            for result in regression_stats.get('regression_results', []):
                new_section += "<tr>"
                gamma_col = result.get('gamma_col', '')
                if '1st' in gamma_col:
                    order = "1st"
                elif '2nd' in gamma_col:
                    order = "2nd"
                elif '3rd' in gamma_col:
                    order = "3rd"
                else:
                    order = gamma_col

                slope = result.get('q^2_coef', 0)
                std_err = result.get('q^2_se', 0)
                r2 = result.get('R_squared', 0)

                new_section += f"<td>{order}</td>"
                new_section += f"<td>{slope:.6f}</td>"
                new_section += f"<td>{std_err:.6f}</td>"
                new_section += f"<td>{r2:.6f}</td>"

                if r2 > 0.99:
                    quality = "✓ Excellent"
                    color = "green"
                elif r2 > 0.95:
                    quality = "○ Good"
                    color = "orange"
                else:
                    quality = "✗ Check"
                    color = "red"
                new_section += f"<td style='color: {color}; font-weight: bold;'>{quality}</td>"
                new_section += "</tr>"

            new_section += "</table>"

        elif regression_stats and 'summary' in regression_stats:
            # Methods B/C: Extract key statistics and display in formatted table
            new_section += "<h4>Detailed Fit Results: Γ vs q² Linear Regression</h4>"

            # Model Statistics Table
            model_header_bg = "#5080A0" if is_dark else "#d0e0f0"
            new_section += f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse: collapse; width: 100%; margin-bottom: 15px; color: {text_color};'>"
            new_section += f"<tr style='background-color: {model_header_bg}; font-weight: bold;'><th colspan='2'>Model Statistics</th></tr>"
            new_section += f"<tr><td style='width: 40%;'><b>R-squared</b></td><td>{regression_stats['rsquared']:.6f}</td></tr>"
            new_section += f"<tr><td><b>Adj. R-squared</b></td><td>{regression_stats['rsquared_adj']:.6f}</td></tr>"
            new_section += f"<tr><td><b>F-statistic</b></td><td>{regression_stats['fvalue']:.3e}</td></tr>"
            new_section += f"<tr><td><b>Prob (F-statistic)</b></td><td>{regression_stats['f_pvalue']:.3e}</td></tr>"
            new_section += f"<tr><td><b>AIC</b></td><td>{regression_stats['aic']:.2f}</td></tr>"
            new_section += f"<tr><td><b>BIC</b></td><td>{regression_stats['bic']:.2f}</td></tr>"
            new_section += "</table>"

            # Coefficients Table
            new_section += f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse: collapse; width: 100%; color: {text_color};'>"
            new_section += f"<tr style='background-color: {model_header_bg}; font-weight: bold;'>"
            new_section += "<th>Coefficient</th><th>Value</th><th>Std Error</th></tr>"

            params = regression_stats.get('params', {})
            if 'const' in params and 'x1' in params:
                # We have intercept and slope - get stderr if available
                const_val = params['const']
                slope_val = params['x1']
                stderr_slope = regression_stats.get('stderr_slope', '-')
                stderr_intercept = regression_stats.get('stderr_intercept', '-')

                if stderr_intercept != '-':
                    new_section += f"<tr><td><b>Intercept</b></td><td>{const_val:.4e}</td><td>{stderr_intercept:.4e}</td></tr>"
                else:
                    new_section += f"<tr><td><b>Intercept</b></td><td>{const_val:.4e}</td><td>-</td></tr>"

                if stderr_slope != '-':
                    new_section += f"<tr><td><b>Slope (q²)</b></td><td>{slope_val:.4e}</td><td>{stderr_slope:.4e}</td></tr>"
                else:
                    new_section += f"<tr><td><b>Slope (q²)</b></td><td>{slope_val:.4e}</td><td>-</td></tr>"

            new_section += "</table>"

            # Add Fit Quality Assessment Table with Optimal Ranges
            new_section += "<h4>Fit Quality Assessment</h4>"
            new_section += f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse: collapse; width: 100%; margin-bottom: 15px; color: {text_color};'>"
            new_section += f"<tr style='background-color: {model_header_bg}; font-weight: bold;'>"
            new_section += "<th>Metric</th><th>Value</th><th>Optimal Range</th><th>Assessment</th></tr>"

            # Helper function to assess fit quality
            def assess_metric(value, good_min=None, good_max=None, acceptable_min=None, acceptable_max=None):
                """Assess a metric value against optimal ranges"""
                if good_min is not None and good_max is not None:
                    if good_min <= value <= good_max:
                        return ("✓ Excellent", "green")
                elif good_min is not None:
                    if value >= good_min:
                        return ("✓ Excellent", "green")
                elif good_max is not None:
                    if value <= good_max:
                        return ("✓ Excellent", "green")

                if acceptable_min is not None and acceptable_max is not None:
                    if acceptable_min <= value <= acceptable_max:
                        return ("○ Acceptable", "orange")
                elif acceptable_min is not None:
                    if value >= acceptable_min:
                        return ("○ Acceptable", "orange")
                elif acceptable_max is not None:
                    if value <= acceptable_max:
                        return ("○ Acceptable", "orange")

                return ("✗ Poor", "red")

            # R²
            r2 = regression_stats['rsquared']
            r2_assessment, r2_color = assess_metric(r2, good_min=0.95, acceptable_min=0.90)
            new_section += f"<tr><td><b>R²</b></td><td>{r2:.4f}</td>"
            new_section += f"<td>> 0.95 (Excellent)<br>0.90-0.95 (Acceptable)<br>< 0.90 (Poor)</td>"
            new_section += f"<td style='color: {r2_color}; font-weight: bold;'>{r2_assessment}</td></tr>"

            # Adjusted R²
            r2_adj = regression_stats['rsquared_adj']
            r2_adj_assessment, r2_adj_color = assess_metric(r2_adj, good_min=0.95, acceptable_min=0.90)
            new_section += f"<tr><td><b>Adjusted R²</b></td><td>{r2_adj:.4f}</td>"
            new_section += f"<td>> 0.95 (Excellent)<br>0.90-0.95 (Acceptable)<br>< 0.90 (Poor)</td>"
            new_section += f"<td style='color: {r2_adj_color}; font-weight: bold;'>{r2_adj_assessment}</td></tr>"

            # F-statistic
            fvalue = regression_stats['fvalue']
            f_assessment, f_color = assess_metric(fvalue, good_min=100, acceptable_min=10)
            new_section += f"<tr><td><b>F-statistic</b></td><td>{fvalue:.2f}</td>"
            new_section += f"<td>> 100 (Excellent)<br>10-100 (Acceptable)<br>< 10 (Poor)</td>"
            new_section += f"<td style='color: {f_color}; font-weight: bold;'>{f_assessment}</td></tr>"

            # p-value (F-test)
            f_pvalue = regression_stats['f_pvalue']
            # For p-value, lower is better
            if f_pvalue < 0.001:
                p_assessment, p_color = ("✓ Excellent", "green")
            elif f_pvalue < 0.05:
                p_assessment, p_color = ("○ Acceptable", "orange")
            else:
                p_assessment, p_color = ("✗ Poor", "red")
            new_section += f"<tr><td><b>p-value (F-test)</b></td><td>{f_pvalue:.3e}</td>"
            new_section += f"<td>< 0.001 (Excellent)<br>0.001-0.05 (Acceptable)<br>> 0.05 (Poor)</td>"
            new_section += f"<td style='color: {p_color}; font-weight: bold;'>{p_assessment}</td></tr>"

            # Overall Assessment
            new_section += "</table>"

            # Overall quality message
            overall_good = sum([
                r2 > 0.95,
                r2_adj > 0.95,
                fvalue > 100,
                f_pvalue < 0.001
            ])

            if overall_good >= 3:
                quality_msg = "<p style='color: green; font-weight: bold; font-size: 14pt;'>✓ Overall Fit Quality: EXCELLENT</p>"
            elif overall_good >= 2:
                quality_msg = "<p style='color: orange; font-weight: bold; font-size: 14pt;'>○ Overall Fit Quality: ACCEPTABLE</p>"
            else:
                quality_msg = "<p style='color: red; font-weight: bold; font-size: 14pt;'>⚠ Overall Fit Quality: POOR - Consider adjusting fit parameters</p>"
            new_section += quality_msg

            new_section += "<p style='font-size: 9pt; color: gray;'><i>Note: AIC and BIC are informational. Lower values indicate better fit when comparing models, but absolute values don't have universal thresholds.</i></p>"

        # Append to existing HTML or create new
        if current_html and "<body" in current_html:
            # Extract body content and append
            import re
            body_match = re.search(r'<body[^>]*>(.*)</body>', current_html, re.DOTALL)
            if body_match:
                body_content = body_match.group(1)
                new_html = current_html.replace(body_match.group(1), body_content + new_section)
                self.details_text.setHtml(new_html)
            else:
                self.details_text.setHtml(current_html + new_section)
        else:
            self.details_text.setHtml(new_section)

    def _load_plots(self, method_name, plots_dict, fit_quality):
        """Load plots into the plotting system"""
        print(f"[ANALYSIS VIEW] Loading {len(plots_dict)} plots for {method_name}")

        # Add to existing plots instead of replacing
        if not hasattr(self, 'current_plots'):
            self.current_plots = {}
        if not hasattr(self, 'current_fit_quality'):
            self.current_fit_quality = {}

        # Merge new plots with existing ones
        self.current_plots.update(plots_dict)
        # Also update all_plots to include new plots
        if not hasattr(self, 'all_plots'):
            self.all_plots = {}
        self.all_plots.update(plots_dict)

        if fit_quality:
            self.current_fit_quality.update(fit_quality)

        # Add separator if there are already plots in the list
        if self.plot_list.count() > 0:
            separator = QListWidgetItem(f"─── {method_name} ───")
            separator.setFlags(separator.flags() & ~Qt.ItemIsSelectable)
            from PyQt5.QtGui import QFont
            font = QFont()
            font.setBold(True)
            separator.setFont(font)
            self.plot_list.addItem(separator)

        # Get starting index for numbering
        start_index = self.plot_list.count()

        # Populate plot list (append, don't clear)
        filenames = list(plots_dict.keys())

        for i, filename in enumerate(filenames):
            # Add quality indicator
            quality_str = ""
            if filename in (fit_quality or {}):
                r2 = fit_quality[filename].get('R2', 0)
                quality_str = f" (R²={r2:.3f})"

            item = QListWidgetItem(f"{start_index + i + 1}. {filename}{quality_str}")
            # Store filename in item data for retrieval
            item.setData(Qt.UserRole, filename)
            self.plot_list.addItem(item)
            print(f"[ANALYSIS VIEW] Added plot {start_index + i + 1}: {filename}{quality_str}")

        # Show navigation widget
        self.nav_widget.show()
        self.plot_placeholder.hide()

        # Show first real plot (skip separators)
        if filenames and self.plot_list.count() > 0:
            # Find first non-separator item
            for row in range(self.plot_list.count()):
                item = self.plot_list.item(row)
                if item and item.data(Qt.UserRole):  # Has filename data
                    self.plot_list.setCurrentRow(row)
                    self._show_plot_by_item(item)
                    print(f"[ANALYSIS VIEW] Displayed first plot")
                    break
        else:
            print(f"[ANALYSIS VIEW WARNING] No plots to display!")

    def _on_plot_selected(self, index):
        """Handle plot selection from list"""
        if index >= 0:
            item = self.plot_list.item(index)
            if item and item.data(Qt.UserRole):  # Only show if it's not a separator
                self._show_plot_by_item(item)

    def _show_plot_by_item(self, item):
        """Show plot based on QListWidgetItem"""
        filename = item.data(Qt.UserRole)
        if not filename:
            return

        print(f"[ANALYSIS VIEW] Showing plot for {filename}")

        # Get plot data
        if filename in self.current_plots:
            fig, data = self.current_plots[filename]
            print(f"[ANALYSIS VIEW] Retrieved plot: fig={fig is not None}, data keys={list(data.keys()) if isinstance(data, dict) else 'not dict'}")

            # Clear and redraw
            self.figure.clear()

            if fig is not None:
                # Copy the plot
                source_axes = fig.get_axes()
                print(f"[ANALYSIS VIEW] Copying {len(source_axes)} subplots")

                for ax_idx, source_ax in enumerate(source_axes):
                    # Create subplot in same position
                    if len(source_axes) == 1:
                        ax = self.figure.add_subplot(111)
                    elif len(source_axes) == 2:
                        ax = self.figure.add_subplot(1, 2, ax_idx + 1)
                    elif len(source_axes) == 4:
                        ax = self.figure.add_subplot(2, 2, ax_idx + 1)
                    else:
                        ax = self.figure.add_subplot(111)

                    # Copy scatter plots (collections) - MUST come before lines for proper z-order
                    for collection in source_ax.collections:
                        offsets = collection.get_offsets()
                        if len(offsets) > 0:
                            ax.scatter(offsets[:, 0], offsets[:, 1],
                                     label=collection.get_label(),
                                     c=collection.get_facecolors(),
                                     s=collection.get_sizes(),
                                     alpha=collection.get_alpha(),
                                     edgecolors=collection.get_edgecolors(),
                                     linewidths=collection.get_linewidths(),
                                     marker=collection.get_paths()[0] if collection.get_paths() else 'o')

                    # Copy lines (fit lines, etc.)
                    for line in source_ax.get_lines():
                        ax.plot(line.get_xdata(), line.get_ydata(),
                               label=line.get_label(),
                               color=line.get_color(),
                               linestyle=line.get_linestyle(),
                               marker=line.get_marker(),
                               markersize=line.get_markersize() if line.get_marker() else 1,
                               alpha=line.get_alpha())

                    # Copy labels and title
                    ax.set_xlabel(source_ax.get_xlabel())
                    ax.set_ylabel(source_ax.get_ylabel())
                    ax.set_title(source_ax.get_title())

                    # Copy scale
                    if source_ax.get_xscale() == 'log':
                        ax.set_xscale('log')
                    if source_ax.get_yscale() == 'log':
                        ax.set_yscale('log')

                    # Copy legend if exists
                    if source_ax.get_legend():
                        ax.legend()

                    # Copy grid (matplotlib 3.9+ compatible)
                    # Check if grid is visible by examining gridlines
                    x_gridlines = source_ax.xaxis.get_gridlines()
                    grid_visible = any(line.get_visible() for line in x_gridlines) if x_gridlines else False
                    ax.grid(grid_visible)

            self.figure.tight_layout()
            self.canvas.draw()
            print(f"[ANALYSIS VIEW] Plot drawn successfully")

            # Update info label
            total_plots = len(self.current_plots)
            current_num = self.plot_list.currentRow() + 1
            info_text = f"<b>Plot {current_num} of {self.plot_list.count()}</b>: {filename}<br>"
            if filename in self.current_fit_quality:
                quality = self.current_fit_quality[filename]
                info_text += f"<b>Fit Quality:</b> R² = {quality.get('R2', 'N/A')}"
                if 'residuals' in quality:
                    info_text += f", Residuals = {quality.get('residuals', 'N/A')}"
            self.plot_info_label.setText(info_text)

        # Update button states based on list position
        current_row = self.plot_list.currentRow()
        self._update_nav_buttons(current_row)

    def _update_nav_buttons(self, current_row):
        """Update navigation button states"""
        # Find previous non-separator
        has_prev = False
        for row in range(current_row - 1, -1, -1):
            item = self.plot_list.item(row)
            if item and item.data(Qt.UserRole):
                has_prev = True
                break

        # Find next non-separator
        has_next = False
        for row in range(current_row + 1, self.plot_list.count()):
            item = self.plot_list.item(row)
            if item and item.data(Qt.UserRole):
                has_next = True
                break

        self.prev_plot_btn.setEnabled(has_prev)
        self.next_plot_btn.setEnabled(has_next)

    def _show_previous_plot(self):
        """Show previous plot"""
        current_row = self.plot_list.currentRow()
        # Find previous non-separator item
        for row in range(current_row - 1, -1, -1):
            item = self.plot_list.item(row)
            if item and item.data(Qt.UserRole):
                self.plot_list.setCurrentRow(row)
                self._show_plot_by_item(item)
                break

    def _show_next_plot(self):
        """Show next plot"""
        current_row = self.plot_list.currentRow()
        # Find next non-separator item
        for row in range(current_row + 1, self.plot_list.count()):
            item = self.plot_list.item(row)
            if item and item.data(Qt.UserRole):
                self.plot_list.setCurrentRow(row)
                self._show_plot_by_item(item)
                break

    def _on_plot_filter_changed(self, filter_text):
        """Handle plot filter change"""
        # Save current all plots if not already saved
        if not self.all_plots and self.current_plots:
            self.all_plots = self.current_plots.copy()

        # If we have no plots saved, nothing to filter
        if not self.all_plots:
            return

        # Apply filter
        if filter_text == "All Plots":
            # Show all plots
            self.current_plots = self.all_plots.copy()
        elif filter_text == "Diffusion Analysis Plots":
            # Show only summary/diffusion plots (keys ending with "Summary")
            self.current_plots = {k: v for k, v in self.all_plots.items()
                                 if "Summary" in k or "Diffusion" in k}
        elif filter_text == "Fit Plots":
            # Show only fit plots (keys NOT ending with "Summary")
            self.current_plots = {k: v for k, v in self.all_plots.items()
                                 if "Summary" not in k and "Diffusion" not in k}

        # Rebuild plot list
        self._rebuild_plot_list()

    def _rebuild_plot_list(self):
        """Rebuild the plot list widget based on current_plots"""
        self.plot_list.clear()

        if not self.current_plots:
            return

        # Add plots to list
        for i, (filename, (fig, data)) in enumerate(self.current_plots.items()):
            # Check if we have fit quality info
            quality_str = ""
            if hasattr(self, 'current_fit_quality') and filename in self.current_fit_quality:
                r2 = self.current_fit_quality[filename].get('R2', 0)
                quality_str = f" (R²={r2:.3f})"

            item = QListWidgetItem(f"{i + 1}. {filename}{quality_str}")
            item.setData(Qt.UserRole, filename)
            self.plot_list.addItem(item)

        # Show first plot
        if self.plot_list.count() > 0:
            self.plot_list.setCurrentRow(0)
            item = self.plot_list.item(0)
            if item and item.data(Qt.UserRole):
                self._show_plot_by_item(item)

    def _show_grid_view(self):
        """Show all plots in grid view"""
        from PyQt5.QtWidgets import QDialog, QScrollArea, QGridLayout
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

        if not self.current_plots:
            return

        # Create grid dialog
        grid_dialog = QDialog(self)
        grid_dialog.setWindowTitle(f"All Plots - {self.current_method_name}")
        grid_dialog.setMinimumSize(1200, 800)

        layout = QVBoxLayout()

        # Info
        info_label = QLabel(
            f"<b>Showing all {len(self.current_plots)} datasets</b>"
        )
        layout.addWidget(info_label)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        # Grid widget
        from PyQt5.QtWidgets import QWidget
        grid_widget = QWidget()
        grid_layout = QGridLayout()

        # Calculate grid dimensions
        num_plots = len(self.current_plots)
        cols = min(4, num_plots)
        rows = (num_plots + cols - 1) // cols

        # Create grid of plots
        for i, (filename, (fig, _)) in enumerate(self.current_plots.items()):
            row = i // cols
            col = i % cols

            if fig is not None:
                # Create small canvas
                small_fig = plt.Figure(figsize=(3, 2.5))
                small_canvas = FigureCanvasQTAgg(small_fig)

                # Copy plot to small figure
                source_axes = fig.get_axes()
                if source_axes:
                    ax = small_fig.add_subplot(111)
                    source_ax = source_axes[0]

                    # Copy lines
                    for line in source_ax.get_lines():
                        ax.plot(line.get_xdata(), line.get_ydata(),
                               color=line.get_color(),
                               linestyle=line.get_linestyle(),
                               linewidth=0.5)

                    ax.set_title(f"{i+1}. {filename[:20]}...", fontsize=8)
                    ax.tick_params(labelsize=6)
                    if source_ax.get_xscale() == 'log':
                        ax.set_xscale('log')
                    if source_ax.get_yscale() == 'log':
                        ax.set_yscale('log')

                small_fig.tight_layout()

                # Add to grid
                grid_layout.addWidget(small_canvas, row, col)

        grid_widget.setLayout(grid_layout)
        scroll.setWidget(grid_widget)
        layout.addWidget(scroll)

        # Close button
        from PyQt5.QtWidgets import QPushButton
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(grid_dialog.accept)
        layout.addWidget(close_btn)

        grid_dialog.setLayout(layout)
        grid_dialog.exec_()

    def _add_postfilter_button(self, method_name, results_df, plots_dict=None, fit_quality=None):
        """Add a postfilter button for the given method"""
        # Check if button already exists for this method
        if method_name in self.postfilter_buttons:
            return

        # Create button
        btn = QPushButton(f"🔧 Post-Filter {method_name}")
        btn.setToolTip(f"Remove bad fits and recalculate results for {method_name}")

        # Different behavior for Method C vs Method B
        if 'Method C' in method_name:
            btn.clicked.connect(lambda: self._open_postfilter_dialog_method_c(
                method_name, results_df, plots_dict, fit_quality))
        else:
            btn.clicked.connect(lambda: self._open_postfilter_dialog(method_name, results_df))

        # Add to layout
        layout = self.postfilter_widget.layout()
        # Insert before the stretch (which is the last item)
        layout.insertWidget(layout.count() - 1, btn)

        # Store button reference
        self.postfilter_buttons[method_name] = btn

        # Show the postfilter widget
        self.postfilter_widget.show()

    def _open_postfilter_dialog(self, method_name, results_df):
        """Open the postfilter dialog for the given method"""
        from gui.dialogs.postfilter_dialog import show_postfilter_dialog

        # Show postfilter dialog
        new_results_df = show_postfilter_dialog(results_df, method_name, self)

        if new_results_df is not None:
            # Update the results display with filtered results
            # Note: This will update the table, but won't rerun the full analysis
            # The user may want to rerun the analysis after seeing filtered results
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Post-Filter Applied",
                f"Filtered results preview shown for {method_name}.\n\n"
                "Note: This is a preview only. To fully reprocess the data,\n"
                "please re-run the cumulant analysis with the filtered data."
            )

    def _open_postfilter_dialog_method_c(self, method_name, results_df, plots_dict, fit_quality):
        """Open the Method C post-fit filtering dialog with visual inspection"""
        from gui.dialogs import MethodCPostFitDialog
        from PyQt5.QtWidgets import QMessageBox

        # Check if analyzer is available
        if not hasattr(self, 'method_c_analyzer') or self.method_c_analyzer is None:
            QMessageBox.warning(
                self,
                "Analyzer Not Available",
                "Cannot perform post-fit filtering: analyzer instance not available.\n"
                "Please re-run Method C analysis."
            )
            return

        # Check if we have plots
        if not plots_dict or not fit_quality:
            QMessageBox.warning(
                self,
                "No Fit Data",
                "Cannot perform post-fit filtering: no fit plots or quality data available."
            )
            return

        # Open the visual inspection dialog
        dialog = MethodCPostFitDialog(
            plots_dict=plots_dict,
            fit_quality=fit_quality,
            results_df=results_df,
            analyzer=self.method_c_analyzer,
            parent=self
        )

        if dialog.exec_():
            # User confirmed - get the filtered results
            filter_results = dialog.get_filtered_results()

            if filter_results:
                included_files = filter_results['included_files']
                excluded_files = filter_results['excluded_files']

                print(f"[POST-FIT FILTER] Recomputing with {len(included_files)} files")
                print(f"[POST-FIT FILTER] Excluded {len(excluded_files)} files: {excluded_files}")

                try:
                    # Get q_range if it was used in original analysis
                    q_range = getattr(self.method_c_analyzer, 'last_q_range', None)

                    # Recompute diffusion analysis with only selected files
                    recomputed_results = self.method_c_analyzer.recompute_method_c_diffusion(
                        included_files=included_files,
                        q_range=q_range
                    )

                    # Update the results table with the new filtered results
                    self._update_results_table(
                        f"Method C (filtered, N={len(included_files)})",
                        recomputed_results,
                        regression_stats=None
                    )

                    # Show success message
                    QMessageBox.information(
                        self,
                        "Post-Fit Filtering Complete",
                        f"Method C diffusion analysis recomputed with {len(included_files)} selected fits.\n\n"
                        f"Results added as: 'Method C (filtered, N={len(included_files)})'\n\n"
                        f"Excluded files ({len(excluded_files)}): {', '.join(excluded_files[:5])}"
                        f"{'...' if len(excluded_files) > 5 else ''}"
                    )

                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Recomputation Failed",
                        f"Failed to recompute diffusion analysis:\n{str(e)}"
                    )
                    print(f"[ERROR] Recomputation failed: {e}")
                    import traceback
                    traceback.print_exc()

    def clear_results(self):
        """Clear all results"""
        self.results_table.setRowCount(0)
        self.details_text.clear()
        self.current_plots = {}
        self.all_plots = {}
        self.plot_list.clear()
        self.nav_widget.hide()
        self.plot_placeholder.show()
        # Reset filter to "All Plots"
        if hasattr(self, 'plot_filter_combo'):
            self.plot_filter_combo.setCurrentText("All Plots")
        # Clear postfilter buttons
        if hasattr(self, 'postfilter_buttons'):
            for btn in self.postfilter_buttons.values():
                btn.deleteLater()
            self.postfilter_buttons.clear()
        if hasattr(self, 'postfilter_widget'):
            self.postfilter_widget.hide()

    def _show_results_context_menu(self, position):
        """Show context menu for results table"""
        from PyQt5.QtWidgets import QMenu, QAction, QApplication

        menu = QMenu()

        # Copy actions
        copy_cell_action = QAction("📋 Copy Cell", self)
        copy_cell_action.triggered.connect(self._copy_cell)
        menu.addAction(copy_cell_action)

        copy_row_action = QAction("📋 Copy Row", self)
        copy_row_action.triggered.connect(self._copy_row)
        menu.addAction(copy_row_action)

        copy_column_action = QAction("📋 Copy Column", self)
        copy_column_action.triggered.connect(self._copy_column)
        menu.addAction(copy_column_action)

        menu.addSeparator()

        copy_table_action = QAction("📋 Copy Entire Table", self)
        copy_table_action.triggered.connect(self._copy_table)
        menu.addAction(copy_table_action)

        # Show menu
        menu.exec_(self.results_table.mapToGlobal(position))

    def _copy_cell(self):
        """Copy selected cell to clipboard"""
        from PyQt5.QtWidgets import QApplication

        current_item = self.results_table.currentItem()
        if current_item:
            QApplication.clipboard().setText(current_item.text())

    def _copy_row(self):
        """Copy selected row to clipboard"""
        from PyQt5.QtWidgets import QApplication

        current_row = self.results_table.currentRow()
        if current_row >= 0:
            row_data = []
            for col in range(self.results_table.columnCount()):
                item = self.results_table.item(current_row, col)
                if item:
                    row_data.append(item.text())
                else:
                    row_data.append("")
            QApplication.clipboard().setText("\t".join(row_data))

    def _copy_column(self):
        """Copy selected column to clipboard"""
        from PyQt5.QtWidgets import QApplication

        current_col = self.results_table.currentColumn()
        if current_col >= 0:
            # Include header
            header_item = self.results_table.horizontalHeaderItem(current_col)
            col_data = [header_item.text() if header_item else ""]

            # Add all cells in column
            for row in range(self.results_table.rowCount()):
                item = self.results_table.item(row, current_col)
                if item:
                    col_data.append(item.text())
                else:
                    col_data.append("")

            QApplication.clipboard().setText("\n".join(col_data))

    def _copy_table(self):
        """Copy entire table to clipboard"""
        from PyQt5.QtWidgets import QApplication

        # Include headers
        headers = []
        for col in range(self.results_table.columnCount()):
            header_item = self.results_table.horizontalHeaderItem(col)
            headers.append(header_item.text() if header_item else "")

        table_data = ["\t".join(headers)]

        # Add all rows
        for row in range(self.results_table.rowCount()):
            row_data = []
            for col in range(self.results_table.columnCount()):
                item = self.results_table.item(row, col)
                if item:
                    row_data.append(item.text())
                else:
                    row_data.append("")
            table_data.append("\t".join(row_data))

        QApplication.clipboard().setText("\n".join(table_data))

    def _show_plot_context_menu(self, position):
        """Show context menu for plot canvas"""
        from PyQt5.QtWidgets import QMenu, QAction

        menu = QMenu()

        # Legend submenu
        legend_menu = menu.addMenu("📊 Legend")

        toggle_legend_action = QAction("Toggle On/Off", self)
        toggle_legend_action.triggered.connect(self._toggle_legend)
        legend_menu.addAction(toggle_legend_action)

        legend_menu.addSeparator()

        positions = [
            ("Upper Right", "upper right"),
            ("Upper Left", "upper left"),
            ("Lower Right", "lower right"),
            ("Lower Left", "lower left"),
            ("Center", "center"),
            ("Best (auto)", "best")
        ]

        for label, pos in positions:
            action = QAction(label, self)
            action.triggered.connect(lambda checked, p=pos: self._set_legend_position(p))
            legend_menu.addAction(action)

        # Grid submenu
        grid_menu = menu.addMenu("⚏ Grid")

        toggle_grid_action = QAction("Toggle On/Off", self)
        toggle_grid_action.triggered.connect(self._toggle_grid)
        grid_menu.addAction(toggle_grid_action)

        grid_menu.addSeparator()

        major_grid_action = QAction("Major Grid", self)
        major_grid_action.triggered.connect(lambda: self._set_grid('major'))
        grid_menu.addAction(major_grid_action)

        minor_grid_action = QAction("Minor Grid", self)
        minor_grid_action.triggered.connect(lambda: self._set_grid('minor'))
        grid_menu.addAction(minor_grid_action)

        both_grid_action = QAction("Both Grids", self)
        both_grid_action.triggered.connect(lambda: self._set_grid('both'))
        grid_menu.addAction(both_grid_action)

        # Scale submenu
        scale_menu = menu.addMenu("📐 Scale")

        scales = [
            ("Linear-Linear", "linear", "linear"),
            ("Log-Linear", "log", "linear"),
            ("Linear-Log", "linear", "log"),
            ("Log-Log", "log", "log")
        ]

        for label, xscale, yscale in scales:
            action = QAction(label, self)
            action.triggered.connect(lambda checked, xs=xscale, ys=yscale: self._set_scale(xs, ys))
            scale_menu.addAction(action)

        # Color scheme submenu
        color_menu = menu.addMenu("🎨 Color Scheme")

        schemes = ["default", "viridis", "plasma", "inferno", "magma", "cividis"]

        for scheme in schemes:
            action = QAction(scheme.capitalize(), self)
            action.triggered.connect(lambda checked, s=scheme: self._set_color_scheme(s))
            color_menu.addAction(action)

        menu.addSeparator()

        # Reset action
        reset_action = QAction("🔄 Reset Plot Style", self)
        reset_action.triggered.connect(self._reset_plot_style)
        menu.addAction(reset_action)

        # Show menu
        menu.exec_(self.canvas.mapToGlobal(position))

    def _toggle_legend(self):
        """Toggle legend on/off"""
        if not self.figure.axes:
            return

        for ax in self.figure.axes:
            legend = ax.get_legend()
            if legend:
                legend.set_visible(not legend.get_visible())
            else:
                ax.legend()

        self.canvas.draw()

    def _set_legend_position(self, position):
        """Set legend position"""
        if not self.figure.axes:
            return

        for ax in self.figure.axes:
            legend = ax.get_legend()
            if legend:
                ax.legend(loc=position)

        self.canvas.draw()

    def _toggle_grid(self):
        """Toggle grid on/off"""
        if not self.figure.axes:
            return

        for ax in self.figure.axes:
            ax.grid(not ax.xaxis._gridOnMajor)

        self.canvas.draw()

    def _set_grid(self, which='major'):
        """Set grid type"""
        if not self.figure.axes:
            return

        for ax in self.figure.axes:
            ax.grid(True, which=which, alpha=0.3)

        self.canvas.draw()

    def _set_scale(self, xscale, yscale):
        """Set axis scales"""
        if not self.figure.axes:
            return

        for ax in self.figure.axes:
            try:
                ax.set_xscale(xscale)
                ax.set_yscale(yscale)
            except Exception as e:
                print(f"Could not set scale: {e}")

        self.canvas.draw()

    def _set_color_scheme(self, scheme):
        """Set color scheme for plots"""
        if not self.figure.axes:
            return

        try:
            import matplotlib.pyplot as plt
            if scheme != "default":
                plt.style.use(scheme)
            else:
                plt.style.use('default')

            # Redraw current plot
            current_item = self.plot_list.currentItem()
            if current_item:
                self._show_plot_by_item(current_item)

        except Exception as e:
            print(f"Could not set color scheme: {e}")

    def _reset_plot_style(self):
        """Reset plot style to default"""
        if not self.figure.axes:
            return

        # Reset to default matplotlib style
        try:
            import matplotlib.pyplot as plt
            plt.style.use('default')

            # Redraw current plot
            current_item = self.plot_list.currentItem()
            if current_item:
                self._show_plot_by_item(current_item)

        except Exception as e:
            print(f"Could not reset plot style: {e}")
