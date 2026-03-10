"""
Analysis View Widget
Main central panel for displaying data, plots, and results
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QTableWidget, QTableWidgetItem, QLabel, QTextEdit,
                             QGroupBox, QScrollArea, QListWidget, QListWidgetItem,
                             QPushButton, QSplitter, QComboBox)
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

        method_label = QLabel("Method:")
        self.plot_method_combo = QComboBox()
        self.plot_method_combo.addItems(["All Methods", "Method A", "Method B", "Method C", "Method D", "NNLS"])
        self.plot_method_combo.currentTextChanged.connect(self._on_method_filter_changed)
        filter_layout.addWidget(method_label)
        filter_layout.addWidget(self.plot_method_combo)

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

        nav_buttons.addWidget(self.prev_plot_btn)
        nav_buttons.addWidget(self.next_plot_btn)
        nav_buttons.addStretch()

        # ScatterForge export button (only visible when ScatterForge is available)
        from ade_dls.gui.export.scatterforge_bridge import is_available as _sf_available
        if _sf_available():
            self.sf_gamma_btn = QPushButton("📤 Γ vs q² → ScatterForge")
            self.sf_gamma_btn.setToolTip(
                "Γ vs q² Datensätze (Messpunkte + Fit) in ScatterForge-Plot öffnen"
            )
            self.sf_gamma_btn.clicked.connect(
                lambda: self._on_send_to_scatterforge('gamma')
            )
            self.sf_diffusion_btn = QPushButton("📤 D vs q² → ScatterForge")
            self.sf_diffusion_btn.setToolTip(
                "Diffusionskoeffizient D = Γ/q² je Messwinkel in ScatterForge-Plot öffnen"
            )
            self.sf_diffusion_btn.clicked.connect(
                lambda: self._on_send_to_scatterforge('diffusion')
            )
            nav_buttons.addWidget(self.sf_gamma_btn)
            nav_buttons.addWidget(self.sf_diffusion_btn)

        nav_layout.addLayout(nav_buttons)

        # Analyzer reference (set by main_window after analysis completes)
        self.cumulant_analyzer = None

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
        self.all_plot_methods = {}  # filename → method_name tag
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
        results_group = QGroupBox("Results Overview")
        results_layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            "Method", "Rh (nm)", "Error (nm)", "D (m²/s)", "D err (m²/s)", "R²", "PDI", "Residuals"
        ])
        # Enable sorting and better column sizing
        self.results_table.setSortingEnabled(True)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setAlternatingRowColors(True)
        # Enable context menu
        self.results_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.results_table.customContextMenuRequested.connect(self._show_results_context_menu)
        self.results_table.currentCellChanged.connect(
            lambda cur_row, cur_col, prev_row, prev_col: self._on_result_row_selected(cur_row)
        )
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

        # Post-Fit Refinement button section (initially hidden)
        self.refinement_widget = QWidget()
        refinement_layout = QHBoxLayout()

        # Label
        refine_label = QLabel("Post-Fit Refinement:")
        refine_label.setStyleSheet("font-weight: bold;")
        refinement_layout.addWidget(refine_label)

        # Global refinement button
        self.refinement_btn = QPushButton("⚙️ Open Refinement")
        self.refinement_btn.setToolTip(
            "Adjust q² range and exclude fits after analysis\n"
            "(Available after running analysis)"
        )
        self.refinement_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #BDBDBD;
            }
        """)
        self.refinement_btn.setEnabled(False)  # Enabled when a row is selected
        self.refinement_btn.clicked.connect(self._open_selected_refinement)
        refinement_layout.addWidget(self.refinement_btn)

        refinement_layout.addStretch()
        self.refinement_widget.setLayout(refinement_layout)
        self.refinement_widget.hide()  # Initially hidden
        layout.addWidget(self.refinement_widget)

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

    def display_cumulant_results(self, method_name, results_df, plots_dict=None, fit_quality=None, switch_tab=True, regression_stats=None, analyzer=None, replace_existing=False):
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
            replace_existing: If True, replace existing plots for this method (for post-refinement)
        """
        # Store analyzer for all methods (used by post-fit refinement)
        if analyzer is not None:
            # Determine analyzer type and store appropriately
            if 'NNLS' in method_name or 'Regularized' in method_name:
                self.laplace_analyzer = analyzer
            else:
                self.cumulant_analyzer = analyzer

                # Also keep Method C specific analyzer for backwards compatibility
                if 'Method C' in method_name:
                    self.method_c_analyzer = analyzer

        # Update Results tab
        self._update_results_table(method_name, results_df, regression_stats)

        # Update Plots tab if plots are available
        if plots_dict:
            self._load_plots(method_name, plots_dict, fit_quality, replace_method=replace_existing)
            if switch_tab:
                # Switch to Plots tab
                self.tabs.setCurrentIndex(1)
        elif switch_tab:
            # Switch to Results tab
            self.tabs.setCurrentIndex(2)

        # Show post-fit refinement button if any analyzer is available
        if (hasattr(self, 'cumulant_analyzer') and self.cumulant_analyzer is not None) or \
           (hasattr(self, 'laplace_analyzer') and self.laplace_analyzer is not None):
            self.refinement_widget.show()

    def show_results_tab(self):
        """Switch to Results tab"""
        self.tabs.setCurrentIndex(2)

    def _on_result_row_selected(self, row):
        """Show detailed HTML for the selected result row"""
        if row < 0:
            return
        item = self.results_table.item(row, 0)
        if item:
            html = item.data(Qt.UserRole)
            if html:
                self.details_text.setHtml(html)

        # Enable refinement button only for methods that support it
        _REFINABLE = {'Method A', 'Method B', 'Method C', 'NNLS', 'Regularized NNLS', 'Method D'}
        method_name = item.data(Qt.UserRole + 1) if item else None
        if hasattr(self, 'refinement_btn'):
            self.refinement_btn.setEnabled(method_name in _REFINABLE)

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

        # Capture prior detail HTML before any row removal (for refinement history chain)
        prior_html = None

        # For Method C, determine order-specific tag for row identity
        _method_c_order_tag = None
        if 'Method C' in method_name and not results_df.empty:
            import re as _re
            _fit_text = str(results_df['Fit'].iloc[0])
            _m = _re.search(r'(2nd|3rd|4th) order', _fit_text)
            _method_c_order_tag = f"Method C ({_m.group(1)} order)" if _m else "Method C (unknown)"

        if 'Method C' in method_name and _method_c_order_tag:
            # Replace existing row with the same order; append if new order
            for r in reversed(range(self.results_table.rowCount())):
                cell = self.results_table.item(r, 0)
                if cell and cell.data(Qt.UserRole + 2) == _method_c_order_tag:
                    h = cell.data(Qt.UserRole)
                    if h:
                        prior_html = h
                    self.results_table.removeRow(r)
        elif 'Method A' in method_name or 'Method B' in method_name:
            # Method A and B replace their own row on re-run
            for r in reversed(range(self.results_table.rowCount())):
                cell = self.results_table.item(r, 0)
                if cell and cell.data(Qt.UserRole + 1) == method_name:
                    h = cell.data(Qt.UserRole)
                    if h:
                        prior_html = h
                    self.results_table.removeRow(r)

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

            method_item.setData(Qt.UserRole + 1, method_name)  # tag for future replacement
            if _method_c_order_tag:
                method_item.setData(Qt.UserRole + 2, _method_c_order_tag)  # order-specific tag
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

            # D [m²/s]
            d_val = row.get('D [m²/s]', float('nan'))
            if isinstance(d_val, (list, pd.Series)):
                d_val = d_val[0] if len(d_val) > 0 else float('nan')
            try:
                d_item = QTableWidgetItem(f"{float(d_val):.3e}")
                d_item.setData(0x0100, float(d_val))
            except (TypeError, ValueError):
                d_item = QTableWidgetItem('N/A')
            self.results_table.setItem(current_rows + i, 3, d_item)

            # D error [m²/s]
            d_err_val = row.get('D error [m²/s]', float('nan'))
            if isinstance(d_err_val, (list, pd.Series)):
                d_err_val = d_err_val[0] if len(d_err_val) > 0 else float('nan')
            try:
                d_err_item = QTableWidgetItem(f"{float(d_err_val):.3e}")
                d_err_item.setData(0x0100, float(d_err_val))
            except (TypeError, ValueError):
                d_err_item = QTableWidgetItem('N/A')
            self.results_table.setItem(current_rows + i, 4, d_err_item)

            # R²
            r2_val = row.get('R_squared', row.get('R-squared', 0))
            if isinstance(r2_val, (list, pd.Series)):
                r2_val = r2_val[0] if len(r2_val) > 0 else 0
            r2_item = QTableWidgetItem(f"{r2_val:.4f}")
            r2_item.setData(0x0100, float(r2_val))
            self.results_table.setItem(current_rows + i, 5, r2_item)

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
            self.results_table.setItem(current_rows + i, 6, pdi_item)

            # Residuals (normality)
            res_val = str(row.get('Residuals', 'N/A'))
            res_item = QTableWidgetItem(res_val)
            self.results_table.setItem(current_rows + i, 7, res_item)

        new_row_count = self.results_table.rowCount()
        print(f"[ANALYSIS VIEW] Finished adding rows. New row count: {new_row_count}")
        self.results_table.resizeColumnsToContents()

        # Build detail HTML for this method run (stored per-row, shown on click)

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
                    elif val != 0 and abs(val) < 1e-3:
                        cell_text = f"{val:.3e}"
                    else:
                        cell_text = f"{val:.4f}"
                else:
                    cell_text = str(val)
                new_section += f"<td style='padding: 8px;'>{cell_text}</td>"
            new_section += "</tr>"
        new_section += "</table>"

        # --- Post-fit Refinement Details section (only for refined rows) ---
        if regression_stats and 'refinement_info' in regression_stats:
            ri = regression_stats['refinement_info']
            new_section += f"<h4 style='color:{title_color};'>Post-fit Refinement Details</h4>"

            # Summary: files and q-range
            new_section += (
                f"<table border='1' cellpadding='5' cellspacing='0' "
                f"style='border-collapse:collapse; width:100%; color:{text_color};'>"
                f"<tr style='background-color:{header_bg}; font-weight:bold;'>"
                f"<th>Parameter</th><th>Value</th></tr>"
            )
            _ri_n_total     = ri.get('n_total')
            _ri_n_excluded  = ri.get('n_excluded', 0)
            _ri_in_range    = ri.get('n_files_in_range')
            _ri_n_pts       = ri.get('n_data_points')
            if _ri_n_total is not None:
                new_section += (
                    f"<tr><td>Files manually excluded</td><td>{_ri_n_excluded}</td></tr>"
                )
            new_section += (
                f"<tr><td>q-range applied</td><td>{ri['q_range_str']}</td></tr>"
            )
            if _ri_in_range is not None and _ri_n_total is not None:
                new_section += (
                    f"<tr><td><b>Files in q-range</b></td>"
                    f"<td><b>{_ri_in_range} / {_ri_n_total}</b></td></tr>"
                )
            if _ri_n_pts is not None:
                new_section += (
                    f"<tr><td><b>Data points in regression</b></td>"
                    f"<td><b>{_ri_n_pts}</b></td></tr>"
                )
            new_section += "</table>"

            # Excluded files list
            if ri['excluded_files']:
                new_section += "<p><b>Excluded files:</b><br>"
                new_section += "<br>".join(f"&nbsp;&nbsp;• {f}" for f in ri['excluded_files'])
                new_section += "</p>"

            # Comparison table: Original vs Refined
            if ri.get('original_rh') is not None:
                orig_rh = ri['original_rh']
                orig_d  = ri.get('original_d')
                orig_r2 = ri.get('original_r2')
                ref_rh  = results_df['Rh [nm]'].iloc[0] if 'Rh [nm]' in results_df.columns else None
                ref_d   = results_df['D [m²/s]'].iloc[0] if 'D [m²/s]' in results_df.columns else None
                ref_r2  = results_df['R_squared'].iloc[0] if 'R_squared' in results_df.columns else None

                new_section += (
                    f"<h4>Original vs. Refined:</h4>"
                    f"<table border='1' cellpadding='5' cellspacing='0' "
                    f"style='border-collapse:collapse; width:100%; color:{text_color};'>"
                    f"<tr style='background-color:{header_bg}; font-weight:bold;'>"
                    f"<th>Quantity</th><th>Original</th><th>Refined</th></tr>"
                )
                cmp_rows = [
                    ("Rh [nm]",  f"{orig_rh:.3f}" if orig_rh is not None else "N/A",
                                 f"{ref_rh:.3f}"  if ref_rh  is not None else "N/A"),
                    ("D [m²/s]", f"{orig_d:.3e}"  if orig_d  is not None else "N/A",
                                 f"{ref_d:.3e}"   if ref_d   is not None else "N/A"),
                    ("R²",       f"{orig_r2:.4f}" if orig_r2 is not None else "N/A",
                                 f"{ref_r2:.4f}"  if ref_r2  is not None else "N/A"),
                ]
                for label, orig_val, ref_val in cmp_rows:
                    new_section += (f"<tr><td><b>{label}</b></td>"
                                    f"<td>{orig_val}</td><td>{ref_val}</td></tr>")
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

        # Distribution Moments section (Method C only – shows Skewness and Kurtosis)
        if 'Method C' in method_name and not results_df.empty:
            import math as _math
            model_header_bg = "#5080A0" if is_dark else "#d0e0f0"
            skewness_val = results_df['Skewness'].iloc[0] if 'Skewness' in results_df.columns else float('nan')
            kurtosis_val = results_df['Kurtosis'].iloc[0] if 'Kurtosis' in results_df.columns else float('nan')

            skewness_val = float(skewness_val) if not isinstance(skewness_val, float) else skewness_val
            kurtosis_val = float(kurtosis_val) if not isinstance(kurtosis_val, float) else kurtosis_val

            import pandas as _pd
            sk_text = "N/A (requires 3rd or 4th order)" if _pd.isna(skewness_val) else f"{skewness_val:.4f}"
            ku_text = "N/A (requires 4th order)" if _pd.isna(kurtosis_val) else f"{kurtosis_val:.4f}"

            new_section += "<h4>Distribution Moments (Cumulant Parameters)</h4>"
            new_section += (
                f"<table border='1' cellpadding='8' cellspacing='0' "
                f"style='border-collapse: collapse; width: 100%; color: {text_color};'>"
                f"<tr style='background-color: {model_header_bg}; font-weight: bold;'>"
                f"<th>Moment</th><th>Value</th><th>Formula</th><th>Available for</th></tr>"
                f"<tr><td><b>Skewness</b></td><td>{sk_text}</td>"
                f"<td>\u03ba\u2083 / \u03ba\u2082^(3/2)</td><td>3rd &amp; 4th order</td></tr>"
                f"<tr><td><b>Kurtosis</b></td><td>{ku_text}</td>"
                f"<td>\u03ba\u2084 / \u03ba\u2082\u00b2</td><td>4th order only</td></tr>"
                f"</table>"
                f"<p style='font-size: 9pt; color: gray;'><i>"
                f"\u03ba\u2082 = 2nd cumulant (c parameter), "
                f"\u03ba\u2083 = 3rd cumulant (d parameter), "
                f"\u03ba\u2084 = 4th cumulant (e parameter)."
                f"</i></p>"
            )

        # Store detail HTML in each newly-added row's col-0 item
        # (displayed when user clicks the row, not automatically)
        # Chain prior HTML when this is a refinement (regression_stats is not None)
        if regression_stats is not None and prior_html:
            new_section = (
                prior_html
                + f"<hr style='border:2px solid {title_color}; margin:20px 0;'>"
                + f"<h3 style='color:{title_color}; margin-top:0;'>&#9658; Post-fit Refinement</h3>"
                + new_section
            )
        detail_html = new_section

        # --- Method D: build per-row HTML (population rows get compact view) ---
        # Map from results_df index → html to store. Default to full detail_html.
        _per_row_html = {}
        if 'Method D' in method_name:
            for idx, row in results_df.iterrows():
                fit_str = str(row.get('Fit', ''))
                if '(Population' in fit_str:
                    pop_num = fit_str  # e.g. "Rh from Method D (Population 1)"
                    rh   = row.get('Rh [nm]',    float('nan'))
                    rh_e = row.get('Rh error [nm]', float('nan'))
                    d    = row.get('D [m²/s]',   float('nan'))
                    d_e  = row.get('D error [m²/s]', float('nan'))
                    r2   = row.get('R_squared',  float('nan'))
                    res  = str(row.get('Residuals', 'N/A'))
                    model_header_bg = "#5080A0" if is_dark else "#d0e0f0"
                    compact = (
                        f"<h3 style='color:{title_color}; margin-top:15px;'>{fit_str}</h3>"
                        f"<table border='1' cellpadding='8' cellspacing='0' "
                        f"style='border-collapse:collapse; width:100%; color:{text_color};'>"
                        f"<tr style='background-color:{model_header_bg}; font-weight:bold;'>"
                        f"<th>Parameter</th><th>Value</th></tr>"
                        f"<tr><td><b>Rh [nm]</b></td><td>{rh:.3f} ± {rh_e:.3f}</td></tr>"
                        f"<tr><td><b>D [m²/s]</b></td><td>{d:.3e} ± {d_e:.3e}</td></tr>"
                        f"<tr><td><b>R²</b></td><td>{r2:.4f}</td></tr>"
                        f"<tr><td><b>Residuals</b></td><td>{res}</td></tr>"
                        f"</table>"
                        f"<p style='font-size:9pt; color:gray;'><i>"
                        f"Population-specific OLS: D = &Gamma;/q&sup2;, then R<sub>h</sub> = k<sub>B</sub>T / (6&pi;&eta;D)."
                        f"</i></p>"
                    )
                    _per_row_html[idx] = compact
                else:
                    _per_row_html[idx] = detail_html  # combined row gets full HTML

        new_start = self.results_table.rowCount() - len(results_df)
        for r in range(new_start, self.results_table.rowCount()):
            cell = self.results_table.item(r, 0)
            if cell:
                df_idx = r - new_start  # positional index in results_df
                if _per_row_html:
                    row_idx = results_df.index[df_idx]
                    cell.setData(Qt.UserRole, _per_row_html.get(row_idx, detail_html))
                else:
                    cell.setData(Qt.UserRole, detail_html)

        # Auto-select the first new row so details appear immediately
        if 0 <= new_start < self.results_table.rowCount():
            self.results_table.blockSignals(True)
            self.results_table.setCurrentCell(new_start, 0)
            self.results_table.blockSignals(False)
            first_idx = results_df.index[0]
            first_html = _per_row_html.get(first_idx, detail_html) if _per_row_html else detail_html
            self.details_text.setHtml(first_html)

    def _load_plots(self, method_name, plots_dict, fit_quality, replace_method=False):
        """
        Load plots into the plotting system

        Args:
            method_name: Name of the method
            plots_dict: Dictionary of plots
            fit_quality: Dictionary of fit quality info
            replace_method: If True, replace all plots for this method (used for post-refinement)
        """
        print(f"[ANALYSIS VIEW] Loading {len(plots_dict)} plots for {method_name}")

        # Add to existing plots instead of replacing
        if not hasattr(self, 'current_plots'):
            self.current_plots = {}
        if not hasattr(self, 'current_fit_quality'):
            self.current_fit_quality = {}
        if not hasattr(self, 'all_plots'):
            self.all_plots = {}
        if not hasattr(self, 'all_plot_methods'):
            self.all_plot_methods = {}

        # Always replace plots for the same method (mirrors results table behavior)
        old_keys = [k for k, m in self.all_plot_methods.items() if m == method_name]
        for k in old_keys:
            self.all_plots.pop(k, None)
            self.current_plots.pop(k, None)
            self.all_plot_methods.pop(k, None)

        # Remove matching list items (plots tagged with this method + its separator)
        items_to_remove = []
        for row in range(self.plot_list.count()):
            item = self.plot_list.item(row)
            filename = item.data(Qt.UserRole)
            if filename and filename in old_keys:
                items_to_remove.append(row)
            elif item and method_name in item.text() and "───" in item.text():
                items_to_remove.append(row)
        for row in reversed(items_to_remove):
            self.plot_list.takeItem(row)
        if items_to_remove:
            print(f"[ANALYSIS VIEW] Replaced {len(items_to_remove)} existing items for {method_name}")

        # Merge new plots with existing ones
        self.current_plots.update(plots_dict)
        self.all_plots.update(plots_dict)
        for filename in plots_dict:
            self.all_plot_methods[filename] = method_name

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
                    n = len(source_axes)
                    if n == 1:
                        ax = self.figure.add_subplot(111)
                    elif n == 2:
                        ax = self.figure.add_subplot(1, 2, ax_idx + 1)
                    elif n == 3:
                        ax = self.figure.add_subplot(1, 3, ax_idx + 1)
                    elif n == 4:
                        ax = self.figure.add_subplot(2, 2, ax_idx + 1)
                    else:
                        ax = self.figure.add_subplot(1, n, ax_idx + 1)

                    # Copy scatter plots (collections) - MUST come before lines for proper z-order
                    for collection in source_ax.collections:
                        offsets = collection.get_offsets()
                        if len(offsets) > 0:
                            # Get sizes and ensure they match offsets length
                            sizes = collection.get_sizes()
                            if len(sizes) == 1:
                                sizes = sizes[0]  # Use scalar if only one size
                            elif len(sizes) != len(offsets):
                                sizes = 20  # Default size if mismatch

                            # Get marker path safely
                            paths = collection.get_paths()
                            marker = paths[0] if paths and len(paths) > 0 else 'o'

                            ax.scatter(offsets[:, 0], offsets[:, 1],
                                     label=collection.get_label(),
                                     c=collection.get_facecolors(),
                                     s=sizes,
                                     alpha=collection.get_alpha(),
                                     edgecolors=collection.get_edgecolors(),
                                     linewidths=collection.get_linewidths(),
                                     marker=marker)

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

    def _on_plot_filter_changed(self, _):
        """Handle plot type filter change"""
        self._apply_plot_filters()

    def _on_method_filter_changed(self, _):
        """Handle plot method filter change"""
        self._apply_plot_filters()

    def _apply_plot_filters(self):
        """Apply both type and method filters to the plot list"""
        if not self.all_plots:
            return
        type_filter = self.plot_filter_combo.currentText()
        method_filter = self.plot_method_combo.currentText() if hasattr(self, 'plot_method_combo') else "All Methods"

        filtered = self.all_plots.copy()

        # Apply type filter
        if type_filter == "Diffusion Analysis Plots":
            filtered = {k: v for k, v in filtered.items() if "Summary" in k or "Diffusion" in k}
        elif type_filter == "Fit Plots":
            filtered = {k: v for k, v in filtered.items() if "Summary" not in k and "Diffusion" not in k}

        # Apply method filter
        if method_filter != "All Methods":
            all_methods = getattr(self, 'all_plot_methods', {})
            filtered = {k: v for k, v in filtered.items()
                        if all_methods.get(k, "").startswith(method_filter)}

        self.current_plots = filtered
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

    def clear_results(self):
        """Clear all results"""
        self.results_table.setRowCount(0)
        self.details_text.clear()
        self.current_plots = {}
        self.all_plots = {}
        self.plot_list.clear()
        self.nav_widget.hide()
        self.plot_placeholder.show()
        # Reset filters
        if hasattr(self, 'plot_filter_combo'):
            self.plot_filter_combo.setCurrentText("All Plots")
        if hasattr(self, 'plot_method_combo'):
            self.plot_method_combo.setCurrentText("All Methods")
        self.all_plot_methods = {}
        # Hide refinement button
        if hasattr(self, 'refinement_widget'):
            self.refinement_widget.hide()
        # Clear stored analyzer
        self.cumulant_analyzer = None

    def set_cumulant_analyzer(self, analyzer):
        """
        Store the CumulantAnalyzer instance so the ScatterForge export
        buttons can access the analysis data.

        Called by main_window after cumulant analysis completes.
        """
        self.cumulant_analyzer = analyzer

    def _on_send_to_scatterforge(self, export_type: str):
        """
        Export the current cumulant Method A results to ScatterForge-Plot.

        Args:
            export_type: 'gamma' for Γ vs q², 'diffusion' for D vs q²
        """
        from ade_dls.gui.export.scatterforge_bridge import (
            from_cumulant_method_a,
            from_diffusion_vs_q2,
            send_to_scatterforge,
        )
        from PyQt5.QtWidgets import QMessageBox

        if self.cumulant_analyzer is None:
            QMessageBox.warning(
                self,
                "No Analyzer",
                "Please run Cumulants Method A first.",
            )
            return

        try:
            if export_type == 'gamma':
                groups = from_cumulant_method_a(self.cumulant_analyzer)
                plot_type = 'DLS - \u0393 vs q\u00b2'
            else:
                groups = from_diffusion_vs_q2(self.cumulant_analyzer)
                plot_type = 'DLS - D vs q\u00b2'

            send_to_scatterforge(groups, plot_type=plot_type)

        except Exception as exc:
            QMessageBox.critical(
                self,
                "ScatterForge Export Failed",
                f"Export error:\n{exc}",
            )

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

    def _open_selected_refinement(self):
        """Open refinement for the method of the currently selected Results Overview row"""
        from PyQt5.QtWidgets import QMessageBox
        from PyQt5.QtCore import Qt

        selected_row = self.results_table.currentRow()
        if selected_row < 0:
            QMessageBox.warning(self, "No Row Selected",
                                "Please select a row in Results Overview first.")
            return

        item = self.results_table.item(selected_row, 0)
        method_name = item.data(Qt.UserRole + 1) if item else None
        order_tag   = item.data(Qt.UserRole + 2) if item else None  # e.g. 'Method C (3rd order)'

        # For Method C: activate the correct per-order data before opening refinement
        if method_name == 'Method C' and order_tag and hasattr(self, 'cumulant_analyzer') and self.cumulant_analyzer is not None:
            import re as _re
            _m = _re.search(r'\((.+?)\)', order_tag)
            if _m:
                order_label = _m.group(1)  # e.g. '3rd order'
                self.cumulant_analyzer.activate_method_c_order(order_label)

        # Store order_tag so _open_refinement_for_method can use it for in-place update
        self._pending_method_c_order_tag = order_tag if method_name == 'Method C' else None

        method_map = {
            'Method A': 'cumulant_a',
            'Method B': 'cumulant_b',
            'Method C': 'cumulant_c',
            'Method D': 'cumulant_d',
            'NNLS':     'nnls',
            'Regularized NNLS': 'regularized',
        }
        selected_method = method_map.get(method_name)

        if selected_method is None:
            QMessageBox.warning(self, "Not Supported",
                                f"Refinement is not available for '{method_name}'.")
            return

        self._open_refinement_for_method(selected_method)

    def _open_refinement_dialog(self):
        """Open the post-fit refinement dialog with method selection (legacy method - kept for compatibility)"""
        from PyQt5.QtWidgets import (QMessageBox, QDialog, QVBoxLayout, QHBoxLayout,
                                    QLabel, QPushButton, QComboBox, QGroupBox)
        from gui.dialogs import PostFitRefinementDialog

        # Collect all available analysis methods
        available_methods = []

        # Check for Cumulant methods
        if hasattr(self, 'cumulant_analyzer') and self.cumulant_analyzer is not None:
            if hasattr(self.cumulant_analyzer, 'method_a_results') and self.cumulant_analyzer.method_a_results is not None:
                available_methods.append(('Cumulant Method A', 'cumulant_a'))
            if hasattr(self.cumulant_analyzer, 'method_b_results') and self.cumulant_analyzer.method_b_results is not None:
                available_methods.append(('Cumulant Method B', 'cumulant_b'))
            if hasattr(self.cumulant_analyzer, 'method_c_results') and self.cumulant_analyzer.method_c_results is not None:
                available_methods.append(('Cumulant Method C', 'cumulant_c'))

        # Check for Laplace methods
        if hasattr(self, 'laplace_analyzer') and self.laplace_analyzer is not None:
            if hasattr(self.laplace_analyzer, 'nnls_final_results') and self.laplace_analyzer.nnls_final_results is not None:
                available_methods.append(('NNLS', 'nnls'))
            if hasattr(self.laplace_analyzer, 'regularized_final_results') and self.laplace_analyzer.regularized_final_results is not None:
                available_methods.append(('Regularized NNLS', 'regularized'))

        if not available_methods:
            QMessageBox.warning(
                self,
                "No Analysis Results",
                "No analysis results available for refinement.\n"
                "Please run an analysis first."
            )
            return

        # If only one method available, open directly
        if len(available_methods) == 1:
            method_name, method_type = available_methods[0]
            self._open_refinement_for_method(method_type)
            return

        # Create selection dialog
        selection_dialog = QDialog(self)
        selection_dialog.setWindowTitle("Select Method to Refine")
        selection_dialog.setMinimumWidth(400)
        layout = QVBoxLayout()

        # Title
        title = QLabel("<b>Post-Fit Refinement</b>")
        title.setStyleSheet("font-size: 12pt;")
        layout.addWidget(title)

        # Description
        desc = QLabel(
            "Multiple analysis methods have results available.\n"
            "Select which method you want to refine:"
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Method selection
        method_group = QGroupBox("Available Methods")
        method_layout = QVBoxLayout()

        method_combo = QComboBox()
        for display_name, method_type in available_methods:
            method_combo.addItem(display_name, method_type)
        method_layout.addWidget(method_combo)

        method_group.setLayout(method_layout)
        layout.addWidget(method_group)

        # Buttons
        button_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(selection_dialog.reject)
        button_layout.addWidget(cancel_btn)

        ok_btn = QPushButton("Open Refinement")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(selection_dialog.accept)
        ok_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)
        selection_dialog.setLayout(layout)

        # Show dialog and get selection
        if selection_dialog.exec_() != QDialog.Accepted:
            return

        selected_method = method_combo.currentData()
        self._open_refinement_for_method(selected_method)

    def _open_refinement_for_method(self, method_type):
        """
        Open refinement dialog for specific method

        Args:
            method_type: 'cumulant_a', 'cumulant_b', 'cumulant_c', 'nnls', or 'regularized'
        """
        from PyQt5.QtWidgets import QMessageBox
        from gui.dialogs import PostFitRefinementDialog

        if method_type in ['cumulant_a', 'cumulant_b', 'cumulant_c']:
            # Cumulant refinement
            method_letter = method_type.split('_')[1].upper()
            available_methods = [method_letter]

            dialog = PostFitRefinementDialog(self.cumulant_analyzer, available_methods, self)
            if dialog.exec_() != dialog.Accepted:
                return

            # Continue with existing cumulant refinement code (kept as-is)...
            # Get refinement parameters
            params = dialog.get_refinement_params()

            print("\n" + "="*60)
            print("POST-FIT REFINEMENT")
            print("="*60)

            # Re-compute results for each method
            results_updated = []

            try:
                # Method A
                if 'A' in available_methods and params['q_ranges']['A'] is not None:
                    print(f"\nRe-computing Method A with q² range: {params['q_ranges']['A']}")
                    result_a = self.cumulant_analyzer.run_method_a(q_range=params['q_ranges']['A'])
                    if result_a is not None:
                        results_updated.append(('Method A', result_a))
                        q_range_a = params['q_ranges']['A']
                        q_str_a = (f"{q_range_a[0]:.3e} \u2013 {q_range_a[1]:.3e} nm\u207b\u00b2"
                                   if q_range_a is not None else "Full range")
                        refinement_stats_a = {
                            'refinement_info': {
                                'q_range_str': q_str_a,
                                'q_range': q_range_a,
                                'excluded_files': [],
                                'n_total': None, 'n_included': None, 'n_excluded': 0,
                            }
                        }
                        self._update_results_table('Method A', result_a,
                                                   regression_stats=refinement_stats_a)

                # Method B
                if 'B' in available_methods and params['q_ranges']['B'] is not None:
                    print(f"\nRe-computing Method B with q² range: {params['q_ranges']['B']}")
                    # Get original fit limits from analyzer
                    if hasattr(self.cumulant_analyzer, 'method_b_fit_limits'):
                        fit_limits = self.cumulant_analyzer.method_b_fit_limits
                    else:
                        fit_limits = (0.0, 0.0002)  # Default
                    result_b = self.cumulant_analyzer.run_method_b(
                        fit_limits=fit_limits,
                        q_range=params['q_ranges']['B']
                    )
                    if result_b is not None:
                        results_updated.append(('Method B', result_b))
                        q_range_b = params['q_ranges']['B']
                        q_str_b = (f"{q_range_b[0]:.3e} \u2013 {q_range_b[1]:.3e} nm\u207b\u00b2"
                                   if q_range_b is not None else "Full range")
                        refinement_stats_b = {
                            'refinement_info': {
                                'q_range_str': q_str_b,
                                'q_range': q_range_b,
                                'excluded_files': [],
                                'n_total': None, 'n_included': None, 'n_excluded': 0,
                            }
                        }
                        self._update_results_table('Method B', result_b,
                                                   regression_stats=refinement_stats_b)

                # Method C
                if 'C' in available_methods:
                    q_range = params['q_ranges']['C']
                    excluded_fits = params['excluded_fits_c']

                    # Handle exclusions and q-range
                    if q_range is not None or excluded_fits:
                        print(f"\nRe-computing Method C:")
                        if q_range:
                            print(f"  q² range: {q_range}")
                        if excluded_fits:
                            print(f"  Excluded fits: {len(excluded_fits)}")

                        # For Method C, we need to recompute with filtered data
                        recompute_output = self._recompute_method_c(q_range, excluded_fits)
                        if recompute_output is not None:
                            result_c, refinement_stats = recompute_output
                            results_updated.append(('Method C', result_c))
                            # Update the existing row in-place (values + appended HTML detail)
                            _pending_tag = getattr(self, '_pending_method_c_order_tag', None)
                            self._update_method_c_row_in_place(_pending_tag, result_c,
                                                               refinement_stats)

                            # Add post-refinement step to pipeline
                            from gui.main_window import JADEDLSMainWindow
                            parent = self.parent()
                            while parent and not isinstance(parent, JADEDLSMainWindow):
                                parent = parent.parent()

                            if parent and hasattr(parent, 'pipeline'):
                                parent.pipeline.add_post_refinement_step(
                                    method_name="Method C",
                                    q_range=q_range,
                                    excluded_files=excluded_fits,
                                    is_laplace=False
                                )

                print("="*60 + "\n")

                # Show summary
                if results_updated:
                    QMessageBox.information(
                        self,
                        "Refinement Complete",
                        f"Successfully refined {len(results_updated)} method(s).\n"
                        "Results have been updated."
                    )
                else:
                    QMessageBox.information(
                        self,
                        "No Changes",
                        "No refinement parameters were changed."
                    )

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Refinement Failed",
                    f"Failed to refine results:\n{str(e)}"
                )
                import traceback
                traceback.print_exc()

        elif method_type == 'cumulant_d':
            # Method D two-stage refinement
            from ade_dls.gui.dialogs import MethodDPostFitDialog

            if not hasattr(self, 'cumulant_analyzer') or self.cumulant_analyzer is None:
                QMessageBox.warning(self, "No Analyzer", "No cumulant analyzer available.")
                return
            if not hasattr(self.cumulant_analyzer, 'method_d_clustered_df') or \
               self.cumulant_analyzer.method_d_clustered_df is None:
                QMessageBox.warning(self, "No Method D Results",
                                    "Please run Method D first before refining.")
                return

            dialog = MethodDPostFitDialog(self.cumulant_analyzer, self)
            if dialog.exec_() != dialog.Accepted:
                return

            ref_params = dialog.get_refinement_params()

            try:
                print("\n" + "="*60)
                print("POST-FIT REFINEMENT – METHOD D")
                print("="*60)

                result_d = self.cumulant_analyzer.refine_method_d(
                    clustering_params=ref_params.get('clustering'),
                    mode_params=ref_params.get('mode_params') or None,
                    combined_q_range=ref_params.get('combined_q_range'),
                )

                self._update_results_table(
                    'Method D', result_d,
                    regression_stats=self.cumulant_analyzer.method_d_regression_stats
                )

                # Add post-refinement step to pipeline
                try:
                    from gui.main_window import JADEDLSMainWindow
                    parent = self.parent()
                    while parent and not isinstance(parent, JADEDLSMainWindow):
                        parent = parent.parent()
                    if parent and hasattr(parent, 'pipeline'):
                        parent.pipeline.add_post_refinement_step(
                            method_name="Method D",
                            q_range=ref_params.get('combined_q_range'),
                            excluded_files=[],
                            is_laplace=False,
                        )
                except Exception:
                    pass

                print("="*60 + "\n")
                QMessageBox.information(self, "Refinement Complete",
                                        "Method D results have been updated.")

            except Exception as e:
                QMessageBox.critical(self, "Refinement Failed",
                                     f"Failed to refine Method D results:\n{str(e)}")
                import traceback
                traceback.print_exc()

        elif method_type in ['nnls', 'regularized']:
            # Laplace (NNLS/Regularized) refinement
            method_name = "NNLS" if method_type == 'nnls' else "Regularized"
            self._open_laplace_refinement_for_method(method_name)

        else:
            QMessageBox.warning(
                self,
                "Unknown Method",
                f"Unknown method type: {method_type}"
            )

    def _open_laplace_refinement_for_method(self, method_name):
        """
        Open Laplace refinement dialog for specific method (NNLS or Regularized)

        Args:
            method_name: 'NNLS' or 'Regularized'
        """
        from PyQt5.QtWidgets import QMessageBox
        from gui.dialogs.laplace_postfit_dialog import LaplacePostFitRefinementDialog

        # Open the refinement dialog
        dialog = LaplacePostFitRefinementDialog(self.laplace_analyzer, method_name, self)

        if dialog.exec_() != dialog.Accepted:
            return

        # Get refinement parameters
        params = dialog.get_refinement_params()
        q_range = params['q_range']
        excluded_files = params['excluded_files']

        # Check if any changes were made
        if q_range is None and not excluded_files:
            QMessageBox.information(
                self,
                "No Changes",
                "No changes were made to the analysis."
            )
            return

        # Proceed with refinement
        print("\n" + "="*60)
        print(f"{method_name.upper()} POST-FIT REFINEMENT")
        print("="*60)

        if excluded_files:
            print(f"Excluding {len(excluded_files)} distribution(s): {excluded_files}")

        if q_range:
            print(f"Applying q² range: {q_range[0]:.4f} to {q_range[1]:.4f} nm⁻²")

        try:
            is_nnls = (method_name == "NNLS")

            # Filter data by excluded files if specified
            if excluded_files:
                if is_nnls and hasattr(self.laplace_analyzer, 'nnls_data'):
                    # Remove rows where filename is in excluded_files
                    self.laplace_analyzer.nnls_data = self.laplace_analyzer.nnls_data[
                        ~self.laplace_analyzer.nnls_data['filename'].isin(excluded_files)
                    ].reset_index(drop=True)
                    self.laplace_analyzer.nnls_data.index = self.laplace_analyzer.nnls_data.index + 1
                    print(f"  Removed {len(excluded_files)} datasets, {len(self.laplace_analyzer.nnls_data)} remaining")

                elif not is_nnls and hasattr(self.laplace_analyzer, 'regularized_data'):
                    self.laplace_analyzer.regularized_data = self.laplace_analyzer.regularized_data[
                        ~self.laplace_analyzer.regularized_data['filename'].isin(excluded_files)
                    ].reset_index(drop=True)
                    self.laplace_analyzer.regularized_data.index = self.laplace_analyzer.regularized_data.index + 1
                    print(f"  Removed {len(excluded_files)} datasets, {len(self.laplace_analyzer.regularized_data)} remaining")

            # Recalculate diffusion coefficients with new q² range
            if is_nnls:
                self.laplace_analyzer.calculate_nnls_diffusion_coefficients(x_range=q_range)
                self.laplace_analyzer._calculate_nnls_final_results(
                    append_mode=True,
                    label_suffix=" (Post-Refined)"
                )

                # Update display with replace_existing=False to append plots
                from gui.main_window import JADEDLSMainWindow
                parent = self.parent()
                while parent and not isinstance(parent, JADEDLSMainWindow):
                    parent = parent.parent()

                if parent:
                    parent._display_nnls_results(replace_existing=False)
                    # Add post-refinement step to pipeline
                    if hasattr(parent, 'pipeline'):
                        parent.pipeline.add_post_refinement_step(
                            method_name="NNLS",
                            q_range=q_range,
                            excluded_files=excluded_files,
                            is_laplace=True
                        )

            else:  # Regularized
                self.laplace_analyzer.calculate_regularized_diffusion_coefficients(x_range=q_range)
                self.laplace_analyzer._calculate_regularized_final_results()

                # Update display with replace_existing=False to append plots
                from gui.main_window import JADEDLSMainWindow
                parent = self.parent()
                while parent and not isinstance(parent, JADEDLSMainWindow):
                    parent = parent.parent()

                if parent:
                    parent._display_regularized_results(replace_existing=False)
                    # Add post-refinement step to pipeline
                    if hasattr(parent, 'pipeline'):
                        parent.pipeline.add_post_refinement_step(
                            method_name="Regularized",
                            q_range=q_range,
                            excluded_files=excluded_files,
                            is_laplace=True
                        )

            print("="*60 + "\n")

            success_msg = f"Successfully refined {method_name} results.\n\n"
            if excluded_files:
                success_msg += f"Excluded {len(excluded_files)} distribution(s).\n"
            if q_range:
                success_msg += f"Applied q² range: {q_range[0]:.4f} to {q_range[1]:.4f} nm⁻².\n"
            success_msg += "\nDiffusion coefficients have been recalculated."

            QMessageBox.information(
                self,
                "Refinement Complete",
                success_msg
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Refinement Failed",
                f"Failed to refine results:\n{str(e)}"
            )
            import traceback
            traceback.print_exc()

    def _update_method_c_row_in_place(self, order_tag, result_df, regression_stats):
        """
        Update an existing Method C row's numeric values and append refinement info to its HTML.

        Used after post-refinement so the existing row is updated in-place rather than
        a new row being appended.

        Args:
            order_tag: Qt.UserRole + 2 tag, e.g. 'Method C (3rd order)'. If None or row not
                       found, falls back to appending a new row via _update_results_table.
            result_df: DataFrame with recomputed results (one row).
            regression_stats: Dict with regression stats and refinement_info.
        """
        import pandas as pd
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QTableWidgetItem

        # Find the target row
        target_row = -1
        if order_tag:
            for r in range(self.results_table.rowCount()):
                cell = self.results_table.item(r, 0)
                if cell and cell.data(Qt.UserRole + 2) == order_tag:
                    target_row = r
                    break

        if target_row < 0:
            # Fallback: append new row (edge case – row not found)
            print(f"[REFINEMENT] Row not found for tag '{order_tag}', appending new row.")
            self._update_results_table('Method C', result_df, regression_stats)
            return

        row = result_df.iloc[0]

        def _make_numeric_item(val, fmt):
            if isinstance(val, (list, pd.Series)):
                val = val[0] if len(val) > 0 else float('nan')
            try:
                fval = float(val)
                item = QTableWidgetItem(fmt.format(fval))
                item.setData(0x0100, fval)
            except (TypeError, ValueError):
                item = QTableWidgetItem('N/A')
            return item

        # Update numeric columns 1–7
        self.results_table.setItem(target_row, 1, _make_numeric_item(row.get('Rh [nm]', 0), '{:.2f}'))
        self.results_table.setItem(target_row, 2, _make_numeric_item(row.get('Rh error [nm]', 0), '{:.2f}'))
        self.results_table.setItem(target_row, 3, _make_numeric_item(row.get('D [m²/s]', float('nan')), '{:.3e}'))
        self.results_table.setItem(target_row, 4, _make_numeric_item(row.get('D error [m²/s]', float('nan')), '{:.3e}'))
        self.results_table.setItem(target_row, 5, _make_numeric_item(
            row.get('R_squared', row.get('R-squared', 0)), '{:.4f}'))
        self.results_table.setItem(target_row, 6, _make_numeric_item(row.get('PDI', float('nan')), '{:.4f}'))
        res_val = str(row.get('Residuals', 'N/A'))
        self.results_table.setItem(target_row, 7, QTableWidgetItem(res_val))

        # Build new HTML section for the refinement and append to existing detail HTML
        # Re-use the HTML-building logic from _update_results_table by calling it with a
        # temporary approach: build a minimal new section with refinement info.
        from PyQt5.QtWidgets import QApplication
        palette = QApplication.palette()
        is_dark = palette.base().color().lightness() < 128
        title_color = "#66B3FF" if is_dark else "#0066cc"
        header_bg = "#2D2D2D" if is_dark else "#e0e0e0"
        row_bg_1 = "#1E1E1E" if is_dark else "#f9f9f9"
        row_bg_2 = "#252525" if is_dark else "#ffffff"
        text_color = "#E0E0E0" if is_dark else "#000000"

        # Only show core result columns – metadata columns (N files, Original *, q-range)
        # are already shown in the Post-fit Refinement Details block below.
        _CORE_COLS = {'Rh [nm]', 'Rh error [nm]', 'D [m²/s]', 'D error [m²/s]',
                      'R_squared', 'Fit', 'Residuals', 'PDI', 'Skewness', 'Kurtosis'}
        display_cols = [c for c in result_df.columns if c in _CORE_COLS]

        def _fmt_cell(val):
            """Format a cell value: scientific notation for very small/large numbers."""
            if isinstance(val, (list, pd.Series)):
                val = val[0] if len(val) > 0 else val
            if isinstance(val, float):
                if pd.isna(val):
                    return "N/A"
                if val != 0 and abs(val) < 1e-3:
                    return f"{val:.3e}"
                return f"{val:.4f}"
            return str(val)

        new_section = f"<h3 style='color: {title_color}; margin-top: 15px;'>Post-Refinement Update</h3>"
        new_section += "<h4>Updated Results:</h4>"
        new_section += (
            f"<table border='1' cellpadding='5' cellspacing='0' "
            f"style='border-collapse: collapse; width: 100%; color: {text_color};'>"
            f"<tr style='background-color: {header_bg}; font-weight: bold;'>"
        )
        for col in display_cols:
            new_section += f"<th style='padding: 8px; text-align: left;'>{col}</th>"
        new_section += "</tr>"
        for idx, df_row in result_df.iterrows():
            bg_color = row_bg_1 if idx % 2 == 0 else row_bg_2
            new_section += f"<tr style='background-color: {bg_color};'>"
            for col in display_cols:
                new_section += f"<td style='padding: 8px;'>{_fmt_cell(df_row[col])}</td>"
            new_section += "</tr>"
        new_section += "</table>"

        # Add refinement details block
        if regression_stats and 'refinement_info' in regression_stats:
            ri = regression_stats['refinement_info']
            new_section += f"<h4 style='color:{title_color};'>Post-fit Refinement Details</h4>"
            new_section += (
                f"<table border='1' cellpadding='5' cellspacing='0' "
                f"style='border-collapse:collapse; width:100%; color:{text_color};'>"
                f"<tr style='background-color:{header_bg}; font-weight:bold;'>"
                f"<th>Parameter</th><th>Value</th></tr>"
            )
            _n_pts        = ri.get('n_data_points')
            _n_in_range   = ri.get('n_files_in_range')
            _n_total      = ri.get('n_total')
            _n_included   = ri.get('n_included')  # manually selected (pre-q-filter)
            _n_excluded   = ri.get('n_excluded')
            if _n_total is not None:
                new_section += (
                    f"<tr><td>Files manually excluded</td><td>{_n_excluded}</td></tr>"
                )
            new_section += (
                f"<tr><td>q-range applied</td><td>{ri['q_range_str']}</td></tr>"
            )
            if _n_in_range is not None and _n_total is not None:
                new_section += (
                    f"<tr><td><b>Files in q-range</b></td>"
                    f"<td><b>{_n_in_range} / {_n_total}</b></td></tr>"
                )
            if _n_pts is not None:
                new_section += (
                    f"<tr><td><b>Data points in regression</b></td>"
                    f"<td><b>{_n_pts}</b></td></tr>"
                )
            new_section += "</table>"
            if ri['excluded_files']:
                new_section += "<p><b>Excluded files:</b><br>"
                new_section += "<br>".join(f"&nbsp;&nbsp;• {f}" for f in ri['excluded_files'])
                new_section += "</p>"
            if ri.get('original_rh') is not None:
                orig_rh = ri['original_rh']
                orig_d  = ri.get('original_d')
                orig_r2 = ri.get('original_r2')
                ref_rh  = result_df['Rh [nm]'].iloc[0] if 'Rh [nm]' in result_df.columns else None
                ref_d   = result_df['D [m²/s]'].iloc[0] if 'D [m²/s]' in result_df.columns else None
                ref_r2  = result_df['R_squared'].iloc[0] if 'R_squared' in result_df.columns else None
                new_section += (
                    f"<h4>Original vs. Refined:</h4>"
                    f"<table border='1' cellpadding='5' cellspacing='0' "
                    f"style='border-collapse:collapse; width:100%; color:{text_color};'>"
                    f"<tr style='background-color:{header_bg}; font-weight:bold;'>"
                    f"<th>Quantity</th><th>Original</th><th>Refined</th></tr>"
                )
                for label, orig_val, ref_val in [
                    ("Rh [nm]",  f"{orig_rh:.3f}" if orig_rh is not None else "N/A",
                                 f"{ref_rh:.3f}"  if ref_rh  is not None else "N/A"),
                    ("D [m²/s]", f"{orig_d:.3e}"  if orig_d  is not None else "N/A",
                                 f"{ref_d:.3e}"   if ref_d   is not None else "N/A"),
                    ("R²",       f"{orig_r2:.4f}" if orig_r2 is not None else "N/A",
                                 f"{ref_r2:.4f}"  if ref_r2  is not None else "N/A"),
                ]:
                    new_section += (f"<tr><td><b>{label}</b></td>"
                                    f"<td>{orig_val}</td><td>{ref_val}</td></tr>")
                new_section += "</table>"

        # Append new section to existing HTML
        existing_html = self.results_table.item(target_row, 0).data(Qt.UserRole) or ''
        chained_html = existing_html + "<hr>" + new_section
        self.results_table.item(target_row, 0).setData(Qt.UserRole, chained_html)

        # Refresh detail panel if this row is currently selected
        if self.results_table.currentRow() == target_row:
            self.details_text.setHtml(chained_html)

        self.results_table.resizeColumnsToContents()
        print(f"[REFINEMENT] Updated row in-place for tag '{order_tag}'")

    def _recompute_method_c(self, q_range, excluded_fits):
        """
        Re-compute Method C with filtered data.

        Returns:
            Tuple (result_df, refinement_stats) or None on error.
        """
        import numpy as np

        analyzer = self.cumulant_analyzer
        if not hasattr(analyzer, 'method_c_fit') or analyzer.method_c_fit is None:
            print("[ERROR] No Method C data available")
            return None

        # Build included_files (complement of excluded)
        all_files = analyzer.method_c_fit['filename'].unique().tolist()
        included_files = [f for f in all_files if f not in (excluded_fits or [])]

        if len(included_files) < 2:
            print("[ERROR] Too few files remaining after exclusion")
            return None

        try:
            result = analyzer.recompute_method_c_diffusion(included_files, q_range)
        except Exception as e:
            print(f"[ERROR] recompute_method_c_diffusion failed: {e}")
            return None

        # Read actual data points and contributing files (set by recompute_method_c_diffusion)
        n_data_points     = getattr(analyzer, 'method_c_recompute_n_points', None)
        n_files_in_range  = getattr(analyzer, 'method_c_recompute_n_files_in_range', None)

        # --- Add refinement metadata columns ---
        n_total    = len(all_files)
        n_included = len(included_files)
        n_excluded = n_total - n_included
        result['N files (total)']    = [n_total]
        result['N files (included)'] = [n_included]
        result['N files (excluded)'] = [n_excluded]
        q_str = (f"{q_range[0]:.3e} – {q_range[1]:.3e} nm⁻²"
                 if q_range is not None else "Full range")
        result['q-range applied'] = [q_str]

        # --- Add original results for comparison ---
        orig = getattr(analyzer, 'method_c_results', None)
        if orig is not None and not orig.empty:
            result['Original Rh [nm]']  = [orig['Rh [nm]'].iloc[0]
                                            if 'Rh [nm]' in orig.columns else np.nan]
            result['Original D [m²/s]'] = [orig['D [m²/s]'].iloc[0]
                                            if 'D [m²/s]' in orig.columns else np.nan]
            result['Original R²']       = [orig['R_squared'].iloc[0]
                                            if 'R_squared' in orig.columns else np.nan]

        # --- Build regression_stats dict for HTML detail ---
        refinement_stats = dict(getattr(analyzer, 'method_c_regression_stats', {}) or {})
        refinement_stats['refinement_info'] = {
            'n_total':        n_total,
            'n_included':     n_included,
            'n_excluded':     n_excluded,
            'n_data_points':    n_data_points,     # actual (q², Γ) pairs in OLS regression
            'n_files_in_range': n_files_in_range,  # unique files with data within q-range
            'excluded_files': excluded_fits or [],
            'q_range':        q_range,
            'q_range_str':    q_str,
            'original_rh':    float(orig['Rh [nm]'].iloc[0])
                              if orig is not None and 'Rh [nm]' in orig.columns else None,
            'original_d':     float(orig['D [m²/s]'].iloc[0])
                              if orig is not None and 'D [m²/s]' in orig.columns else None,
            'original_r2':    float(orig['R_squared'].iloc[0])
                              if orig is not None and 'R_squared' in orig.columns else None,
        }

        return result, refinement_stats

