"""
Analysis View Widget
Main central panel for displaying data, plots, and results
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QTableWidget, QTableWidgetItem, QLabel, QTextEdit,
                             QGroupBox, QScrollArea)
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
        nav_layout = QHBoxLayout()

        from PyQt5.QtWidgets import QPushButton, QListWidget

        self.plot_list = QListWidget()
        self.plot_list.setMaximumWidth(200)
        self.plot_list.currentRowChanged.connect(self._on_plot_selected)
        nav_layout.addWidget(self.plot_list)

        # Right side: plot and navigation
        right_layout = QVBoxLayout()

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

        right_layout.addLayout(nav_buttons)

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

            plot_layout.addWidget(self.toolbar)
            plot_layout.addWidget(self.canvas)
            self.plot_placeholder.hide()
        except ImportError:
            # Matplotlib not available, use placeholder
            pass

        plot_group.setLayout(plot_layout)
        right_layout.addWidget(plot_group)

        # Plot info label
        self.plot_info_label = QLabel("")
        self.plot_info_label.setWordWrap(True)
        right_layout.addWidget(self.plot_info_label)

        nav_layout.addLayout(right_layout)
        self.nav_widget.setLayout(nav_layout)
        self.nav_widget.hide()  # Initially hidden

        layout.addWidget(self.nav_widget)

        widget.setLayout(layout)

        # Store current plots
        self.current_plots = {}
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
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Method", "Rh (nm)", "Error (nm)", "R²", "PDI", "Residuals"
        ])
        # Enable sorting and better column sizing
        self.results_table.setSortingEnabled(True)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setAlternatingRowColors(True)
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

    def display_cumulant_results(self, method_name, results_df, plots_dict=None, fit_quality=None):
        """
        Display cumulant analysis results in Results and Plots tabs

        Args:
            method_name: Name of the method (e.g., "Method B")
            results_df: DataFrame with final results
            plots_dict: Dictionary {filename: (fig, data)} or None
            fit_quality: Dictionary {filename: {'R2': float, ...}} or None
        """
        # Update Results tab
        self._update_results_table(method_name, results_df)

        # Update Plots tab if plots are available
        if plots_dict:
            self._load_plots(method_name, plots_dict, fit_quality)
            # Switch to Plots tab
            self.tabs.setCurrentIndex(1)
        else:
            # Switch to Results tab
            self.tabs.setCurrentIndex(2)

    def _update_results_table(self, method_name, results_df):
        """Update results table with new results"""
        if results_df.empty:
            return

        # Get existing row count
        current_rows = self.results_table.rowCount()

        # Add rows for new results
        for i, row in results_df.iterrows():
            self.results_table.insertRow(current_rows + i)

            # Method - use the 'Fit' column for the full description
            fit_name = str(row.get('Fit', method_name))
            method_item = QTableWidgetItem(fit_name)

            # Set background color based on method type
            from PyQt5.QtGui import QColor
            if 'Method A' in method_name or '1st order' in fit_name or '2nd order' in fit_name or '3rd order' in fit_name:
                method_item.setBackground(QColor(230, 240, 255))  # Light blue
            elif 'Method B' in method_name or 'linear' in fit_name:
                method_item.setBackground(QColor(240, 255, 230))  # Light green
            elif 'Method C' in method_name or 'iterative' in fit_name or 'non-linear' in fit_name:
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

            # Residuals
            res_val = row.get('Residuals', 'N/A')
            self.results_table.setItem(current_rows + i, 5,
                                      QTableWidgetItem(str(res_val)))

        self.results_table.resizeColumnsToContents()

        # Update details text with formatted output
        # Get current HTML content
        current_html = self.details_text.toHtml()

        # Start building new method section
        new_section = f"<h3 style='color: #0066cc; margin-top: 15px;'>{method_name}</h3>"
        new_section += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%;'>"
        new_section += "<tr style='background-color: #e0e0e0; font-weight: bold;'>"

        # Add headers
        for col in results_df.columns:
            new_section += f"<th style='padding: 8px; text-align: left;'>{col}</th>"
        new_section += "</tr>"

        # Add data rows with alternating colors
        for idx, row in results_df.iterrows():
            bg_color = "#f9f9f9" if idx % 2 == 0 else "#ffffff"
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

        self.current_plots = plots_dict
        self.current_fit_quality = fit_quality or {}
        self.current_method_name = method_name

        # Clear and populate plot list
        self.plot_list.clear()
        filenames = list(plots_dict.keys())

        for i, filename in enumerate(filenames):
            # Add quality indicator
            quality_str = ""
            if filename in self.current_fit_quality:
                r2 = self.current_fit_quality[filename].get('R2', 0)
                quality_str = f" (R²={r2:.3f})"

            self.plot_list.addItem(f"{i+1}. {filename}{quality_str}")
            print(f"[ANALYSIS VIEW] Added plot {i+1}: {filename}{quality_str}")

        # Show navigation widget
        self.nav_widget.show()
        self.plot_placeholder.hide()

        # Show first plot
        if filenames:
            self.current_plot_index = 0
            self.plot_list.setCurrentRow(0)
            self._show_plot(0)
            print(f"[ANALYSIS VIEW] Displayed first plot")
        else:
            print(f"[ANALYSIS VIEW WARNING] No plots to display!")

    def _on_plot_selected(self, index):
        """Handle plot selection from list"""
        if index >= 0:
            self._show_plot(index)

    def _show_plot(self, index):
        """Show plot at given index"""
        print(f"[ANALYSIS VIEW] _show_plot called with index={index}")

        filenames = list(self.current_plots.keys())
        if index < 0 or index >= len(filenames):
            print(f"[ANALYSIS VIEW ERROR] Index {index} out of range (0-{len(filenames)-1})")
            return

        self.current_plot_index = index
        filename = filenames[index]
        print(f"[ANALYSIS VIEW] Showing plot for {filename}")

        # Update selection
        self.plot_list.setCurrentRow(index)

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

                    # Copy lines
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
            info_text = f"<b>Dataset {index + 1} of {len(filenames)}</b>: {filename}<br>"
            if filename in self.current_fit_quality:
                quality = self.current_fit_quality[filename]
                info_text += f"<b>Fit Quality:</b> R² = {quality.get('R2', 'N/A')}"
                if 'residuals' in quality:
                    info_text += f", Residuals = {quality.get('residuals', 'N/A')}"
            self.plot_info_label.setText(info_text)

        # Update button states
        self.prev_plot_btn.setEnabled(index > 0)
        self.next_plot_btn.setEnabled(index < len(filenames) - 1)

    def _show_previous_plot(self):
        """Show previous plot"""
        if self.current_plot_index > 0:
            self._show_plot(self.current_plot_index - 1)

    def _show_next_plot(self):
        """Show next plot"""
        filenames = list(self.current_plots.keys())
        if self.current_plot_index < len(filenames) - 1:
            self._show_plot(self.current_plot_index + 1)

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

    def clear_results(self):
        """Clear all results"""
        self.results_table.setRowCount(0)
        self.details_text.clear()
        self.current_plots = {}
        self.nav_widget.hide()
        self.plot_placeholder.show()
