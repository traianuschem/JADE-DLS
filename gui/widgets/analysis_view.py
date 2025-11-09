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
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from matplotlib.figure import Figure

            self.figure = Figure(figsize=(8, 6))
            self.canvas = FigureCanvasQTAgg(self.figure)
            plot_layout.addWidget(self.canvas)
            self.plot_placeholder.hide()
        except ImportError:
            # Matplotlib not available, use placeholder
            pass

        plot_group.setLayout(plot_layout)
        layout.addWidget(plot_group)

        widget.setLayout(layout)
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
        results_layout.addWidget(self.results_table)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        # Details text
        details_group = QGroupBox("Details")
        details_layout = QVBoxLayout()

        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setMaximumHeight(150)
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
        if 'files' in self.pipeline.data:
            files = self.pipeline.data['files']
            num_files = len(files)

            stats_text = f"""
<b>Files Loaded:</b> {num_files}<br>
<b>Data Folder:</b> {self.pipeline.data.get('data_folder', 'N/A')}<br>
<b>Status:</b> Ready for analysis
"""
            self.stats_label.setText(stats_text)

            # Update files table
            self.files_table.setRowCount(min(num_files, 50))  # Limit display
            for i, filepath in enumerate(files[:50]):
                import os
                filename = os.path.basename(filepath)
                self.files_table.setItem(i, 0, QTableWidgetItem(filename))
                self.files_table.setItem(i, 1, QTableWidgetItem("-"))
                self.files_table.setItem(i, 2, QTableWidgetItem("-"))
                self.files_table.setItem(i, 3, QTableWidgetItem("✓ Loaded"))

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
