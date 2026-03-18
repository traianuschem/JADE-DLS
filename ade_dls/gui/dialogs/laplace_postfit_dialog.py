"""
Post-Fit Refinement Dialog for Laplace Methods (NNLS and Regularized)

Allows refinement of NNLS/Regularized analysis results after initial fitting:
- Adjust q² range for diffusion coefficient calculation
- Visual Γ vs q² plots with interactive range selection
- Exclude individual distribution plots based on quality
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QDoubleSpinBox, QFormLayout, QTabWidget,
                             QWidget, QGroupBox, QMessageBox, QSizePolicy,
                             QListWidget, QListWidgetItem, QCheckBox, QScrollArea,
                             QSplitter, QSpinBox, QComboBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QFileDialog)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


class InteractivePlotWidget(QWidget):
    """
    Widget with interactive Γ vs q² plot for range selection
    """

    def __init__(self, data, gamma_cols, q_squared_col, method_name, parent=None):
        super().__init__(parent)
        self.data = data
        self.gamma_cols = gamma_cols if isinstance(gamma_cols, list) else [gamma_cols]
        self.q_squared_col = q_squared_col
        self.method_name = method_name
        self.parent_dialog = parent

        # Range selection
        self.q_min = None
        self.q_max = None
        self.selection_rect = None
        self.click_start = None

        self.init_ui()

    def init_ui(self):
        """Initialize UI with matplotlib plot"""
        layout = QVBoxLayout()

        # Create matplotlib figure
        self.figure, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Plot initial data
        self.plot_data()

        # Connect mouse events for range selection
        self.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_release_event', self.on_mouse_release)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Instructions
        instructions = QLabel(
            "📌 <b>Interactive Selection:</b> Click and drag on the plot to select q² range. "
            "The spinboxes below will update automatically."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("padding: 5px; background-color: #FFF3E0; border-radius: 3px;")
        layout.addWidget(instructions)

        self.setLayout(layout)

    def plot_data(self):
        """Plot Γ vs q² data for all peaks"""
        self.ax.clear()

        # Extract data
        q_squared = self.data[self.q_squared_col].values

        # Color cycle
        colors = plt.cm.tab10(range(len(self.gamma_cols)))

        # Plot each gamma column (each peak)
        for i, gamma_col in enumerate(self.gamma_cols):
            if gamma_col in self.data.columns:
                gamma = self.data[gamma_col].values

                # Sort by q²
                sort_idx = np.argsort(q_squared)
                q_sorted = q_squared[sort_idx]
                gamma_sorted = gamma[sort_idx]

                # Plot data points
                self.ax.plot(q_sorted, gamma_sorted, 'o', markersize=8,
                            label=f'{gamma_col}', color=colors[i], alpha=0.7)

                # Simple linear regression for visualization
                if len(q_sorted) > 1:
                    coeffs = np.polyfit(q_sorted, gamma_sorted, 1)
                    fit_line = np.poly1d(coeffs)
                    q_fit = np.linspace(q_sorted.min(), q_sorted.max(), 100)
                    self.ax.plot(q_fit, fit_line(q_fit), '--', color=colors[i],
                                linewidth=2, alpha=0.6)

        self.ax.set_xlabel('q² [nm⁻²]', fontsize=11, fontweight='bold')
        self.ax.set_ylabel('Γ [s⁻¹]', fontsize=11, fontweight='bold')
        self.ax.set_title(f'{self.method_name}: Diffusion Coefficient Fit', fontsize=12, fontweight='bold')
        self.ax.legend(loc='best')
        self.ax.grid(True, alpha=0.3)

        self.canvas.draw()

    def on_mouse_press(self, event):
        """Handle mouse press event"""
        if event.inaxes == self.ax and event.button == 1:  # Left click
            self.click_start = event.xdata
            # Remove previous selection rectangle if exists
            if self.selection_rect:
                self.selection_rect.remove()
                self.selection_rect = None

    def on_mouse_move(self, event):
        """Handle mouse move event - show preview of selection"""
        if self.click_start is not None and event.inaxes == self.ax and event.xdata is not None:
            # Remove old rectangle
            if self.selection_rect:
                self.selection_rect.remove()

            # Draw new rectangle
            x_min = min(self.click_start, event.xdata)
            x_max = max(self.click_start, event.xdata)
            y_limits = self.ax.get_ylim()

            self.selection_rect = Rectangle(
                (x_min, y_limits[0]),
                x_max - x_min,
                y_limits[1] - y_limits[0],
                fill=True,
                facecolor='green',
                alpha=0.2,
                edgecolor='green',
                linewidth=2
            )
            self.ax.add_patch(self.selection_rect)
            self.canvas.draw()

    def on_mouse_release(self, event):
        """Handle mouse release event - finalize selection"""
        if self.click_start is not None and event.inaxes == self.ax and event.xdata is not None:
            # Calculate selected range
            self.q_min = min(self.click_start, event.xdata)
            self.q_max = max(self.click_start, event.xdata)

            # Update parent dialog spinboxes
            if hasattr(self.parent_dialog, 'update_q_range_from_plot'):
                self.parent_dialog.update_q_range_from_plot(self.q_min, self.q_max)

        self.click_start = None


class DistributionInspectorWidget(QWidget):
    """
    Widget to display and select distributions to exclude
    """

    def __init__(self, plots_dict, method_name, parent=None):
        super().__init__(parent)
        self.plots_dict = plots_dict  # {filename: (fig, data)}
        self.method_name = method_name
        self.excluded_files = []

        self.init_ui()

    def init_ui(self):
        """Initialize the distribution inspector UI"""
        layout = QVBoxLayout()

        # Info
        info_label = QLabel(
            f"<b>Select distributions to exclude</b><br>"
            f"Use arrow keys or mouse to navigate. Check the boxes to exclude distributions from analysis.<br>"
            f"Press Space to toggle exclusion for the selected distribution."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Create horizontal split: List on left, Plot on right
        splitter = QSplitter(Qt.Horizontal)

        # Left side: List with checkboxes
        left_widget = QWidget()
        left_layout = QVBoxLayout()

        # Create QListWidget for distribution list (better keyboard navigation)
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SingleSelection)

        # Filter and add items
        filenames = sorted(self.plots_dict.keys())
        self.valid_filenames = []

        # Debug: Print available plots
        print(f"[DistributionInspector] Available plots: {len(filenames)}")

        for filename in filenames:
            # Skip summary/diffusion analysis plots but keep individual distribution plots
            if 'Summary' in filename or 'Diffusion Analysis' in filename or 'Analysis' in filename:
                print(f"[DistributionInspector] Skipping summary plot: {filename}")
                continue  # Skip summary plots

            self.valid_filenames.append(filename)

            # Create list item with checkbox
            item = QListWidgetItem(filename)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.list_widget.addItem(item)

        print(f"[DistributionInspector] Created {len(self.valid_filenames)} items")

        # Connect selection changed signal
        self.list_widget.currentItemChanged.connect(self._on_selection_changed)
        self.list_widget.itemChanged.connect(self._update_count)

        left_layout.addWidget(self.list_widget)

        # Selection controls
        control_layout = QHBoxLayout()

        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all)
        control_layout.addWidget(select_all_btn)

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all)
        control_layout.addWidget(deselect_all_btn)

        control_layout.addStretch()

        # Count label
        self.count_label = QLabel("0 distributions selected for exclusion")
        control_layout.addWidget(self.count_label)

        left_layout.addLayout(control_layout)
        left_widget.setLayout(left_layout)

        # Right side: Plot display
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

        right_widget = QWidget()
        right_layout = QVBoxLayout()

        # Placeholder figure
        from matplotlib.figure import Figure
        self.current_figure = Figure(figsize=(8, 6))
        ax = self.current_figure.add_subplot(111)
        ax.text(0.5, 0.5, 'Click on a distribution to view its plot',
                ha='center', va='center', fontsize=12, color='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        self.canvas = FigureCanvas(self.current_figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)
        right_widget.setLayout(right_layout)

        # Add both sides to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)  # Left: 1 part
        splitter.setStretchFactor(1, 2)  # Right: 2 parts

        layout.addWidget(splitter)

        # Select first item by default
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

        self.setLayout(layout)

    def _on_selection_changed(self, current, previous):
        """Handle selection change in list widget"""
        if current is not None:
            filename = current.text()
            self._show_plot(filename)

    def _select_all(self):
        """Select all distributions for exclusion"""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Checked)

    def _deselect_all(self):
        """Deselect all distributions"""
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setCheckState(Qt.Unchecked)

    def _show_plot(self, filename):
        """Display the plot for the selected distribution"""
        if filename not in self.plots_dict:
            print(f"[DistributionInspector] Plot not found for: {filename}")
            return

        # Clear current figure
        self.current_figure.clear()

        # Get the stored figure (plots_dict contains tuples: (fig, data))
        plot_entry = self.plots_dict[filename]
        if isinstance(plot_entry, tuple):
            stored_fig, _ = plot_entry  # Unpack tuple
        else:
            stored_fig = plot_entry  # Handle case where it's just a figure

        # Copy axes from stored figure to current figure
        if hasattr(stored_fig, 'axes') and len(stored_fig.axes) > 0:
            for i, ax_src in enumerate(stored_fig.axes):
                # Create subplot in same position
                ax_dest = self.current_figure.add_subplot(len(stored_fig.axes), 1, i+1)

                # Copy plot contents
                for line in ax_src.get_lines():
                    ax_dest.plot(line.get_xdata(), line.get_ydata(),
                               color=line.get_color(),
                               linestyle=line.get_linestyle(),
                               linewidth=line.get_linewidth(),
                               label=line.get_label(),
                               marker=line.get_marker(),
                               markersize=line.get_markersize())

                # Copy axis labels and title
                ax_dest.set_xlabel(ax_src.get_xlabel())
                ax_dest.set_ylabel(ax_src.get_ylabel())
                ax_dest.set_title(ax_src.get_title())

                # Copy scale
                ax_dest.set_xscale(ax_src.get_xscale())
                ax_dest.set_yscale(ax_src.get_yscale())

                # Copy grid settings
                # Check if grid is enabled by inspecting gridlines visibility
                grid_visible = (len(ax_src.xaxis.get_gridlines()) > 0 and
                              ax_src.xaxis.get_gridlines()[0].get_visible())
                ax_dest.grid(grid_visible)

                # Copy legend if exists
                if ax_src.get_legend() is not None:
                    ax_dest.legend()

                # Copy text annotations
                for text in ax_src.texts:
                    # Properly handle bbox patch
                    bbox_dict = None
                    if text.get_bbox_patch():
                        bbox_patch = text.get_bbox_patch()
                        bbox_dict = dict(
                            boxstyle='round,pad=0.3',
                            facecolor=bbox_patch.get_facecolor(),
                            edgecolor=bbox_patch.get_edgecolor(),
                            alpha=bbox_patch.get_alpha() or 0.7
                        )

                    ax_dest.text(text.get_position()[0], text.get_position()[1],
                               text.get_text(),
                               transform=ax_dest.transData if text.get_transform() == ax_src.transData else ax_dest.transAxes,
                               fontsize=text.get_fontsize(),
                               color=text.get_color(),
                               ha=text.get_ha(),
                               va=text.get_va(),
                               bbox=bbox_dict)

        else:
            # Fallback: show message
            ax = self.current_figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Plot loaded: {filename}',
                   ha='center', va='center', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        try:
            self.current_figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw()

        print(f"[DistributionInspector] Displayed plot: {filename}")

    def _update_count(self, item=None):
        """Update the count of excluded distributions"""
        count = 0
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                count += 1
        self.count_label.setText(f"{count} distribution(s) selected for exclusion")

    def get_excluded_files(self):
        """Get list of files to exclude"""
        excluded = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                excluded.append(item.text())
        return excluded


class LaplacePostFitRefinementDialog(QDialog):
    """
    Dialog for post-fit refinement of Laplace analysis results (NNLS or Regularized)

    Provides two-stage refinement:
    - Tab 1: Q² range adjustment with interactive plot
    - Tab 2: Distribution exclusion
    """

    def __init__(self, analyzer, method_name, parent=None):
        """
        Initialize the post-fit refinement dialog

        Args:
            analyzer: LaplaceAnalyzer instance with existing results
            method_name: 'NNLS' or 'Regularized'
            parent: Parent widget
        """
        super().__init__(parent)
        self.analyzer = analyzer
        self.method_name = method_name
        self.is_nnls = (method_name == 'NNLS')

        # Store refinement parameters
        self.q_range = None
        self.excluded_files = []
        self._clustering_params = {}
        self._per_pop_ranges = {}

        # Pre-compute data and clustering metadata
        if self.is_nnls:
            self._data = getattr(analyzer, 'nnls_data', None)
            self._cluster_info = getattr(analyzer, 'nnls_cluster_info', None)
        else:
            self._data = getattr(analyzer, 'regularized_data', None)
            self._cluster_info = getattr(analyzer, 'regularized_cluster_info', None)

        self._pop_cols = (
            sorted([c for c in self._data.columns if c.startswith('gamma_pop')])
            if self._data is not None else []
        )
        self._pop_widgets = {}   # pop_num -> dict of widget refs

        self.setWindowTitle(f"{method_name} Post-Fit Refinement")
        self.setModal(True)
        self.resize(950, 750)

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()

        # Title and description
        title = QLabel(f"{self.method_name} Post-Fit Refinement")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        description = QLabel(
            f"Refine your {self.method_name} analysis results: adjust the fitting q² range, "
            "exclude distributions, re-configure clustering, or tune per-population q² ranges."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        # Create tabs
        self.tabs = QTabWidget()

        # Tab 1: Q² range adjustment (combined)
        self.tabs.addTab(self.create_q_range_tab(), "1. Fitting Range")

        # Tab 2: Distribution exclusion
        self.tabs.addTab(self.create_exclusion_tab(), "2. Distribution Selection")

        # Tab 3: Clustering settings
        self.tabs.addTab(self.create_clustering_tab(), "3. Clustering")

        # Per-population tabs (one per detected population)
        for i, col in enumerate(self._pop_cols):
            pop_num = i + 1
            self.tabs.addTab(self.create_population_tab(pop_num, col), f"Pop. {pop_num}")

        # SLS Analysis tab — only for Regularized (requires regularized_data)
        if not self.is_nnls:
            self.tabs.addTab(self.create_sls_tab(), "📡 SLS Analysis")

        layout.addWidget(self.tabs)

        # Buttons
        button_layout = QHBoxLayout()

        self.reset_btn = QPushButton("Reset All")
        self.reset_btn.clicked.connect(self.reset_all)
        button_layout.addWidget(self.reset_btn)

        button_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setDefault(True)
        self.apply_btn.clicked.connect(self.apply_refinement)
        self.apply_btn.setStyleSheet("""
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
        button_layout.addWidget(self.apply_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_q_range_tab(self):
        """Create q² range adjustment tab with interactive plot"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Get data for plotting
        if self.is_nnls:
            data = self.analyzer.nnls_data
        else:
            data = self.analyzer.regularized_data
        gamma_cols = [col for col in data.columns if col.startswith('gamma_')] if data is not None else []

        # Add interactive Γ vs q² plot
        if data is not None and gamma_cols:
            self.plot_widget = InteractivePlotWidget(
                data,
                gamma_cols,
                'q^2',
                self.method_name,
                self
            )
            layout.addWidget(self.plot_widget)

        # Q² range group
        q_range_group = QGroupBox("Q² Range for Diffusion Coefficient Fit")
        q_range_layout = QFormLayout()

        # Get current q² range from data
        if data is not None and 'q^2' in data.columns:
            q_squared_values = data['q^2'].values
            min_q = float(q_squared_values.min())
            max_q = float(q_squared_values.max())
        else:
            min_q = 0.0
            max_q = 0.01

        # Min Q²
        self.q_min = QDoubleSpinBox()
        self.q_min.setRange(0, 1)
        self.q_min.setValue(min_q)
        self.q_min.setDecimals(6)
        self.q_min.setSingleStep(0.0001)
        self.q_min.setSuffix(" nm⁻²")
        q_range_layout.addRow("Min Q²:", self.q_min)

        # Max Q²
        self.q_max = QDoubleSpinBox()
        self.q_max.setRange(0, 1)
        self.q_max.setValue(max_q)
        self.q_max.setDecimals(6)
        self.q_max.setSingleStep(0.0001)
        self.q_max.setSuffix(" nm⁻²")
        q_range_layout.addRow("Max Q²:", self.q_max)

        q_range_group.setLayout(q_range_layout)
        layout.addWidget(q_range_group)

        # Info
        info = QLabel(
            "<b>Effect:</b> Re-calculate diffusion coefficient using only data points "
            "within the specified q² range. Individual distribution fits remain unchanged."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 10px; background-color: #e3f2fd; border-radius: 4px;")
        layout.addWidget(info)

        tab.setLayout(layout)
        return tab

    def create_exclusion_tab(self):
        """Create distribution exclusion tab"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Get plots
        if self.is_nnls:
            plots_dict = self.analyzer.nnls_plots if hasattr(self.analyzer, 'nnls_plots') else {}
        else:
            plots_dict = self.analyzer.regularized_plots if hasattr(self.analyzer, 'regularized_plots') else {}

        if plots_dict:
            self.exclusion_widget = DistributionInspectorWidget(plots_dict, self.method_name, self)
            layout.addWidget(self.exclusion_widget)
        else:
            layout.addWidget(QLabel("No distribution plots available for inspection."))

        # Info
        info = QLabel(
            "<b>Effect:</b> Excluded distributions will be removed from the diffusion "
            "coefficient analysis. The q² vs Γ regression will be recalculated without them."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 10px; background-color: #e3f2fd; border-radius: 4px;")
        layout.addWidget(info)

        tab.setLayout(layout)
        return tab

    # ------------------------------------------------------------------ #
    #  Clustering tab                                                      #
    # ------------------------------------------------------------------ #

    def create_clustering_tab(self):
        """Create clustering settings tab with live preview (mirrors Method D)."""
        tab = QWidget()
        layout = QVBoxLayout()

        info = QLabel(
            "<b>Adjust clustering parameters</b> to control how decay rates are grouped into "
            "populations before diffusion coefficient fitting. "
            "Click <i>Refresh Preview</i> to see the effect without applying."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 8px; background-color: #e3f2fd; border-radius: 4px;")
        layout.addWidget(info)

        settings_group = QGroupBox("Clustering Settings")
        form = QFormLayout()

        self._cl_use = QCheckBox("Enable clustering")
        current_use = self._cluster_info.get('enable_clustering', True) if self._cluster_info else True
        self._cl_use.setChecked(current_use)
        form.addRow("", self._cl_use)

        self._cl_strategy = QComboBox()
        self._cl_strategy.addItems(["Hierarchical – Ward linkage", "Simple – gap-based"])
        current_strategy = self._cluster_info.get('clustering_strategy', 'silhouette_refined') if self._cluster_info else 'silhouette_refined'
        if 'simple' in current_strategy.lower():
            self._cl_strategy.setCurrentIndex(1)
        form.addRow("Method:", self._cl_strategy)

        self._cl_distance = QDoubleSpinBox()
        self._cl_distance.setRange(0.01, 10.0)
        self._cl_distance.setSingleStep(0.1)
        self._cl_distance.setDecimals(3)
        current_dist = self._cluster_info.get('distance_threshold', 2.0) if self._cluster_info else 2.0
        self._cl_distance.setValue(current_dist)
        self._cl_distance.setToolTip("Log-space distance threshold. Lower = more populations detected.")
        form.addRow("Distance threshold:", self._cl_distance)

        self._cl_silhouette = QCheckBox("Silhouette-based cluster refinement")
        self._cl_silhouette.setChecked('silhouette' in current_strategy.lower())
        form.addRow("", self._cl_silhouette)

        settings_group.setLayout(form)
        layout.addWidget(settings_group)

        refresh_btn = QPushButton("↺  Refresh Clustering Preview")
        refresh_btn.clicked.connect(self._refresh_clustering_preview)
        layout.addWidget(refresh_btn)

        self._cl_fig, self._cl_axes = plt.subplots(1, 2, figsize=(10, 4))
        self._cl_canvas = FigureCanvas(self._cl_fig)
        self._cl_canvas.setMinimumHeight(280)
        layout.addWidget(self._cl_canvas)

        self._cl_stats_label = QLabel(
            "Click '↺ Refresh Clustering Preview' to preview the clustering result."
        )
        self._cl_stats_label.setWordWrap(True)
        layout.addWidget(self._cl_stats_label)

        tab.setLayout(layout)
        return tab

    def _refresh_clustering_preview(self):
        """Re-run clustering with current dialog settings and update the preview plot."""
        if self._data is None:
            self._cl_stats_label.setText("No data available for preview.")
            return

        gamma_cols = sorted([
            c for c in self._data.columns
            if c.startswith('gamma_') and not c.startswith('gamma_pop')
        ])
        if not gamma_cols:
            self._cl_stats_label.setText("No gamma columns found in data for clustering preview.")
            return

        strategy_text = self._cl_strategy.currentText()
        if 'Ward' in strategy_text:
            strategy = 'silhouette_refined' if self._cl_silhouette.isChecked() else 'hierarchical'
        else:
            strategy = 'simple'

        try:
            from ade_dls.analysis.clustering import cluster_all_gammas
            preview_df, cluster_info = cluster_all_gammas(
                self._data.copy(),
                gamma_cols=gamma_cols,
                q_squared_col='q^2',
                enable_clustering=self._cl_use.isChecked(),
                normalize_by_q2=True,
                distance_threshold=self._cl_distance.value(),
                clustering_strategy=strategy,
                interactive=False,
                plot=False,
                experiment_name=self.method_name
            )
            self._draw_clustering_preview(preview_df, cluster_info)
        except Exception as e:
            self._cl_stats_label.setText(f"Preview failed: {e}")
            import traceback
            traceback.print_exc()

    def _draw_clustering_preview(self, df, cluster_info):
        """Draw the clustering preview on the embedded matplotlib figure."""
        for ax in self._cl_axes:
            ax.clear()

        pop_cols = sorted([c for c in df.columns if c.startswith('gamma_pop')])
        q2 = df['q^2'].values if 'q^2' in df.columns else None
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        stats_parts = []

        for i, col in enumerate(pop_cols):
            pop_num = i + 1
            vals = df[col].values
            if q2 is not None:
                mask = ~np.isnan(vals) & ~np.isnan(q2) & (q2 > 0)
                gv, q2v = vals[mask], q2[mask]
            else:
                mask = ~np.isnan(vals)
                gv = vals[mask]
                q2v = np.ones(mask.sum())

            color = colors[i % len(colors)]
            label = f"Pop. {pop_num}"

            self._cl_axes[0].scatter(q2v, gv, color=color, alpha=0.7, s=30, label=label)

            D_vals = gv / np.maximum(q2v, 1e-30)
            log_D = np.log10(np.maximum(D_vals, 1e-30))
            if len(log_D) > 1:
                self._cl_axes[1].hist(log_D, bins=min(20, len(log_D)), color=color, alpha=0.5, label=label)

            abundance = cluster_info.get(f'abundance_pop_{pop_num}', np.nan) if cluster_info else np.nan
            if not np.isnan(abundance):
                stats_parts.append(f"Pop. {pop_num}: {abundance*100:.1f}%")

        self._cl_axes[0].set_xlabel("q² [nm⁻²]")
        self._cl_axes[0].set_ylabel("Γ [s⁻¹]")
        self._cl_axes[0].set_title("Γ vs q² by Population")
        if pop_cols:
            self._cl_axes[0].legend(fontsize=8)

        self._cl_axes[1].set_xlabel("log₁₀(D [nm²/s])")
        self._cl_axes[1].set_ylabel("Count")
        self._cl_axes[1].set_title("D Distribution by Population")
        if pop_cols:
            self._cl_axes[1].legend(fontsize=8)

        self._cl_fig.tight_layout()
        self._cl_canvas.draw()

        n_pops = len(pop_cols)
        sil = cluster_info.get('silhouette_score', None) if cluster_info else None
        stats_str = f"Populations found: {n_pops}"
        if sil is not None and not (isinstance(sil, float) and np.isnan(sil)):
            stats_str += f"  |  Silhouette score: {sil:.3f}"
        if stats_parts:
            stats_str += "  |  " + "  ".join(stats_parts)
        self._cl_stats_label.setText(stats_str)

    # ------------------------------------------------------------------ #
    #  Per-population tabs                                                 #
    # ------------------------------------------------------------------ #

    def create_population_tab(self, pop_num, col):
        """Create a per-population q² range tab (mirrors Method D population tabs)."""
        tab = QWidget()
        layout = QVBoxLayout()

        if self._data is not None:
            plot_widget = InteractivePlotWidget(
                self._data, [col], 'q^2', f"Pop. {pop_num}", self
            )
            layout.addWidget(plot_widget)

        q_group = QGroupBox(f"Q² Range for Population {pop_num}")
        q_form = QFormLayout()

        enable_q = QCheckBox("Apply custom q² range for this population")
        q_form.addRow("", enable_q)

        q_min_sb = QDoubleSpinBox()
        q_min_sb.setRange(0, 1e9)
        q_min_sb.setDecimals(6)
        q_min_sb.setSuffix(" nm⁻²")

        q_max_sb = QDoubleSpinBox()
        q_max_sb.setRange(0, 1e9)
        q_max_sb.setDecimals(6)
        q_max_sb.setSuffix(" nm⁻²")

        if self._data is not None and 'q^2' in self._data.columns:
            q2_vals = self._data['q^2'].dropna()
            if not q2_vals.empty:
                q_min_sb.setValue(float(q2_vals.min()))
                q_max_sb.setValue(float(q2_vals.max()))

        q_min_sb.setEnabled(False)
        q_max_sb.setEnabled(False)

        def _toggle(checked, mn=q_min_sb, mx=q_max_sb):
            mn.setEnabled(checked)
            mx.setEnabled(checked)
        enable_q.toggled.connect(_toggle)

        q_form.addRow("Min q²:", q_min_sb)
        q_form.addRow("Max q²:", q_max_sb)
        q_group.setLayout(q_form)
        layout.addWidget(q_group)

        info = QLabel(
            "<b>Effect:</b> Recalculates the diffusion coefficient for this population "
            "using only data points within the specified q² range."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 8px; background-color: #e3f2fd; border-radius: 4px;")
        layout.addWidget(info)

        layout.addStretch()
        tab.setLayout(layout)

        self._pop_widgets[pop_num] = {
            'enable_q': enable_q,
            'q_min': q_min_sb,
            'q_max': q_max_sb,
        }
        return tab

    # ------------------------------------------------------------------ #

    def update_q_range_from_plot(self, q_min, q_max):
        """
        Update spinboxes when user selects range in plot

        Args:
            q_min: Minimum q² value
            q_max: Maximum q² value
        """
        self.q_min.setValue(q_min)
        self.q_max.setValue(q_max)

    def reset_all(self):
        """Reset all refinement parameters to defaults"""
        reply = QMessageBox.question(
            self,
            "Reset All",
            "Reset all refinement parameters to their default values?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # Get default q² range
            if self.is_nnls:
                data = self.analyzer.nnls_data
            else:
                data = self.analyzer.regularized_data

            if data is not None and 'q^2' in data.columns:
                q_squared_values = data['q^2'].values
                min_q = float(q_squared_values.min())
                max_q = float(q_squared_values.max())
            else:
                min_q = 0.0
                max_q = 0.01

            self.q_min.setValue(min_q)
            self.q_max.setValue(max_q)

            # Reset exclusions
            if hasattr(self, 'exclusion_widget'):
                self.exclusion_widget._deselect_all()

    def apply_refinement(self):
        """Apply refinement and recalculate results"""
        # Validate combined q² range
        # min > max is an error; min == max means "use full range" (no override)
        if self.q_min.value() > self.q_max.value():
            QMessageBox.warning(
                self,
                "Invalid Range",
                f"{self.method_name}: Minimum q² must be less than or equal to maximum q²."
            )
            return

        if self.q_min.value() < self.q_max.value():
            self.q_range = (self.q_min.value(), self.q_max.value())
        else:
            # min == max → no custom range
            self.q_range = None

        # Get excluded files
        if hasattr(self, 'exclusion_widget'):
            self.excluded_files = self.exclusion_widget.get_excluded_files()

        # Collect clustering params
        if hasattr(self, '_cl_use'):
            strategy_text = self._cl_strategy.currentText()
            if 'Ward' in strategy_text:
                strategy = 'silhouette_refined' if self._cl_silhouette.isChecked() else 'hierarchical'
            else:
                strategy = 'simple'
            self._clustering_params = {
                'use_clustering': self._cl_use.isChecked(),
                'clustering_strategy': strategy,
                'distance_threshold': self._cl_distance.value(),
            }

        # Collect per-population q² ranges
        self._per_pop_ranges = {}
        for pop_num, refs in self._pop_widgets.items():
            if refs['enable_q'].isChecked():
                q_min = refs['q_min'].value()
                q_max = refs['q_max'].value()
                if q_min >= q_max:
                    QMessageBox.warning(
                        self,
                        "Invalid Range",
                        f"Population {pop_num}: Minimum q² must be less than maximum q²."
                    )
                    return
                self._per_pop_ranges[pop_num] = (q_min, q_max)

        self.accept()

    def get_refinement_params(self):
        """
        Get refinement parameters

        Returns:
            dict with keys: q_range, excluded_files, clustering, per_pop_ranges
        """
        return {
            'q_range': self.q_range,
            'excluded_files': self.excluded_files,
            'clustering': self._clustering_params,
            'per_pop_ranges': self._per_pop_ranges,
        }

    # ------------------------------------------------------------------
    # SLS Analysis Tab (Regularized only)
    # ------------------------------------------------------------------

    def create_sls_tab(self):
        """Create the SLS analysis tab."""
        widget = QWidget()
        outer = QVBoxLayout()

        # ── Configuration ─────────────────────────────────────────────
        config_group = QGroupBox("Configuration")
        form = QFormLayout()

        n_auto = len(self.analyzer.regularized_final_results) \
            if self.analyzer.regularized_final_results is not None else 1
        self._sls_npop_spin = QSpinBox()
        self._sls_npop_spin.setRange(1, 4)
        self._sls_npop_spin.setValue(min(n_auto, 4))
        self._sls_npop_spin.setToolTip("Number of populations to analyse")
        form.addRow("Populations:", self._sls_npop_spin)

        q2_layout = QHBoxLayout()
        self._sls_q2_min = QDoubleSpinBox()
        self._sls_q2_min.setRange(0.0, 1000.0)
        self._sls_q2_min.setDecimals(6)
        self._sls_q2_min.setValue(0.0)
        self._sls_q2_min.setToolTip("Lower q² bound for Guinier fit (0 = no limit)")
        self._sls_q2_max = QDoubleSpinBox()
        self._sls_q2_max.setRange(0.0, 1000.0)
        self._sls_q2_max.setDecimals(6)
        self._sls_q2_max.setValue(0.0)
        self._sls_q2_max.setToolTip("Upper q² bound for Guinier fit (0 = no limit)")
        q2_layout.addWidget(QLabel("min:"))
        q2_layout.addWidget(self._sls_q2_min)
        q2_layout.addWidget(QLabel("max:"))
        q2_layout.addWidget(self._sls_q2_max)
        form.addRow("q² Guinier range [nm⁻²]:", q2_layout)

        self._sls_exponent_spin = QSpinBox()
        self._sls_exponent_spin.setRange(1, 10)
        self._sls_exponent_spin.setValue(6)
        self._sls_exponent_spin.setToolTip(
            "Rh exponent for number-weighting:\n"
            "6 = Rayleigh (compact spheres)\n"
            "5 = Daoud-Cotton (star polymers)"
        )
        form.addRow("Rh exponent:", self._sls_exponent_spin)

        self._sls_use_nw_cb = QCheckBox("Apply number-weighting correction")
        self._sls_use_nw_cb.setChecked(True)
        form.addRow("", self._sls_use_nw_cb)

        config_group.setLayout(form)
        outer.addWidget(config_group)

        # ── Action buttons ─────────────────────────────────────────────
        btn_layout = QHBoxLayout()

        self._sls_run_btn = QPushButton("▶ Run SLS Analysis")
        self._sls_run_btn.clicked.connect(self._sls_run_analysis)
        btn_layout.addWidget(self._sls_run_btn)

        self._sls_status_label = QLabel("Initialising intensity data…")
        self._sls_status_label.setStyleSheet("color: #666; font-style: italic;")
        btn_layout.addWidget(self._sls_status_label)
        btn_layout.addStretch()

        outer.addLayout(btn_layout)

        # Auto-load intensity from already-loaded base data
        self._sls_auto_load_intensity()

        # ── Guinier Plot ───────────────────────────────────────────────
        plot_group = QGroupBox("Guinier Plot")
        plot_layout = QVBoxLayout()

        try:
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _FC
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as _NT
            import matplotlib.pyplot as _plt

            self._sls_figure, self._sls_ax = _plt.subplots(figsize=(7, 4))
            self._sls_canvas = _FC(self._sls_figure)
            self._sls_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self._sls_nav = _NT(self._sls_canvas, widget)

            self._sls_ax.text(0.5, 0.5, "Run SLS Analysis to see the Guinier plot",
                              ha='center', va='center', fontsize=11, color='#aaa',
                              transform=self._sls_ax.transAxes)
            self._sls_ax.set_xticks([])
            self._sls_ax.set_yticks([])

            plot_layout.addWidget(self._sls_nav)
            plot_layout.addWidget(self._sls_canvas)
            self._sls_plot_available = True
        except Exception:
            self._sls_plot_available = False
            plot_layout.addWidget(QLabel(
                "Matplotlib Qt5 backend not available. "
                "Plot will be shown in a separate window."
            ))

        plot_group.setLayout(plot_layout)
        outer.addWidget(plot_group)

        # ── Summary Table ──────────────────────────────────────────────
        summary_group = QGroupBox("SLS Summary")
        summary_layout = QVBoxLayout()

        self._sls_summary_table = QTableWidget()
        self._sls_summary_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._sls_summary_table.setAlternatingRowColors(True)
        self._sls_summary_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        summary_layout.addWidget(self._sls_summary_table)

        export_csv_btn = QPushButton("💾 Export SLS Summary (CSV)")
        export_csv_btn.clicked.connect(self._sls_export_csv)
        summary_layout.addWidget(export_csv_btn)

        summary_group.setLayout(summary_layout)
        outer.addWidget(summary_group)

        widget.setLayout(outer)
        return widget

    def _sls_auto_load_intensity(self):
        """
        Auto-load intensity data from the files already loaded into df_basedata.

        Strategy (in order of preference):
          1. Already cached in analyzer.df_intensity → reuse
          2. df_basedata has a 'folder' column (added by data_loader >= this version)
          3. df_basedata has no 'folder' → fall back to a QFileDialog
        """
        import os

        # 1. Already loaded from a previous run
        if self.analyzer.df_intensity is not None:
            n = len(self.analyzer.df_intensity)
            self._sls_status_label.setText(f"✅ Intensity data ready ({n} records).")
            self._sls_status_label.setStyleSheet("color: green; font-style: normal;")
            return

        df_base = getattr(self.analyzer, 'df_basedata', None)
        if df_base is None or df_base.empty:
            self._sls_status_label.setText("⚠ No base data available.")
            self._sls_status_label.setStyleSheet("color: orange;")
            return

        # 2. Build file paths from df_basedata['folder'] + df_basedata['filename']
        file_paths = []
        if 'folder' in df_base.columns:
            for _, row in df_base.drop_duplicates(subset=['filename']).iterrows():
                folder = str(row.get('folder', '') or '')
                fname  = str(row.get('filename', '') or '')
                if folder and fname:
                    fp = os.path.join(folder, fname)
                    if os.path.isfile(fp):
                        file_paths.append(fp)

        # 3. Fallback: ask the user for the folder
        if not file_paths:
            self._sls_status_label.setText(
                "⚠ File paths not in base data — please select the data folder.")
            self._sls_status_label.setStyleSheet("color: orange;")
            folder = QFileDialog.getExistingDirectory(
                self,
                "Select folder with ALV .ASC files (intensity)",
                "",
                QFileDialog.ShowDirsOnly,
            )
            if not folder:
                return
            file_paths = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith('.asc')
            ]
            if not file_paths:
                self._sls_status_label.setText("⚠ No .ASC files found in selected folder.")
                return

        try:
            ok = self.analyzer.load_intensity_data(file_paths)
            if ok:
                n = len(self.analyzer.df_intensity)
                self._sls_status_label.setText(
                    f"✅ Intensity data loaded ({n} records from {len(file_paths)} files).")
                self._sls_status_label.setStyleSheet("color: green; font-style: normal;")
            else:
                self._sls_status_label.setText(
                    "⚠ Could not extract intensity data from the loaded files.")
                self._sls_status_label.setStyleSheet("color: orange;")
        except Exception as e:
            self._sls_status_label.setText(f"⚠ Error loading intensity: {e}")
            self._sls_status_label.setStyleSheet("color: red;")

    def _sls_run_analysis(self):
        """Slot: run SLS analysis and update plot + table."""
        n_pop = self._sls_npop_spin.value()
        q2_min = self._sls_q2_min.value()
        q2_max = self._sls_q2_max.value()
        exponent = self._sls_exponent_spin.value()
        use_nw = self._sls_use_nw_cb.isChecked()

        q2_range = None
        if q2_min > 0 or q2_max > 0:
            q2_range = (q2_min if q2_min > 0 else 0.0,
                        q2_max if q2_max > 0 else float('inf'))

        try:
            summary_df = self.analyzer.run_sls_analysis(
                n_populations=n_pop,
                q2_range=q2_range,
                exponent=exponent,
                use_nw=use_nw,
            )
            self._sls_update_plot()
            self._sls_populate_table(summary_df)
            self._sls_status_label.setText("✅ SLS analysis complete.")
        except RuntimeError as e:
            QMessageBox.warning(self, "SLS Analysis Error", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Unexpected Error", str(e))

    def _sls_update_plot(self):
        """Redraw the embedded Guinier plot."""
        from ade_dls.analysis.sls import plot_guinier

        if not self._sls_plot_available:
            plot_guinier(
                self.analyzer.guinier_results,
                total_result=self.analyzer.guinier_total,
            )
            return

        self._sls_ax.clear()
        plot_guinier(
            self.analyzer.guinier_results,
            total_result=self.analyzer.guinier_total,
            ax=self._sls_ax,
        )
        self._sls_figure.tight_layout()
        self._sls_canvas.draw()

    def _sls_populate_table(self, df):
        """Fill the SLS summary QTableWidget."""
        if df is None or df.empty:
            return

        self._sls_summary_table.setRowCount(len(df))
        self._sls_summary_table.setColumnCount(len(df.columns))
        self._sls_summary_table.setHorizontalHeaderLabels(list(df.columns))

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
                if col == 'qRg_max':
                    try:
                        if float(row[col]) > 1.3:
                            item.setBackground(QColor('#FFB6C1'))
                    except (TypeError, ValueError):
                        pass
                self._sls_summary_table.setItem(r, c, item)

    def _sls_export_csv(self):
        """Export SLS summary to CSV."""
        if self.analyzer.sls_summary is None:
            QMessageBox.warning(self, "No Data", "Run SLS Analysis first.")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export SLS Summary", "", "CSV Files (*.csv)")
        if filename:
            try:
                self.analyzer.sls_summary.to_csv(filename, index=False)
                QMessageBox.information(self, "Export Successful",
                                        f"SLS summary saved to:\n{filename}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))
