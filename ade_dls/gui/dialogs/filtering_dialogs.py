"""
Data Filtering Dialogs
Interactive dialogs for filtering countrates and correlations
"""

import numpy as np
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QListWidget, QListWidgetItem, QSplitter,
                             QMessageBox, QCheckBox, QGroupBox, QAbstractItemView,
                             QSpinBox, QFrame)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class CountrateFilterDialog(QDialog):
    """
    Dialog for visualizing and filtering countrate data

    Shows all countrate plots and allows user to select bad measurements
    for exclusion
    """

    def __init__(self, countrate_data, parent=None):
        super().__init__(parent)
        self.countrate_data = countrate_data
        self.original_data = countrate_data.copy()
        self.excluded_files = set()
        self.current_file = None

        self.setWindowTitle("Filter Count Rates")
        self.setModal(True)
        self.resize(1200, 700)

        self.init_ui()
        self.update_plot()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QHBoxLayout()

        # Left panel: File list
        left_panel = self.create_file_list_panel()

        # Right panel: Plot
        right_panel = self.create_plot_panel()

        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])

        layout.addWidget(splitter)
        self.setLayout(layout)

    def create_file_list_panel(self):
        """Create left panel with file list"""
        widget = QGroupBox("Files")
        layout = QVBoxLayout()

        # Info label
        info = QLabel(
            "Select files with bad count rates to exclude.\n"
            "Click on a file to view its count rate plot."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Keyboard shortcuts banner
        shortcuts_label = QLabel(
            "<b>Keyboard Shortcuts:</b><br>"
            "Ctrl+X = Toggle exclude/include current file"
        )
        shortcuts_label.setStyleSheet("background-color: #e8f4f8; padding: 8px; border: 1px solid #b8d4e8; border-radius: 4px;")
        shortcuts_label.setWordWrap(True)
        layout.addWidget(shortcuts_label)

        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        for filename in sorted(self.countrate_data.keys()):
            item = QListWidgetItem(filename)
            item.setData(Qt.UserRole, filename)
            self.file_list.addItem(item)

        self.file_list.currentItemChanged.connect(self.on_file_selected)
        layout.addWidget(self.file_list)

        # Statistics
        self.stats_label = QLabel(f"Total files: {len(self.countrate_data)}\nExcluded: 0")
        layout.addWidget(self.stats_label)

        # Mark as bad button
        mark_bad_btn = QPushButton("❌ Mark Selected as Bad")
        mark_bad_btn.clicked.connect(self.mark_selected_as_bad)
        layout.addWidget(mark_bad_btn)

        # Unmark button
        unmark_btn = QPushButton("✓ Unmark Selected")
        unmark_btn.clicked.connect(self.unmark_selected)
        layout.addWidget(unmark_btn)

        # Accept/Cancel buttons
        button_layout = QHBoxLayout()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        accept_btn = QPushButton("Apply Filter")
        accept_btn.setDefault(True)
        accept_btn.clicked.connect(self.accept)
        button_layout.addWidget(accept_btn)

        layout.addLayout(button_layout)

        widget.setLayout(layout)
        return widget

    def create_plot_panel(self):
        """Create right panel with plot"""
        widget = QGroupBox("Count Rate Plot")
        layout = QVBoxLayout()

        # Matplotlib figure
        self.figure = Figure(figsize=(10, 9))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        widget.setLayout(layout)
        return widget

    def on_file_selected(self, current, previous):
        """Handle file selection"""
        if current:
            filename = current.data(Qt.UserRole)
            self.current_file = filename
            self.update_plot()

    def update_plot(self):
        """Update the plot"""
        self.figure.clear()

        if self.current_file and self.current_file in self.countrate_data:
            df = self.countrate_data[self.current_file]

            ax1 = self.figure.add_subplot(211)
            ax2 = self.figure.add_subplot(212)

            detector_cols = [col for col in
                             ['detectorslot 1', 'detectorslot 2', 'detectorslot 3', 'detectorslot 4']
                             if col in df.columns and not df[col].isna().all()]

            if 'time [s]' in df.columns:
                time = df['time [s]'].values
                dt = np.mean(np.diff(time)) if len(time) > 1 else 1.0

                for col in detector_cols:
                    values = df[col].dropna().values
                    ax1.plot(time[:len(values)], values, label=col, alpha=0.7)

                    # FFT (DC-Anteil bei Index 0 weglassen)
                    N = len(values)
                    freqs = np.fft.rfftfreq(N, d=dt)
                    fft_mag = np.abs(np.fft.rfft(values))
                    ax2.semilogy(freqs[1:], fft_mag[1:], label=col, alpha=0.7)

            ax1.set_xlabel('Time [s]')
            ax1.set_ylabel('Count Rate [kHz]')
            ax1.set_title(f'Count Rate: {self.current_file}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.set_xlabel('Frequency [Hz]')
            ax2.set_ylabel('Magnitude [a.u.]')
            ax2.set_title('Frequency Spectrum (FFT)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Mark if excluded
            if self.current_file in self.excluded_files:
                ax1.text(0.5, 0.95, '❌ EXCLUDED',
                        transform=ax1.transAxes,
                        ha='center', va='top',
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
                        fontsize=14, fontweight='bold')

            self.figure.tight_layout()

        self.canvas.draw()

    def mark_selected_as_bad(self):
        """Mark selected files as bad"""
        selected_items = self.file_list.selectedItems()

        for item in selected_items:
            filename = item.data(Qt.UserRole)
            self.excluded_files.add(filename)
            item.setForeground(Qt.red)
            item.setText(f"❌ {filename}")

        self.update_stats()
        self.update_plot()

    def unmark_selected(self):
        """Unmark selected files"""
        selected_items = self.file_list.selectedItems()

        for item in selected_items:
            filename = item.data(Qt.UserRole)
            if filename in self.excluded_files:
                self.excluded_files.remove(filename)
            item.setForeground(Qt.black)
            item.setText(filename)

        self.update_stats()
        self.update_plot()

    def update_stats(self):
        """Update statistics label"""
        total = len(self.countrate_data)
        excluded = len(self.excluded_files)
        remaining = total - excluded

        self.stats_label.setText(
            f"Total files: {total}\n"
            f"Excluded: {excluded}\n"
            f"Remaining: {remaining}"
        )

    def toggle_exclude_current(self):
        """Toggle exclusion status of current file"""
        current_item = self.file_list.currentItem()
        if current_item:
            filename = current_item.data(Qt.UserRole)
            if filename in self.excluded_files:
                # Unmark
                self.excluded_files.remove(filename)
                current_item.setForeground(Qt.black)
                current_item.setText(filename)
            else:
                # Mark as bad
                self.excluded_files.add(filename)
                current_item.setForeground(Qt.red)
                current_item.setText(f"❌ {filename}")

            self.update_stats()
            self.update_plot()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        # Ctrl+X: Toggle exclude/include current file
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_X:
            self.toggle_exclude_current()
        else:
            super().keyPressEvent(event)

    def get_filtered_data(self):
        """Get filtered countrate data (excluding marked files)"""
        return {k: v for k, v in self.countrate_data.items()
                if k not in self.excluded_files}


class CorrelationFilterDialog(QDialog):
    """
    Dialog for visualizing and filtering correlation data

    Shows all correlation plots and allows user to select bad measurements
    for exclusion
    """

    def __init__(self, correlation_data, parent=None):
        super().__init__(parent)
        self.correlation_data = correlation_data
        self.original_data = correlation_data.copy()
        self.excluded_files = set()
        self.current_file = None

        # Noise correction state (off by default)
        self.noise_correction_active = False
        self.noise_params = {
            'baseline_correction': False,
            'baseline_pct': 5,
            'intercept_correction': False,
            'intercept_pct': 2,
        }
        self._noise_preview_active = False  # True while showing before/after preview

        self.setWindowTitle("Filter Correlations")
        self.setModal(True)
        self.resize(1200, 800)

        self.init_ui()
        self.update_plot()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QHBoxLayout()

        # Left panel: File list
        left_panel = self.create_file_list_panel()

        # Right panel: Plot
        right_panel = self.create_plot_panel()

        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])

        layout.addWidget(splitter)
        self.setLayout(layout)

    def create_file_list_panel(self):
        """Create left panel with file list"""
        widget = QGroupBox("Files")
        layout = QVBoxLayout()

        # Info label
        info = QLabel(
            "Select files with bad correlations to exclude.\n"
            "Click on a file to view its correlation plot."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Keyboard shortcuts banner
        shortcuts_label = QLabel(
            "<b>Keyboard Shortcuts:</b><br>"
            "Ctrl+X = Toggle exclude/include current file"
        )
        shortcuts_label.setStyleSheet("background-color: #e8f4f8; padding: 8px; border: 1px solid #b8d4e8; border-radius: 4px;")
        shortcuts_label.setWordWrap(True)
        layout.addWidget(shortcuts_label)

        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)

        for filename in sorted(self.correlation_data.keys()):
            item = QListWidgetItem(filename)
            item.setData(Qt.UserRole, filename)
            self.file_list.addItem(item)

        self.file_list.currentItemChanged.connect(self.on_file_selected)
        layout.addWidget(self.file_list)

        # Statistics
        self.stats_label = QLabel(f"Total files: {len(self.correlation_data)}\nExcluded: 0")
        layout.addWidget(self.stats_label)

        # Mark as bad button
        mark_bad_btn = QPushButton("❌ Mark Selected as Bad")
        mark_bad_btn.clicked.connect(self.mark_selected_as_bad)
        layout.addWidget(mark_bad_btn)

        # Unmark button
        unmark_btn = QPushButton("✓ Unmark Selected")
        unmark_btn.clicked.connect(self.unmark_selected)
        layout.addWidget(unmark_btn)

        # --- Noise Reduction Section ---
        noise_group = QGroupBox("Noise Reduction")
        noise_layout = QVBoxLayout()
        noise_layout.setSpacing(4)

        # Baseline correction row
        baseline_row = QHBoxLayout()
        self.baseline_cb = QCheckBox("Baseline correction")
        self.baseline_cb.setChecked(False)
        self.baseline_pct_spin = QSpinBox()
        self.baseline_pct_spin.setRange(1, 20)
        self.baseline_pct_spin.setValue(5)
        self.baseline_pct_spin.setSuffix(" %")
        self.baseline_pct_spin.setFixedWidth(60)
        baseline_row.addWidget(self.baseline_cb)
        baseline_row.addStretch()
        baseline_row.addWidget(self.baseline_pct_spin)
        noise_layout.addLayout(baseline_row)

        # Intercept correction row
        intercept_row = QHBoxLayout()
        self.intercept_cb = QCheckBox("Intercept correction")
        self.intercept_cb.setChecked(False)
        self.intercept_pct_spin = QSpinBox()
        self.intercept_pct_spin.setRange(1, 10)
        self.intercept_pct_spin.setValue(2)
        self.intercept_pct_spin.setSuffix(" %")
        self.intercept_pct_spin.setFixedWidth(60)
        intercept_row.addWidget(self.intercept_cb)
        intercept_row.addStretch()
        intercept_row.addWidget(self.intercept_pct_spin)
        noise_layout.addLayout(intercept_row)

        # Preview / Apply / Undo buttons
        noise_btn_row = QHBoxLayout()
        preview_btn = QPushButton("Preview")
        preview_btn.setToolTip("Show before/after comparison in plot")
        preview_btn.clicked.connect(self.on_preview_noise)
        self.apply_noise_btn = QPushButton("Apply")
        self.apply_noise_btn.setToolTip("Apply noise correction to all files")
        self.apply_noise_btn.clicked.connect(self.on_apply_noise)
        self.undo_noise_btn = QPushButton("Undo")
        self.undo_noise_btn.setToolTip("Revert noise correction")
        self.undo_noise_btn.setEnabled(False)
        self.undo_noise_btn.clicked.connect(self.on_undo_noise)
        noise_btn_row.addWidget(preview_btn)
        noise_btn_row.addWidget(self.apply_noise_btn)
        noise_btn_row.addWidget(self.undo_noise_btn)
        noise_layout.addLayout(noise_btn_row)

        # Status label
        self.noise_status_label = QLabel("Status: Not applied")
        self.noise_status_label.setStyleSheet("font-style: italic; color: gray;")
        noise_layout.addWidget(self.noise_status_label)

        noise_group.setLayout(noise_layout)
        layout.addWidget(noise_group)

        # Accept/Cancel buttons
        button_layout = QHBoxLayout()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        accept_btn = QPushButton("Apply Filter")
        accept_btn.setDefault(True)
        accept_btn.clicked.connect(self.accept)
        button_layout.addWidget(accept_btn)

        layout.addLayout(button_layout)

        widget.setLayout(layout)
        return widget

    def create_plot_panel(self):
        """Create right panel with plot"""
        widget = QGroupBox("Correlation Plot (normalized g²-1)")
        layout = QVBoxLayout()

        # Matplotlib figure
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        widget.setLayout(layout)
        return widget

    def on_file_selected(self, current, previous):
        """Handle file selection"""
        if current:
            filename = current.data(Qt.UserRole)
            self.current_file = filename
            self.update_plot()

    def update_plot(self):
        """Update the plot. If noise correction is active, overlay the corrected mean curve."""
        self._noise_preview_active = False
        self.figure.clear()

        if self.current_file and self.current_file in self.correlation_data:
            # Plot single file
            ax = self.figure.add_subplot(111)
            df = self.correlation_data[self.current_file]

            # Raw ALV .asc data stores g(2)-1 directly (decays from ~β to 0).
            # process_correlation_data() uses only correlation 1 + 2 (the two detectors).
            # Channels 3 and 4 may be all-zero (unused) — skip those.
            if 'time [ms]' in df.columns:
                for col in ['correlation 1', 'correlation 2', 'correlation 3', 'correlation 4']:
                    if col in df.columns:
                        vals = df[col].values
                        # Skip channels that are all-NaN or all-zero (unused ALV channels)
                        if not np.all(np.isnan(vals)) and not np.all(vals == 0):
                            ax.semilogx(df['time [ms]'], vals, label=col, alpha=0.7)

            # If noise correction is applied, overlay the corrected mean as dashed line
            if self.noise_correction_active:
                t, g2 = self._get_raw_g2(self.current_file)  # already g(2)-1
                if g2 is not None:
                    g2_corr = self._apply_noise_inline(
                        g2,
                        baseline_pct=self.noise_params['baseline_pct'],
                        intercept_pct=self.noise_params['intercept_pct'],
                        do_baseline=self.noise_params['baseline_correction'],
                        do_intercept=self.noise_params['intercept_correction'],
                    )
                    ax.semilogx(t, g2_corr, 'k--', linewidth=1.5,
                                label='Noise-corrected (mean)', alpha=0.9)

            ax.set_xlabel('Time [ms]')
            ax.set_ylabel('g\u00b2 - 1')
            ax.set_title(f'Correlation: {self.current_file}')
            ax.legend()
            ax.grid(True, alpha=0.3, which='both')
            ax.set_ylim(bottom=-0.1)

            # Mark if excluded
            if self.current_file in self.excluded_files:
                ax.text(0.5, 0.95, '\u274c EXCLUDED',
                       transform=ax.transAxes,
                       ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='red', alpha=0.5),
                       fontsize=14, fontweight='bold')

        self.canvas.draw()

    def mark_selected_as_bad(self):
        """Mark selected files as bad"""
        selected_items = self.file_list.selectedItems()

        for item in selected_items:
            filename = item.data(Qt.UserRole)
            self.excluded_files.add(filename)
            item.setForeground(Qt.red)
            item.setText(f"❌ {filename}")

        self.update_stats()
        self.update_plot()

    def unmark_selected(self):
        """Unmark selected files"""
        selected_items = self.file_list.selectedItems()

        for item in selected_items:
            filename = item.data(Qt.UserRole)
            if filename in self.excluded_files:
                self.excluded_files.remove(filename)
            item.setForeground(Qt.black)
            item.setText(filename)

        self.update_stats()
        self.update_plot()

    def update_stats(self):
        """Update statistics label"""
        total = len(self.correlation_data)
        excluded = len(self.excluded_files)
        remaining = total - excluded

        self.stats_label.setText(
            f"Total files: {total}\n"
            f"Excluded: {excluded}\n"
            f"Remaining: {remaining}"
        )

    # --- Noise Reduction Methods ---

    def _get_raw_g2(self, filename):
        """Return (time_ms, g2_values) for preview.

        Mirrors process_correlation_data() EXACTLY:
            g(2)-1 = (correlation 1 + correlation 2) / 2
        Only the first two detector channels are averaged — exactly as
        JADE 2.0 preprocessing.py does. The raw ALV values are already
        in the g(2)-1 convention (decay from ~β to 0).
        """
        df = self.correlation_data[filename]
        if 'correlation 1' not in df.columns or 'correlation 2' not in df.columns:
            return None, None
        # Exact replication of process_correlation_data():
        g2 = (df['correlation 1'] + df['correlation 2']).values / 2.0
        t = df['time [ms]'].values
        return t, g2

    def _apply_noise_inline(self, g2, baseline_pct, intercept_pct, do_baseline, do_intercept):
        """Apply noise corrections inline on a g(2)-1 numpy array.

        Identical logic to JADE 2.0 apply_noise_corrections():
        1. Baseline: subtract mean of last baseline_pct% of points
        2. Intercept: replace first intercept_pct% of points with their mean
        Baseline is applied first so intercept correction acts on shifted data.
        """
        g2c = g2.copy()
        n = len(g2c)
        if do_baseline:
            n_base = max(1, int(np.floor(n * baseline_pct / 100)))
            baseline_mean = g2c[-n_base:].mean()
            g2c = g2c - baseline_mean
        if do_intercept:
            n_int = max(1, int(np.floor(n * intercept_pct / 100)))
            intercept_mean = g2c[:n_int].mean()
            g2c[:n_int] = intercept_mean
        return g2c

    def on_preview_noise(self):
        """Show before/after noise correction in the plot canvas for the selected file."""
        if not self.current_file:
            QMessageBox.information(self, "No file selected", "Please select a file from the list first.")
            return

        t, g2 = self._get_raw_g2(self.current_file)
        if t is None:
            return

        g2_corr = self._apply_noise_inline(
            g2,
            baseline_pct=self.baseline_pct_spin.value(),
            intercept_pct=self.intercept_pct_spin.value(),
            do_baseline=self.baseline_cb.isChecked(),
            do_intercept=self.intercept_cb.isChecked(),
        )

        self._noise_preview_active = True
        self.figure.clear()
        # Shared y-axis so both panels are directly comparable
        ax1 = self.figure.add_subplot(1, 2, 1)
        ax2 = self.figure.add_subplot(1, 2, 2, sharey=ax1)

        ax1.semilogx(t, g2, color='steelblue', linewidth=1.5)
        ax1.set_xlabel('Time [ms]')
        ax1.set_ylabel('g\u00b2 - 1')
        ax1.set_title('Original')
        ax1.grid(True, alpha=0.3, which='both')
        # Leave matplotlib to auto-scale; just ensure zero is visible
        ax1.set_ylim(bottom=min(-0.05, g2.min() - 0.05 * abs(g2.max())))

        ax2.semilogx(t, g2_corr, color='darkorange', linewidth=1.5)
        ax2.set_xlabel('Time [ms]')
        ax2.set_title('After Noise Correction')
        ax2.grid(True, alpha=0.3, which='both')

        # Mark active corrections in title
        active = []
        if self.baseline_cb.isChecked():
            active.append(f'baseline -{self.baseline_pct_spin.value()}%')
        if self.intercept_cb.isChecked():
            active.append(f'intercept {self.intercept_pct_spin.value()}%')
        correction_str = ', '.join(active) if active else 'none selected'
        self.figure.suptitle(
            f'Noise Preview: {self.current_file}\n({correction_str})',
            fontsize=9
        )
        self.figure.tight_layout()
        self.canvas.draw()

    def on_apply_noise(self):
        """Save noise correction parameters and mark as active."""
        self.noise_correction_active = True
        self.noise_params = {
            'baseline_correction': self.baseline_cb.isChecked(),
            'baseline_pct': self.baseline_pct_spin.value(),
            'intercept_correction': self.intercept_cb.isChecked(),
            'intercept_pct': self.intercept_pct_spin.value(),
        }
        self.noise_status_label.setText("Status: Applied \u2713")
        self.noise_status_label.setStyleSheet("font-style: italic; color: green;")
        self.apply_noise_btn.setEnabled(False)
        self.undo_noise_btn.setEnabled(True)

    def on_undo_noise(self):
        """Revert noise correction."""
        self.noise_correction_active = False
        self.noise_status_label.setText("Status: Not applied")
        self.noise_status_label.setStyleSheet("font-style: italic; color: gray;")
        self.apply_noise_btn.setEnabled(True)
        self.undo_noise_btn.setEnabled(False)
        self._noise_preview_active = False
        self.update_plot()

    def get_noise_params(self):
        """Return (active: bool, params: dict) for use in main_window after preprocessing."""
        return self.noise_correction_active, self.noise_params.copy()

    def toggle_exclude_current(self):
        """Toggle exclusion status of current file"""
        current_item = self.file_list.currentItem()
        if current_item:
            filename = current_item.data(Qt.UserRole)
            if filename in self.excluded_files:
                # Unmark
                self.excluded_files.remove(filename)
                current_item.setForeground(Qt.black)
                current_item.setText(filename)
            else:
                # Mark as bad
                self.excluded_files.add(filename)
                current_item.setForeground(Qt.red)
                current_item.setText(f"❌ {filename}")

            self.update_stats()
            self.update_plot()

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        # Ctrl+X: Toggle exclude/include current file
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_X:
            self.toggle_exclude_current()
        else:
            super().keyPressEvent(event)

    def get_filtered_data(self):
        """Get filtered correlation data (excluding marked files)"""
        return {k: v for k, v in self.correlation_data.items()
                if k not in self.excluded_files}
