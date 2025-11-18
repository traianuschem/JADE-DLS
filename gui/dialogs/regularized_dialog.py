"""
Regularized NNLS Parameter Dialog with Interactive Preview
Allows users to configure Tikhonov-Phillips regularization parameters
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QLineEdit, QPushButton, QCheckBox,
                             QSlider, QSpinBox, QDoubleSpinBox, QTabWidget,
                             QWidget, QMessageBox, QSizePolicy, QComboBox)
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class RegularizedDialog(QDialog):
    """
    Dialog for Regularized NNLS parameter configuration

    Features:
        - Configure decay time range
        - Alpha parameter with optional optimization
        - Peak detection parameters with live preview
        - Advanced regularization options
    """

    parameters_accepted = pyqtSignal(dict)

    def __init__(self, laplace_analyzer, parent=None):
        super().__init__(parent)
        self.laplace_analyzer = laplace_analyzer

        # Default parameters
        self.params = {
            'decay_times': np.logspace(-8, 1, 200),
            'alpha': 1.0,
            'prominence': 0.05,
            'distance': 1,
            'normalize': True,
            'sparsity_penalty': 0.0,
            'enforce_unimodality': False,
            'num_preview': 5
        }

        # Storage for preview
        self.preview_figure = None
        self.preview_datasets = []

        self.init_ui()
        self.setWindowTitle("Regularized NNLS Configuration")
        self.resize(1200, 800)

    def init_ui(self):
        """Initialize the UI"""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Regularized NNLS with Tikhonov-Phillips Regularization")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        # Tabs for different sections
        tabs = QTabWidget()

        # Tab 1: Basic Parameters
        basic_tab = self.create_basic_parameters_tab()
        tabs.addTab(basic_tab, "Basic Parameters")

        # Tab 2: Peak Detection
        peak_tab = self.create_peak_detection_tab()
        tabs.addTab(peak_tab, "Peak Detection")

        # Tab 3: Advanced Options
        advanced_tab = self.create_advanced_tab()
        tabs.addTab(advanced_tab, "Advanced Options")

        # Tab 4: Preview
        preview_tab = self.create_preview_tab()
        tabs.addTab(preview_tab, "Interactive Preview")

        layout.addWidget(tabs)

        # Bottom buttons
        button_layout = QHBoxLayout()

        self.run_btn = QPushButton("Run Regularized Analysis")
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.run_btn.clicked.connect(self.accept_parameters)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(self.run_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def create_basic_parameters_tab(self):
        """Create basic parameters tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Decay Times Group
        decay_group = QGroupBox("Decay Time Range")
        decay_layout = QVBoxLayout()

        # Start
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Start (seconds):"))
        self.decay_start_spin = QDoubleSpinBox()
        self.decay_start_spin.setDecimals(10)
        self.decay_start_spin.setRange(1e-10, 1)
        self.decay_start_spin.setValue(1e-8)
        self.decay_start_spin.setSingleStep(1e-9)
        self.decay_start_spin.setStyleSheet("QDoubleSpinBox { min-width: 150px; }")
        start_layout.addWidget(self.decay_start_spin)
        start_layout.addWidget(QLabel("(e.g., 1e-8)"))
        start_layout.addStretch()
        decay_layout.addLayout(start_layout)

        # End
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("End (seconds):"))
        self.decay_end_spin = QDoubleSpinBox()
        self.decay_end_spin.setDecimals(2)
        self.decay_end_spin.setRange(1e-6, 100)
        self.decay_end_spin.setValue(10)
        self.decay_end_spin.setSingleStep(1)
        self.decay_end_spin.setStyleSheet("QDoubleSpinBox { min-width: 150px; }")
        end_layout.addWidget(self.decay_end_spin)
        end_layout.addWidget(QLabel("(e.g., 10)"))
        end_layout.addStretch()
        decay_layout.addLayout(end_layout)

        # Number of points
        num_layout = QHBoxLayout()
        num_layout.addWidget(QLabel("Number of points:"))
        self.decay_num_spin = QSpinBox()
        self.decay_num_spin.setRange(50, 1000)
        self.decay_num_spin.setValue(200)
        self.decay_num_spin.setSingleStep(10)
        num_layout.addWidget(self.decay_num_spin)
        num_layout.addWidget(QLabel("(logarithmic spacing)"))
        num_layout.addStretch()
        decay_layout.addLayout(num_layout)

        decay_group.setLayout(decay_layout)
        layout.addWidget(decay_group)

        # Alpha Parameter Group
        alpha_group = QGroupBox("Regularization Parameter (Alpha)")
        alpha_layout = QVBoxLayout()

        alpha_info = QLabel(
            "Alpha controls the smoothness of the distribution:\n"
            "• Low alpha (< 0.1): Less smoothing, more detail (may be noisy)\n"
            "• Medium alpha (0.1 - 1): Balanced smoothing (recommended)\n"
            "• High alpha (> 1): Strong smoothing (may lose detail)"
        )
        alpha_info.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        alpha_info.setWordWrap(True)
        alpha_layout.addWidget(alpha_info)

        alpha_input_layout = QHBoxLayout()
        alpha_input_layout.addWidget(QLabel("Alpha value:"))

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setDecimals(4)
        self.alpha_spin.setRange(0.0001, 100.0)
        self.alpha_spin.setValue(1.0)
        self.alpha_spin.setSingleStep(0.1)
        self.alpha_spin.setStyleSheet("QDoubleSpinBox { min-width: 120px; }")
        alpha_input_layout.addWidget(self.alpha_spin)

        self.find_alpha_btn = QPushButton("🔍 Find Optimal Alpha")
        self.find_alpha_btn.setToolTip("Run alpha analysis to find optimal value (may take a few minutes)")
        self.find_alpha_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 8px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.find_alpha_btn.clicked.connect(self.find_optimal_alpha)
        alpha_input_layout.addWidget(self.find_alpha_btn)

        alpha_input_layout.addStretch()
        alpha_layout.addLayout(alpha_input_layout)

        alpha_group.setLayout(alpha_layout)
        layout.addWidget(alpha_group)

        # Processing Options
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout()

        self.show_plots_check = QCheckBox("Show individual plots during processing")
        self.show_plots_check.setChecked(True)
        options_layout.addWidget(self.show_plots_check)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_peak_detection_tab(self):
        """Create peak detection parameters tab with interactive sliders"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Info
        info = QLabel(
            "Adjust peak detection parameters and use the preview to see the effect.\n"
            "These parameters work the same as in NNLS analysis."
        )
        info.setStyleSheet("color: #666; font-style: italic; padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Prominence
        prom_group = QGroupBox("Peak Prominence")
        prom_layout = QVBoxLayout()

        prom_info = QLabel("Prominence: Minimum height of peaks relative to baseline.\n"
                          "Lower values = more sensitive (finds smaller peaks)")
        prom_info.setWordWrap(True)
        prom_info.setStyleSheet("color: #666; font-style: italic;")
        prom_layout.addWidget(prom_info)

        prom_slider_layout = QHBoxLayout()
        prom_slider_layout.addWidget(QLabel("Prominence:"))

        self.prominence_slider = QSlider(Qt.Horizontal)
        self.prominence_slider.setRange(1, 500)  # 0.001 to 0.5 (scaled by 1000)
        self.prominence_slider.setValue(50)  # 0.05
        self.prominence_slider.setTickPosition(QSlider.TicksBelow)
        self.prominence_slider.setTickInterval(50)
        self.prominence_slider.valueChanged.connect(self.on_prominence_slider_changed)
        prom_slider_layout.addWidget(self.prominence_slider)

        # Manual input field
        self.prominence_input = QDoubleSpinBox()
        self.prominence_input.setDecimals(3)
        self.prominence_input.setRange(0.001, 0.5)
        self.prominence_input.setValue(0.05)
        self.prominence_input.setSingleStep(0.001)
        self.prominence_input.setMinimumWidth(80)
        self.prominence_input.valueChanged.connect(self.on_prominence_input_changed)
        prom_slider_layout.addWidget(self.prominence_input)

        prom_layout.addLayout(prom_slider_layout)
        prom_group.setLayout(prom_layout)
        layout.addWidget(prom_group)

        # Distance
        dist_group = QGroupBox("Peak Distance")
        dist_layout = QVBoxLayout()

        dist_info = QLabel("Distance: Minimum separation between peaks (in index points).\n"
                          "Higher values = fewer peaks (forces peaks to be further apart)")
        dist_info.setWordWrap(True)
        dist_info.setStyleSheet("color: #666; font-style: italic;")
        dist_layout.addWidget(dist_info)

        dist_slider_layout = QHBoxLayout()
        dist_slider_layout.addWidget(QLabel("Distance:"))

        self.distance_slider = QSlider(Qt.Horizontal)
        self.distance_slider.setRange(1, 50)
        self.distance_slider.setValue(1)
        self.distance_slider.setTickPosition(QSlider.TicksBelow)
        self.distance_slider.setTickInterval(5)
        self.distance_slider.valueChanged.connect(self.on_distance_slider_changed)
        dist_slider_layout.addWidget(self.distance_slider)

        # Manual input field
        self.distance_input = QSpinBox()
        self.distance_input.setRange(1, 50)
        self.distance_input.setValue(1)
        self.distance_input.setSingleStep(1)
        self.distance_input.setMinimumWidth(80)
        self.distance_input.valueChanged.connect(self.on_distance_input_changed)
        dist_slider_layout.addWidget(self.distance_input)

        dist_layout.addLayout(dist_slider_layout)
        dist_group.setLayout(dist_layout)
        layout.addWidget(dist_group)

        # Quick presets
        preset_group = QGroupBox("Quick Presets")
        preset_layout = QHBoxLayout()

        monodisperse_btn = QPushButton("Monodisperse\n(1 peak)")
        monodisperse_btn.clicked.connect(lambda: self.apply_preset('monodisperse'))
        preset_layout.addWidget(monodisperse_btn)

        bidisperse_btn = QPushButton("Bidisperse\n(2 peaks)")
        bidisperse_btn.clicked.connect(lambda: self.apply_preset('bidisperse'))
        preset_layout.addWidget(bidisperse_btn)

        polydisperse_btn = QPushButton("Polydisperse\n(multiple peaks)")
        polydisperse_btn.clicked.connect(lambda: self.apply_preset('polydisperse'))
        preset_layout.addWidget(polydisperse_btn)

        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_advanced_tab(self):
        """Create advanced options tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Regularization Options
        reg_group = QGroupBox("Regularization Options")
        reg_layout = QVBoxLayout()

        self.normalize_check = QCheckBox("Normalize distribution (sum = 1)")
        self.normalize_check.setChecked(True)
        self.normalize_check.setToolTip("Ensures that the distribution integrates to 1")
        reg_layout.addWidget(self.normalize_check)

        self.unimodal_check = QCheckBox("Enforce unimodality (single peak)")
        self.unimodal_check.setChecked(False)
        self.unimodal_check.setToolTip("Forces distribution to have only one peak (for monodisperse samples)")
        reg_layout.addWidget(self.unimodal_check)

        sparsity_layout = QHBoxLayout()
        sparsity_layout.addWidget(QLabel("Sparsity penalty:"))
        self.sparsity_spin = QDoubleSpinBox()
        self.sparsity_spin.setDecimals(3)
        self.sparsity_spin.setRange(0.0, 1.0)
        self.sparsity_spin.setValue(0.0)
        self.sparsity_spin.setSingleStep(0.01)
        self.sparsity_spin.setToolTip("L1 penalty to promote sparse solutions (0 = off)")
        sparsity_layout.addWidget(self.sparsity_spin)
        sparsity_layout.addWidget(QLabel("(0 = off, >0 = sparse)"))
        sparsity_layout.addStretch()
        reg_layout.addLayout(sparsity_layout)

        reg_group.setLayout(reg_layout)
        layout.addWidget(reg_group)

        # Info box
        info_group = QGroupBox("Advanced Options Information")
        info_layout = QVBoxLayout()

        info_text = QLabel(
            "<b>Normalize:</b> Scales the distribution so that it sums to 1. "
            "This is useful for probability distributions.<br><br>"

            "<b>Enforce Unimodality:</b> Restricts the solution to have a single peak. "
            "Use this for monodisperse samples where you expect only one particle size.<br><br>"

            "<b>Sparsity Penalty:</b> Adds an L1 regularization term that encourages "
            "fewer non-zero components in the distribution. Higher values lead to sparser solutions."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        info_layout.addWidget(info_text)

        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        widget.setLayout(layout)
        return widget

    def create_preview_tab(self):
        """Create interactive preview tab"""
        widget = QWidget()
        layout = QVBoxLayout()

        # Controls
        controls_layout = QHBoxLayout()

        controls_layout.addWidget(QLabel("Preview datasets:"))
        self.preview_num_spin = QSpinBox()
        self.preview_num_spin.setRange(1, 10)
        self.preview_num_spin.setValue(5)
        controls_layout.addWidget(self.preview_num_spin)

        self.preview_btn = QPushButton("🔍 Generate Preview")
        self.preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 8px 15px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.preview_btn.clicked.connect(self.generate_preview)
        controls_layout.addWidget(self.preview_btn)

        self.update_preview_btn = QPushButton("↻ Update Preview")
        self.update_preview_btn.setEnabled(False)
        self.update_preview_btn.clicked.connect(self.update_preview)
        controls_layout.addWidget(self.update_preview_btn)

        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Info label
        self.preview_info_label = QLabel("Click 'Generate Preview' to see Regularized NNLS results on random datasets")
        self.preview_info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(self.preview_info_label)

        # Matplotlib canvas
        self.canvas = FigureCanvas(Figure(figsize=(12, 8)))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        widget.setLayout(layout)
        return widget

    def on_prominence_slider_changed(self, value):
        """Handle prominence slider change"""
        prominence = value / 1000.0
        self.prominence_input.blockSignals(True)
        self.prominence_input.setValue(prominence)
        self.prominence_input.blockSignals(False)
        self.params['prominence'] = prominence

    def on_prominence_input_changed(self, value):
        """Handle prominence manual input change"""
        slider_value = int(value * 1000)
        self.prominence_slider.blockSignals(True)
        self.prominence_slider.setValue(slider_value)
        self.prominence_slider.blockSignals(False)
        self.params['prominence'] = value

    def on_distance_slider_changed(self, value):
        """Handle distance slider change"""
        self.distance_input.blockSignals(True)
        self.distance_input.setValue(value)
        self.distance_input.blockSignals(False)
        self.params['distance'] = value

    def on_distance_input_changed(self, value):
        """Handle distance manual input change"""
        self.distance_slider.blockSignals(True)
        self.distance_slider.setValue(value)
        self.distance_slider.blockSignals(False)
        self.params['distance'] = value

    def apply_preset(self, preset_name):
        """Apply parameter preset"""
        presets = {
            'monodisperse': {'prominence': 0.1, 'distance': 20},
            'bidisperse': {'prominence': 0.05, 'distance': 10},
            'polydisperse': {'prominence': 0.02, 'distance': 3}
        }

        if preset_name in presets:
            prom = presets[preset_name]['prominence']
            dist = presets[preset_name]['distance']

            # Update sliders
            self.prominence_slider.setValue(int(prom * 1000))
            self.distance_slider.setValue(dist)

            # Update params
            self.params['prominence'] = prom
            self.params['distance'] = dist

            QMessageBox.information(self, "Preset Applied",
                                   f"Applied '{preset_name}' preset:\n"
                                   f"Prominence: {prom}\n"
                                   f"Distance: {dist}\n\n"
                                   f"Click 'Update Preview' to see changes.")

    def find_optimal_alpha(self):
        """Open alpha analysis dialog"""
        from gui.dialogs.alpha_analysis_dialog import AlphaAnalysisDialog

        dialog = AlphaAnalysisDialog(self.laplace_analyzer, self)
        if dialog.exec_() == dialog.Accepted:
            recommended_alpha = dialog.get_recommended_alpha()
            self.alpha_spin.setValue(recommended_alpha)

            QMessageBox.information(
                self,
                "Alpha Optimized",
                f"Optimal alpha value found: {recommended_alpha:.4f}\n\n"
                f"This value has been set in the Alpha field."
            )

    def generate_preview(self):
        """Generate preview with current parameters"""
        # Similar to NNLS preview but using regularized fit
        self.update_params_from_inputs()

        try:
            self.preview_info_label.setText("Generating preview... Please wait.")
            self.preview_info_label.setStyleSheet("color: orange; font-weight: bold;")

            # Run preview
            from regularized_optimized import nnls_preview_random
            import random

            # Select random datasets
            all_keys = list(self.laplace_analyzer.processed_correlations.keys())
            num_to_select = min(self.preview_num_spin.value(), len(all_keys))
            selected_keys = random.sample(all_keys, num_to_select)
            self.preview_datasets = selected_keys

            # Create parameters for regularized fit
            reg_params = {
                'decay_times': self.params['decay_times'],
                'prominence': self.params['prominence'],
                'distance': self.params['distance'],
                'alpha': self.params['alpha'],
                'normalize': self.params['normalize'],
                'sparsity_penalty': self.params['sparsity_penalty'],
                'enforce_unimodality': self.params['enforce_unimodality']
            }

            # Generate preview using regularized fit
            self._create_regularized_preview(selected_keys, reg_params)

            self.preview_info_label.setText(
                f"Preview generated for {len(selected_keys)} datasets. "
                f"Adjust parameters and click 'Update Preview' to see changes."
            )
            self.preview_info_label.setStyleSheet("color: green; font-weight: bold;")

            self.update_preview_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Preview Error",
                               f"Error generating preview:\n{str(e)}")
            import traceback
            traceback.print_exc()
            self.preview_info_label.setText("Error generating preview")
            self.preview_info_label.setStyleSheet("color: red; font-weight: bold;")

    def _create_regularized_preview(self, selected_keys, params):
        """Create preview plots using regularized fit"""
        from regularized import nnls_reg_simple

        decay_times = params['decay_times']

        # Calculate grid size
        num_datasets = len(selected_keys)
        cols = min(3, num_datasets)
        rows = (num_datasets + cols - 1) // cols

        self.canvas.figure.clear()

        # Process each dataset
        for idx, key in enumerate(selected_keys):
            df = self.laplace_analyzer.processed_correlations[key]

            # Run regularized fit
            try:
                _, f_optimized, _, _, peaks = nnls_reg_simple(df, key, params)
            except Exception as e:
                print(f"Error processing {key}: {e}")
                continue

            # Create subplot
            ax = self.canvas.figure.add_subplot(rows, cols, idx + 1)

            # Plot distribution
            ax.semilogx(decay_times, f_optimized, 'b-', linewidth=2, label='Distribution')

            # Mark peaks
            if len(peaks) > 0:
                peak_amplitudes = f_optimized[peaks]
                normalized_amplitudes = peak_amplitudes / np.sum(peak_amplitudes) if len(peak_amplitudes) > 0 else np.array([])

                colors = ['red', 'green', 'orange', 'purple', 'cyan']
                for i, peak_idx in enumerate(peaks):
                    color = colors[i % len(colors)]
                    ax.plot(decay_times[peak_idx], f_optimized[peak_idx], 'o',
                           color=color, markersize=10, label=f'Peak {i+1} ({normalized_amplitudes[i]*100:.1f}%)')

                    # Annotate
                    ax.annotate(f'τ={decay_times[peak_idx]:.2e}',
                               xy=(decay_times[peak_idx], f_optimized[peak_idx]),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=8, bbox=dict(boxstyle="round", fc=color, alpha=0.3))

            ax.set_xlabel('Decay Time [s]')
            ax.set_ylabel('Intensity')
            ax.set_title(f'{key}\n{len(peaks)} peaks found')
            ax.grid(True, which='both', alpha=0.3)
            ax.legend(fontsize=8)

        self.canvas.figure.suptitle(
            f'Regularized NNLS Preview - Alpha: {params["alpha"]:.4f}, '
            f'Prominence: {params["prominence"]:.3f}, Distance: {params["distance"]}',
            fontsize=14, fontweight='bold'
        )
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def update_preview(self):
        """Update preview with current parameter values"""
        self.generate_preview()

    def update_params_from_inputs(self):
        """Update parameters dictionary from input widgets"""
        # Decay times
        start = self.decay_start_spin.value()
        end = self.decay_end_spin.value()
        num = self.decay_num_spin.value()
        self.params['decay_times'] = np.logspace(np.log10(start), np.log10(end), num)

        # Alpha
        self.params['alpha'] = self.alpha_spin.value()

        # Peak detection - use values from input fields (they're synced with sliders)
        self.params['prominence'] = self.prominence_input.value()
        self.params['distance'] = self.distance_input.value()

        # Advanced options
        self.params['normalize'] = self.normalize_check.isChecked()
        self.params['sparsity_penalty'] = self.sparsity_spin.value()
        self.params['enforce_unimodality'] = self.unimodal_check.isChecked()

        # Processing options
        self.params['show_plots'] = self.show_plots_check.isChecked()

    def accept_parameters(self):
        """Accept parameters and emit signal"""
        self.update_params_from_inputs()

        # Emit parameters
        self.parameters_accepted.emit(self.params)
        self.accept()

    def get_parameters(self):
        """Get current parameters"""
        self.update_params_from_inputs()
        return self.params
