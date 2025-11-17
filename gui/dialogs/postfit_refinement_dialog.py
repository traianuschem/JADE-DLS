"""
Post-Fit Refinement Dialog

Allows refinement of cumulant analysis results after initial fitting:
- Adjust q² range for diffusion coefficient calculation (all methods)
- Exclude individual fits (Method C only)
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QDoubleSpinBox, QFormLayout, QTabWidget,
                             QWidget, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt


class PostFitRefinementDialog(QDialog):
    """
    Dialog for post-fit refinement of cumulant analysis results

    Provides method-specific refinement options:
    - Method A: Q² range adjustment
    - Method B: Q² range adjustment
    - Method C: Q² range adjustment + fit exclusion
    """

    def __init__(self, analyzer, available_methods, parent=None):
        """
        Initialize the post-fit refinement dialog

        Args:
            analyzer: CumulantAnalyzer instance with existing results
            available_methods: List of methods with results ['A', 'B', 'C']
            parent: Parent widget
        """
        super().__init__(parent)
        self.analyzer = analyzer
        self.available_methods = available_methods

        # Store current q-ranges for each method
        self.q_ranges = {
            'A': None,
            'B': None,
            'C': None
        }

        # Store excluded fits for Method C
        self.excluded_fits_c = []

        self.setWindowTitle("Post-Fit Refinement")
        self.setModal(True)
        self.resize(600, 400)

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout()

        # Title and description
        title = QLabel("Post-Fit Refinement")
        title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        layout.addWidget(title)

        description = QLabel(
            "Refine your cumulant analysis results by adjusting the q² range for "
            "diffusion coefficient calculation and/or excluding individual fits."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        # Create tabs for each method
        self.tabs = QTabWidget()

        if 'A' in self.available_methods:
            self.tabs.addTab(self.create_method_a_tab(), "Method A")

        if 'B' in self.available_methods:
            self.tabs.addTab(self.create_method_b_tab(), "Method B")

        if 'C' in self.available_methods:
            self.tabs.addTab(self.create_method_c_tab(), "Method C")

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

    def create_method_a_tab(self):
        """Create refinement tab for Method A"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Q² range group
        q_range_group = QGroupBox("Q² Range for Diffusion Coefficient Fit")
        q_range_layout = QFormLayout()

        # Get current q² range from basedata
        if hasattr(self.analyzer, 'df_basedata') and self.analyzer.df_basedata is not None:
            q_squared_values = self.analyzer.df_basedata['q^2'].values
            min_q = float(q_squared_values.min())
            max_q = float(q_squared_values.max())
        else:
            min_q = 0.0
            max_q = 0.01

        # Min Q²
        self.a_q_min = QDoubleSpinBox()
        self.a_q_min.setRange(0, 1)
        self.a_q_min.setValue(min_q)
        self.a_q_min.setDecimals(6)
        self.a_q_min.setSingleStep(0.0001)
        self.a_q_min.setSuffix(" nm⁻²")
        q_range_layout.addRow("Min Q²:", self.a_q_min)

        # Max Q²
        self.a_q_max = QDoubleSpinBox()
        self.a_q_max.setRange(0, 1)
        self.a_q_max.setValue(max_q)
        self.a_q_max.setDecimals(6)
        self.a_q_max.setSingleStep(0.0001)
        self.a_q_max.setSuffix(" nm⁻²")
        q_range_layout.addRow("Max Q²:", self.a_q_max)

        q_range_group.setLayout(q_range_layout)
        layout.addWidget(q_range_group)

        # Info
        info = QLabel(
            "<b>Effect:</b> Re-calculate diffusion coefficient using only data points "
            "within the specified q² range. Individual cumulant fits remain unchanged."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 10px; background-color: #e3f2fd; border-radius: 4px;")
        layout.addWidget(info)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_method_b_tab(self):
        """Create refinement tab for Method B"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Q² range group
        q_range_group = QGroupBox("Q² Range for Diffusion Coefficient Fit")
        q_range_layout = QFormLayout()

        # Get current q² range from basedata
        if hasattr(self.analyzer, 'df_basedata') and self.analyzer.df_basedata is not None:
            q_squared_values = self.analyzer.df_basedata['q^2'].values
            min_q = float(q_squared_values.min())
            max_q = float(q_squared_values.max())
        else:
            min_q = 0.0
            max_q = 0.01

        # Min Q²
        self.b_q_min = QDoubleSpinBox()
        self.b_q_min.setRange(0, 1)
        self.b_q_min.setValue(min_q)
        self.b_q_min.setDecimals(6)
        self.b_q_min.setSingleStep(0.0001)
        self.b_q_min.setSuffix(" nm⁻²")
        q_range_layout.addRow("Min Q²:", self.b_q_min)

        # Max Q²
        self.b_q_max = QDoubleSpinBox()
        self.b_q_max.setRange(0, 1)
        self.b_q_max.setValue(max_q)
        self.b_q_max.setDecimals(6)
        self.b_q_max.setSingleStep(0.0001)
        self.b_q_max.setSuffix(" nm⁻²")
        q_range_layout.addRow("Max Q²:", self.b_q_max)

        q_range_group.setLayout(q_range_layout)
        layout.addWidget(q_range_group)

        # Info
        info = QLabel(
            "<b>Effect:</b> Re-calculate diffusion coefficient using only data points "
            "within the specified q² range. Individual linear fits remain unchanged."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 10px; background-color: #e3f2fd; border-radius: 4px;")
        layout.addWidget(info)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_method_c_tab(self):
        """Create refinement tab for Method C"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Q² range group
        q_range_group = QGroupBox("Q² Range for Diffusion Coefficient Fit")
        q_range_layout = QFormLayout()

        # Get current q² range from basedata
        if hasattr(self.analyzer, 'df_basedata') and self.analyzer.df_basedata is not None:
            q_squared_values = self.analyzer.df_basedata['q^2'].values
            min_q = float(q_squared_values.min())
            max_q = float(q_squared_values.max())
        else:
            min_q = 0.0
            max_q = 0.01

        # Min Q²
        self.c_q_min = QDoubleSpinBox()
        self.c_q_min.setRange(0, 1)
        self.c_q_min.setValue(min_q)
        self.c_q_min.setDecimals(6)
        self.c_q_min.setSingleStep(0.0001)
        self.c_q_min.setSuffix(" nm⁻²")
        q_range_layout.addRow("Min Q²:", self.c_q_min)

        # Max Q²
        self.c_q_max = QDoubleSpinBox()
        self.c_q_max.setRange(0, 1)
        self.c_q_max.setValue(max_q)
        self.c_q_max.setDecimals(6)
        self.c_q_max.setSingleStep(0.0001)
        self.c_q_max.setSuffix(" nm⁻²")
        q_range_layout.addRow("Max Q²:", self.c_q_max)

        q_range_group.setLayout(q_range_layout)
        layout.addWidget(q_range_group)

        # Fit exclusion group
        exclusion_group = QGroupBox("Exclude Individual Fits")
        exclusion_layout = QVBoxLayout()

        self.inspect_fits_btn = QPushButton("Inspect & Exclude Fits...")
        self.inspect_fits_btn.clicked.connect(self.open_fit_inspector_c)
        exclusion_layout.addWidget(self.inspect_fits_btn)

        self.excluded_label_c = QLabel("No fits excluded")
        self.excluded_label_c.setStyleSheet("font-style: italic; color: #666;")
        exclusion_layout.addWidget(self.excluded_label_c)

        exclusion_group.setLayout(exclusion_layout)
        layout.addWidget(exclusion_group)

        # Info
        info = QLabel(
            "<b>Effect:</b> Re-calculate diffusion coefficient using only data points "
            "within the specified q² range AND excluding selected fits. "
            "Individual non-linear fits remain unchanged."
        )
        info.setWordWrap(True)
        info.setStyleSheet("padding: 10px; background-color: #e3f2fd; border-radius: 4px;")
        layout.addWidget(info)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def open_fit_inspector_c(self):
        """Open fit inspection dialog for Method C"""
        from gui.dialogs.method_c_postfit_dialog import MethodCPostFitDialog

        if not hasattr(self.analyzer, 'method_c_plots') or not self.analyzer.method_c_plots:
            QMessageBox.warning(
                self,
                "No Plots Available",
                "No Method C fit plots are available for inspection."
            )
            return

        dialog = MethodCPostFitDialog(
            self.analyzer.method_c_plots,
            self.analyzer.method_c_fit_quality,
            self.excluded_fits_c,
            self
        )

        if dialog.exec_() == QDialog.Accepted:
            self.excluded_fits_c = dialog.get_excluded_files()

            # Update label
            if self.excluded_fits_c:
                self.excluded_label_c.setText(f"{len(self.excluded_fits_c)} fits excluded")
                self.excluded_label_c.setStyleSheet("font-weight: bold; color: #d32f2f;")
            else:
                self.excluded_label_c.setText("No fits excluded")
                self.excluded_label_c.setStyleSheet("font-style: italic; color: #666;")

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
            if hasattr(self.analyzer, 'df_basedata') and self.analyzer.df_basedata is not None:
                q_squared_values = self.analyzer.df_basedata['q^2'].values
                min_q = float(q_squared_values.min())
                max_q = float(q_squared_values.max())
            else:
                min_q = 0.0
                max_q = 0.01

            # Reset Method A
            if 'A' in self.available_methods:
                self.a_q_min.setValue(min_q)
                self.a_q_max.setValue(max_q)

            # Reset Method B
            if 'B' in self.available_methods:
                self.b_q_min.setValue(min_q)
                self.b_q_max.setValue(max_q)

            # Reset Method C
            if 'C' in self.available_methods:
                self.c_q_min.setValue(min_q)
                self.c_q_max.setValue(max_q)
                self.excluded_fits_c = []
                self.excluded_label_c.setText("No fits excluded")
                self.excluded_label_c.setStyleSheet("font-style: italic; color: #666;")

    def apply_refinement(self):
        """Apply refinement and recalculate results"""
        # Validate inputs
        for method in self.available_methods:
            if method == 'A':
                if self.a_q_min.value() >= self.a_q_max.value():
                    QMessageBox.warning(
                        self,
                        "Invalid Range",
                        "Method A: Minimum q² must be less than maximum q²."
                    )
                    return
                self.q_ranges['A'] = (self.a_q_min.value(), self.a_q_max.value())

            elif method == 'B':
                if self.b_q_min.value() >= self.b_q_max.value():
                    QMessageBox.warning(
                        self,
                        "Invalid Range",
                        "Method B: Minimum q² must be less than maximum q²."
                    )
                    return
                self.q_ranges['B'] = (self.b_q_min.value(), self.b_q_max.value())

            elif method == 'C':
                if self.c_q_min.value() >= self.c_q_max.value():
                    QMessageBox.warning(
                        self,
                        "Invalid Range",
                        "Method C: Minimum q² must be less than maximum q²."
                    )
                    return
                self.q_ranges['C'] = (self.c_q_min.value(), self.c_q_max.value())

        # Accept dialog
        self.accept()

    def get_refinement_params(self):
        """
        Get refinement parameters

        Returns:
            dict: Dictionary with refinement parameters for each method
        """
        return {
            'q_ranges': self.q_ranges,
            'excluded_fits_c': self.excluded_fits_c
        }
