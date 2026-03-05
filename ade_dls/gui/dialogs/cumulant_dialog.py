"""
Cumulant Analysis Dialogs
One focused dialog per method (A, B, C)
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QLabel, QPushButton, QComboBox, QDoubleSpinBox,
                             QFormLayout, QWidget, QCheckBox, QMessageBox)
from PyQt5.QtCore import Qt


# ---------------------------------------------------------------------------
# Shared helper: q² range widget block
# ---------------------------------------------------------------------------

def _build_q_range_group(parent_dialog):
    """
    Build a QGroupBox with q² range controls and attach them as attributes
    of *parent_dialog*:  q_range_enabled, q_min, q_max.

    Returns the QGroupBox ready to be inserted into a layout.
    """
    group = QGroupBox("Diffusion Analysis q² Range (optional)")
    layout = QFormLayout()

    parent_dialog.q_range_enabled = QCheckBox("Restrict q² range for Γ vs q² fit")
    layout.addRow(parent_dialog.q_range_enabled)

    q_limit_layout = QHBoxLayout()

    parent_dialog.q_min = QDoubleSpinBox()
    parent_dialog.q_min.setRange(0, 1000)
    parent_dialog.q_min.setValue(0.0)
    parent_dialog.q_min.setDecimals(6)
    parent_dialog.q_min.setSuffix(" nm⁻²")
    parent_dialog.q_min.setEnabled(False)

    parent_dialog.q_max = QDoubleSpinBox()
    parent_dialog.q_max.setRange(0, 1000)
    parent_dialog.q_max.setValue(100.0)
    parent_dialog.q_max.setDecimals(6)
    parent_dialog.q_max.setSuffix(" nm⁻²")
    parent_dialog.q_max.setEnabled(False)

    q_limit_layout.addWidget(QLabel("Min:"))
    q_limit_layout.addWidget(parent_dialog.q_min)
    q_limit_layout.addWidget(QLabel("Max:"))
    q_limit_layout.addWidget(parent_dialog.q_max)
    layout.addRow("q² Range:", q_limit_layout)

    hint = QLabel(
        "<i>Use this to exclude outliers at very low or high q² values<br>"
        "from the Diffusion Coefficient analysis (Γ vs q² linear fit).</i>"
    )
    hint.setWordWrap(True)
    layout.addRow("", hint)

    group.setLayout(layout)

    parent_dialog.q_range_enabled.toggled.connect(parent_dialog.q_min.setEnabled)
    parent_dialog.q_range_enabled.toggled.connect(parent_dialog.q_max.setEnabled)

    return group


def _collect_q_range(dialog):
    """
    Read q² range from *dialog* (which must have q_range_enabled/q_min/q_max).
    Returns (min, max) tuple or None.  Shows a QMessageBox on invalid input
    and returns the sentinel value False to signal validation failure.
    """
    if not dialog.q_range_enabled.isChecked():
        return None
    q_min_val = dialog.q_min.value()
    q_max_val = dialog.q_max.value()
    if q_min_val >= q_max_val:
        QMessageBox.warning(
            dialog,
            "Invalid q² Range",
            f"Minimum q² ({q_min_val}) must be less than maximum ({q_max_val})."
        )
        return False  # sentinel: validation failed
    return (q_min_val, q_max_val)


# ---------------------------------------------------------------------------
# Method A
# ---------------------------------------------------------------------------

class CumulantADialog(QDialog):
    """
    Parameter dialog for Cumulant Method A (ALV Software Cumulant Data).

    The method extracts 1st/2nd/3rd order cumulant fit results directly
    from the ALV correlator .asc files – no additional fit parameters are
    required.  The only user-configurable option is an optional q² range
    restriction for the Γ vs q² diffusion analysis.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Method A – ALV Cumulant Data")
        self.setMinimumWidth(480)

        self.q_range = None  # filled in accept()
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        info = QLabel(
            "<b>Method A: ALV Software Cumulant Data</b><br><br>"
            "Extracts cumulant fit results written by the ALV correlator software "
            "directly from the <code>.asc</code> files.<br><br>"
            "Results include:<br>"
            "• 1st order cumulant fit<br>"
            "• 2nd order cumulant fit (with PDI)<br>"
            "• 3rd order cumulant fit (with PDI)"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        layout.addWidget(_build_q_range_group(self))
        layout.addStretch()
        layout.addLayout(self._button_row())
        self.setLayout(layout)

    def _button_row(self):
        row = QHBoxLayout()
        row.addStretch()
        run_btn = QPushButton("Run Method A")
        run_btn.setDefault(True)
        run_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(run_btn)
        row.addWidget(cancel_btn)
        return row

    def accept(self):
        q = _collect_q_range(self)
        if q is False:
            return  # validation failed – stay open
        self.q_range = q
        super().accept()

    def get_configuration(self):
        return {'q_range': self.q_range}


# ---------------------------------------------------------------------------
# Method B
# ---------------------------------------------------------------------------

class CumulantBDialog(QDialog):
    """
    Parameter dialog for Cumulant Method B (Linear Cumulant Fit).

    Fits ln(√(g²(τ)−1)) vs τ over a narrow user-defined time window.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Method B – Linear Cumulant Fit")
        self.setMinimumWidth(520)

        self.fit_limits = None  # filled in accept()
        self.q_range = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # Fit time range
        time_group = QGroupBox("Fit Time Range")
        time_form = QFormLayout()

        row = QHBoxLayout()
        self.b_fit_min = QDoubleSpinBox()
        self.b_fit_min.setRange(0, 1_000_000)
        self.b_fit_min.setValue(0.0)
        self.b_fit_min.setDecimals(1)
        self.b_fit_min.setSingleStep(10.0)
        self.b_fit_min.setSuffix(" µs")

        self.b_fit_max = QDoubleSpinBox()
        self.b_fit_max.setRange(0, 1_000_000)
        self.b_fit_max.setValue(200.0)  # 200 µs default
        self.b_fit_max.setDecimals(1)
        self.b_fit_max.setSingleStep(10.0)
        self.b_fit_max.setSuffix(" µs")

        row.addWidget(QLabel("Min:"))
        row.addWidget(self.b_fit_min)
        row.addWidget(QLabel("Max:"))
        row.addWidget(self.b_fit_max)
        time_form.addRow("Time Range:", row)

        hint = QLabel(
            "<i>Keep the range narrow (e.g. 0 – 200 µs).<br>"
            "Only the initial decay region needs to be fitted.</i>"
        )
        hint.setWordWrap(True)
        time_form.addRow("", hint)
        time_group.setLayout(time_form)
        layout.addWidget(time_group)

        layout.addWidget(_build_q_range_group(self))
        layout.addStretch()
        layout.addLayout(self._button_row())
        self.setLayout(layout)

    def _button_row(self):
        row = QHBoxLayout()
        row.addStretch()
        run_btn = QPushButton("Run Method B")
        run_btn.setDefault(True)
        run_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(run_btn)
        row.addWidget(cancel_btn)
        return row

    def accept(self):
        t_min_us = self.b_fit_min.value()
        t_max_us = self.b_fit_max.value()
        if t_min_us >= t_max_us:
            QMessageBox.warning(
                self,
                "Invalid Fit Range",
                f"Minimum fit time ({t_min_us:.1f} µs) must be less than maximum ({t_max_us:.1f} µs)."
            )
            return

        q = _collect_q_range(self)
        if q is False:
            return

        self.fit_limits = (t_min_us * 1e-6, t_max_us * 1e-6)
        self.q_range = q
        super().accept()

    def get_configuration(self):
        return {
            'fit_limits': self.fit_limits,
            'q_range': self.q_range,
        }


# ---------------------------------------------------------------------------
# Method C
# ---------------------------------------------------------------------------

class CumulantCDialog(QDialog):
    """
    Parameter dialog for Cumulant Method C (Iterative Non-Linear Fit).

    Iterative non-linear least-squares fit of g²(τ) using 2nd, 3rd, or
    4th order cumulant expansion functions.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Method C – Iterative Non-Linear Fit")
        self.setMinimumWidth(580)

        self.params = None   # filled in accept()
        self.q_range = None
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        # --- Fit time range ---
        time_group = QGroupBox("Fit Time Range")
        time_form = QFormLayout()

        time_row = QHBoxLayout()
        self.c_fit_min = QDoubleSpinBox()
        self.c_fit_min.setRange(1e-9, 100)
        self.c_fit_min.setValue(1e-9)
        self.c_fit_min.setDecimals(12)
        self.c_fit_min.setSuffix(" s")

        self.c_fit_max = QDoubleSpinBox()
        self.c_fit_max.setRange(1e-9, 100)
        self.c_fit_max.setValue(10.0)
        self.c_fit_max.setDecimals(6)
        self.c_fit_max.setSuffix(" s")

        time_row.addWidget(QLabel("Min:"))
        time_row.addWidget(self.c_fit_min)
        time_row.addWidget(QLabel("Max:"))
        time_row.addWidget(self.c_fit_max)
        time_form.addRow("Time Range:", time_row)
        time_group.setLayout(time_form)
        layout.addWidget(time_group)

        # --- Fit options ---
        options_group = QGroupBox("Fit Options")
        options_form = QFormLayout()

        self.c_fit_function = QComboBox()
        self.c_fit_function.addItems([
            "2nd Cumulant (up to c parameter)",
            "3rd Cumulant (up to d parameter)",
            "4th Cumulant (up to e parameter) – Recommended",
        ])
        self.c_fit_function.setCurrentIndex(2)
        options_form.addRow("Fit Function:", self.c_fit_function)

        self.c_adaptive = QCheckBox("Use adaptive initial parameter guesses")
        self.c_adaptive.setChecked(True)
        self.c_adaptive.setToolTip(
            "Automatically determine good initial parameters for each dataset."
        )
        options_form.addRow("", self.c_adaptive)

        self.c_strategy = QComboBox()
        self.c_strategy.addItems([
            "Individual – Adapt for each dataset separately",
            "Global – Use median values across all datasets",
            "Representative – Use best signal-to-noise dataset",
        ])
        self.c_strategy.setCurrentIndex(0)
        options_form.addRow("Adaptation Strategy:", self.c_strategy)

        self.c_optimizer = QComboBox()
        self.c_optimizer.addItems([
            "Levenberg-Marquardt (lm) – Recommended",
            "Trust Region Reflective (trf)",
            "Dogbox",
        ])
        self.c_optimizer.setCurrentIndex(0)
        options_form.addRow("Optimization Method:", self.c_optimizer)

        options_group.setLayout(options_form)
        layout.addWidget(options_group)

        # --- Advanced: initial parameters ---
        self.init_params_group = QGroupBox("Initial Parameters (Advanced)")
        init_form = QFormLayout()

        self.c_init_a = QDoubleSpinBox()
        self.c_init_a.setRange(0, 1)
        self.c_init_a.setValue(0.8)
        self.c_init_a.setDecimals(2)
        self.c_init_a.setSingleStep(0.1)
        init_form.addRow("a (baseline):", self.c_init_a)

        self.c_init_b = QDoubleSpinBox()
        self.c_init_b.setRange(0, 1_000_000)
        self.c_init_b.setValue(10_000)
        self.c_init_b.setDecimals(0)
        init_form.addRow("b (decay rate):", self.c_init_b)

        self.c_init_c = QDoubleSpinBox()
        self.c_init_c.setRange(-1000, 1000)
        self.c_init_c.setValue(0)
        self.c_init_c.setDecimals(2)
        init_form.addRow("c (2nd cumulant):", self.c_init_c)

        hint_adv = QLabel(
            "<i>These values are only used when adaptive guesses are disabled.</i>"
        )
        hint_adv.setWordWrap(True)
        init_form.addRow("", hint_adv)

        self.init_params_group.setLayout(init_form)
        self.init_params_group.setVisible(False)   # hidden while adaptive=True
        layout.addWidget(self.init_params_group)

        # Show/hide advanced block based on adaptive checkbox
        self.c_adaptive.toggled.connect(
            lambda checked: self.init_params_group.setVisible(not checked)
        )

        # --- Processing options ---
        proc_group = QGroupBox("Processing Options")
        proc_layout = QVBoxLayout()
        self.c_multiprocessing = QCheckBox(
            "Use multiprocessing (faster for large datasets)"
        )
        self.c_multiprocessing.setChecked(False)
        self.c_multiprocessing.setToolTip(
            "Enable parallel processing via joblib (cross-platform)."
        )
        proc_layout.addWidget(self.c_multiprocessing)
        proc_group.setLayout(proc_layout)
        layout.addWidget(proc_group)

        # --- q² range ---
        layout.addWidget(_build_q_range_group(self))

        layout.addStretch()
        layout.addLayout(self._button_row())
        self.setLayout(layout)

    def _button_row(self):
        row = QHBoxLayout()
        row.addStretch()
        run_btn = QPushButton("Run Method C")
        run_btn.setDefault(True)
        run_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(run_btn)
        row.addWidget(cancel_btn)
        return row

    def accept(self):
        t_min = self.c_fit_min.value()
        t_max = self.c_fit_max.value()
        if t_min >= t_max:
            QMessageBox.warning(
                self,
                "Invalid Fit Range",
                "Minimum fit time must be less than maximum."
            )
            return

        q = _collect_q_range(self)
        if q is False:
            return

        fit_func_map = {0: 'fit_function2', 1: 'fit_function3', 2: 'fit_function4'}
        optimizer_map = {0: 'lm', 1: 'trf', 2: 'dogbox'}
        strategy_map  = {0: 'individual', 1: 'global', 2: 'representative'}

        self.params = {
            'fit_limits': (t_min, t_max),
            'fit_function': fit_func_map[self.c_fit_function.currentIndex()],
            'adaptive_initial_guesses': self.c_adaptive.isChecked(),
            'adaptation_strategy': strategy_map[self.c_strategy.currentIndex()],
            'optimizer': optimizer_map[self.c_optimizer.currentIndex()],
            'use_multiprocessing': self.c_multiprocessing.isChecked(),
            'initial_parameters': [
                self.c_init_a.value(),
                self.c_init_b.value(),
                self.c_init_c.value(),
                0,   # d
                0,   # e
                0,   # f (baseline offset)
            ],
        }
        self.q_range = q
        super().accept()

    def get_configuration(self):
        return {
            'method_c_params': self.params,
            'q_range': self.q_range,
        }
