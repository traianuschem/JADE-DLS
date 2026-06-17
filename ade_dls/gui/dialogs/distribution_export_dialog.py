"""
Distribution CSV Export Dialog
==============================
Small preset chooser for exporting multimodal distribution plots as CSV.

Lets the user pick one of three presets before a target folder is chosen:
  - all distributions,
  - a number (X) of randomly selected distributions,
  - one averaged distribution per scattering angle.

Returns ``(mode, n_random)`` via :meth:`get_selection`, where ``mode`` is one of
``'all'`` / ``'random'`` / ``'average_per_angle'`` (matching
``csv_export.build_distribution_tables``) and ``n_random`` is the spin-box value
(only meaningful for ``'random'``).
"""

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QRadioButton,
                             QButtonGroup, QSpinBox, QPushButton, QLabel,
                             QGroupBox)
from PyQt5.QtCore import Qt


class DistributionExportDialog(QDialog):
    """Preset chooser for distribution CSV export."""

    def __init__(self, n_available: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Distributions as CSV")
        self._n_available = max(int(n_available), 1)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()

        info = QLabel(
            f"{self._n_available} distribution(s) available. "
            "Please select an export preset:"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        group = QGroupBox("Preset")
        group_layout = QVBoxLayout()

        self.button_group = QButtonGroup(self)

        self.radio_all = QRadioButton("All distributions")
        self.radio_all.setChecked(True)
        self.button_group.addButton(self.radio_all)
        group_layout.addWidget(self.radio_all)

        # Random X row
        random_row = QHBoxLayout()
        self.radio_random = QRadioButton("Random selection:")
        self.button_group.addButton(self.radio_random)
        random_row.addWidget(self.radio_random)

        self.spin_random = QSpinBox()
        self.spin_random.setMinimum(1)
        self.spin_random.setMaximum(self._n_available)
        self.spin_random.setValue(min(5, self._n_available))
        self.spin_random.setEnabled(False)
        random_row.addWidget(self.spin_random)
        random_row.addWidget(QLabel("distributions"))
        random_row.addStretch()
        group_layout.addLayout(random_row)

        self.radio_average = QRadioButton("One averaged distribution per scattering angle")
        self.button_group.addButton(self.radio_average)
        group_layout.addWidget(self.radio_average)

        group.setLayout(group_layout)
        layout.addWidget(group)

        # Enable the spin box only while the random preset is active.
        self.radio_random.toggled.connect(self.spin_random.setEnabled)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        ok_btn = QPushButton("Export…")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        self.setLayout(layout)

    def get_selection(self):
        """Return ``(mode, n_random)`` for the chosen preset."""
        if self.radio_random.isChecked():
            return 'random', self.spin_random.value()
        if self.radio_average.isChecked():
            return 'average_per_angle', None
        return 'all', None
