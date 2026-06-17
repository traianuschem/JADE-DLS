"""
Dialog for selecting the data folder and instrument format.

Shows the auto-detected format and allows the user to manually override it.
Format descriptions explain which folder level to select for each instrument.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QComboBox, QTextEdit, QDialogButtonBox, QFileDialog, QFrame
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QColor


class LoadDataDialog(QDialog):
    """Folder-picker + instrument-format selector.

    Usage::

        dlg = LoadDataDialog(parent, initial_folder="")
        if dlg.exec_() == QDialog.Accepted:
            folder, parser = dlg.folder, dlg.parser
    """

    def __init__(self, parent=None, initial_folder=""):
        super().__init__(parent)
        self.setWindowTitle("Load Data")
        self.setMinimumWidth(520)
        self.setModal(True)

        from ade_dls.core.parsers import INSTRUMENT_PARSERS, detect_parser
        self._PARSERS = INSTRUMENT_PARSERS
        self._detect_parser = detect_parser

        self.folder = initial_folder
        self.parser = None

        self._build_ui()
        if initial_folder:
            self._on_folder_changed(initial_folder)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # --- Folder row ---
        folder_label = QLabel("Data folder:")
        folder_label.setFont(self._bold())
        layout.addWidget(folder_label)

        folder_row = QHBoxLayout()
        self._folder_edit = QLineEdit(self.folder)
        self._folder_edit.setReadOnly(True)
        self._folder_edit.setPlaceholderText("No folder selected…")
        folder_row.addWidget(self._folder_edit)

        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(90)
        browse_btn.clicked.connect(self._browse)
        folder_row.addWidget(browse_btn)
        layout.addLayout(folder_row)

        # --- Detection badge ---
        self._detection_label = QLabel("")
        self._detection_label.setWordWrap(True)
        layout.addWidget(self._detection_label)

        # --- Separator ---
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        layout.addWidget(sep)

        # --- Format selector ---
        fmt_label = QLabel("Instrument format:")
        fmt_label.setFont(self._bold())
        layout.addWidget(fmt_label)

        self._fmt_combo = QComboBox()
        self._fmt_combo.addItem("— auto-detect —", userData=None)
        for p in self._PARSERS:
            self._fmt_combo.addItem(p.INSTRUMENT_NAME, userData=p)
        self._fmt_combo.currentIndexChanged.connect(self._on_format_changed)
        layout.addWidget(self._fmt_combo)

        # --- Description box ---
        self._desc_box = QTextEdit()
        self._desc_box.setReadOnly(True)
        self._desc_box.setFixedHeight(90)
        self._desc_box.setStyleSheet("background: transparent; border: none;")
        layout.addWidget(self._desc_box)

        # --- OK / Cancel ---
        self._buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self._buttons.accepted.connect(self._accept)
        self._buttons.rejected.connect(self.reject)
        self._ok_btn = self._buttons.button(QDialogButtonBox.Ok)
        self._ok_btn.setEnabled(False)
        layout.addWidget(self._buttons)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Data Folder", self.folder or "",
            QFileDialog.ShowDirsOnly
        )
        if folder:
            self._folder_edit.setText(folder)
            self._on_folder_changed(folder)

    def _on_folder_changed(self, folder: str):
        self.folder = folder
        detected = self._detect_parser(folder)

        if detected:
            self._detection_label.setText(
                f"✔  Auto-detected: <b>{detected.INSTRUMENT_NAME}</b>"
            )
            self._detection_label.setStyleSheet("color: #2d7a2d;")
            # Pre-select the detected format in the combo
            for i in range(self._fmt_combo.count()):
                if self._fmt_combo.itemData(i) is detected.__class__:
                    self._fmt_combo.setCurrentIndex(i)
                    break
        else:
            self._detection_label.setText(
                "⚠  Format not recognised — please select manually below."
            )
            self._detection_label.setStyleSheet("color: #b85c00;")
            self._fmt_combo.setCurrentIndex(0)

        self._update_ok_state()

    def _on_format_changed(self, index: int):
        parser_cls = self._fmt_combo.itemData(index)
        if parser_cls is not None:
            self._desc_box.setPlainText(
                getattr(parser_cls, "DESCRIPTION", "") or
                f"No description available for {parser_cls.INSTRUMENT_NAME}."
            )
        else:
            # "auto-detect" selected: show description of detected parser if any
            if self.folder:
                detected = self._detect_parser(self.folder)
                if detected:
                    self._desc_box.setPlainText(
                        getattr(detected.__class__, "DESCRIPTION", "") or ""
                    )
                else:
                    self._desc_box.setPlainText("")
            else:
                self._desc_box.setPlainText("")
        self._update_ok_state()

    def _update_ok_state(self):
        has_folder = bool(self.folder)
        combo_idx = self._fmt_combo.currentIndex()
        parser_cls = self._fmt_combo.itemData(combo_idx)

        # OK enabled if folder is set AND either a format was detected/selected
        if not has_folder:
            self._ok_btn.setEnabled(False)
            return

        if parser_cls is not None:
            self._ok_btn.setEnabled(True)
        else:
            # "auto-detect": enable only if detection succeeds
            detected = self._detect_parser(self.folder)
            self._ok_btn.setEnabled(detected is not None)

    def _accept(self):
        combo_idx = self._fmt_combo.currentIndex()
        parser_cls = self._fmt_combo.itemData(combo_idx)

        if parser_cls is not None:
            self.parser = parser_cls()
        else:
            self.parser = self._detect_parser(self.folder)

        if self.parser is None:
            # Should not happen (OK is disabled), but be defensive
            return

        self.accept()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bold() -> QFont:
        f = QFont()
        f.setBold(True)
        return f
