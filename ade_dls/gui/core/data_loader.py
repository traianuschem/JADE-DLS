"""
Data Loader with Preprocessing Integration
Handles loading and processing of DLS data files
"""

import os
import pandas as pd
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread

from ade_dls.engine.loading import SeriesLoader, DataLoadError

# Ensure the package root is importable (parsers wrap the preprocessing layer).
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class DataLoadWorker(QThread):
    """
    Worker thread for loading data without freezing GUI.

    Thin Qt wrapper around ade_dls.engine.loading.SeriesLoader (all loading logic lives there).

    Signals:
        progress: Emitted with (current, total, message)
        step_complete: Emitted when a step is complete
        finished: Emitted when all loading is complete
        error: Emitted on error
    """

    progress = pyqtSignal(int, int, str)  # current, total, message
    step_complete = pyqtSignal(str, object)  # step_name, data
    finished = pyqtSignal(dict)  # all_data
    error = pyqtSignal(str)  # error_message

    def __init__(self, data_folder, parser, load_countrates=True, load_correlations=True,
                 load_cumulants=True):
        super().__init__()
        self.loader = SeriesLoader(
            data_folder, parser, load_countrates, load_correlations, load_cumulants,
            on_progress=self.progress.emit, on_step=self.step_complete.emit)

    @property
    def errors(self):
        return self.loader.errors

    @property
    def is_cancelled(self):
        return self.loader.is_cancelled

    def run(self):
        """Run the data loading process"""
        try:
            all_data = self.loader.run()
        except DataLoadError as e:
            self.error.emit(str(e))
            return
        if all_data is not None:  # None = cancelled
            self.finished.emit(all_data)

    def cancel(self):
        """Cancel the loading process"""
        self.loader.cancel()


class DataLoader(QObject):
    """
    Main data loader class that manages the worker thread

    Signals:
        progress: Progress updates
        data_loaded: Emitted when data is loaded
        error: Error messages
    """

    progress = pyqtSignal(int, int, str)
    data_loaded = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.worker = None

    def load_data(self, data_folder, parser=None, load_countrates=True, load_correlations=True,
                  load_cumulants=True):
        """
        Load data from folder.

        Args:
            data_folder: Path to the data folder.
            parser: Optional pre-selected InstrumentParser instance (from the
                    load dialog). When None, format detection runs automatically.
            load_countrates: Whether to load count rate data.
            load_correlations: Whether to load correlation data.
            load_cumulants: Whether to load instrument-software cumulant fits.
        """
        if parser is None:
            from ade_dls.core.parsers import detect_parser
            parser = detect_parser(data_folder)
            if parser is None:
                self.error.emit(
                    f"No known instrument format detected in:\n{data_folder}"
                )
                return
        print(f"[DATA LOADER] Format: {parser.INSTRUMENT_NAME}")

        # Create worker thread
        self.worker = DataLoadWorker(data_folder, parser, load_countrates, load_correlations,
                                     load_cumulants)

        # Connect signals
        self.worker.progress.connect(self.progress.emit)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self.error.emit)

        # Start worker
        self.worker.start()

    def _on_finished(self, data):
        """Handle worker completion"""
        self.data_loaded.emit(data)
        self.worker = None

    def cancel(self):
        """Cancel current loading operation"""
        if self.worker:
            self.worker.cancel()
            self.worker.wait()
            self.worker = None
