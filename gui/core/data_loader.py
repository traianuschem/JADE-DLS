"""
Data Loader with Preprocessing Integration
Handles loading and processing of DLS data files
"""

import os
import glob
import pandas as pd
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, QThread

# Import preprocessing functions
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from preprocessing import (extract_data, extract_correlation, extract_countrate,
                          find_correlation_row, find_countrate_row,
                          process_correlation_data)


class DataLoadWorker(QThread):
    """
    Worker thread for loading data without freezing GUI

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

    def __init__(self, data_folder, load_countrates=True, load_correlations=True):
        super().__init__()
        self.data_folder = data_folder
        self.load_countrates = load_countrates
        self.load_correlations = load_correlations
        self.is_cancelled = False
        self.errors = []  # Track all errors for reporting

    def run(self):
        """Run the data loading process"""
        try:
            all_data = {}
            all_data['data_folder'] = self.data_folder

            # Step 1: Find files
            self.progress.emit(0, 5, "Searching for .asc files...")
            datafiles = glob.glob(os.path.join(self.data_folder, "*.asc"))

            # Filter out averaged files
            filtered_files = [f for f in datafiles
                            if "averaged" not in os.path.basename(f).lower()]

            if not filtered_files:
                self.error.emit(f"No .asc files found in {self.data_folder}")
                return

            all_data['files'] = filtered_files
            all_data['num_files'] = len(filtered_files)

            self.progress.emit(1, 5, f"Found {len(filtered_files)} data files")
            self.step_complete.emit("find_files", filtered_files)

            if self.is_cancelled:
                return

            # Step 2: Extract base data
            self.progress.emit(2, 5, "Extracting base data (angle, temperature, etc.)...")
            df_basedata = self._extract_basedata(filtered_files)

            if df_basedata is None or df_basedata.empty:
                self.error.emit("Failed to extract base data from files")
                return

            all_data['basedata'] = df_basedata
            self.step_complete.emit("basedata", df_basedata)

            if self.is_cancelled:
                return

            # Step 3: Extract countrates (optional)
            if self.load_countrates:
                self.progress.emit(3, 5, "Extracting count rates...")
                all_countrates = self._extract_countrates(filtered_files)
                all_data['countrates'] = all_countrates
                self.step_complete.emit("countrates", all_countrates)
            else:
                self.progress.emit(3, 5, "Skipping count rate extraction")

            if self.is_cancelled:
                return

            # Step 4: Extract correlations
            if self.load_correlations:
                self.progress.emit(4, 5, "Extracting correlation data...")
                all_correlations = self._extract_correlations(filtered_files)

                if not all_correlations:
                    self.error.emit("Failed to extract correlation data")
                    return

                all_data['correlations'] = all_correlations
                self.step_complete.emit("correlations", all_correlations)
            else:
                self.progress.emit(4, 5, "Skipping correlation extraction")

            if self.is_cancelled:
                return

            # Step 5: Calculate q and q^2
            self.progress.emit(5, 5, "Calculating scattering vectors...")
            df_basedata = self._calculate_q_vectors(df_basedata)
            all_data['basedata'] = df_basedata

            # Add error summary to results
            all_data['errors'] = self.errors
            all_data['total_files'] = len(filtered_files)
            all_data['successful_files'] = len(filtered_files) - len([e for e in self.errors if e['step'] == 'basedata'])

            # Complete
            self.finished.emit(all_data)

        except Exception as e:
            self.error.emit(f"Error loading data: {str(e)}")

    def _extract_basedata(self, files):
        """Extract base data from all files"""
        all_data = []
        total = len(files)

        for i, file in enumerate(files):
            if self.is_cancelled:
                return None

            self.progress.emit(2, 5, f"Extracting base data: {i+1}/{total} files")

            try:
                extracted_data = extract_data(file)
                if extracted_data is not None:
                    filename = os.path.basename(file)
                    extracted_data['filename'] = filename
                    all_data.append(extracted_data)
                else:
                    # File couldn't be processed
                    error_msg = f"Could not extract data from {os.path.basename(file)}"
                    self.errors.append({'file': os.path.basename(file), 'step': 'basedata', 'error': error_msg})
                    print(f"Warning: {error_msg}")
            except Exception as e:
                error_msg = f"Failed to extract data: {str(e)}"
                self.errors.append({'file': os.path.basename(file), 'step': 'basedata', 'error': error_msg})
                print(f"Warning: Failed to extract data from {file}: {e}")
                continue

        if all_data:
            df = pd.concat(all_data, ignore_index=True)
            df.index = df.index + 1
            return df
        else:
            return pd.DataFrame()

    def _extract_countrates(self, files):
        """Extract count rate data from all files"""
        all_countrates = {}
        total = len(files)

        for i, file in enumerate(files):
            if self.is_cancelled:
                return all_countrates

            self.progress.emit(3, 5, f"Extracting count rates: {i+1}/{total} files")

            try:
                extracted_countrate = extract_countrate(file)
                if extracted_countrate is not None:
                    filename = os.path.basename(file)
                    # Rename columns
                    new_column_names = {
                        0: 'time [s]',
                        1: 'detectorslot 1',
                        2: 'detectorslot 2',
                        3: 'detectorslot 3',
                        4: 'detectorslot 4'
                    }
                    extracted_countrate = extracted_countrate.rename(columns=new_column_names)
                    all_countrates[filename] = extracted_countrate
                else:
                    # File couldn't be processed but not critical
                    error_msg = f"Could not extract countrate data"
                    self.errors.append({'file': os.path.basename(file), 'step': 'countrates', 'error': error_msg})
            except Exception as e:
                error_msg = f"Failed to extract countrate: {str(e)}"
                self.errors.append({'file': os.path.basename(file), 'step': 'countrates', 'error': error_msg})
                print(f"Warning: Failed to extract countrate from {file}: {e}")
                continue

        return all_countrates

    def _extract_correlations(self, files):
        """Extract correlation data from all files"""
        all_correlations = {}
        total = len(files)

        for i, file in enumerate(files):
            if self.is_cancelled:
                return all_correlations

            self.progress.emit(4, 5, f"Extracting correlations: {i+1}/{total} files")

            try:
                extracted_correlation = extract_correlation(file)
                if extracted_correlation is not None:
                    filename = os.path.basename(file)
                    # Rename columns
                    new_column_names = {
                        0: 'time [ms]',
                        1: 'correlation 1',
                        2: 'correlation 2',
                        3: 'correlation 3',
                        4: 'correlation 4'
                    }
                    extracted_correlation = extracted_correlation.rename(columns=new_column_names)
                    all_correlations[filename] = extracted_correlation
                else:
                    # File couldn't be processed but not critical
                    error_msg = f"Could not extract correlation data"
                    self.errors.append({'file': os.path.basename(file), 'step': 'correlations', 'error': error_msg})
            except Exception as e:
                error_msg = f"Failed to extract correlation: {str(e)}"
                self.errors.append({'file': os.path.basename(file), 'step': 'correlations', 'error': error_msg})
                print(f"Warning: Failed to extract correlation from {file}: {e}")
                continue

        return all_correlations

    def _calculate_q_vectors(self, df):
        """Calculate scattering vector q and q^2"""
        if df.empty:
            return df

        # Calculate q
        df['q'] = abs(
            ((4 * np.pi * df['refractive_index']) / df['wavelength [nm]']) *
            np.sin(np.radians(df['angle [°]']) / 2)
        )

        # Calculate q^2
        df['q^2'] = df['q'] ** 2

        return df

    def cancel(self):
        """Cancel the loading process"""
        self.is_cancelled = True


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

    def load_data(self, data_folder, load_countrates=True, load_correlations=True):
        """
        Load data from folder

        Args:
            data_folder: Path to folder with .asc files
            load_countrates: Whether to load count rate data
            load_correlations: Whether to load correlation data
        """
        # Create worker thread
        self.worker = DataLoadWorker(data_folder, load_countrates, load_correlations)

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
