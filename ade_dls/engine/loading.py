"""
Qt-free series loader (extracted from gui/core/data_loader.py).

Contains the loading logic that used to live inside the QThread worker so that it can run
headless (scripts, tests, CI) and is not tied to PyQt5.
"""

import os
import pandas as pd
import numpy as np


class DataLoadError(Exception):
    """Loading failed; the message is meant for the user (same texts the GUI showed before)."""


class SeriesLoader:
    """
    Qt-free loader for one measurement series (folder -> basedata, countrates, correlations,
    instrument cumulants, q / q^2). The GUI wraps it in a QThread (gui/core/data_loader.py),
    scripts and tests call ``run()`` directly.

    Callbacks (all optional):
        on_progress(current, total, message)
        on_step(step_name, data)   after find_files / basedata / countrates / correlations / cumulants
    ``run()`` returns the result dict, ``None`` if cancelled, and raises DataLoadError on failure.
    """

    def __init__(self, data_folder, parser, load_countrates=True, load_correlations=True,
                 load_cumulants=True, on_progress=None, on_step=None):
        self.data_folder = data_folder
        self.parser = parser  # InstrumentParser instance (selected by format detection)
        self.load_countrates = load_countrates
        self.load_correlations = load_correlations
        self.load_cumulants = load_cumulants
        self.is_cancelled = False
        self.errors = []  # Track all errors for reporting
        self._on_progress = on_progress or (lambda current, total, message: None)
        self._on_step = on_step or (lambda name, data: None)

    def run(self):
        """Run the loading process; returns the result dict (None if cancelled)."""
        try:
            print(f"[DATA LOADER] Starting data loading from: {self.data_folder}")
            print(f"[DATA LOADER] Using parser: {self.parser.INSTRUMENT_NAME}")
            all_data = {}
            all_data['data_folder'] = self.data_folder
            all_data['instrument'] = self.parser.INSTRUMENT_NAME

            # Step 1: Find measurement units via the parser plugin
            self._on_progress(0, 5, "Searching for data files...")
            print(f"[DATA LOADER] Searching for data files...")
            filtered_files = self.parser.get_file_list(self.data_folder)

            print(f"[DATA LOADER] Found {len(filtered_files)} measurement units")

            if not filtered_files:
                print(f"[DATA LOADER ERROR] No data files found in {self.data_folder}")
                raise DataLoadError(f"No data files found in {self.data_folder}")

            all_data['files'] = filtered_files
            all_data['num_files'] = len(filtered_files)

            print(f"[DATA LOADER] Processing {len(filtered_files)} data files:")
            for i, f in enumerate(filtered_files, 1):
                print(f"[DATA LOADER]   {i}. {os.path.basename(f)}")

            self._on_progress(1, 5, f"Found {len(filtered_files)} data files")
            self._on_step("find_files", filtered_files)

            if self.is_cancelled:
                return

            # Step 2: Extract base data
            print(f"[DATA LOADER] Step 2: Extracting base data...")
            self._on_progress(2, 5, "Extracting base data (angle, temperature, etc.)...")
            df_basedata = self._extract_basedata(filtered_files)

            if df_basedata is None or df_basedata.empty:
                print(f"[DATA LOADER ERROR] Failed to extract base data from files")
                raise DataLoadError("Failed to extract base data from files")

            print(f"[DATA LOADER] Successfully extracted base data from {len(df_basedata)} files")
            all_data['basedata'] = df_basedata
            self._on_step("basedata", df_basedata)

            if self.is_cancelled:
                return

            # Step 3: Extract countrates (optional)
            if self.load_countrates:
                print(f"[DATA LOADER] Step 3: Extracting count rates...")
                self._on_progress(3, 5, "Extracting count rates...")
                all_countrates = self._extract_countrates(filtered_files)
                print(f"[DATA LOADER] Extracted count rates from {len(all_countrates)} files")
                all_data['countrates'] = all_countrates
                self._on_step("countrates", all_countrates)
            else:
                print(f"[DATA LOADER] Step 3: Skipping count rate extraction")
                self._on_progress(3, 5, "Skipping count rate extraction")

            if self.is_cancelled:
                return

            # Step 4: Extract correlations
            if self.load_correlations:
                print(f"[DATA LOADER] Step 4: Extracting correlation data...")
                self._on_progress(4, 5, "Extracting correlation data...")
                all_correlations = self._extract_correlations(filtered_files)

                if not all_correlations:
                    print(f"[DATA LOADER ERROR] Failed to extract correlation data")
                    raise DataLoadError("Failed to extract correlation data")

                print(f"[DATA LOADER] Extracted correlations from {len(all_correlations)} files")
                all_data['correlations'] = all_correlations
                self._on_step("correlations", all_correlations)
            else:
                print(f"[DATA LOADER] Step 4: Skipping correlation extraction")
                self._on_progress(4, 5, "Skipping correlation extraction")

            if self.is_cancelled:
                return

            # Step 4b: Extract instrument-software cumulant fit results (Method A).
            # Non-critical: instruments without a cumulant export just yield {}.
            all_data['cumulant_mode'] = self.parser.CUMULANT_MODE
            if self.load_cumulants:
                print(f"[DATA LOADER] Step 4b: Extracting cumulant results...")
                all_cumulants = self._extract_cumulants(filtered_files)
                print(f"[DATA LOADER] Extracted cumulant results from {len(all_cumulants)} files")
                all_data['cumulants'] = all_cumulants
                self._on_step("cumulants", all_cumulants)
            else:
                print(f"[DATA LOADER] Step 4b: Skipping cumulant extraction")
                all_data['cumulants'] = {}

            if self.is_cancelled:
                return

            # Step 5: Calculate q and q^2
            print(f"[DATA LOADER] Step 5: Calculating scattering vectors...")
            self._on_progress(5, 5, "Calculating scattering vectors...")
            df_basedata = self._calculate_q_vectors(df_basedata)
            all_data['basedata'] = df_basedata

            # Add error summary to results
            all_data['errors'] = self.errors
            all_data['total_files'] = len(filtered_files)
            all_data['successful_files'] = len(filtered_files) - len([e for e in self.errors if e['step'] == 'basedata'])

            # Print summary
            print(f"[DATA LOADER] ===== Data Loading Complete =====")
            print(f"[DATA LOADER] Total files: {all_data['total_files']}")
            print(f"[DATA LOADER] Successful files: {all_data['successful_files']}")
            if self.errors:
                print(f"[DATA LOADER] Errors encountered: {len(self.errors)}")
                for error in self.errors:
                    print(f"[DATA LOADER]   - {error['file']}: {error['error']}")
            else:
                print(f"[DATA LOADER] No errors!")

            return all_data

        except DataLoadError:
            raise
        except Exception as e:
            raise DataLoadError(f"Error loading data: {str(e)}") from e

    def _extract_basedata(self, files):
        """
        Extract base data from all files with batch processing
        to avoid memory issues with large datasets
        """
        import gc  # Garbage collection

        all_data = []
        total = len(files)
        batch_size = 20  # Process in batches to avoid memory issues
        failed_files = []

        for i, file in enumerate(files):
            if self.is_cancelled:
                return None

            # Update progress more frequently
            if i % 5 == 0 or i == total - 1:
                self._on_progress(2, 5, f"Extracting base data: {i+1}/{total} files")

            try:
                # Extract data via the parser plugin (sets filename/folder itself)
                extracted_data = self.parser.extract_basedata(file)

                if extracted_data is not None and not extracted_data.empty:
                    filename = self.parser.get_label(file)
                    all_data.append(extracted_data)
                    print(f"[DATA LOADER]   ✓ {filename}")
                else:
                    # File couldn't be processed
                    filename = self.parser.get_label(file)
                    error_msg = f"No data extracted (file may be corrupt or empty)"
                    self.errors.append({
                        'file': filename,
                        'step': 'basedata',
                        'error': error_msg
                    })
                    failed_files.append(filename)
                    print(f"[DATA LOADER]   ✗ {filename} - {error_msg}")

            except MemoryError:
                filename = self.parser.get_label(file)
                error_msg = "Out of memory"
                self.errors.append({
                    'file': filename,
                    'step': 'basedata',
                    'error': error_msg
                })
                failed_files.append(filename)
                print(f"[DATA LOADER]   ✗ {filename} - MEMORY ERROR")
                # Force garbage collection
                gc.collect()
                continue

            except Exception as e:
                filename = self.parser.get_label(file)
                error_msg = f"{type(e).__name__}: {str(e)}"
                self.errors.append({
                    'file': filename,
                    'step': 'basedata',
                    'error': error_msg
                })
                failed_files.append(filename)
                print(f"[DATA LOADER]   ✗ {filename} - ERROR: {error_msg}")
                import traceback
                traceback.print_exc()
                continue

            # Batch processing: concatenate every batch_size files
            # This prevents memory buildup
            if len(all_data) >= batch_size:
                try:
                    # Concatenate batch and clear list
                    batch_df = pd.concat(all_data, ignore_index=True)
                    all_data = [batch_df]  # Keep only the concatenated result
                    gc.collect()  # Free memory
                except Exception as e:
                    print(f"[DATA LOADER ERROR] Failed to concatenate batch: {e}")
                    # Continue anyway - data is still in all_data list

        # Final concatenation
        if all_data:
            try:
                if len(all_data) == 1 and isinstance(all_data[0], pd.DataFrame):
                    df = all_data[0]
                else:
                    df = pd.concat(all_data, ignore_index=True)

                df.index = df.index + 1

                # Print summary
                if failed_files:
                    print(f"[DATA LOADER] Base data extraction complete:")
                    print(f"[DATA LOADER]   ✓ Success: {len(df)} files")
                    print(f"[DATA LOADER]   ✗ Failed: {len(failed_files)} files")

                # Force garbage collection
                gc.collect()

                return df
            except Exception as e:
                print(f"[DATA LOADER ERROR] Failed to create final DataFrame: {e}")
                import traceback
                traceback.print_exc()
                return pd.DataFrame()
        else:
            print(f"[DATA LOADER ERROR] No data extracted from any file")
            return pd.DataFrame()

    def _extract_countrates(self, files):
        """Extract count rate data from all files"""
        all_countrates = {}
        total = len(files)

        for i, file in enumerate(files):
            if self.is_cancelled:
                return all_countrates

            self._on_progress(3, 5, f"Extracting count rates: {i+1}/{total} files")

            try:
                extracted_countrate = self.parser.extract_countrates(file)
                if extracted_countrate is not None:
                    filename = self.parser.get_label(file)
                    all_countrates[filename] = extracted_countrate
                else:
                    # File couldn't be processed but not critical
                    error_msg = f"Could not extract countrate data"
                    self.errors.append({'file': self.parser.get_label(file), 'step': 'countrates', 'error': error_msg})
                    print(f"[DATA LOADER WARNING] {error_msg} for {self.parser.get_label(file)}")
            except Exception as e:
                error_msg = f"Failed to extract countrate: {str(e)}"
                self.errors.append({'file': self.parser.get_label(file), 'step': 'countrates', 'error': error_msg})
                print(f"[DATA LOADER ERROR] Failed to extract countrate from {self.parser.get_label(file)}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return all_countrates

    def _extract_correlations(self, files):
        """Extract correlation data from all files"""
        all_correlations = {}
        total = len(files)

        for i, file in enumerate(files):
            if self.is_cancelled:
                return all_correlations

            self._on_progress(4, 5, f"Extracting correlations: {i+1}/{total} files")

            try:
                extracted_correlation = self.parser.extract_correlations(file)
                if extracted_correlation is not None:
                    filename = self.parser.get_label(file)
                    all_correlations[filename] = extracted_correlation
                else:
                    # File couldn't be processed but not critical
                    error_msg = f"Could not extract correlation data"
                    self.errors.append({'file': self.parser.get_label(file), 'step': 'correlations', 'error': error_msg})
                    print(f"[DATA LOADER WARNING] {error_msg} for {self.parser.get_label(file)}")
            except Exception as e:
                error_msg = f"Failed to extract correlation: {str(e)}"
                self.errors.append({'file': self.parser.get_label(file), 'step': 'correlations', 'error': error_msg})
                print(f"[DATA LOADER ERROR] Failed to extract correlation from {self.parser.get_label(file)}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return all_correlations

    def _extract_cumulants(self, files):
        """Extract instrument-software cumulant fit results from all files.

        Non-critical (like correlations/countrates): failures are logged in
        self.errors and the file is skipped. Keyed by parser label so it lines
        up with basedata['filename'] and the correlations/countrates dicts."""
        all_cumulants = {}
        total = len(files)

        for i, file in enumerate(files):
            if self.is_cancelled:
                return all_cumulants

            self._on_progress(4, 5, f"Extracting cumulants: {i+1}/{total} files")

            try:
                extracted = self.parser.extract_cumulants(file)
                if extracted is not None:
                    all_cumulants[self.parser.get_label(file)] = extracted
                else:
                    error_msg = "Could not extract cumulant data"
                    self.errors.append({'file': self.parser.get_label(file), 'step': 'cumulants', 'error': error_msg})
                    print(f"[DATA LOADER WARNING] {error_msg} for {self.parser.get_label(file)}")
            except Exception as e:
                error_msg = f"Failed to extract cumulants: {str(e)}"
                self.errors.append({'file': self.parser.get_label(file), 'step': 'cumulants', 'error': error_msg})
                print(f"[DATA LOADER ERROR] Failed to extract cumulants from {self.parser.get_label(file)}: {e}")
                continue

        return all_cumulants

    def _calculate_q_vectors(self, df):
        """
        Calculate scattering vector q and q^2
        Includes robust handling of missing/invalid data
        """
        if df.empty:
            return df

        try:
            # Check if required columns exist
            required_cols = ['angle [°]', 'wavelength [nm]', 'refractive_index']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"[DATA LOADER ERROR] Missing required columns for q calculation: {missing_cols}")
                return df

            # Count rows before filtering
            rows_before = len(df)

            # Remove rows with None/NaN values in required columns
            df = df.dropna(subset=required_cols)

            rows_after = len(df)
            if rows_before != rows_after:
                print(f"[DATA LOADER] Removed {rows_before - rows_after} rows with missing metadata")

            if df.empty:
                print(f"[DATA LOADER ERROR] No valid rows remaining after removing incomplete data")
                return df

            # Calculate q (scattering vector)
            # q = (4π n / λ) * sin(θ/2)
            df['q'] = abs(
                ((4 * np.pi * df['refractive_index']) / df['wavelength [nm]']) *
                np.sin(np.radians(df['angle [°]']) / 2)
            )

            # Calculate q^2
            df['q^2'] = df['q'] ** 2

            print(f"[DATA LOADER] Calculated q and q² for {len(df)} files")

            return df

        except Exception as e:
            print(f"[DATA LOADER ERROR] Failed to calculate q vectors: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            # Return original dataframe without q columns
            return df

    def cancel(self):
        """Cancel the loading process"""
        self.is_cancelled = True
