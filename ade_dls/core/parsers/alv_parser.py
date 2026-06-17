"""
ALV instrument parser (reference implementation).

Wraps the existing ``ade_dls.core.preprocessing`` functions without changing
them. File discovery and column renaming that used to live in the data-loader
worker now live here.
"""

import glob
import os
from typing import Optional

import pandas as pd

from .base_parser import InstrumentParser
from ade_dls.core.preprocessing import (extract_data, extract_correlation,
                                        extract_countrate)
from ade_dls.analysis.cumulants import extract_cumulants as _extract_alv_cumulants


class ALVParser(InstrumentParser):

    INSTRUMENT_NAME = "ALV"
    DESCRIPTION = (
        "ALV correlator (ALV-5000, ALV-6000, ALV-7004, …)\n\n"
        "Select the folder that directly contains the .asc measurement files."
    )
    FILE_GLOB = "*.asc"
    HEADER_SIGNATURE = "ALV-"          # first line, e.g. "ALV-7004/USB"
    IS_DIRECTORY_BASED = False
    CUMULANT_MODE = "frequency"        # ALV reports Γ [1/ms] + µ₂/µ₃ directly

    def get_file_list(self, folder_path: str) -> list[str]:
        files = (glob.glob(os.path.join(folder_path, "*.asc")) +
                 glob.glob(os.path.join(folder_path, "*.ASC")))
        # Deduplicate (Windows is case-insensitive and can yield duplicates).
        seen, unique = set(), []
        for f in files:
            key = os.path.abspath(f).lower()
            if key not in seen:
                seen.add(key)
                unique.append(f)
        # Filter out averaged files.
        return [f for f in unique
                if "averaged" not in os.path.basename(f).lower()]

    def extract_basedata(self, measurement_id: str) -> Optional[pd.DataFrame]:
        df = extract_data(measurement_id)
        if df is None or df.empty:
            return None
        df['filename'] = self.get_label(measurement_id)   # = basename (ALV default)
        df['folder'] = os.path.dirname(os.path.abspath(measurement_id))
        return df

    def extract_correlations(self, measurement_id: str) -> Optional[pd.DataFrame]:
        df = extract_correlation(measurement_id)
        if df is None:
            return None
        return df.rename(columns={0: 'time [ms]', 1: 'correlation 1',
                                  2: 'correlation 2', 3: 'correlation 3',
                                  4: 'correlation 4'})

    def extract_countrates(self, measurement_id: str) -> Optional[pd.DataFrame]:
        df = extract_countrate(measurement_id)
        if df is None:
            return None
        return df.rename(columns={0: 'time [s]', 1: 'detectorslot 1',
                                  2: 'detectorslot 2', 3: 'detectorslot 3',
                                  4: 'detectorslot 4'})

    def extract_cumulants(self, measurement_id: str) -> Optional[pd.DataFrame]:
        # ALV stores the software cumulant fit (Γ orders 1-3 + µ₂/µ₃) inside the
        # same .asc file; the existing parser already returns the right columns.
        df = _extract_alv_cumulants(measurement_id)
        if df is None or df.empty:
            return None
        df['filename'] = self.get_label(measurement_id)   # = basename (ALV default)
        return df
