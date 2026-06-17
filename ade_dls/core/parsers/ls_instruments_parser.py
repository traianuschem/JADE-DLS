"""
LS Instruments parser (directory-based).

LS Instruments stores one ``Measurement N/`` folder per measurement, each with
``Repetition M/`` sub-folders. A measurement unit = one Repetition folder.

Differences from ALV:
- Lag time is in seconds -> x1000 for milliseconds.
- No wavelength stored -> back-calculated from q and angle.
- The "Scattering vector [1/nm]" column is mislabelled; the value is in 1/m
  (verified: 7.7e6 => lambda ~ 561 nm).
- Viscosity "[mPas]" == "[cp]" (1:1).
- 1 correlation channel, 2 count-rate channels (CHA/CHB).
"""

import glob
import os
from typing import Optional

import numpy as np
import pandas as pd

from .base_parser import InstrumentParser


class LSInstrumentsParser(InstrumentParser):

    INSTRUMENT_NAME = "LSInstruments"
    DESCRIPTION = (
        "LS Instruments spectrometer (LSI, exported via LSLab as CSV)\n\n"
        "Select the folder that contains the 'Measurement 1', 'Measurement 2', … "
        "sub-folders (typically the 'csv' export folder, NOT a single Measurement "
        "or Repetition sub-folder)."
    )
    IS_DIRECTORY_BASED = True
    CUMULANT_MODE = "radius"        # LS reports Rh per order directly, not Γ.
    # FILE_GLOB / HEADER_SIGNATURE unused -> custom detect().

    @classmethod
    def detect(cls, folder_path: str) -> float:
        """Structure detection: Measurement*/Repetition*/Correlation Function.csv."""
        for meas in glob.glob(os.path.join(folder_path, "Measurement*")):
            if not os.path.isdir(meas):
                continue
            for rep in glob.glob(os.path.join(meas, "Repetition*")):
                if os.path.exists(os.path.join(rep, "Correlation Function.csv")):
                    return 1.0
        return 0.0

    def get_file_list(self, folder_path: str) -> list[str]:
        reps = []
        for meas in sorted(glob.glob(os.path.join(folder_path, "Measurement*"))):
            if not os.path.isdir(meas):
                continue
            for rep in sorted(glob.glob(os.path.join(meas, "Repetition*"))):
                if os.path.exists(os.path.join(rep, "Correlation Function.csv")):
                    reps.append(rep)
        return reps

    def get_label(self, measurement_id: str) -> str:
        rep = os.path.basename(measurement_id)                       # "Repetition1"
        meas = os.path.basename(os.path.dirname(measurement_id))     # "Measurement 1"
        return f"{meas} - {rep}"

    def extract_basedata(self, measurement_id: str) -> Optional[pd.DataFrame]:
        path = os.path.join(measurement_id, "Summary.csv")
        if not os.path.exists(path):
            return None
        s = pd.read_csv(path, encoding='utf-8-sig').iloc[0]
        angle = float(s['Scattering angle [deg]'])
        n = float(s['Solvent refractive index'])
        q_per_m = float(s['Scattering vector [1/nm]'])   # header wrong; value is 1/m
        # Back-calculate lambda:  q = 4*pi*n / lambda * sin(theta/2)
        wavelength_nm = (4 * np.pi * n * np.sin(np.radians(angle) / 2)
                         / q_per_m) * 1e9
        return pd.DataFrame({
            'angle [°]':        [angle],
            'temperature [K]':  [float(s['Temperature [K]'])],
            'wavelength [nm]':  [wavelength_nm],
            'refractive_index': [n],
            'viscosity [cp]':   [float(s['Solvent viscosity [mPas]'])],  # mPas = cP
            'filename':         [self.get_label(measurement_id)],
            'folder':           [measurement_id],
        })

    def extract_correlations(self, measurement_id: str) -> Optional[pd.DataFrame]:
        path = os.path.join(measurement_id, "Correlation Function.csv")
        if not os.path.exists(path):
            return None
        # Header: line 1 description, line 2 "Correlation function",
        #         line 3 "Lag time [s],Value".
        df = pd.read_csv(path, skiprows=2, encoding='utf-8-sig')
        df.columns = ['time [ms]', 'correlation 1']
        df['time [ms]'] = df['time [ms]'] * 1000.0            # s -> ms
        df = df.replace([np.inf, -np.inf], np.nan).dropna()   # catch LS special values
        return df

    def extract_countrates(self, measurement_id: str) -> Optional[pd.DataFrame]:
        path = os.path.join(measurement_id, "Count Trace.csv")
        if not os.path.exists(path):
            return None
        # Header: line 1 "Count Rate History", line 2 "Correlation Type:",
        #         line 3 column names.
        df = pd.read_csv(path, skiprows=2, encoding='utf-8-sig')
        df.columns = ['time [s]', 'detectorslot 1', 'detectorslot 2']
        return df

    # Maps the LS per-order radius columns onto Method A's "radius" schema.
    _RH_VALUE_COL = 'Hydrodynamic radius [nm]Value'
    _RH_COV_COL = 'Hydrodynamic radius [nm]Coefficient of variation'
    _RH_STD_COL = 'Hydrodynamic radius [nm]Standard deviation'

    def extract_cumulants(self, measurement_id: str) -> Optional[pd.DataFrame]:
        path = os.path.join(measurement_id, "Cumulants Results.csv")
        if not os.path.exists(path):
            return None
        # Layout: line 1 description, line 2 header, lines 3-5 = cumulant order
        # 1/2/3, then a blank line + a "Fitted results" block (not needed here).
        meta = pd.read_csv(path, skiprows=1, nrows=3, encoding='utf-8-sig')

        order_names = {1: '1st', 2: '2nd', 3: '3rd'}
        out: dict[str, list] = {}
        for _, r in meta.iterrows():
            try:
                order = int(r['Cumulant order'])
            except (ValueError, TypeError):
                continue
            label = order_names.get(order)
            if label is None:
                continue
            # Only keep successful fits; failed ones become NaN.
            ok = str(r.get('Fit succeeded', 'True')).strip().lower() == 'true'
            out[f'{label} order Rh [nm]'] = [float(r[self._RH_VALUE_COL]) if ok else np.nan]
            out[f'{label} order Rh CoV'] = [float(r[self._RH_COV_COL]) if ok else np.nan]
            out[f'{label} order Rh std [nm]'] = [float(r[self._RH_STD_COL]) if ok else np.nan]

        if not out:
            return None
        out['filename'] = [self.get_label(measurement_id)]
        return pd.DataFrame(out)
