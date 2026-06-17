"""
Instrument parser plugin system — base class / template.

This module defines the abstract base class every instrument parser inherits
from. Subclasses are registered automatically (PEP 487 ``__init_subclass__``),
so adding a new instrument is a matter of writing one class and importing it in
``parsers/__init__.py``.

The data contract (column names of the returned DataFrames) is documented on
the abstract methods below and MUST be honoured — the analysis and GUI layers
depend on it.
"""

from abc import ABC, abstractmethod
import glob
import os
from typing import Optional

import pandas as pd

# Central registry — populated automatically via __init_subclass__.
INSTRUMENT_PARSERS: list[type["InstrumentParser"]] = []


class InstrumentParser(ABC):
    """Base class (template) for all instrument parsers.

    ADDING A NEW INSTRUMENT:
    1. Copy ``alv_parser.py`` as a template into a new file in ``parsers/``.
    2. Set the declarative fields (INSTRUMENT_NAME, FILE_GLOB,
       HEADER_SIGNATURE, IS_DIRECTORY_BASED).
    3. Implement the three ``extract_*`` methods + ``get_file_list()``.
    4. Import the new file in ``parsers/__init__.py``.
    The rest of the application finds the parser automatically (auto-registration).
    """

    # --- Declarative fields (cover the common case) ---
    INSTRUMENT_NAME: str = "Unknown"     # display name
    DESCRIPTION: str = ""               # shown in the load dialog; explain what folder to select
    FILE_GLOB: str = ""                  # e.g. "*.asc" (empty = no file-based detection)
    HEADER_SIGNATURE: str = ""           # unique string in the file header, e.g. "ALV-"
    IS_DIRECTORY_BASED: bool = False     # True if 1 measurement = 1 sub-folder (LS Instruments)
    CUMULANT_MODE: str = "frequency"     # format of extract_cumulants(): "frequency" (Γ in
                                         # 1/ms + µ₂/µ₃, ALV) or "radius" (Rh per order, LS)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        INSTRUMENT_PARSERS.append(cls)   # automatic registration

    # --- Auto-detection (declarative default; override when needed) ---
    @classmethod
    def detect(cls, folder_path: str) -> float:
        """Return a confidence 0.0-1.0 that this parser can handle ``folder_path``.

        The default covers the "folder with matching files" case (e.g. ALV).
        Directory-based formats (IS_DIRECTORY_BASED) override this method and
        check structural features (sub-folders, filename patterns).

        IMPORTANT: do not perform expensive operations — read only the first
        ~2 KB of a single file.
        """
        if not cls.FILE_GLOB:
            return 0.0
        matches = (glob.glob(os.path.join(folder_path, cls.FILE_GLOB.lower())) +
                   glob.glob(os.path.join(folder_path, cls.FILE_GLOB.upper())))
        if not matches:
            return 0.0
        # Content beats extension: check header signature in the first file.
        if cls.HEADER_SIGNATURE:
            try:
                with open(matches[0], 'r', errors='ignore') as f:
                    head = f.read(2048)
                if cls.HEADER_SIGNATURE in head:
                    return 1.0
            except OSError:
                pass
            return 0.5          # extension matches, signature not confirmed
        return 0.8              # glob only, no signature defined

    # --- File discovery ---
    @abstractmethod
    def get_file_list(self, folder_path: str) -> list[str]:
        """Return the list of measurement units (unique IDs / paths).

        ALV: list of .asc file paths.
        Directory-based: list of sub-folder paths (1 measurement = 1 folder).
        """
        ...

    def get_label(self, measurement_id: str) -> str:
        """Unique, human-readable key for a measurement unit.

        Used by the worker as the dict key (correlations/countrates) AND as
        ``basedata['filename']`` — both MUST match, otherwise the analysis loses
        the link between correlation data and metadata.

        Default: the basename. Directory-based parsers override this, because
        otherwise identically named sub-folders (e.g. 'Repetition1') collide.
        """
        return os.path.basename(measurement_id)

    # --- Parsing per measurement unit (required columns: see docstrings) ---
    @abstractmethod
    def extract_basedata(self, measurement_id: str) -> Optional[pd.DataFrame]:
        """Metadata. Required columns: 'angle [°]', 'temperature [K]',
        'wavelength [nm]', 'refractive_index', 'viscosity [cp]',
        'filename', 'folder'. Returns None on error."""
        ...

    @abstractmethod
    def extract_correlations(self, measurement_id: str) -> Optional[pd.DataFrame]:
        """Correlation function. Required columns: 'time [ms]', 'correlation 1'
        (optional 'correlation 2..4'). Returns None on error."""
        ...

    @abstractmethod
    def extract_countrates(self, measurement_id: str) -> Optional[pd.DataFrame]:
        """Count rates. Required columns: 'time [s]', 'detectorslot 1'
        (optional 'detectorslot 2..4'). Returns None on error."""
        ...

    def extract_cumulants(self, measurement_id: str) -> Optional[pd.DataFrame]:
        """Instrument-software cumulant fit results (used by Cumulant Method A).

        Returns a single-row DataFrame whose columns depend on ``CUMULANT_MODE``:

        - "frequency" (ALV): '1st/2nd/3rd order frequency [1/ms]',
          '2nd/3rd order frequency exp param [ms^2]'.
        - "radius" (LS Instruments): '1st/2nd/3rd order Rh [nm]',
          '1st/2nd/3rd order Rh CoV', '1st/2nd/3rd order Rh std [nm]'.

        Both add a 'filename' column matching extract_basedata()'s label.
        Default returns None (instrument has no software cumulant export)."""
        return None
