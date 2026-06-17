# Data Formats and Extending Parsers

## Supported Instruments

ADE-DLS currently ships with parsers for two instrument families:

| Instrument | File format | Extension | Parser |
|------------|-------------|-----------|--------|
| ALV correlators (ALV-5000, ALV-7004, …) | ASCII header + data blocks | `.ASC` | `ALVParser` |
| LS Instruments (NanoLab, …) | Directory per measurement | directory | `LSInstrumentsParser` |

The correct parser is selected automatically based on file extension, header signature, and directory structure — no manual selection is required.

---

## ALV .ASC Format

Each `.ASC` file contains one DLS measurement at one scattering angle. The file is a plain-text format with a keyword header followed by data blocks.

Relevant header fields extracted by the parser:

| Header keyword | Maps to |
|----------------|---------|
| `ALV-...` | Header signature (detection) |
| `Angle` | `angle [°]` |
| `Temperature` | `temperature [K]` |
| `Wavelength` | `wavelength [nm]` |
| `Refractive Index` | `refractive_index` |
| `Viscosity` | `viscosity [cp]` |
| `Count Rate` section | `detectorslot 1..4` vs. time |
| `Correlation` section | `correlation 1..4` vs. `time [ms]` |
| Cumulant fit block | `1st/2nd/3rd order frequency [1/ms]` |

Files from a multi-angle measurement series should all be placed in a single folder. The parser discovers all `.ASC` files in the selected folder.

---

## LS Instruments Format

Each measurement is stored as a **sub-directory** containing multiple files (one per repetition and channel). The parser treats each sub-directory as one measurement unit.

Auto-detection checks for characteristic subdirectory naming conventions.

---

## Parser Architecture

All parsers inherit from `InstrumentParser` ([`ade_dls/core/parsers/base_parser.py`](../ade_dls/core/parsers/base_parser.py)) and are registered automatically. The load dialog queries every registered parser and selects the one with the highest detection confidence.

### Auto-detection

The `detect(folder_path)` class method returns a confidence value (0.0 – 1.0):

- `1.0` — file glob matches **and** header signature confirmed
- `0.8` — file glob matches, no signature check defined
- `0.5` — file glob matches but signature not found
- `0.0` — no matching files

### Data contract

Every parser must return DataFrames with the following column sets:

**`extract_basedata()`** — one-row metadata DataFrame:

| Column | Type | Description |
|--------|------|-------------|
| `angle [°]` | float | Scattering angle |
| `temperature [K]` | float | Sample temperature |
| `wavelength [nm]` | float | Laser wavelength in vacuo |
| `refractive_index` | float | Solvent refractive index |
| `viscosity [cp]` | float | Solvent dynamic viscosity |
| `filename` | str | Unique label (used as dict key) |
| `folder` | str | Source folder path |

**`extract_correlations()`** — g₂(τ) data:

| Column | Type | Description |
|--------|------|-------------|
| `time [ms]` | float | Lag time τ |
| `correlation 1` | float | g₂(τ) for detector 1 |
| `correlation 2..4` | float | (optional) additional detectors |

**`extract_countrates()`** — photon count rates:

| Column | Type | Description |
|--------|------|-------------|
| `time [s]` | float | Measurement time |
| `detectorslot 1` | float | Count rate [kHz] for detector 1 |
| `detectorslot 2..4` | float | (optional) additional detectors |

**`extract_cumulants()`** — optional instrument-software cumulant results (used by Cumulant Method A):

- ALV (`CUMULANT_MODE = "frequency"`): columns `1st order frequency [1/ms]`, `2nd/3rd order frequency [1/ms]`, `2nd/3rd order frequency exp param [ms^2]`
- LS Instruments (`CUMULANT_MODE = "radius"`): columns `1st/2nd/3rd order Rh [nm]`, CoV, std

---

## Adding a New Instrument Parser

1. **Create a new file** in `ade_dls/core/parsers/`, e.g. `malvern_parser.py`. Use `alv_parser.py` as a template.

2. **Subclass `InstrumentParser`** and set the declarative fields:

```python
from .base_parser import InstrumentParser

class MalvernParser(InstrumentParser):
    INSTRUMENT_NAME = "Malvern Zetasizer"
    DESCRIPTION = "Select folder containing .dls files from Malvern Zetasizer"
    FILE_GLOB = "*.dls"
    HEADER_SIGNATURE = "Zetasizer"   # unique string in the file header
    IS_DIRECTORY_BASED = False
    CUMULANT_MODE = "radius"         # or "frequency"
```

3. **Implement the three required methods:**

```python
def get_file_list(self, folder_path: str) -> list[str]:
    import glob, os
    return glob.glob(os.path.join(folder_path, "*.dls"))

def extract_basedata(self, measurement_id: str):
    # Parse header → return single-row DataFrame with required columns
    ...

def extract_correlations(self, measurement_id: str):
    # Parse correlation block → return DataFrame with time [ms] + correlation columns
    ...

def extract_countrates(self, measurement_id: str):
    # Parse count rate block → return DataFrame with time [s] + detectorslot columns
    ...
```

4. **Register the parser** by importing the new module in `ade_dls/core/parsers/__init__.py`:

```python
from . import malvern_parser  # noqa: F401 — registers MalvernParser automatically
```

The parser is now available in the load dialog. No other changes are required.

### Detection confidence tips

- If your format has a unique header keyword, set `HEADER_SIGNATURE` — this gives confidence 1.0.
- For directory-based formats, override `detect()` and check for structural features (sub-folder names, mandatory files).
- Return `None` from any `extract_*` method on parse error; the loader will skip that file and log a warning.
