"""
Instrument parser plugin package.

Importing the concrete parser modules below triggers their auto-registration
(via ``InstrumentParser.__init_subclass__``). The explicit imports keep
registration deterministic and PyInstaller-safe (no entry-point / autodiscovery
magic that breaks in a frozen build).
"""

from typing import Optional

from .base_parser import InstrumentParser, INSTRUMENT_PARSERS
from . import alv_parser              # noqa: F401 - import triggers registration
from . import ls_instruments_parser   # noqa: F401 - import triggers registration


def detect_parser(folder_path: str) -> Optional[InstrumentParser]:
    """Return an instance of the parser with the highest confidence, or None.

    Confidence scoring (Mantid pattern): each registered parser scores the
    folder; the highest wins. A score of 0.0 means "not my format".
    """
    scored = [(parser_cls.detect(folder_path), parser_cls)
              for parser_cls in INSTRUMENT_PARSERS]
    scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best_cls = scored[0]
    return best_cls() if best_score > 0.0 else None
