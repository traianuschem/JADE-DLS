"""Moved to ade_dls.engine.provenance (GUI/core separation). This alias keeps the old import path working."""
import sys as _sys
from importlib import import_module as _import_module

_sys.modules[__name__] = _import_module("ade_dls.engine.provenance")
