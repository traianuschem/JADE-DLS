"""Qt-free orchestration layer (analyzers, loading, preprocessing, provenance, exports).

Sits between the numerical kernels (ade_dls.analysis, ade_dls.core, ade_dls.utils) and the
PyQt5 GUI (ade_dls.gui). Nothing in this package may import ade_dls.gui or PyQt;
tests/test_architecture.py enforces that.
"""
