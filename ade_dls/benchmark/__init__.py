"""Synthetic ground-truth benchmark for ADE-DLS (Qt-free).

``ade_dls.benchmark.synthetic`` simulates multi-angle DLS data (lognormal populations, Biganzoli-Ferri noise,
ALV-7004 lag grid) with known truth and can write ALV ``.ASC`` files, so that the complete chain
(parser -> preprocessing -> methods) can be tested against the truth.
"""
