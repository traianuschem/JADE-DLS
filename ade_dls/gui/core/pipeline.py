"""
Transparent Analysis Pipeline (Qt wrapper)

All logic lives in ade_dls.engine.pipeline.Pipeline; this class only exposes its events as
pyqtSignals for the GUI.
"""

from PyQt5.QtCore import QObject, pyqtSignal

from ade_dls.engine.pipeline import AnalysisStep, Pipeline  # noqa: F401  (AnalysisStep re-exported)


class TransparentPipeline(QObject, Pipeline):
    """
    Transparent analysis pipeline that tracks all steps and can export to
    Jupyter notebooks or Python scripts

    Signals:
        step_added: Emitted when a new step is added
        analysis_completed: Emitted when analysis is complete
        data_loaded: Emitted when data is loaded
    """

    step_added = pyqtSignal(dict)
    analysis_completed = pyqtSignal(dict)
    data_loaded = pyqtSignal(str)

    def __init__(self):
        QObject.__init__(self)
        Pipeline.__init__(self)

    def _emit(self, event: str, payload) -> None:
        getattr(self, event).emit(payload)
        super()._emit(event, payload)
