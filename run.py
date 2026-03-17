"""
ADE-DLS Launcher
Run this file to start the application:
    python run.py
"""
import sys
import os

# Ensure the project root is on the path so package imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication, QSplashScreen
from PyQt5.QtGui import QPixmap, QColor, QPainter, QFont
from PyQt5.QtCore import Qt


def _make_splash_pixmap():
    px = QPixmap(420, 160)
    px.fill(QColor("#1e3a5f"))
    p = QPainter(px)
    p.setPen(QColor("white"))
    p.setFont(QFont("Arial", 28, QFont.Bold))
    p.drawText(px.rect().adjusted(0, -20, 0, -20), Qt.AlignCenter, "ADE-DLS")
    p.setFont(QFont("Arial", 11))
    p.setPen(QColor("#aac4e0"))
    p.drawText(px.rect().adjusted(0, 40, 0, 40), Qt.AlignCenter, "Loading, please wait\u2026")
    p.end()
    return px


if __name__ == '__main__':
    app = QApplication(sys.argv)
    splash = QSplashScreen(_make_splash_pixmap())
    splash.show()
    app.processEvents()  # Splash sofort rendern

    from ade_dls.gui.main_window import main  # schwere Imports passieren hier
    main(app=app, splash=splash)
