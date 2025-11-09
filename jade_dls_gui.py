#!/usr/bin/env python3
"""
JADE-DLS GUI Application
Main entry point for the graphical user interface

Usage:
    python jade_dls_gui.py
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import JADEDLSMainWindow


def main():
    """Main application entry point"""

    # Enable High DPI scaling
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("JADE-DLS")
    app.setOrganizationName("JADE-DLS Team")

    # Set application style
    app.setStyle('Fusion')

    # Create and show main window
    window = JADEDLSMainWindow()
    window.show()

    # Run application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
