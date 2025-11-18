#!/usr/bin/env python3
"""
JADE-DLS GUI Application
Main entry point for the graphical user interface

Usage:
    python jade-dls-gui.py
"""

import sys
import os

print("=" * 70)
print("JADE-DLS GUI - Dynamic Light Scattering Analysis")
print("=" * 70)
print("Initializing application... This may take a few moments.")
print("Loading modules and dependencies...")

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

print("✓ Qt modules loaded")

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Loading analysis modules (matplotlib, scipy, pandas)...")
from gui.main_window import JADEDLSMainWindow
print("✓ Analysis modules loaded")


def main():
    """Main application entry point"""
    print("\nStarting JADE-DLS GUI...")

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
    print("Creating main window...")
    window = JADEDLSMainWindow()
    print("✓ Initialization complete!")
    print("=" * 70)
    print()

    window.show()

    # Run application
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
