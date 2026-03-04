"""
ADE-DLS Launcher
Run this file to start the application:
    python run.py
"""
import sys
import os

# Ensure the project root is on the path so package imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ade_dls.gui.main_window import main

if __name__ == '__main__':
    main()
