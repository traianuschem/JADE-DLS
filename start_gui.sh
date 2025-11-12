#!/bin/bash
# JADE-DLS GUI Launcher Script

echo "========================================="
echo "  JADE-DLS GUI Launcher"
echo "========================================="
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if PyQt5 is installed
echo "Checking dependencies..."
python3 -c "import PyQt5" 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "PyQt5 not found. Installing dependencies..."
    pip3 install -r requirements_gui.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
fi

echo "Dependencies OK"
echo ""
echo "Starting JADE-DLS GUI..."
echo ""

# Launch the GUI
python3 jade_dls_gui.py

echo ""
echo "JADE-DLS GUI closed."
