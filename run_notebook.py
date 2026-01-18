#!/usr/bin/env python
"""
Simple notebook runner script
"""
import subprocess
import sys

# Run the notebook using nbconvert with the venv Python
result = subprocess.run([
    sys.executable, 
    "-m", "nbconvert",
    "--to", "notebook",
    "--execute",
    "notebooks/06_ml_models.ipynb",
    "--output", "notebooks/06_ml_models.ipynb",
    "--ExecutePreprocessor.timeout=300"
], cwd="d:/Klypto_ML_assignmant")

sys.exit(result.returncode)
