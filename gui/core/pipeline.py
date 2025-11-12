"""
Transparent Analysis Pipeline
Tracks all analysis steps, parameters, and generates reproducible code
"""

import inspect
import json
from datetime import datetime
from typing import List, Dict, Any, Callable
from PyQt5.QtCore import QObject, pyqtSignal
import pandas as pd
import numpy as np


class AnalysisStep:
    """Represents a single step in the analysis pipeline"""

    def __init__(self, name: str, function: Callable = None, params: Dict[str, Any] = None,
                 step_type: str = "function", custom_code: str = None):
        self.name = name
        self.function = function
        self.function_name = function.__name__ if function else None
        self.params = params or {}
        self.timestamp = datetime.now()
        self.result = None
        self.code = None
        self.metadata = {}
        self.step_type = step_type  # 'function' or 'filter' or 'custom'
        self.custom_code = custom_code

        # Extract function source code
        if function:
            try:
                self.code = inspect.getsource(function)
            except (OSError, TypeError):
                self.code = f"# Built-in function: {self.function_name}"

    def execute(self, *args, **kwargs):
        """Execute this analysis step"""
        if self.function:
            self.result = self.function(*args, **self.params, **kwargs)
            return self.result
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization"""
        return {
            'name': self.name,
            'function': self.function_name,
            'params': self.params,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'step_type': self.step_type
        }

    def generate_code(self) -> str:
        """Generate Python code for this step"""
        # If we have custom code (for filters or manual steps), use that
        if self.custom_code:
            return f"""
# Step: {self.name}
# Timestamp: {self.timestamp}
{self.custom_code}
"""
        elif self.function:
            params_str = ', '.join([f"{k}={repr(v)}" for k, v in self.params.items()])
            return f"""
# Step: {self.name}
# Timestamp: {self.timestamp}
result_{self.name.replace(' ', '_')} = {self.function_name}({params_str})
"""
        else:
            return f"""
# Step: {self.name}
# Timestamp: {self.timestamp}
# Manual step - no automatic code generation
"""


class TransparentPipeline(QObject):
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
        super().__init__()
        self.steps: List[AnalysisStep] = []
        self.data = {}
        self.results = {}
        self.metadata = {
            'created': datetime.now(),
            'version': '2.0',
            'description': 'JADE-DLS Analysis Pipeline'
        }

    def add_step(self, name: str, function: Callable, params: Dict[str, Any] = None) -> AnalysisStep:
        """
        Add a step to the pipeline

        Args:
            name: Human-readable step name
            function: Function to execute
            params: Parameters for the function

        Returns:
            The created AnalysisStep
        """
        if params is None:
            params = {}

        step = AnalysisStep(name, function, params)
        self.steps.append(step)

        # Emit signal
        self.step_added.emit(step.to_dict())

        return step

    def add_filter_step(self, filter_type: str, excluded_files: List[str],
                       original_count: int, remaining_count: int) -> AnalysisStep:
        """
        Add a data filtering step to the pipeline

        This generates reproducible code that excludes specific files

        Args:
            filter_type: Type of filter ('countrate' or 'correlation')
            excluded_files: List of filenames that were excluded
            original_count: Number of files before filtering
            remaining_count: Number of files after filtering

        Returns:
            The created AnalysisStep
        """
        # Generate code for this filtering step
        if len(excluded_files) > 0:
            files_list_str = ',\n    '.join([f"'{f}'" for f in excluded_files])
            code = f"""
# Filtering step: Exclude {len(excluded_files)} files based on {filter_type}
# Original files: {original_count}, Remaining: {remaining_count}
files_to_exclude_{filter_type} = [
    {files_list_str}
]

# Filter {filter_type} data
{filter_type}_data = {{k: v for k, v in {filter_type}_data.items()
                if k not in files_to_exclude_{filter_type}}}

print(f"Filtered {filter_type}: {{len(files_to_exclude_{filter_type})}} files excluded, {{len({filter_type}_data)}} files remaining")

# Update basedata to match (if it exists)
if 'df_basedata' in locals() and len(df_basedata) > 0:
    df_basedata = df_basedata[~df_basedata['filename'].isin(files_to_exclude_{filter_type})]
    df_basedata = df_basedata.reset_index(drop=True)
    df_basedata.index = df_basedata.index + 1
"""
        else:
            code = f"""
# Filtering step: No files excluded from {filter_type}
# All {original_count} files kept
print(f"No files excluded from {filter_type} filtering - {{len({filter_type}_data)}} files kept")
"""

        # Create step with custom code
        step = AnalysisStep(
            name=f"Filter {filter_type.capitalize()}",
            step_type='filter',
            custom_code=code,
            params={
                'filter_type': filter_type,
                'excluded_files': excluded_files,
                'original_count': original_count,
                'remaining_count': remaining_count
            }
        )

        self.steps.append(step)

        # Emit signal
        self.step_added.emit(step.to_dict())

        return step

    def execute_step(self, step_index: int, *args, **kwargs):
        """Execute a specific step by index"""
        if 0 <= step_index < len(self.steps):
            step = self.steps[step_index]
            result = step.execute(*args, **kwargs)
            self.results[step.name] = result
            return result
        else:
            raise IndexError(f"Step index {step_index} out of range")

    def execute_all(self):
        """Execute all steps in the pipeline"""
        for i, step in enumerate(self.steps):
            print(f"Executing step {i+1}/{len(self.steps)}: {step.name}")
            self.execute_step(i)

        self.analysis_completed.emit(self.results)

    def load_data(self, data_folder: str):
        """
        Load data from folder

        Args:
            data_folder: Path to folder containing .asc files
        """
        import glob
        import os
        from preprocessing import extract_data, extract_correlation

        # Find all .asc files
        datafiles = glob.glob(os.path.join(data_folder, "*.asc"))
        filtered_files = [f for f in datafiles
                          if "averaged" not in os.path.basename(f).lower()]

        # Record this as a step
        self.add_step(
            "Load Data",
            self._load_data_internal,
            {'data_folder': data_folder, 'files': filtered_files}
        )

        # Store data
        self.data['data_folder'] = data_folder
        self.data['files'] = filtered_files
        self.data['num_files'] = len(filtered_files)

        # Emit signal
        self.data_loaded.emit(data_folder)

    def _load_data_internal(self, data_folder: str, files: List[str]):
        """Internal data loading function"""
        # This would call the actual preprocessing functions
        return {'folder': data_folder, 'count': len(files)}

    def get_current_code(self) -> str:
        """
        Generate Python code for all steps executed so far

        Returns:
            Python code as string
        """
        code_parts = []

        # Header
        code_parts.append(f"""
# JADE-DLS Analysis Script
# Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Generated from GUI session

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import extract_data, extract_correlation
from cumulants import analyze_diffusion_coefficient
from regularized import nnls_reg_all
import glob
import os
""")

        # Add each step
        for i, step in enumerate(self.steps):
            code_parts.append(f"\n# ========== Step {i+1}: {step.name} ==========")
            code_parts.append(step.generate_code())

        return '\n'.join(code_parts)

    def export_to_script(self, filename: str):
        """
        Export pipeline to Python script

        Args:
            filename: Output filename
        """
        code = self.get_current_code()

        with open(filename, 'w') as f:
            f.write(code)

    def export_to_notebook(self, filename: str):
        """
        Export pipeline to Jupyter notebook

        Args:
            filename: Output filename (.ipynb)
        """
        import nbformat
        from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

        nb = new_notebook()

        # Title cell
        nb.cells.append(new_markdown_cell(f"""# JADE-DLS Analysis

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data Folder:** {self.data.get('data_folder', 'N/A')}
**Number of Files:** {self.data.get('num_files', 0)}

This notebook was auto-generated from the JADE-DLS GUI and contains all analysis steps.
"""))

        # Imports cell
        nb.cells.append(new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import extract_data, extract_correlation
from cumulants import analyze_diffusion_coefficient
from regularized import nnls_reg_all
import glob
import os

%matplotlib inline"""))

        # Add each step as a separate cell
        for i, step in enumerate(self.steps):
            # Markdown description
            nb.cells.append(new_markdown_cell(f"""## Step {i+1}: {step.name}

**Timestamp:** {step.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Function:** `{step.function_name}`
**Parameters:**
```python
{json.dumps(step.params, indent=2)}
```
"""))

            # Code cell
            nb.cells.append(new_code_cell(step.generate_code()))

        # Write notebook
        with open(filename, 'w') as f:
            nbformat.write(nb, f)

    def export_to_pdf(self, filename: str):
        """
        Export complete analysis report to PDF

        Args:
            filename: Output PDF filename
        """
        # This would use reportlab or similar to create a PDF
        # For now, we'll create a simple text-based report
        report = f"""
JADE-DLS Analysis Report
========================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data Folder: {self.data.get('data_folder', 'N/A')}
Number of Files: {self.data.get('num_files', 0)}

Analysis Steps:
---------------
"""
        for i, step in enumerate(self.steps):
            report += f"\n{i+1}. {step.name}\n"
            report += f"   Function: {step.function_name}\n"
            report += f"   Parameters: {step.params}\n"

        # For now, save as text (PDF generation would need reportlab)
        text_filename = filename.replace('.pdf', '.txt')
        with open(text_filename, 'w') as f:
            f.write(report)

    def get_parameter_history(self) -> pd.DataFrame:
        """
        Get history of all parameters used in the pipeline

        Returns:
            DataFrame with parameter history
        """
        history = []
        for step in self.steps:
            for param_name, param_value in step.params.items():
                history.append({
                    'step': step.name,
                    'timestamp': step.timestamp,
                    'parameter': param_name,
                    'value': param_value
                })

        return pd.DataFrame(history)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of the pipeline

        Returns:
            Dictionary with pipeline summary
        """
        return {
            'total_steps': len(self.steps),
            'created': self.metadata['created'],
            'data_folder': self.data.get('data_folder'),
            'num_files': self.data.get('num_files', 0),
            'steps': [step.name for step in self.steps]
        }

    def clear(self):
        """Clear all steps and results"""
        self.steps = []
        self.results = {}

    def save_session(self, filename: str):
        """
        Save pipeline session to JSON

        Args:
            filename: Output JSON filename
        """
        session_data = {
            'metadata': {
                **self.metadata,
                'created': self.metadata['created'].isoformat()
            },
            'data': self.data,
            'steps': [step.to_dict() for step in self.steps]
        }

        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)

    def load_session(self, filename: str):
        """
        Load pipeline session from JSON

        Args:
            filename: Input JSON filename
        """
        with open(filename, 'r') as f:
            session_data = json.load(f)

        self.metadata = session_data['metadata']
        self.data = session_data['data']

        # Note: Can't fully restore functions from JSON,
        # would need to re-create steps manually
        print("Session metadata loaded. Note: Step functions need manual restoration.")
