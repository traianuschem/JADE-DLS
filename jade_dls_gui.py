#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JADE-DLS GUI Application
Jupyter-based Angular Dependent Evaluator for Dynamic Light Scattering
with Dark Mode Support
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import glob
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

# Import custom modules
from preprocessing import (extract_data, extract_countrate, extract_correlation,
                          process_correlation_data, remove_from_data, remove_dataframes)
from cumulants import (extract_cumulants, analyze_diffusion_coefficient,
                       calculate_cumulant_results_A, create_zero_cumulant_results_A,
                       calculate_g2_B, plot_processed_correlations, remove_rows_by_index)
from cumulants_C import (plot_processed_correlations_iterative, get_adaptive_initial_parameters,
                        get_meaningful_parameters, calculate_mean_fit_metrics)
from regularized import nnls_all, nnls_reg_all, calculate_decay_rates
from scipy.constants import k

class DarkModeStyle:
    """Dark mode color scheme for the GUI"""
    BG_DARK = "#2b2b2b"
    BG_MEDIUM = "#3c3c3c"
    BG_LIGHT = "#4a4a4a"
    FG_TEXT = "#e0e0e0"
    FG_HIGHLIGHT = "#ffffff"
    ACCENT = "#4a90e2"
    ACCENT_HOVER = "#6aa3e8"
    BORDER = "#555555"
    ERROR = "#ff6b6b"
    SUCCESS = "#51cf66"

class LightModeStyle:
    """Light mode color scheme for the GUI"""
    BG_DARK = "#f0f0f0"
    BG_MEDIUM = "#ffffff"
    BG_LIGHT = "#fafafa"
    FG_TEXT = "#000000"
    FG_HIGHLIGHT = "#000000"
    ACCENT = "#0066cc"
    ACCENT_HOVER = "#0052a3"
    BORDER = "#cccccc"
    ERROR = "#d32f2f"
    SUCCESS = "#2e7d32"

class JADEDLSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("JADE-DLS: Dynamic Light Scattering Analysis")
        self.root.geometry("1200x800")

        # Data storage
        self.df_basedata = None
        self.all_correlations = {}
        self.processed_correlations = {}
        self.method_A_data = None
        self.method_B_data = None
        self.method_C_data = None
        self.c_value = None
        self.delta_c = None

        # Dark mode state
        self.dark_mode = tk.BooleanVar(value=True)
        self.current_style = DarkModeStyle()

        # Configure matplotlib for dark mode
        self.setup_matplotlib_style()

        # Create GUI
        self.create_menu()
        self.create_widgets()
        self.apply_theme()

    def setup_matplotlib_style(self):
        """Configure matplotlib for current theme"""
        if self.dark_mode.get():
            plt.style.use('dark_background')
        else:
            plt.style.use('default')

    def toggle_dark_mode(self):
        """Toggle between dark and light mode"""
        self.dark_mode.set(not self.dark_mode.get())
        if self.dark_mode.get():
            self.current_style = DarkModeStyle()
        else:
            self.current_style = LightModeStyle()
        self.setup_matplotlib_style()
        self.apply_theme()

    def apply_theme(self):
        """Apply the current theme to all widgets"""
        style = ttk.Style()

        # Configure ttk styles
        style.theme_use('clam')

        # Main colors
        style.configure('.',
                       background=self.current_style.BG_MEDIUM,
                       foreground=self.current_style.FG_TEXT,
                       fieldbackground=self.current_style.BG_LIGHT,
                       bordercolor=self.current_style.BORDER)

        # Button style
        style.configure('TButton',
                       background=self.current_style.ACCENT,
                       foreground=self.current_style.FG_HIGHLIGHT,
                       borderwidth=1,
                       focuscolor=self.current_style.ACCENT_HOVER,
                       lightcolor=self.current_style.ACCENT_HOVER,
                       darkcolor=self.current_style.ACCENT)
        style.map('TButton',
                 background=[('active', self.current_style.ACCENT_HOVER)])

        # Frame style
        style.configure('TFrame',
                       background=self.current_style.BG_MEDIUM,
                       borderwidth=1,
                       relief='flat')

        # Label style
        style.configure('TLabel',
                       background=self.current_style.BG_MEDIUM,
                       foreground=self.current_style.FG_TEXT)

        # Entry style
        style.configure('TEntry',
                       fieldbackground=self.current_style.BG_LIGHT,
                       foreground=self.current_style.FG_TEXT,
                       bordercolor=self.current_style.BORDER,
                       lightcolor=self.current_style.BORDER,
                       darkcolor=self.current_style.BORDER)

        # Notebook style
        style.configure('TNotebook',
                       background=self.current_style.BG_MEDIUM,
                       borderwidth=0)
        style.configure('TNotebook.Tab',
                       background=self.current_style.BG_DARK,
                       foreground=self.current_style.FG_TEXT,
                       padding=[10, 5])
        style.map('TNotebook.Tab',
                 background=[('selected', self.current_style.BG_MEDIUM)],
                 foreground=[('selected', self.current_style.FG_HIGHLIGHT)])

        # Apply to root window
        self.root.configure(bg=self.current_style.BG_MEDIUM)

        # Update text widgets - keep them white with black text for readability
        text_widgets = []
        if hasattr(self, 'log_text'):
            text_widgets.append(self.log_text)
        if hasattr(self, 'summary_text'):
            text_widgets.append(self.summary_text)
        if hasattr(self, 'method_a_results'):
            text_widgets.append(self.method_a_results)
        if hasattr(self, 'method_b_results'):
            text_widgets.append(self.method_b_results)
        if hasattr(self, 'method_c_results'):
            text_widgets.append(self.method_c_results)

        # Always use white background with black text for readability
        for widget in text_widgets:
            widget.configure(
                bg='#ffffff',
                fg='#000000',
                insertbackground='#000000')

    def create_menu(self):
        """Create menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Data", command=self.load_data)
        file_menu.add_command(label="Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Dark Mode",
                                  variable=self.dark_mode,
                                  command=self.toggle_dark_mode)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def create_widgets(self):
        """Create main GUI widgets"""
        # Main container with padding
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(1, weight=1)

        # Top control panel
        control_frame = ttk.LabelFrame(main_container, text="Data Loading", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)

        # Data folder selection
        ttk.Label(control_frame, text="Data Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.folder_var = tk.StringVar()
        folder_entry = ttk.Entry(control_frame, textvariable=self.folder_var)
        folder_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(control_frame, text="Browse...",
                  command=self.browse_folder).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(control_frame, text="Load Data",
                  command=self.load_data).grid(row=0, column=3, padx=5, pady=5)

        # Notebook for different analysis methods
        self.notebook = ttk.Notebook(main_container)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create tabs
        self.create_overview_tab()
        self.create_method_a_tab()
        self.create_method_b_tab()
        self.create_method_c_tab()
        self.create_regularized_tab()
        self.create_log_tab()

    def create_overview_tab(self):
        """Create overview tab"""
        overview_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(overview_frame, text="Overview")

        overview_frame.columnconfigure(0, weight=1)
        overview_frame.rowconfigure(1, weight=1)

        # Info label
        info_text = """
        JADE-DLS: Jupyter-based Angular Dependent Evaluator for Dynamic Light Scattering

        Steps:
        1. Load your .asc data files using the 'Data Folder' browser
        2. Select an analysis method from the tabs
        3. Configure parameters and run analysis
        4. Export results when complete
        """
        info_label = ttk.Label(overview_frame, text=info_text, justify=tk.LEFT)
        info_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N), pady=10)

        # Data summary
        self.summary_frame = ttk.LabelFrame(overview_frame, text="Data Summary", padding="10")
        self.summary_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.summary_text = scrolledtext.ScrolledText(
            self.summary_frame,
            height=20,
            wrap=tk.WORD,
            font=('Courier', 10))
        self.summary_text.pack(fill=tk.BOTH, expand=True)

    def create_method_a_tab(self):
        """Create Cumulant Method A tab"""
        method_a_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(method_a_frame, text="Method A (Cumulant)")

        method_a_frame.columnconfigure(0, weight=1)
        method_a_frame.rowconfigure(2, weight=1)

        # Description
        desc = ttk.Label(method_a_frame,
                        text="Method A: Uses cumulant fit data from ALV-Software",
                        font=('TkDefaultFont', 10, 'bold'))
        desc.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        # Control frame
        control = ttk.Frame(method_a_frame)
        control.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # q-range inputs
        ttk.Label(control, text="q² Fit Range (optional):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Label(control, text="Min:").grid(row=0, column=1, padx=(10, 5))
        self.method_a_qmin = ttk.Entry(control, width=10)
        self.method_a_qmin.grid(row=0, column=2, padx=5)
        ttk.Label(control, text="Max:").grid(row=0, column=3, padx=5)
        self.method_a_qmax = ttk.Entry(control, width=10)
        self.method_a_qmax.grid(row=0, column=4, padx=5)

        ttk.Button(control, text="Run Analysis",
                  command=self.run_method_a).grid(row=0, column=5, padx=20)

        # Results frame
        results_frame = ttk.LabelFrame(method_a_frame, text="Results", padding="10")
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.method_a_results = scrolledtext.ScrolledText(
            results_frame,
            height=15,
            wrap=tk.NONE,
            font=('Courier', 9))
        self.method_a_results.pack(fill=tk.BOTH, expand=True)

    def create_method_b_tab(self):
        """Create Cumulant Method B tab with postfiltering"""
        method_b_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(method_b_frame, text="Method B (Linear)")

        method_b_frame.columnconfigure(0, weight=1)
        method_b_frame.rowconfigure(2, weight=1)

        # Description
        desc = ttk.Label(method_b_frame,
                        text="Method B: Simplest method using linear fit",
                        font=('TkDefaultFont', 10, 'bold'))
        desc.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        # Control frame
        control = ttk.Frame(method_b_frame)
        control.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Fit limits
        ttk.Label(control, text="Fit Limits (s):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Label(control, text="Min:").grid(row=0, column=1, padx=(10, 5))
        self.method_b_tmin = ttk.Entry(control, width=10)
        self.method_b_tmin.insert(0, "0")
        self.method_b_tmin.grid(row=0, column=2, padx=5)
        ttk.Label(control, text="Max:").grid(row=0, column=3, padx=5)
        self.method_b_tmax = ttk.Entry(control, width=10)
        self.method_b_tmax.insert(0, "0.0002")
        self.method_b_tmax.grid(row=0, column=4, padx=5)

        # q-range inputs
        ttk.Label(control, text="q² Fit Range:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(control, text="Min:").grid(row=1, column=1, padx=(10, 5))
        self.method_b_qmin = ttk.Entry(control, width=10)
        self.method_b_qmin.grid(row=1, column=2, padx=5)
        ttk.Label(control, text="Max:").grid(row=1, column=3, padx=5)
        self.method_b_qmax = ttk.Entry(control, width=10)
        self.method_b_qmax.grid(row=1, column=4, padx=5)

        # Buttons
        button_frame = ttk.Frame(control)
        button_frame.grid(row=0, column=5, rowspan=2, padx=20)
        ttk.Button(button_frame, text="Run Analysis",
                  command=self.run_method_b).pack(pady=2)
        ttk.Button(button_frame, text="Post-Filter Results",
                  command=self.postfilter_method_b).pack(pady=2)

        # Results frame
        results_frame = ttk.LabelFrame(method_b_frame, text="Results", padding="10")
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.method_b_results = scrolledtext.ScrolledText(
            results_frame,
            height=15,
            wrap=tk.NONE,
            font=('Courier', 9))
        self.method_b_results.pack(fill=tk.BOTH, expand=True)

    def create_method_c_tab(self):
        """Create Cumulant Method C tab with postfiltering"""
        method_c_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(method_c_frame, text="Method C (Iterative)")

        method_c_frame.columnconfigure(0, weight=1)
        method_c_frame.rowconfigure(2, weight=1)

        # Description
        desc = ttk.Label(method_c_frame,
                        text="Method C: Iterative nonlinear fit method",
                        font=('TkDefaultFont', 10, 'bold'))
        desc.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        # Control frame
        control = ttk.Frame(method_c_frame)
        control.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Fit limits
        ttk.Label(control, text="Fit Limits (s):").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Label(control, text="Min:").grid(row=0, column=1, padx=(10, 5))
        self.method_c_tmin = ttk.Entry(control, width=10)
        self.method_c_tmin.insert(0, "1e-9")
        self.method_c_tmin.grid(row=0, column=2, padx=5)
        ttk.Label(control, text="Max:").grid(row=0, column=3, padx=5)
        self.method_c_tmax = ttk.Entry(control, width=10)
        self.method_c_tmax.insert(0, "10")
        self.method_c_tmax.grid(row=0, column=4, padx=5)

        # q-range inputs
        ttk.Label(control, text="q² Fit Range:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Label(control, text="Min:").grid(row=1, column=1, padx=(10, 5))
        self.method_c_qmin = ttk.Entry(control, width=10)
        self.method_c_qmin.grid(row=1, column=2, padx=5)
        ttk.Label(control, text="Max:").grid(row=1, column=3, padx=5)
        self.method_c_qmax = ttk.Entry(control, width=10)
        self.method_c_qmax.grid(row=1, column=4, padx=5)

        # Buttons
        button_frame = ttk.Frame(control)
        button_frame.grid(row=0, column=5, rowspan=2, padx=20)
        ttk.Button(button_frame, text="Run Analysis",
                  command=self.run_method_c).pack(pady=2)
        ttk.Button(button_frame, text="Post-Filter Results",
                  command=self.postfilter_method_c).pack(pady=2)

        # Results frame
        results_frame = ttk.LabelFrame(method_c_frame, text="Results", padding="10")
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.method_c_results = scrolledtext.ScrolledText(
            results_frame,
            height=15,
            wrap=tk.NONE,
            font=('Courier', 9))
        self.method_c_results.pack(fill=tk.BOTH, expand=True)

    def create_regularized_tab(self):
        """Create Regularized Fit tab"""
        reg_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(reg_frame, text="Regularized Fit")

        ttk.Label(reg_frame,
                 text="Regularized NNLS Fit - To be implemented",
                 font=('TkDefaultFont', 10, 'bold')).pack(pady=20)

    def create_log_tab(self):
        """Create log/console tab"""
        log_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(log_frame, text="Log")

        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=20,
            wrap=tk.WORD,
            font=('Courier', 9))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Clear log button
        ttk.Button(log_frame, text="Clear Log",
                  command=lambda: self.log_text.delete(1.0, tk.END)).grid(
                      row=1, column=0, sticky=tk.E, pady=(5, 0))

    def log(self, message):
        """Add message to log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def browse_folder(self):
        """Browse for data folder"""
        folder = filedialog.askdirectory()
        if folder:
            self.folder_var.set(folder)

    def load_data(self):
        """Load data from selected folder"""
        folder = self.folder_var.get()
        if not folder:
            messagebox.showerror("Error", "Please select a data folder first!")
            return

        try:
            self.log("Loading data from: " + folder)

            # Get all .asc files
            datafiles = glob.glob(os.path.join(folder, "*.asc"))
            filtered_files = [f for f in datafiles
                            if "averaged" not in os.path.basename(f).lower()]

            self.log(f"Found {len(filtered_files)} data files")

            # Extract base data
            all_data = []
            for file in filtered_files:
                extracted_data = extract_data(file)
                if extracted_data is not None:
                    filename = os.path.basename(file)
                    extracted_data['filename'] = filename
                    all_data.append(extracted_data)

            if all_data:
                self.df_basedata = pd.concat(all_data, ignore_index=True)
                self.df_basedata.index = self.df_basedata.index + 1

                # Calculate q and q^2
                self.df_basedata['q'] = abs(((4*np.pi*self.df_basedata['refractive_index'])/
                                            (self.df_basedata['wavelength [nm]']))*
                                           np.sin(np.radians(self.df_basedata['angle [°]'])/2))
                self.df_basedata['q^2'] = (self.df_basedata['q']**2)

                self.log(f"Base data loaded: {len(self.df_basedata)} entries")
            else:
                self.log("ERROR: No data extracted!")
                return

            # Extract correlations
            self.all_correlations = {}
            for file in filtered_files:
                filename = os.path.basename(file)
                extracted_correlation = extract_correlation(file)
                if extracted_correlation is not None:
                    self.all_correlations[filename] = extracted_correlation

            # Rename columns
            new_column_names = {0: 'time [ms]', 1: 'correlation 1',
                              2: 'correlation 2', 3: 'correlation 3', 4: 'correlation 4'}
            self.all_correlations = {key: df.rename(columns=new_column_names)
                                    for key, df in self.all_correlations.items()}

            self.log(f"Extracted correlation data for {len(self.all_correlations)} files")

            # Process correlations
            columns_to_drop = ['time [ms]', 'correlation 1', 'correlation 2',
                             'correlation 3', 'correlation 4']
            self.processed_correlations = process_correlation_data(
                self.all_correlations, columns_to_drop)

            # Calculate c value
            mean_temperature = self.df_basedata['temperature [K]'].mean()
            std_temperature = self.df_basedata['temperature [K]'].std()
            mean_viscosity = self.df_basedata['viscosity [cp]'].mean()
            std_viscosity = self.df_basedata['viscosity [cp]'].std()

            self.c_value = (k*mean_temperature)/(6*np.pi*mean_viscosity*10**(-3))
            fractional_error_c = np.sqrt((std_temperature / mean_temperature)**2 +
                                        (std_viscosity / mean_viscosity)**2)
            self.delta_c = fractional_error_c * self.c_value

            self.log(f"c = {self.c_value:.4e} +/- {self.delta_c:.4e}")

            # Update summary
            self.update_summary()

            messagebox.showinfo("Success", "Data loaded successfully!")

        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")

    def update_summary(self):
        """Update data summary"""
        if self.df_basedata is None:
            return

        summary = "DATA SUMMARY\n"
        summary += "="*60 + "\n\n"
        summary += f"Number of files: {len(self.df_basedata)}\n"
        summary += f"Temperature: {self.df_basedata['temperature [K]'].mean():.2f} ± "
        summary += f"{self.df_basedata['temperature [K]'].std():.4f} K\n"
        summary += f"Viscosity: {self.df_basedata['viscosity [cp]'].mean():.4f} ± "
        summary += f"{self.df_basedata['viscosity [cp]'].std():.6f} cp\n"
        summary += f"Wavelength: {self.df_basedata['wavelength [nm]'].mean():.1f} nm\n"
        summary += f"Refractive index: {self.df_basedata['refractive_index'].mean():.4f}\n"
        summary += f"\nAngles: {sorted(self.df_basedata['angle [°]'].unique())}\n"
        summary += f"\nq² range: {self.df_basedata['q^2'].min():.6f} to {self.df_basedata['q^2'].max():.6f} nm⁻²\n"

        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary)

    def run_method_a(self):
        """Run Cumulant Method A analysis"""
        if self.df_basedata is None:
            messagebox.showerror("Error", "Please load data first!")
            return

        try:
            self.log("\n" + "="*60)
            self.log("Running Cumulant Method A...")

            # Extract cumulants
            all_data = []
            file_to_path = {os.path.basename(f): self.folder_var.get() + "/" + os.path.basename(f)
                          for f in glob.glob(self.folder_var.get() + "/*.asc")}

            for filename in self.all_correlations.keys():
                if filename in file_to_path:
                    file_path = file_to_path[filename]
                    extracted_cumulants = extract_cumulants(file_path)
                    if extracted_cumulants is not None:
                        extracted_cumulants['filename'] = filename
                        all_data.append(extracted_cumulants)

            if not all_data:
                self.log("ERROR: No cumulant data extracted!")
                return

            df_extracted_cumulants = pd.concat(all_data, ignore_index=True)
            df_extracted_cumulants.index = df_extracted_cumulants.index + 1

            # Merge with base data
            cumulant_method_A_data = pd.merge(
                self.df_basedata, df_extracted_cumulants,
                on='filename', how='outer')
            cumulant_method_A_data = cumulant_method_A_data.reset_index(drop=True)
            cumulant_method_A_data.index = cumulant_method_A_data.index + 1

            # Get q-range if specified
            x_range = None
            qmin_text = self.method_a_qmin.get().strip()
            qmax_text = self.method_a_qmax.get().strip()
            if qmin_text and qmax_text:
                try:
                    x_range = (float(qmin_text), float(qmax_text))
                    self.log(f"Using q² range: {x_range}")
                except ValueError:
                    self.log("Warning: Invalid q-range values, using full range")

            # Analyze diffusion coefficient
            cumulant_method_A_diff = analyze_diffusion_coefficient(
                data_df=cumulant_method_A_data,
                q_squared_col='q^2',
                gamma_cols=['1st order frequency [1/ms]',
                           '2nd order frequency [1/ms]',
                           '3rd order frequency [1/ms]'],
                gamma_unit='1/ms',
                x_range=x_range)

            # Create diffusion coefficient dataframe
            A_diff = pd.DataFrame()
            A_diff['D [m^2/s]'] = cumulant_method_A_diff['q^2_coef']*10**(-15)
            A_diff['std err D [m^2/s]'] = cumulant_method_A_diff['q^2_se']*10**(-15)

            # Calculate polydispersity
            cumulant_method_A_data['polydispersity_2nd_order'] = \
                cumulant_method_A_data['2nd order frequency exp param [ms^2]'] / \
                (cumulant_method_A_data['2nd order frequency [1/ms]'])**2
            polydispersity_A_2 = cumulant_method_A_data['polydispersity_2nd_order'].mean()

            cumulant_method_A_data['polydispersity_3rd_order'] = \
                cumulant_method_A_data['3rd order frequency exp param [ms^2]'] / \
                (cumulant_method_A_data['3rd order frequency [1/ms]'])**2
            polydispersity_A_3 = cumulant_method_A_data['polydispersity_3rd_order'].mean()

            # Calculate results
            results = calculate_cumulant_results_A(
                A_diff, cumulant_method_A_diff,
                polydispersity_A_2, polydispersity_A_3,
                self.c_value, self.delta_c)

            # Display results
            self.method_a_results.delete(1.0, tk.END)
            self.method_a_results.insert(1.0, results.to_string())

            self.log("Method A analysis complete!")
            self.log(f"\nResults:\n{results.to_string()}")

        except Exception as e:
            self.log(f"ERROR in Method A: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Error", f"Method A failed:\n{str(e)}")

    def run_method_b(self):
        """Run Cumulant Method B analysis"""
        if self.df_basedata is None:
            messagebox.showerror("Error", "Please load data first!")
            return

        try:
            self.log("\n" + "="*60)
            self.log("Running Cumulant Method B...")

            # Get fit limits
            try:
                fit_limits = (float(self.method_b_tmin.get()),
                            float(self.method_b_tmax.get()))
            except ValueError:
                messagebox.showerror("Error", "Invalid fit limits!")
                return

            # Calculate g2_mod
            processed_correlations = calculate_g2_B(self.processed_correlations)

            # Fit function
            def fit_function(x, a, b, c):
                return 0.5*np.log(a) - b*x + 0.5*c*x**2

            # Plot and fit
            cumulant_method_B_fit = plot_processed_correlations(
                processed_correlations, fit_function, fit_limits)

            # Merge with base data
            self.method_B_data = pd.merge(
                self.df_basedata, cumulant_method_B_fit,
                on='filename', how='outer')
            self.method_B_data = self.method_B_data.reset_index(drop=True)
            self.method_B_data.index = self.method_B_data.index + 1

            # Get q-range
            x_range = None
            qmin_text = self.method_b_qmin.get().strip()
            qmax_text = self.method_b_qmax.get().strip()
            if qmin_text and qmax_text:
                try:
                    x_range = (float(qmin_text), float(qmax_text))
                    self.log(f"Using q² range: {x_range}")
                except ValueError:
                    self.log("Warning: Invalid q-range values, using full range")

            # Analyze diffusion
            cumulant_method_B_diff = analyze_diffusion_coefficient(
                data_df=self.method_B_data,
                q_squared_col='q^2',
                gamma_cols=['b'],
                method_names=['Method B'],
                x_range=x_range)

            # Calculate results
            B_diff = pd.DataFrame()
            B_diff['D [m^2/s]'] = cumulant_method_B_diff['q^2_coef']*10**(-18)
            B_diff['std err D [m^2/s]'] = cumulant_method_B_diff['q^2_se']*10**(-18)

            # Polydispersity
            self.method_B_data['polydispersity'] = \
                self.method_B_data['c'] / (self.method_B_data['b'])**2
            polydispersity_B = self.method_B_data['polydispersity'].mean()

            # Create results
            results = pd.DataFrame()
            results['Rh [nm]'] = self.c_value * (1/B_diff['D [m^2/s]'][0]) * 10**9
            fractional_error_Rh_B = np.sqrt(
                (self.delta_c / self.c_value)**2 +
                (B_diff['std err D [m^2/s]'][0] / B_diff['D [m^2/s]'][0])**2)
            results['Rh error [nm]'] = fractional_error_Rh_B * results['Rh [nm]']
            results['R_squared'] = cumulant_method_B_diff['R_squared']
            results['Fit'] = 'Rh from linear cumulant fit'
            results['Residuals'] = cumulant_method_B_diff['Normality']
            results['PDI'] = polydispersity_B

            # Display results
            self.method_b_results.delete(1.0, tk.END)
            self.method_b_results.insert(1.0, results.to_string())
            self.method_b_results.insert(tk.END, "\n\nData available for post-filtering.\n")
            self.method_b_results.insert(tk.END, f"Total datasets: {len(self.method_B_data)}\n")

            self.log("Method B analysis complete!")
            self.log(f"\nResults:\n{results.to_string()}")

        except Exception as e:
            self.log(f"ERROR in Method B: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Error", f"Method B failed:\n{str(e)}")

    def run_method_c(self):
        """Run Cumulant Method C analysis"""
        if self.df_basedata is None:
            messagebox.showerror("Error", "Please load data first!")
            return

        messagebox.showinfo("Info", "Method C implementation in progress.\n" +
                          "This requires iterative fitting with configurable parameters.")

    def postfilter_method_b(self):
        """Post-filter Method B results"""
        if self.method_B_data is None:
            messagebox.showerror("Error", "Please run Method B analysis first!")
            return

        self.log("\n" + "="*60)
        self.log("Opening post-filter dialog for Method B...")

        # Create dialog for entering indices to remove
        dialog = tk.Toplevel(self.root)
        dialog.title("Post-Filter Method B Results")
        dialog.geometry("600x500")

        # Apply theme to dialog
        dialog.configure(bg=self.current_style.BG_MEDIUM)

        ttk.Label(dialog, text="Current Data - Enter row indices to REMOVE (comma-separated):",
                 font=('TkDefaultFont', 10, 'bold')).pack(pady=10)

        # Show current data with white background
        text_widget = scrolledtext.ScrolledText(
            dialog,
            height=15,
            width=70,
            bg='#ffffff',
            fg='#000000')
        text_widget.pack(padx=10, pady=5)
        text_widget.insert(1.0, self.method_B_data[['filename', 'angle [°]', 'q^2', 'b', 'R-squared']].to_string())
        text_widget.configure(state='disabled')

        ttk.Label(dialog, text="Row indices to remove (e.g., 1,5,8):").pack(pady=5)
        indices_entry = ttk.Entry(dialog, width=40)
        indices_entry.pack(pady=5)

        def apply_filter():
            indices_str = indices_entry.get().strip()
            if not indices_str:
                self.log("Post-filter cancelled - no indices specified")
                dialog.destroy()
                return

            self.log(f"Applying filter to remove indices: {indices_str}")

            # Apply filter
            filtered_data = remove_rows_by_index(self.method_B_data.copy(), indices_str)

            if len(filtered_data) == len(self.method_B_data):
                self.log("No rows were removed - check your indices")
                messagebox.showwarning("Warning", "No rows were removed. Check your indices.")
                return

            self.method_B_data = filtered_data
            self.log(f"Removed rows. Remaining datasets: {len(self.method_B_data)}")

            dialog.destroy()

            # Recalculate diffusion coefficient with filtered data
            self.recalculate_method_b_results()

        ttk.Button(dialog, text="Apply Filter", command=apply_filter).pack(pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack()

    def recalculate_method_b_results(self):
        """Recalculate Method B results after filtering (without re-fitting)"""
        try:
            self.log("\n" + "="*60)
            self.log("Recalculating Method B results with filtered data...")

            # Get q-range if specified
            x_range = None
            qmin_text = self.method_b_qmin.get().strip()
            qmax_text = self.method_b_qmax.get().strip()
            if qmin_text and qmax_text:
                try:
                    x_range = (float(qmin_text), float(qmax_text))
                    self.log(f"Using q² range: {x_range}")
                except ValueError:
                    self.log("Warning: Invalid q-range values, using full range")

            # Analyze diffusion with filtered data
            cumulant_method_B_diff = analyze_diffusion_coefficient(
                data_df=self.method_B_data,
                q_squared_col='q^2',
                gamma_cols=['b'],
                method_names=['Method B'],
                x_range=x_range)

            # Calculate results
            B_diff = pd.DataFrame()
            B_diff['D [m^2/s]'] = cumulant_method_B_diff['q^2_coef']*10**(-18)
            B_diff['std err D [m^2/s]'] = cumulant_method_B_diff['q^2_se']*10**(-18)

            # Polydispersity
            self.method_B_data['polydispersity'] = \
                self.method_B_data['c'] / (self.method_B_data['b'])**2
            polydispersity_B = self.method_B_data['polydispersity'].mean()

            # Create results
            results = pd.DataFrame()
            results['Rh [nm]'] = self.c_value * (1/B_diff['D [m^2/s]'][0]) * 10**9
            fractional_error_Rh_B = np.sqrt(
                (self.delta_c / self.c_value)**2 +
                (B_diff['std err D [m^2/s]'][0] / B_diff['D [m^2/s]'][0])**2)
            results['Rh error [nm]'] = fractional_error_Rh_B * results['Rh [nm]']
            results['R_squared'] = cumulant_method_B_diff['R_squared']
            results['Fit'] = 'Rh from linear cumulant fit (filtered)'
            results['Residuals'] = cumulant_method_B_diff['Normality']
            results['PDI'] = polydispersity_B

            # Display results
            self.method_b_results.delete(1.0, tk.END)
            self.method_b_results.insert(1.0, "FILTERED RESULTS\n")
            self.method_b_results.insert(tk.END, "="*60 + "\n\n")
            self.method_b_results.insert(tk.END, results.to_string())
            self.method_b_results.insert(tk.END, f"\n\nUsing {len(self.method_B_data)} datasets (after filtering)\n")
            self.method_b_results.insert(tk.END, "You can apply additional filters if needed.\n")

            self.log("Method B recalculation complete!")
            self.log(f"\nFiltered Results:\n{results.to_string()}")

        except Exception as e:
            self.log(f"ERROR in Method B recalculation: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Error", f"Recalculation failed:\n{str(e)}")

    def postfilter_method_c(self):
        """Post-filter Method C results"""
        if self.method_C_data is None:
            messagebox.showerror("Error", "Please run Method C analysis first!")
            return

        self.log("\n" + "="*60)
        self.log("Opening post-filter dialog for Method C...")

        # Create dialog for entering indices to remove
        dialog = tk.Toplevel(self.root)
        dialog.title("Post-Filter Method C Results")
        dialog.geometry("600x500")

        # Apply theme to dialog
        dialog.configure(bg=self.current_style.BG_MEDIUM)

        ttk.Label(dialog, text="Current Data - Enter row indices to REMOVE (comma-separated):",
                 font=('TkDefaultFont', 10, 'bold')).pack(pady=10)

        # Show current data with white background
        text_widget = scrolledtext.ScrolledText(
            dialog,
            height=15,
            width=70,
            bg='#ffffff',
            fg='#000000')
        text_widget.pack(padx=10, pady=5)

        # Show relevant columns for Method C
        display_cols = ['filename', 'angle [°]', 'q^2', 'best_b', 'R-squared']
        available_cols = [col for col in display_cols if col in self.method_C_data.columns]
        text_widget.insert(1.0, self.method_C_data[available_cols].to_string())
        text_widget.configure(state='disabled')

        ttk.Label(dialog, text="Row indices to remove (e.g., 1,5,8):").pack(pady=5)
        indices_entry = ttk.Entry(dialog, width=40)
        indices_entry.pack(pady=5)

        def apply_filter():
            indices_str = indices_entry.get().strip()
            if not indices_str:
                self.log("Post-filter cancelled - no indices specified")
                dialog.destroy()
                return

            self.log(f"Applying filter to remove indices: {indices_str}")

            # Apply filter
            filtered_data = remove_rows_by_index(self.method_C_data.copy(), indices_str)

            if len(filtered_data) == len(self.method_C_data):
                self.log("No rows were removed - check your indices")
                messagebox.showwarning("Warning", "No rows were removed. Check your indices.")
                return

            self.method_C_data = filtered_data
            self.log(f"Removed rows. Remaining datasets: {len(self.method_C_data)}")

            dialog.destroy()

            # Recalculate diffusion coefficient with filtered data
            self.recalculate_method_c_results()

        ttk.Button(dialog, text="Apply Filter", command=apply_filter).pack(pady=10)
        ttk.Button(dialog, text="Cancel", command=dialog.destroy).pack()

    def recalculate_method_c_results(self):
        """Recalculate Method C results after filtering (without re-fitting)"""
        try:
            self.log("\n" + "="*60)
            self.log("Recalculating Method C results with filtered data...")

            # Get q-range if specified
            x_range = None
            qmin_text = self.method_c_qmin.get().strip()
            qmax_text = self.method_c_qmax.get().strip()
            if qmin_text and qmax_text:
                try:
                    x_range = (float(qmin_text), float(qmax_text))
                    self.log(f"Using q² range: {x_range}")
                except ValueError:
                    self.log("Warning: Invalid q-range values, using full range")

            # Analyze diffusion with filtered data
            cumulant_method_C_diff = analyze_diffusion_coefficient(
                data_df=self.method_C_data,
                q_squared_col='q^2',
                gamma_cols=['best_b'],
                method_names=['Method C'],
                x_range=x_range)

            # Calculate results
            C_diff = pd.DataFrame()
            C_diff['D [m^2/s]'] = cumulant_method_C_diff['q^2_coef']*10**(-18)
            C_diff['std err D [m^2/s]'] = cumulant_method_C_diff['q^2_se']*10**(-18)

            # Polydispersity
            self.method_C_data['polydispersity'] = \
                self.method_C_data['best_c'] / (self.method_C_data['best_b'])**2
            polydispersity_C = self.method_C_data['polydispersity'].mean()

            # Create results
            results = pd.DataFrame()
            results['Rh [nm]'] = self.c_value * (1/C_diff['D [m^2/s]'][0]) * 10**9
            fractional_error_Rh_C = np.sqrt(
                (self.delta_c / self.c_value)**2 +
                (C_diff['std err D [m^2/s]'][0] / C_diff['D [m^2/s]'][0])**2)
            results['Rh error [nm]'] = fractional_error_Rh_C * results['Rh [nm]']
            results['R_squared'] = cumulant_method_C_diff['R_squared']
            results['Fit'] = 'Rh from iterative non-linear cumulant fit (filtered)'
            results['Residuals'] = cumulant_method_C_diff['Normality']
            results['PDI'] = polydispersity_C

            # Display results
            self.method_c_results.delete(1.0, tk.END)
            self.method_c_results.insert(1.0, "FILTERED RESULTS\n")
            self.method_c_results.insert(tk.END, "="*60 + "\n\n")
            self.method_c_results.insert(tk.END, results.to_string())
            self.method_c_results.insert(tk.END, f"\n\nUsing {len(self.method_C_data)} datasets (after filtering)\n")
            self.method_c_results.insert(tk.END, "You can apply additional filters if needed.\n")

            self.log("Method C recalculation complete!")
            self.log(f"\nFiltered Results:\n{results.to_string()}")

        except Exception as e:
            self.log(f"ERROR in Method C recalculation: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            messagebox.showerror("Error", f"Recalculation failed:\n{str(e)}")

    def export_results(self):
        """Export all results to files"""
        if self.df_basedata is None:
            messagebox.showerror("Error", "No data to export!")
            return

        # Ask for output directory
        output_dir = filedialog.askdirectory(title="Select output directory")
        if not output_dir:
            return

        try:
            # Export base data
            output_file = os.path.join(output_dir, "basedata.txt")
            self.df_basedata.to_csv(output_file, sep='\t', index=False)
            self.log(f"Exported base data to: {output_file}")

            messagebox.showinfo("Success", f"Results exported to:\n{output_dir}")

        except Exception as e:
            self.log(f"ERROR exporting: {str(e)}")
            messagebox.showerror("Error", f"Export failed:\n{str(e)}")

    def show_about(self):
        """Show about dialog"""
        about_text = """
JADE-DLS v1.0
Jupyter-based Angular Dependent Evaluator
for Dynamic Light Scattering

With Dark Mode Support

© 2025
"""
        messagebox.showinfo("About JADE-DLS", about_text)

def main():
    root = tk.Tk()
    app = JADEDLSApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
