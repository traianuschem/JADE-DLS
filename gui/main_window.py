"""
JADE-DLS Main GUI Window
A transparent, user-friendly GUI for Dynamic Light Scattering analysis
"""

from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QMenuBar, QAction, QStatusBar,
                             QMessageBox, QFileDialog, QTabWidget, QProgressBar)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QIcon

from gui.widgets.workflow_panel import WorkflowPanel
from gui.widgets.analysis_view import AnalysisView
from gui.widgets.inspector_panel import InspectorPanel
from gui.core.pipeline import TransparentPipeline
from gui.core.status_manager import StatusManager, ProgressDialog
from gui.core.data_loader import DataLoader


class JADEDLSMainWindow(QMainWindow):
    """
    Main application window for JADE-DLS analysis

    Features:
    - Transparent analysis pipeline
    - Interactive visualization
    - Code export capabilities
    - Parameter tracking
    """

    def __init__(self):
        super().__init__()

        # Initialize the transparent pipeline
        self.pipeline = TransparentPipeline()

        # Initialize status manager
        self.status_manager = None  # Will be set after status bar is created

        # Initialize data loader
        self.data_loader = DataLoader()

        # Progress dialog for long operations
        self.progress_dialog = None

        # Data storage
        self.loaded_data = None

        # Setup UI
        self.init_ui()

        # Connect signals
        self.connect_signals()

    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("JADE-DLS - Dynamic Light Scattering Analysis")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget with splitter layout
        self.setup_central_widget()

        # Create menu bar
        self.create_menu_bar()

        # Create status bar
        self.statusBar().showMessage("Ready")

        # Initialize status manager (after status bar exists)
        self.status_manager = StatusManager(self.statusBar())

    def setup_central_widget(self):
        """Setup the main layout with three panels"""
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # Create main splitter
        self.splitter = QSplitter(Qt.Horizontal)

        # Left panel: Workflow steps
        self.workflow_panel = WorkflowPanel(self.pipeline)
        self.workflow_panel.setMinimumWidth(200)

        # Center panel: Analysis view with tabs
        self.analysis_view = AnalysisView(self.pipeline)
        self.analysis_view.setMinimumWidth(500)

        # Right panel: Inspector (code viewer, parameters)
        self.inspector_panel = InspectorPanel(self.pipeline)
        self.inspector_panel.setMinimumWidth(300)

        # Add panels to splitter
        self.splitter.addWidget(self.workflow_panel)
        self.splitter.addWidget(self.analysis_view)
        self.splitter.addWidget(self.inspector_panel)

        # Set initial sizes (20%, 50%, 30%)
        self.splitter.setSizes([280, 700, 420])

        main_layout.addWidget(self.splitter)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def create_menu_bar(self):
        """Create the application menu bar"""
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu('&File')

        load_action = QAction('&Load Data...', self)
        load_action.setShortcut('Ctrl+O')
        load_action.setStatusTip('Load DLS data files')
        load_action.triggered.connect(self.load_data)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        export_notebook_action = QAction('Export as &Jupyter Notebook...', self)
        export_notebook_action.setShortcut('Ctrl+J')
        export_notebook_action.setStatusTip('Export analysis as Jupyter notebook')
        export_notebook_action.triggered.connect(self.export_notebook)
        file_menu.addAction(export_notebook_action)

        export_script_action = QAction('Export as Python &Script...', self)
        export_script_action.setShortcut('Ctrl+P')
        export_script_action.setStatusTip('Export analysis as Python script')
        export_script_action.triggered.connect(self.export_script)
        file_menu.addAction(export_script_action)

        export_report_action = QAction('Export &Report (PDF)...', self)
        export_report_action.setStatusTip('Export complete analysis report')
        export_report_action.triggered.connect(self.export_report)
        file_menu.addAction(export_report_action)

        file_menu.addSeparator()

        exit_action = QAction('E&xit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Analysis Menu
        analysis_menu = menubar.addMenu('&Analysis')

        cumulant_action = QAction('&Cumulant Analysis...', self)
        cumulant_action.setStatusTip('Run cumulant fitting methods')
        cumulant_action.triggered.connect(self.run_cumulant_analysis)
        analysis_menu.addAction(cumulant_action)

        nnls_action = QAction('&NNLS Analysis...', self)
        nnls_action.setStatusTip('Run NNLS inverse Laplace transform')
        nnls_action.triggered.connect(self.run_nnls_analysis)
        analysis_menu.addAction(nnls_action)

        regularized_action = QAction('&Regularized Fit...', self)
        regularized_action.setStatusTip('Run regularized inverse Laplace transform')
        regularized_action.triggered.connect(self.run_regularized_analysis)
        analysis_menu.addAction(regularized_action)

        analysis_menu.addSeparator()

        compare_action = QAction('C&ompare Results...', self)
        compare_action.setStatusTip('Compare all analysis methods')
        compare_action.triggered.connect(self.compare_results)
        analysis_menu.addAction(compare_action)

        # View Menu
        view_menu = menubar.addMenu('&View')

        show_code_action = QAction('Show &Code', self, checkable=True)
        show_code_action.setChecked(True)
        show_code_action.setStatusTip('Show generated code')
        show_code_action.triggered.connect(self.toggle_code_view)
        view_menu.addAction(show_code_action)

        show_params_action = QAction('Show &Parameters', self, checkable=True)
        show_params_action.setChecked(True)
        show_params_action.setStatusTip('Show analysis parameters')
        show_params_action.triggered.connect(self.toggle_params_view)
        view_menu.addAction(show_params_action)

        view_menu.addSeparator()

        reset_layout_action = QAction('&Reset Layout', self)
        reset_layout_action.setStatusTip('Reset window layout to default')
        reset_layout_action.triggered.connect(self.reset_layout)
        view_menu.addAction(reset_layout_action)

        # Help Menu
        help_menu = menubar.addMenu('&Help')

        tutorial_action = QAction('&Tutorial', self)
        tutorial_action.setShortcut('F1')
        tutorial_action.setStatusTip('Show interactive tutorial')
        tutorial_action.triggered.connect(self.show_tutorial)
        help_menu.addAction(tutorial_action)

        docs_action = QAction('&Documentation', self)
        docs_action.setStatusTip('Open documentation')
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)

        help_menu.addSeparator()

        about_action = QAction('&About JADE-DLS', self)
        about_action.setStatusTip('About this application')
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def connect_signals(self):
        """Connect signals between components"""
        # Connect workflow panel signals
        self.workflow_panel.step_selected.connect(self.on_step_selected)
        self.workflow_panel.run_analysis.connect(self.on_run_analysis)

        # Connect pipeline signals
        self.pipeline.step_added.connect(self.on_pipeline_step_added)
        self.pipeline.analysis_completed.connect(self.on_analysis_completed)

        # Connect data loader signals
        self.data_loader.progress.connect(self.on_load_progress)
        self.data_loader.data_loaded.connect(self.on_data_loaded)
        self.data_loader.error.connect(self.on_load_error)

    # ========== Slot Methods ==========

    def load_data(self):
        """Load DLS data files"""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Data Folder",
            "",
            QFileDialog.ShowDirsOnly
        )

        if folder:
            # Create and show progress dialog
            self.progress_dialog = ProgressDialog("Loading Data", self)
            self.progress_dialog.setRange(0, 5)
            self.progress_dialog.setValue(0)
            self.progress_dialog.show()

            # Update status
            self.status_manager.start_operation(f"Loading data from {folder}")

            # Start loading in background thread
            self.data_loader.load_data(
                folder,
                load_countrates=True,
                load_correlations=True
            )

    def export_notebook(self):
        """Export analysis as Jupyter notebook"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Jupyter Notebook",
            "analysis.ipynb",
            "Jupyter Notebooks (*.ipynb)"
        )

        if filename:
            try:
                self.pipeline.export_to_notebook(filename)
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Analysis exported to:\n{filename}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def export_script(self):
        """Export analysis as Python script"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Python Script",
            "analysis.py",
            "Python Scripts (*.py)"
        )

        if filename:
            try:
                self.pipeline.export_to_script(filename)
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Analysis exported to:\n{filename}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def export_report(self):
        """Export complete analysis report as PDF"""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export PDF Report",
            "analysis_report.pdf",
            "PDF Files (*.pdf)"
        )

        if filename:
            try:
                self.pipeline.export_to_pdf(filename)
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Report exported to:\n{filename}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def run_cumulant_analysis(self):
        """Run cumulant analysis"""
        self.workflow_panel.activate_step('cumulant')

    def run_nnls_analysis(self):
        """Run NNLS analysis"""
        self.workflow_panel.activate_step('nnls')

    def run_regularized_analysis(self):
        """Run regularized analysis"""
        self.workflow_panel.activate_step('regularized')

    def compare_results(self):
        """Compare all analysis results"""
        self.analysis_view.show_comparison()

    def toggle_code_view(self, checked):
        """Toggle code viewer visibility"""
        self.inspector_panel.set_code_view_visible(checked)

    def toggle_params_view(self, checked):
        """Toggle parameters view visibility"""
        self.inspector_panel.set_params_view_visible(checked)

    def reset_layout(self):
        """Reset window layout to default"""
        self.splitter.setSizes([280, 700, 420])

    def show_tutorial(self):
        """Show interactive tutorial"""
        QMessageBox.information(
            self,
            "Tutorial",
            "Interactive tutorial coming soon!\n\n"
            "For now, use File > Load Data to get started."
        )

    def show_documentation(self):
        """Open documentation"""
        QMessageBox.information(
            self,
            "Documentation",
            "Documentation available at:\n"
            "https://github.com/your-repo/jade-dls/docs"
        )

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About JADE-DLS",
            "<h3>JADE-DLS</h3>"
            "<p><b>J</b>upyter-based <b>A</b>ngular <b>D</b>ependent "
            "<b>E</b>valuator for <b>D</b>ynamic <b>L</b>ight <b>S</b>cattering</p>"
            "<p>Version 2.0 (GUI Edition)</p>"
            "<p>A transparent, user-friendly tool for DLS data analysis</p>"
            "<p>© 2025 JADE-DLS Development Team</p>"
        )

    def on_step_selected(self, step_name):
        """Handle workflow step selection"""
        self.analysis_view.show_step(step_name)
        self.inspector_panel.update_for_step(step_name)

    def on_run_analysis(self, analysis_type):
        """Handle run analysis request"""
        self.statusBar().showMessage(f"Running {analysis_type} analysis...")
        # Analysis logic will be implemented in the pipeline

    def on_pipeline_step_added(self, step_info):
        """Handle new pipeline step"""
        self.inspector_panel.update_code_view()
        self.workflow_panel.update_progress()

    def on_analysis_completed(self, results):
        """Handle analysis completion"""
        self.statusBar().showMessage("Analysis completed successfully")
        self.analysis_view.display_results(results)

    def on_load_progress(self, current, total, message):
        """Handle data loading progress updates"""
        # Update status manager
        self.status_manager.update(message, current, total)

        # Update progress dialog if it exists
        if self.progress_dialog:
            self.progress_dialog.update_status(message, current, total)

    def on_data_loaded(self, data):
        """Handle successful data loading"""
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        # Store data
        self.loaded_data = data

        # Update pipeline
        self.pipeline.data = data
        self.pipeline.data_loaded.emit(data.get('data_folder', 'Unknown'))

        # Update status
        num_files = data.get('num_files', 0)
        self.status_manager.complete_operation(
            f"Successfully loaded {num_files} files"
        )

        # Update analysis view
        self.analysis_view.update_data_overview()

        # Mark workflow step as complete
        self.workflow_panel.mark_step_complete('load_data')
        self.workflow_panel.update_progress()

        # Show success message
        QMessageBox.information(
            self,
            "Data Loaded",
            f"Successfully loaded {num_files} files.\n\n"
            f"Base data: {len(data.get('basedata', []))} entries\n"
            f"Correlations: {len(data.get('correlations', {}))} datasets\n"
            f"Count rates: {len(data.get('countrates', {}))} datasets"
        )

    def on_load_error(self, error_message):
        """Handle data loading errors"""
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        # Update status
        self.status_manager.error(error_message)

        # Show error message
        QMessageBox.critical(
            self,
            "Error Loading Data",
            f"Failed to load data:\n\n{error_message}"
        )
