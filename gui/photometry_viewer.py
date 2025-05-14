import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Import from other GUI modules
from .menu_handlers import create_menu
from .file_handlers import open_file, load_file, open_secondary_file, load_secondary_file
from .filter_panel import create_filter_panel, update_filter
from .drift_panel import create_drift_panel, open_drift_window
from .blanking_panel import create_blanking_panel, toggle_blanking_mode
from .navigation import connect_zoom_events, reset_view, zoom_x, zoom_y, pan_x
from .artifact_panel import create_artifact_panel, highlight_artifacts
from .visualization import update_panel_y_limits, update_artifact_markers

# Import from data_processing
from data_processing import read_ppd_file, parse_ppd_data, process_data

# Import from other modules
from plot_utils import create_checkbox_legend
from config import get_font_size_for_resolution

def __init__(self, root, file_path=None):
        self.root = root
        self.file_path = file_path
        self.secondary_file_path = None  # Track secondary file

        # Get font sizes based on screen resolution
        self.font_sizes = get_font_size_for_resolution()

        # Configure fonts for matplotlib
        plt.rcParams.update({
            'font.size': self.font_sizes['base'],
            'axes.titlesize': self.font_sizes['title'],
            'axes.labelsize': self.font_sizes['base'],
            'xtick.labelsize': self.font_sizes['base'],
            'ytick.labelsize': self.font_sizes['base'],
            'legend.fontsize': self.font_sizes['base'],
        })

        # Initialize variables for primary file
        self.header = None
        self.data_bytes = None
        self.time = None
        self.analog_1 = None
        self.analog_2 = None
        self.digital_1 = None
        self.digital_2 = None
        self.raw_analog_1 = None
        self.raw_analog_2 = None
        self.drift_curve = None
        # Add to __init__:
        self.blanking_mode = False
        self.blanking_selection = None
        self.blanking_rectangle = None
        self.blanking_cid = None
        self.blanking_start = None
        self.blanking_file = "primary"
        self.blanking_regions = []
        self.primary_has_blanking = False
        self.secondary_has_blanking = False
        self.ctrl_pressed = False

        # Initialize variables for secondary file
        self.secondary_header = None
        self.secondary_data_bytes = None
        self.secondary_time = None
        self.secondary_analog_1 = None
        self.secondary_analog_2 = None
        self.secondary_digital_1 = None
        self.secondary_digital_2 = None
        self.secondary_raw_analog_1 = None
        self.secondary_raw_analog_2 = None
        self.secondary_drift_curve = None
        self.secondary_processed_signal = None
        self.secondary_processed_time = None
        self.secondary_artifact_mask = None

        # Initialize variables for processed data
        self.processed_time = None
        self.processed_signal = None
        self.processed_analog_2 = None
        self.downsampled_raw_analog_1 = None
        self.downsampled_raw_analog_2 = None
        self.processed_digital_1 = None
        self.processed_digital_2 = None
        self.artifact_mask = None

        # Initialize secondary processed data
        self.secondary_processed_time = None
        self.secondary_processed_signal = None
        self.secondary_processed_analog_2 = None
        self.secondary_downsampled_raw_analog_1 = None
        self.secondary_downsampled_raw_analog_2 = None
        self.secondary_processed_digital_1 = None
        self.secondary_processed_digital_2 = None
        self.secondary_artifact_mask = None

        # Initialize processing parameters with requested defaults
        self.low_cutoff = 0.001  # Changed as requested
        self.high_cutoff = 1.0
        self.downsample_factor = 50  # Changed as requested
        self.artifact_threshold = 3.0
        self.drift_correction = True
        self.drift_degree = 2

        # Store line visibility states
        self.line_visibility = {}

        # Denoise thread control
        self.denoise_thread = None
        self.cancel_denoise_event = threading.Event()
        self.denoise_progress_queue = queue.Queue()
        self.denoising_applied = False  # Flag to track if denoising has been applied

        # UI elements
        self.main_frame = None
        self.nav_frame = None
        self.frame = None
        self.slider_frame = None
        self.fig = None
        self.ax1_legend = None
        self.ax2_legend = None
        self.ax3_legend = None

        # Plot elements
        self.line = None  # Primary processed signal
        self.secondary_line = None  # Secondary processed signal
        self.raw_line = None  # Primary raw analog 1
        self.secondary_raw_line = None  # Secondary raw analog 1
        self.raw_line2 = None  # Primary raw analog 2
        self.secondary_raw_line2 = None  # Secondary raw analog 2
        self.drift_line = None  # Primary drift curve
        self.secondary_drift_line = None  # Secondary drift curve
        self.artifact_markers_processed = None
        self.secondary_artifact_markers_processed = None
        self.artifact_markers_raw = None
        self.secondary_artifact_markers_raw = None
        self.digital_lines = []
        self.secondary_digital_lines = []

        # Initialize advanced denoising parameters
        self.aggressive_var = None
        self.remove_var = None
        self.max_gap_var = None
        self.control_var = None
        self.progress_bar = None
        self.cancel_denoise_button = None

        # Add to __init__
        self.peaks = None
        self.valleys = None
        self.peak_metrics = None
        self.manual_peak_mode = False
        self.manual_valley_mode = False
        self.peak_annotations = []
        self.valley_annotations = []
        self.valley_metrics = None
        self.peak_lines = []
        self.valley_lines = []
        self.peak_feature_annotations = []
        self.selection_cid = None
        self.pick_cid = None

        # Initialize PSTH variables early
        self.psth_signal_var = tk.StringVar(value="Both")  # Default value
        self.reference_event_var = tk.StringVar(value="pvn_peaks")
        self.plot_pvn_var = tk.BooleanVar(value=True)
        self.plot_son_var = tk.BooleanVar(value=True)
        self.psth_before_var = tk.DoubleVar(value=60.0)
        self.psth_after_var = tk.DoubleVar(value=60.0)
        self.show_sem_var = tk.BooleanVar(value=True)
        self.show_individual_var = tk.BooleanVar(value=False)
        self.pvn_psth_data = None
        self.son_psth_data = None

        # Setup the menu bar
        self.create_menu()
        # Setup proper closing behavior
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # If a file path was provided, load it
        if file_path:
            self.load_file(file_path)

        def setup_keyboard_shortcuts(self):
            """Setup keyboard shortcuts for common operations"""
            self.root.bind('<Home>', lambda event: self.reset_view())
            self.root.bind('<Control-0>', lambda event: self.reset_view())  # Ctrl+0 also resets view


    def create_gui(self):
        try:
            # Create main frame and toolbar as before
            self.main_frame = tk.Frame(self.root)
            self.main_frame.pack(fill=tk.BOTH, expand=True)

            # Add a toolbar for quick access to common functions
            self.create_toolbar()

            # Create navigation frame
            self.nav_frame = tk.Frame(self.main_frame)
            self.nav_frame.pack(fill=tk.X, padx=20, pady=10)

            # Create a frame for the matplotlib figure
            self.frame = tk.Frame(self.main_frame)
            self.frame.pack(fill=tk.BOTH, expand=True)

            # Create matplotlib figure
            self.fig = plt.figure(figsize=(10, 8))

            # Create GridSpec with revised layout: separate panels for file 1 and file 2
            # Adjust height ratios to make panels 1 and 2 larger and panel 3 smaller
            gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 1])

            # Get region label for primary file
            primary_region = self.get_region_label_from_filename(self.file_path)

            # --- PANEL 1: File 1 (Primary) with all its signals ---
            self.ax1 = self.fig.add_subplot(gs[0])

            # Plot the main processed signal - ONLY THIS VISIBLE BY DEFAULT
            self.line, = self.ax1.plot(
                self.processed_time / 60,
                self.processed_signal,
                'g-', lw=1.5,
                label=f'{primary_region} Signal'
            )

            # Add the no-control version as a dashed line - HIDDEN BY DEFAULT
            self.line_no_control, = self.ax1.plot(
                self.processed_time / 60,
                self.processed_signal_no_control,
                color='lightgreen',
                linestyle='--',
                lw=1.5,
                alpha=0.9,
                label=f'{primary_region} No-Control'
            )

            # Normalize raw signal to similar scale as processed signal
            normalized_raw = self.normalize_raw_signal(self.processed_signal, self.downsampled_raw_analog_1)

            # Add raw analog 1 signal - HIDDEN BY DEFAULT
            self.raw_line, = self.ax1.plot(
                self.processed_time / 60,
                normalized_raw,
                'b-', lw=1,
                label=f'{primary_region} Raw'
            )

            # Add isosbestic channel (analog 2) - HIDDEN BY DEFAULT
            if self.downsampled_raw_analog_2 is not None:
                normalized_isosbestic = self.normalize_raw_signal(self.processed_signal, self.downsampled_raw_analog_2)
                self.raw_line2, = self.ax1.plot(
                    self.processed_time / 60,
                    normalized_isosbestic,
                    'm-', lw=1,
                    label=f'{primary_region} Isosbestic'
                )

            # Add drift curve if available - HIDDEN BY DEFAULT
            if self.drift_curve is not None and self.drift_correction:
                baseline_idx = min(int(len(self.drift_curve) * 0.1), 10)
                baseline = np.mean(self.drift_curve[:baseline_idx])
                df_drift_curve = 100 * (self.drift_curve - baseline) / baseline if baseline != 0 else self.drift_curve
                self.drift_line, = self.ax1.plot(
                    self.processed_time / 60,
                    df_drift_curve,
                    'r--', lw=1, alpha=0.5,
                    label=f'{primary_region} Drift'
                )

            # Initialize artifact markers for File 1
            self.artifact_markers_processed, = self.ax1.plot(
                [], [], 'ro', ms=4, alpha=0.7,
                label=f'{primary_region} Artifacts'
            )

            # IMPORTANT: Initialize artifact_markers_raw as None initially
            self.artifact_markers_raw = None

            # Set proper y-axis label
            self.ax1.set_ylabel('dF/F%', fontsize=self.font_sizes['base'])
            self.ax1.set_title(f'File 1: {primary_region}', fontsize=self.font_sizes['title'])
            self.ax1.grid(True, alpha=0.3)
            self.ax1.tick_params(labelsize=self.font_sizes['base'])

            # --- PANEL 2: File 2 (Secondary) will be empty until a secondary file is loaded ---
            self.ax2 = self.fig.add_subplot(gs[1], sharex=self.ax1)
            self.ax2.set_ylabel('dF/F%', fontsize=self.font_sizes['base'])
            self.ax2.set_title('File 2: Not Loaded', fontsize=self.font_sizes['title'])
            self.ax2.grid(True, alpha=0.3)
            self.ax2.tick_params(labelsize=self.font_sizes['base'])

            # Initialize secondary signal components as None
            self.secondary_line = None
            self.secondary_raw_line = None
            self.secondary_raw_line2 = None
            self.secondary_drift_line = None
            self.secondary_line_no_control = None
            self.secondary_artifact_markers_processed = None
            self.secondary_artifact_markers_raw = None

            # --- PANEL 3: Digital signals from both files (reduced height) ---
            self.ax3 = self.fig.add_subplot(gs[2], sharex=self.ax1)
            self.digital_lines = []
            self.secondary_digital_lines = []

            if self.processed_digital_1 is not None:
                line, = self.ax3.plot(
                    self.time / 60,
                    self.processed_digital_1,
                    'b-', lw=1,
                    label=f'{primary_region} Digital 1'
                )
                self.digital_lines.append(line)

            if self.processed_digital_2 is not None:
                line, = self.ax3.plot(
                    self.time / 60,
                    self.processed_digital_2,
                    'r-', lw=1,
                    label=f'{primary_region} Digital 2'
                )
                self.digital_lines.append(line)

            self.ax3.set_ylabel('TTL', fontsize=self.font_sizes['base'])
            self.ax3.set_ylim(-0.1, 1.1)
            self.ax3.set_xlabel('Time (minutes)', fontsize=self.font_sizes['base'])
            self.ax3.tick_params(labelsize=self.font_sizes['base'])

            # Set initial visibility states
            self.line_visibility = {}
            self.line_visibility[self.line] = True  # Only processed signal visible by default

            if hasattr(self, 'line_no_control') and self.line_no_control is not None:
                self.line_no_control.set_visible(False)
                self.line_visibility[self.line_no_control] = False

            if hasattr(self, 'raw_line') and self.raw_line is not None:
                self.raw_line.set_visible(False)
                self.line_visibility[self.raw_line] = False

            if hasattr(self, 'raw_line2') and self.raw_line2 is not None:
                self.raw_line2.set_visible(False)
                self.line_visibility[self.raw_line2] = False

            if hasattr(self, 'drift_line') and self.drift_line is not None:
                self.drift_line.set_visible(False)
                self.line_visibility[self.drift_line] = False

            # Create custom legends with checkboxes
            self.ax1_legend = self.create_checkbox_legend(self.ax1)
            self.ax2_legend = self.create_checkbox_legend(self.ax2)
            if self.digital_lines:
                self.ax3_legend = self.create_checkbox_legend(self.ax3)

            # Update artifact markers with safety check
            self.update_artifact_markers()

            # Create canvas and toolbar
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            # Add toolbar
            self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
            self.toolbar.update()

            # Connect zoom events AFTER canvas is created
            self.connect_zoom_events()

            # Create slider frame with more space
            self.slider_frame = tk.Frame(self.main_frame)
            self.slider_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

            # Create sliders
            self.create_sliders()

            # Set appropriate y-axis scaling for each panel with 0 baseline
            self.update_panel_y_limits()

            # Adjust layout
            self.fig.tight_layout()

            # Status label for updates
            self.status_label = tk.Label(self.nav_frame, text="Ready", fg="green",
                                         font=('Arial', self.font_sizes['base']))
            self.status_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=10)

            print("GUI creation successful")
        except Exception as e:
            print(f"Error in GUI creation: {e}")
            import traceback
            traceback.print_exc()

    def on_closing(self):
        """Handle the window closing event properly to ensure the application terminates"""
        print("Closing application...")
        try:
            # If denoising is running, cancel it
            if self.denoise_thread and self.denoise_thread.is_alive():
                print("Cancelling denoising thread...")
                self.cancel_denoise_event.set()
                self.denoise_thread.join(timeout=1.0)
                if self.denoise_thread.is_alive():
                    print("Warning: Denoising thread did not terminate")
                    # Clear peak detection resources
                    if hasattr(self, 'peak_annotations'):
                        self.clear_peak_annotations()

            # Clean up resources
            print("Closing matplotlib figures...")
            plt.close('all')  # Close all matplotlib figures

            # Register an exit handler to force termination
            import atexit
            atexit.register(lambda: os._exit(0))

            print("Destroying Tkinter root...")
            # Destroy the Tkinter root window
            self.root.quit()
            self.root.destroy()

            # Force exit immediately to avoid any hanging threads
            print("Exiting application...")
            import os, sys
            os._exit(0)  # More forceful than sys.exit()
        except Exception as e:
            print(f"Error during closing: {e}")
            import traceback, os
            traceback.print_exc()
            os._exit(1)  # Force exit even on error

