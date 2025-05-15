import json
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
import time as time_module
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve
from scipy.signal import butter, filtfilt, resample, savgol_filter
from scipy.stats import linregress
from scipy.ndimage import median_filter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
import atexit
from scipy.signal import butter, filtfilt, resample, savgol_filter, sosfilt, sosfiltfilt, butter, zpk2sos


matplotlib.use('TkAgg')  # Explicitly set the backend for better interactivity
import sys


def read_ppd_file(file_path):
    """Reads a .ppd file and returns the header and data."""
    try:
        with open(file_path, 'rb') as f:
            # First two bytes are the header length in little-endian
            header_len_bytes = f.read(2)
            header_len = int.from_bytes(header_len_bytes, byteorder='little')

            # Read the header
            header_bytes = f.read(header_len)
            header_str = header_bytes.decode('utf-8')

            # Parse the header JSON
            header = json.loads(header_str)

            # Read the rest of the file as data
            data_bytes = f.read()

            print(f"Successfully read file with header: sampling_rate={header.get('sampling_rate', 'unknown')}")
            return header, data_bytes
    except Exception as e:
        print(f"Error reading PPD file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def parse_ppd_data(header, data_bytes):
    """Parses the data bytes from a .ppd file and returns the analog and digital signals."""
    try:
        if header is None or data_bytes is None or len(data_bytes) == 0:
            print("Error: No valid data found")
            return None, None, None, None, None

        # Get sampling rate from header - ensure it's a numeric value
        sampling_rate = float(header.get('sampling_rate', 1000))  # Default to 1000 Hz
        print(f"Using sampling rate: {sampling_rate} Hz")

        # Get volts per division if available
        volts_per_division = header.get('volts_per_division', [1.0, 1.0])  # Default to 1.0 if not specified

        # Parse data bytes - original format is unsigned 16-bit integers in little-endian
        data = np.frombuffer(data_bytes, dtype=np.dtype('<u2'))

        # Safety check for empty data
        if len(data) == 0:
            print("Warning: No data found in file")
            return None, None, None, None, None

        # Store original data length for reference
        original_data_length = len(data)
        print(f"Original data length: {original_data_length} samples")

        # Extract analog and digital signals
        # The last bit is the digital signal, the rest is the analog value
        analog = data >> 1  # Shift right to remove digital bit
        digital = data & 1  # Mask to get only the last bit

        # Separate channels - even indices are channel 1, odd indices are channel 2
        analog_1 = analog[0::2]
        analog_2 = analog[1::2]
        digital_1 = digital[0::2]
        digital_2 = digital[1::2]

        # CRITICAL FIX: Ensure all arrays have the same length
        min_len = min(len(analog_1), len(analog_2), len(digital_1), len(digital_2))
        analog_1 = analog_1[:min_len]
        analog_2 = analog_2[:min_len]
        digital_1 = digital_1[:min_len]
        digital_2 = digital_2[:min_len]

        # Calculate the original duration based on sampling rate
        original_duration = min_len / sampling_rate  # Duration in seconds
        print(f"Original duration: {original_duration:.2f} seconds")

        # If the data is very large, downsample properly to prevent memory issues
        max_points = 500000  # Reasonable maximum points to process
        if min_len > max_points:
            print(f"Data is very large ({min_len} points), downsampling to prevent memory issues")

            # Calculate downsample factor
            downsample_factor = min_len // max_points + 1

            # FIXED: Create time array with the exact same number of points as the downsampled signals
            # Use exact number division to ensure precise count
            num_points = min_len // downsample_factor

            # Downsample signals using stride
            analog_1 = analog_1[::downsample_factor][:num_points]  # Ensure exact length
            analog_2 = analog_2[::downsample_factor][:num_points]
            digital_1 = digital_1[::downsample_factor][:num_points]
            digital_2 = digital_2[::downsample_factor][:num_points]

            # Create time array with exact same length
            time = np.linspace(0, original_duration, num_points)

            print(f"Downsampled to {len(analog_1)} points. Time range preserved: 0 to {time[-1]:.2f} seconds")
        else:
            # Create time array using the original sampling rate
            time = np.arange(min_len) / sampling_rate  # Time in seconds

        # Verify all arrays have the same length
        assert len(time) == len(analog_1) == len(analog_2) == len(digital_1) == len(digital_2), \
            f"Array length mismatch: time={len(time)}, analog_1={len(analog_1)}, analog_2={len(analog_2)}, " \
            f"digital_1={len(digital_1)}, digital_2={len(digital_2)}"

        # Apply volts per division scaling if available
        if len(volts_per_division) >= 2:
            analog_1 = analog_1 * volts_per_division[0]
            analog_2 = analog_2 * volts_per_division[1]

        print(f"Parsed data: {len(time)} samples, time range: 0 to {time[-1]:.2f} seconds")
        return time, analog_1, analog_2, digital_1, digital_2
    except Exception as e:
        print(f"Error parsing PPD file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def detect_artifacts(signal, threshold=3.0):
    """
    Detect artifacts in a control signal (typically the 405nm channel).

    Parameters:
    -----------
    signal : array
        Signal array (typically analog_2/405nm channel)
    threshold : float
        Threshold for artifact detection (in standard deviations)

    Returns:
    --------
    artifact_mask : array of bool
        Boolean mask where True indicates an artifact
    """
    if signal is None:
        return None

    # Handle NaN values
    valid_signal = np.isfinite(signal)
    if not np.any(valid_signal):
        return np.zeros_like(signal, dtype=bool)

    # Calculate signal statistics on valid data only
    valid_data = signal[valid_signal]
    mean_val = np.mean(valid_data)
    std_val = np.std(valid_data)

    # Identify points that deviate too much from the mean
    artifact_mask = np.zeros_like(signal, dtype=bool)
    artifact_mask[valid_signal] = np.abs(valid_data - mean_val) > threshold * std_val

    # Expand the mask a bit to include neighboring points
    if np.any(artifact_mask):
        expanded_mask = np.convolve(artifact_mask.astype(float), np.ones(5) / 5, mode='same')
        artifact_mask = expanded_mask > 0.2  # Lower threshold to catch neighboring points

    return artifact_mask


def butter_filter(data, cutoff, fs, order=5):
    """Applies a Butterworth filter to the data with reduced edge effects."""
    if data is None or len(data) == 0:
        return data

    try:
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq

        # Use Second-Order Sections (SOS) form for better numerical stability
        sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')

        # Use forward-backward filtering with reflection to reduce edge effects
        # Create mirrored edges to minimize boundary effects
        edge_size = min(2000, len(data) // 4)  # Use a reasonable size for edge padding

        # Mirror the edges of the signal
        data_padded = np.concatenate((data[edge_size:0:-1], data, data[-2:-edge_size - 2:-1]))

        # Apply filter to padded signal
        filtered_padded = sosfiltfilt(sos, data_padded)

        # Extract the original signal region
        filtered = filtered_padded[edge_size:-edge_size]

        return filtered
    except Exception as e:
        print(f"Error in butter_filter: {e}")
        return data  # Return original data if filtering fails


def downsample_data(time, signal, factor=1):
    """
    Downsample the data by the given factor.

    Parameters:
    -----------
    time : array
        Time array
    signal : array
        Signal array
    factor : int
        Downsample factor (1 = no downsampling)

    Returns:
    --------
    downsampled_time, downsampled_signal
    """
    if factor <= 1:
        return time, signal

    # Use stride to downsample
    downsampled_time = time[::factor]
    downsampled_signal = signal[::factor]

    return downsampled_time, downsampled_signal


def process_data(time, analog_1, analog_2, digital_1, digital_2,
                 low_cutoff=0.01,
                 high_cutoff=1.0,
                 downsample_factor=1,
                 artifact_threshold=3.0):
    """Processes the data by applying filters and downsampling."""
    # Compute sampling rate
    dt = np.diff(time).mean()
    fs = 1 / dt

    # Make a copy of the signals for processing
    processed_signal = analog_1.copy()
    processed_analog_2 = analog_2.copy() if analog_2 is not None else None

    # Also make downsampled copies of raw data
    downsampled_raw_analog_1 = analog_1.copy()
    downsampled_raw_analog_2 = analog_2.copy() if analog_2 is not None else None

    # Detect artifacts in the control channel (if available)
    artifact_mask = None
    if analog_2 is not None:
        artifact_mask = detect_artifacts(analog_2, threshold=artifact_threshold)

    # Apply filters to processed signals only
    if low_cutoff > 0 or high_cutoff > 0:
        nyquist = fs / 2
        if low_cutoff > 0 and low_cutoff < nyquist:
            b, a = butter(2, low_cutoff / nyquist, 'high')
            processed_signal = filtfilt(b, a, processed_signal)
            if processed_analog_2 is not None:
                processed_analog_2 = filtfilt(b, a, processed_analog_2)
        if high_cutoff > 0 and high_cutoff < nyquist:
            b, a = butter(2, high_cutoff / nyquist, 'low')
            processed_signal = filtfilt(b, a, processed_signal)
            if processed_analog_2 is not None:
                processed_analog_2 = filtfilt(b, a, processed_analog_2)

    # Apply downsampling if needed
    if downsample_factor > 1:
        # Downsample processed signals
        processed_time, processed_signal = downsample_data(time, processed_signal, int(downsample_factor))
        if processed_analog_2 is not None:
            _, processed_analog_2 = downsample_data(time, processed_analog_2, int(downsample_factor))

        # Downsample raw signals for display (not processing)
        downsampled_time, downsampled_raw_analog_1 = downsample_data(time, downsampled_raw_analog_1,
                                                                     int(downsample_factor))
        if downsampled_raw_analog_2 is not None:
            _, downsampled_raw_analog_2 = downsample_data(time, downsampled_raw_analog_2, int(downsample_factor))

        # Also downsample the artifact mask if it exists
        if artifact_mask is not None:
            _, artifact_mask = downsample_data(time, artifact_mask.astype(float), int(downsample_factor))
            # Convert back to boolean
            artifact_mask = artifact_mask > 0.5
    else:
        processed_time = time
        downsampled_time = time

    # DO NOT downsample digital signals
    processed_digital_1 = digital_1
    processed_digital_2 = digital_2

    # Return the processed data, downsampled raw data, and artifact mask
    return processed_time, processed_signal, processed_analog_2, downsampled_raw_analog_1, downsampled_raw_analog_2, processed_digital_1, processed_digital_2, artifact_mask


class PhotometryViewer:
    def __init__(self, root, file_path):
        self.root = root
        self.file_path = file_path

        # Setup proper closing behavior
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Read and parse data
        self.header, self.data_bytes = read_ppd_file(file_path)
        self.time, self.analog_1, self.analog_2, self.digital_1, self.digital_2 = parse_ppd_data(
            self.header, self.data_bytes)

        if self.time is None or self.analog_1 is None:
            print("Error: Failed to load or parse data")
            messagebox.showerror("Error", "Failed to load or parse data")
            root.destroy()
            return

        # Store original data
        self.raw_analog_1 = self.analog_1.copy()
        self.raw_analog_2 = self.analog_2.copy() if self.analog_2 is not None else None

        # Initial parameters
        self.low_cutoff = 0.01
        self.high_cutoff = 1.0
        self.downsample_factor = 1
        self.artifact_threshold = 3.0

        # Process data with initial parameters
        result = process_data(
            self.time, self.analog_1, self.analog_2, self.digital_1, self.digital_2,
            self.low_cutoff, self.high_cutoff, self.downsample_factor, self.artifact_threshold
        )

        # Unpack results
        self.processed_time, self.processed_signal, self.processed_analog_2, \
            self.downsampled_raw_analog_1, self.downsampled_raw_analog_2, \
            self.processed_digital_1, self.processed_digital_2, self.artifact_mask = result

        # Create the GUI
        self.create_gui()

    def create_gui(self):
        # Create a main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Add zoom buttons at the top
        self.add_navigation_buttons()

        # Create a frame for the matplotlib figure
        self.frame = tk.Frame(self.main_frame)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1])

        # Panel 1: Processed signal plot with artifact markers
        self.ax1 = self.fig.add_subplot(gs[0])
        self.line, = self.ax1.plot(self.processed_time / 60, self.processed_signal, 'g-', lw=1)

        # Initialize artifact markers (will be updated later)
        self.artifact_markers_processed, = self.ax1.plot([], [], 'ro', ms=4, alpha=0.7, label='Artifacts')

        self.ax1.set_ylabel('\u0394F/F')
        self.ax1.set_title('Processed Photometry Signal')
        self.ax1.grid(True, alpha=0.3)

        # Auto-scale y-axis with asymmetric scaling (5*STD positive, 2*STD negative)
        self.update_y_limits()

        # Panel 2: Raw signals plot (both analog channels) with artifact markers
        self.ax2 = self.fig.add_subplot(gs[1], sharex=self.ax1)
        self.raw_line, = self.ax2.plot(self.processed_time / 60, self.downsampled_raw_analog_1, 'b-', lw=1,
                                       label='Analog 1')

        if self.downsampled_raw_analog_2 is not None:
            self.raw_line2, = self.ax2.plot(self.processed_time / 60, self.downsampled_raw_analog_2, 'g-', lw=1,
                                            label='Analog 2')

            # Add artifact markers on the 405nm channel
            self.artifact_markers_raw, = self.ax2.plot([], [], 'ro', ms=4, alpha=0.7, label='Artifacts')

            self.ax2.legend(loc='upper right')

        self.ax2.set_ylabel('Raw Signal')
        self.ax2.set_title('Raw Analog Signals')
        self.ax2.grid(True, alpha=0.3)

        # Panel 3: Combined Digital signals
        self.ax3 = self.fig.add_subplot(gs[2], sharex=self.ax1)
        self.digital_lines = []

        if self.processed_digital_1 is not None:
            line, = self.ax3.plot(self.time / 60, self.processed_digital_1, 'b-', lw=1, label='Digital 1')
            self.digital_lines.append(line)

        if self.processed_digital_2 is not None:
            line, = self.ax3.plot(self.time / 60, self.processed_digital_2, 'r-', lw=1, label='Digital 2')
            self.digital_lines.append(line)

        self.ax3.set_ylabel('Digital Signals')
        self.ax3.set_ylim(-0.1, 1.1)
        self.ax3.set_xlabel('Time (minutes)')
        if self.digital_lines:
            self.ax3.legend(loc='upper right')

        # Update artifact markers
        self.update_artifact_markers()

        # Create canvas and toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.toolbar.update()

        # Create slider frame with more space
        self.slider_frame = tk.Frame(self.main_frame)
        self.slider_frame.pack(fill=tk.X, padx=20, pady=10)

        # Create sliders with larger size
        self.create_sliders()

        # Connect mouse wheel events for zoom (keep this functionality)
        self.connect_zoom_events()

        # Adjust layout
        self.fig.tight_layout()

    def on_closing(self):
        """Handle the window closing event properly"""
        print("Closing application...")
        try:
            # Clean up resources
            plt.close('all')  # Close all matplotlib figures

            # Destroy the Tkinter root window
            self.root.quit()
            self.root.destroy()

            # Force exit if needed
            import sys
            sys.exit(0)
        except Exception as e:
            print(f"Error during closing: {e}")
            import sys
            sys.exit(1)

    def create_sliders(self):
        # Create a better grid layout for sliders
        self.slider_frame.columnconfigure(1, weight=1)  # Make slider column expandable

        # Low cutoff slider - LARGER SIZE
        tk.Label(self.slider_frame, text="Low cutoff (Hz):", font=('Arial', 12)).grid(
            row=0, column=0, sticky="w", padx=10, pady=10)
        self.low_slider = tk.Scale(self.slider_frame, from_=0.001, to=0.5,
                                   resolution=0.001, orient=tk.HORIZONTAL,
                                   length=800, width=25, font=('Arial', 12),
                                   command=self.update_filter)
        self.low_slider.set(self.low_cutoff)
        self.low_slider.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # High cutoff slider - LARGER SIZE
        tk.Label(self.slider_frame, text="High cutoff (Hz):", font=('Arial', 12)).grid(
            row=1, column=0, sticky="w", padx=10, pady=10)
        self.high_slider = tk.Scale(self.slider_frame, from_=0.1, to=5.0,
                                    resolution=0.1, orient=tk.HORIZONTAL,
                                    length=800, width=25, font=('Arial', 12),
                                    command=self.update_filter)
        self.high_slider.set(self.high_cutoff)
        self.high_slider.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Downsample slider - LARGER SIZE
        tk.Label(self.slider_frame, text="Downsample factor:", font=('Arial', 12)).grid(
            row=2, column=0, sticky="w", padx=10, pady=10)
        self.downsample_slider = tk.Scale(self.slider_frame, from_=1, to=100,
                                          resolution=1, orient=tk.HORIZONTAL,
                                          length=800, width=25, font=('Arial', 12),
                                          command=self.update_filter)
        self.downsample_slider.set(self.downsample_factor)
        self.downsample_slider.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        # Artifact threshold slider - LARGER SIZE
        tk.Label(self.slider_frame, text="Artifact threshold:", font=('Arial', 12)).grid(
            row=3, column=0, sticky="w", padx=10, pady=10)
        self.artifact_slider = tk.Scale(self.slider_frame, from_=1.0, to=10.0,
                                        resolution=0.1, orient=tk.HORIZONTAL,
                                        length=800, width=25, font=('Arial', 12),
                                        command=self.update_filter)
        self.artifact_slider.set(self.artifact_threshold)
        self.artifact_slider.grid(row=3, column=1, padx=10, pady=10, sticky="ew")

    def update_artifact_markers(self):
        """Update the artifact markers on both plots"""
        if self.artifact_mask is None or not np.any(self.artifact_mask):
            # No artifacts detected, clear the markers
            self.artifact_markers_processed.set_data([], [])
            if hasattr(self, 'artifact_markers_raw'):
                self.artifact_markers_raw.set_data([], [])
            return

        # Get the time points where artifacts occur (in minutes)
        artifact_times = self.processed_time[self.artifact_mask] / 60

        # Update markers on the processed signal
        artifact_values_processed = self.processed_signal[self.artifact_mask]
        self.artifact_markers_processed.set_data(artifact_times, artifact_values_processed)

        # Update markers on the raw signal (405nm channel)
        if hasattr(self, 'artifact_markers_raw') and self.downsampled_raw_analog_2 is not None:
            artifact_values_raw = self.downsampled_raw_analog_2[self.artifact_mask]
            self.artifact_markers_raw.set_data(artifact_times, artifact_values_raw)

    def add_navigation_buttons(self):
        # Create a frame for navigation buttons at the top
        nav_frame = tk.Frame(self.main_frame)
        nav_frame.pack(fill=tk.X, padx=20, pady=10)

        # Use a more compact layout for the navigation controls
        # Row 1: Zoom and Pan controls
        tk.Label(nav_frame, text="Navigation:", font=('Arial', 12, 'bold')).grid(
            row=0, column=0, padx=10, pady=5, sticky="w")

        # X-axis zoom buttons
        tk.Button(nav_frame, text="Zoom In (X)", font=('Arial', 12), width=10,
                  command=lambda: self.zoom_x(in_out="in")).grid(
            row=0, column=1, padx=5, pady=5)

        tk.Button(nav_frame, text="Zoom Out (X)", font=('Arial', 12), width=10,
                  command=lambda: self.zoom_x(in_out="out")).grid(
            row=0, column=2, padx=5, pady=5)

        # Y-axis zoom buttons (for all panels)
        tk.Button(nav_frame, text="Zoom In (Y)", font=('Arial', 12), width=10,
                  command=lambda: self.zoom_y_all("in")).grid(
            row=0, column=3, padx=5, pady=5)

        tk.Button(nav_frame, text="Zoom Out (Y)", font=('Arial', 12), width=10,
                  command=lambda: self.zoom_y_all("out")).grid(
            row=0, column=4, padx=5, pady=5)

        # Pan buttons
        tk.Button(nav_frame, text="← Pan Left", font=('Arial', 12), width=10,
                  command=lambda: self.pan_x(direction="left")).grid(
            row=0, column=5, padx=5, pady=5)

        tk.Button(nav_frame, text="Pan Right →", font=('Arial', 12), width=10,
                  command=lambda: self.pan_x(direction="right")).grid(
            row=0, column=6, padx=5, pady=5)

        # Reset view button
        tk.Button(nav_frame, text="Reset View", font=('Arial', 12, 'bold'), width=15,
                  command=self.reset_view, bg='lightblue').grid(
            row=0, column=7, padx=10, pady=5)

    def zoom_x(self, in_out="in"):
        """Zoom in or out on the x-axis"""
        # Get current x-limits
        xlim = self.ax1.get_xlim()

        # Calculate midpoint
        mid = (xlim[0] + xlim[1]) / 2

        # Calculate new range
        current_range = xlim[1] - xlim[0]
        if in_out == "in":
            new_range = current_range * 0.7  # Zoom in by 30%
        else:
            new_range = current_range * 1.4  # Zoom out by 40%

        # Calculate new limits
        new_xlim = [mid - new_range / 2, mid + new_range / 2]

        # Apply to all axes
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_xlim(new_xlim)

        # Redraw
        self.canvas.draw_idle()

    def pan_x(self, direction="right"):
        """Pan the view left or right"""
        # Get current x-limits
        xlim = self.ax1.get_xlim()

        # Calculate the amount to shift (25% of visible range)
        shift_amount = 0.25 * (xlim[1] - xlim[0])

        # Apply shift based on direction
        if direction == "left":
            new_xlim = [xlim[0] - shift_amount, xlim[1] - shift_amount]
        else:  # right
            new_xlim = [xlim[0] + shift_amount, xlim[1] + shift_amount]

        # Apply to all axes
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_xlim(new_xlim)

        # Redraw
        self.canvas.draw_idle()

    def zoom_y_all(self, in_out="in"):
        """Zoom in or out on the y-axis for all panels (except digital)"""
        # Zoom the processed signal panel
        self.zoom_y(self.ax1, in_out)

        # Zoom the raw signal panel
        self.zoom_y(self.ax2, in_out)

        # Don't zoom the digital panel - it should stay fixed

        # Redraw
        self.canvas.draw_idle()

    def zoom_y(self, ax, in_out="in"):
        """Zoom in or out on the y-axis for the specified axis"""
        # Get current y-limits
        ylim = ax.get_ylim()

        # Calculate midpoint
        mid = (ylim[0] + ylim[1]) / 2

        # Calculate new range
        current_range = ylim[1] - ylim[0]
        if in_out == "in":
            new_range = current_range * 0.7  # Zoom in by 30%
        else:
            new_range = current_range * 1.4  # Zoom out by 40%

        # Calculate new limits
        new_ylim = [mid - new_range / 2, mid + new_range / 2]

        # Apply to the specified axis
        ax.set_ylim(new_ylim)

    def reset_view(self):
        """Reset the view to show all data"""
        # Reset x-axis to show all data
        self.ax1.set_xlim(self.time[0] / 60, self.time[-1] / 60)

        # Reset y-axis for processed signal (asymmetric: 5*STD positive, 2*STD negative)
        self.update_y_limits()

        # Reset y-axis for raw signal
        raw_data = [self.raw_analog_1]
        if self.raw_analog_2 is not None:
            raw_data.append(self.raw_analog_2)

        min_val = min([data.min() for data in raw_data])
        max_val = max([data.max() for data in raw_data])
        margin = 0.1 * (max_val - min_val)
        self.ax2.set_ylim(min_val - margin, max_val + margin)

        # Reset y-axis for digital signals
        self.ax3.set_ylim(-0.1, 1.1)

        # Redraw
        self.canvas.draw_idle()

    def connect_zoom_events(self):
        """Connect mouse wheel events for custom zooming"""
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.ctrl_pressed = False

    def on_key_press(self, event):
        """Handle key press events to track Ctrl key state"""
        if event.key == 'control':
            self.ctrl_pressed = True

    def on_key_release(self, event):
        """Handle key release events to track Ctrl key state"""
        if event.key == 'control':
            self.ctrl_pressed = False

    def on_scroll(self, event):
        """
        Handle scroll events:
        - Normal scroll: zoom y-axis of all analog plots
        - Ctrl+scroll: zoom x-axis of all plots
        """
        # Check if the event occurred within an axis
        if event.inaxes is None:
            return

        if self.ctrl_pressed:
            # Ctrl+scroll: zoom x-axis (all axes)
            base_scale = 1.3  # Increased zoom factor for better effect

            # Get current x limits
            xlim = self.ax1.get_xlim()  # Use ax1 as the reference
            xdata = event.xdata

            # Calculate new limits
            if event.button == 'up':  # Zoom in
                new_xlim = [xdata - (xdata - xlim[0]) / base_scale,
                            xdata + (xlim[1] - xdata) / base_scale]
            else:  # Zoom out
                new_xlim = [xdata - (xdata - xlim[0]) * base_scale,
                            xdata + (xlim[1] - xdata) * base_scale]

            # Apply new limits to all axes
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.set_xlim(new_xlim)
        else:
            # Normal scroll: zoom y-axis of all analog plots
            base_scale = 1.2

            # Don't allow y zoom on digital signals panel
            if event.inaxes == self.ax3:
                return

            # Get current y-limits of the axis being scrolled
            ylim = event.inaxes.get_ylim()
            ydata = event.ydata

            # Calculate new limits
            if event.button == 'up':  # Zoom in
                new_ylim = [ydata - (ydata - ylim[0]) / base_scale,
                            ydata + (ylim[1] - ydata) / base_scale]
            else:  # Zoom out
                new_ylim = [ydata - (ydata - ylim[0]) * base_scale,
                            ydata + (ylim[1] - ydata) * base_scale]

            # Apply new limits to both analog panels
            if event.inaxes == self.ax1:
                self.ax1.set_ylim(new_ylim)
            elif event.inaxes == self.ax2:
                self.ax2.set_ylim(new_ylim)

        # Redraw the figure
        self.canvas.draw_idle()

    def update_filter(self, _=None):
        """Update filter parameters and reprocess data"""
        # Save current axis limits before updating
        x_lim = self.ax1.get_xlim()
        y1_lim = self.ax1.get_ylim()
        y2_lim = self.ax2.get_ylim()

        # Get updated parameter values
        self.low_cutoff = self.low_slider.get()
        self.high_cutoff = self.high_slider.get()
        self.downsample_factor = int(self.downsample_slider.get())
        self.artifact_threshold = self.artifact_slider.get()

        print(f"Processing with: low={self.low_cutoff}, high={self.high_cutoff}, "
              f"downsample={self.downsample_factor}, artifact_threshold={self.artifact_threshold}")

        # Reprocess with new parameters
        result = process_data(
            self.time, self.raw_analog_1, self.raw_analog_2, self.digital_1, self.digital_2,
            low_cutoff=self.low_cutoff, high_cutoff=self.high_cutoff,
            downsample_factor=self.downsample_factor, artifact_threshold=self.artifact_threshold
        )

        # Unpack results
        self.processed_time, self.processed_signal, self.processed_analog_2, \
            self.downsampled_raw_analog_1, self.downsampled_raw_analog_2, \
            self.processed_digital_1, self.processed_digital_2, self.artifact_mask = result

        # Update artifact markers
        self.update_artifact_markers()

        # Update processed signal plot
        self.line.set_xdata(self.processed_time / 60)
        self.line.set_ydata(self.processed_signal)

        # Update raw analog plots with downsampled RAW data
        self.raw_line.set_xdata(self.processed_time / 60)
        self.raw_line.set_ydata(self.downsampled_raw_analog_1)

        if hasattr(self, 'raw_line2') and self.downsampled_raw_analog_2 is not None:
            self.raw_line2.set_xdata(self.processed_time / 60)
            self.raw_line2.set_ydata(self.downsampled_raw_analog_2)

        # Digital signals use original time base
        if len(self.digital_lines) > 0 and self.processed_digital_1 is not None:
            self.digital_lines[0].set_xdata(self.time / 60)
            self.digital_lines[0].set_ydata(self.processed_digital_1)
        if len(self.digital_lines) > 1 and self.processed_digital_2 is not None:
            self.digital_lines[1].set_xdata(self.time / 60)
            self.digital_lines[1].set_ydata(self.processed_digital_2)

        # Restore previous axis limits instead of auto-scaling
        self.ax1.set_xlim(x_lim)
        self.ax1.set_ylim(y1_lim)
        self.ax2.set_ylim(y2_lim)

        # Redraw
        self.canvas.draw_idle()

    def update_y_limits(self):
        """Update y-axis limits for processed signal with asymmetric scaling.
        Positive side: 5*STD, Negative side: 2*STD"""
        mean_val = np.mean(self.processed_signal)
        std_val = np.std(self.processed_signal)
        self.ax1.set_ylim(mean_val - 2 * std_val, mean_val + 5 * std_val)


def main():
    """Main function to start the application."""
    try:
        # Create the root window
        root = tk.Tk()
        root.title("Photometry Signal Viewer")
        root.geometry("1400x1000")  # Larger window size

        # Register cleanup function to ensure application exits
        def cleanup():
            if root.winfo_exists():
                root.quit()
                root.destroy()

        atexit.register(cleanup)

        # Show file dialog
        file_path = filedialog.askopenfilename(
            title="Select PPD File",
            filetypes=[("PPD files", "*.ppd"), ("All files", "*.*")]
        )

        if not file_path:
            print("No file selected. Exiting.")
            root.destroy()
            return

        # Create viewer application
        app = PhotometryViewer(root, file_path)

        # Start main loop
        root.mainloop()

    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()