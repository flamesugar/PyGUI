import json
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, Menu, ttk
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
import threading
from scipy.interpolate import interp1d
import queue

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

        # Apply volts per division scaling if available (convert to mV)
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
    Uses a simplified approach for better performance.
    """
    if signal is None:
        return None

    # Handle NaN values
    valid_signal = np.isfinite(signal)
    if not np.any(valid_signal):
        return np.zeros_like(signal, dtype=bool)

    # Calculate median and median absolute deviation (more robust to outliers)
    median_val = np.median(signal[valid_signal])
    mad = np.median(np.abs(signal[valid_signal] - median_val))

    # Scale MAD to approximate standard deviation
    mad_to_std = 1.4826  # Constant to convert MAD to std for normal distribution

    # Identify outliers using scaled MAD
    artifact_mask = np.zeros_like(signal, dtype=bool)
    artifact_mask[valid_signal] = np.abs(signal[valid_signal] - median_val) > threshold * mad * mad_to_std

    return artifact_mask


def remove_artifacts_fast(signal, artifact_mask):
    """
    Fast, aggressive artifact removal using median filtering.
    Replaces artifacts with median-filtered values.
    """
    if signal is None or artifact_mask is None or not np.any(artifact_mask):
        return signal.copy()

    # Make a copy of the signal
    corrected_signal = signal.copy()

    # Apply median filter to the entire signal (simple, fast)
    # Use a modest window size to balance artifact removal with signal preservation
    window_size = min(11, len(signal) // 100)
    if window_size % 2 == 0:  # Must be odd for median filter
        window_size += 1

    # Apply median filter and replace artifacts with filtered values
    median_filtered = median_filter(signal, size=window_size)
    corrected_signal[artifact_mask] = median_filtered[artifact_mask]

    return corrected_signal


def advanced_denoise_signal(signal, artifact_mask, time, progress_callback=None, cancel_event=None):
    """
    Apply advanced denoising to the signal with progress reporting.
    Uses both interpolation and adaptive filtering.

    Parameters:
    -----------
    signal : array
        Signal to clean
    artifact_mask : array of bool
        Boolean mask where True indicates artifacts
    time : array
        Time array for interpolation
    progress_callback : callable, optional
        Function to call with progress updates (0-100)
    cancel_event : threading.Event, optional
        Event to check for cancellation requests

    Returns:
    --------
    denoised_signal : array
        The cleaned signal
    """
    if signal is None or artifact_mask is None or not np.any(artifact_mask):
        return signal.copy()

    # Make a copy of the signal
    denoised_signal = signal.copy()

    # Find artifacts
    artifact_indices = np.where(artifact_mask)[0]

    if len(artifact_indices) == 0:
        return denoised_signal

    # Total work units for progress tracking
    total_work = len(artifact_indices)
    work_done = 0

    # 1. Find continuous artifact segments
    segments = []
    current_segment = [artifact_indices[0]]

    for i in range(1, len(artifact_indices)):
        # Check if this is a cancellation point
        if cancel_event and cancel_event.is_set():
            return signal.copy()  # Return original signal if cancelled

        # If indices are adjacent, add to current segment
        if artifact_indices[i] == artifact_indices[i - 1] + 1:
            current_segment.append(artifact_indices[i])
        else:
            # End of segment, add to list and start a new one
            segments.append(current_segment)
            current_segment = [artifact_indices[i]]

    # Add the last segment
    if current_segment:
        segments.append(current_segment)

    # Update progress
    if progress_callback:
        progress_callback(10)  # 10% complete after finding segments

    # Check for cancellation
    if cancel_event and cancel_event.is_set():
        return signal.copy()

    # 2. Process each segment using appropriate method
    valid_indices = np.where(~artifact_mask)[0]

    if len(valid_indices) < 2:  # Need at least 2 valid points for interpolation
        return denoised_signal

    # Create interpolation function for valid points
    try:
        interp_func = interp1d(time[valid_indices], signal[valid_indices],
                               kind='cubic', bounds_error=False, fill_value='extrapolate')
    except Exception as e:
        print(f"Interpolation error: {e}")
        # Fall back to linear interpolation if cubic fails
        try:
            interp_func = interp1d(time[valid_indices], signal[valid_indices],
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
        except Exception as e:
            print(f"Linear interpolation also failed: {e}")
            return denoised_signal

    # Check for cancellation
    if cancel_event and cancel_event.is_set():
        return signal.copy()

    # Process segments based on their length
    segments_processed = 0
    for segment in segments:
        # Check for cancellation periodically
        if cancel_event and cancel_event.is_set():
            return signal.copy()

        segment_len = len(segment)

        # Short segments (1-3 points): Use cubic interpolation
        if segment_len <= 3:
            for idx in segment:
                denoised_signal[idx] = interp_func(time[idx])

        # Medium segments (4-10 points): Use blend of interpolation and median filtering
        elif segment_len <= 10:
            # Get interpolated values
            interp_values = interp_func(time[segment])

            # Get median filtered values
            window_size = min(11, len(signal) // 100)
            if window_size % 2 == 0:
                window_size += 1
            median_values = median_filter(signal, size=window_size)[segment]

            # Blend interpolation (better for preserving shape) with median filtering (better for noise)
            blend_factor = 0.7  # 70% interpolation, 30% median filtering
            blended_values = blend_factor * interp_values + (1 - blend_factor) * median_values

            # Apply blended values
            denoised_signal[segment] = blended_values

        # Longer segments: Use Savitzky-Golay filtering for smoother results
        else:
            try:
                # Calculate appropriate window size based on segment length
                window_size = min(segment_len + 1, 51)
                if window_size % 2 == 0:  # Must be odd
                    window_size -= 1

                # Calculate appropriate polynomial order
                poly_order = min(3, window_size - 2)

                # Get expanded segment with some buffer on each side
                buffer = window_size // 2
                start = max(0, segment[0] - buffer)
                end = min(len(signal), segment[-1] + buffer + 1)
                expanded_segment = np.arange(start, end)

                # Apply Savitzky-Golay filter to expanded segment
                if len(expanded_segment) > window_size:
                    sg_filtered = savgol_filter(signal[expanded_segment], window_size, poly_order)

                    # Use only the filtered values for the actual artifact segment
                    for i, idx in enumerate(segment):
                        if idx in expanded_segment:
                            pos = np.where(expanded_segment == idx)[0][0]
                            denoised_signal[idx] = sg_filtered[pos]
            except Exception as e:
                print(f"Savitzky-Golay filtering failed: {e}")
                # Fall back to median filtering if SG fails
                median_values = median_filter(signal, size=min(11, segment_len))[segment]
                denoised_signal[segment] = median_values

        # Update progress
        segments_processed += 1
        if progress_callback:
            # Scale from 10% to 90% based on segments processed
            progress = 10 + int(80 * segments_processed / len(segments))
            progress_callback(progress)

        # Check for cancellation
        if cancel_event and cancel_event.is_set():
            return signal.copy()

    # 3. Final smoothing pass to reduce discontinuities at segment boundaries
    if len(segments) > 0:
        # Find all segment boundaries
        boundaries = []
        for segment in segments:
            if segment[0] > 0:
                boundaries.append(segment[0] - 1)  # Point before segment
            if segment[-1] < len(signal) - 1:
                boundaries.append(segment[-1] + 1)  # Point after segment

        # Apply light smoothing to boundary regions
        for boundary in boundaries:
            # Create a small window around the boundary
            window_size = 5
            start = max(0, boundary - window_size // 2)
            end = min(len(signal), boundary + window_size // 2 + 1)

            if end - start >= 3:  # Need at least 3 points for smoothing
                window = np.arange(start, end)
                window_values = denoised_signal[window]

                # Apply simple 3-point moving average
                smoothed = np.zeros_like(window_values)
                for i in range(1, len(window_values) - 1):
                    smoothed[i] = np.mean(window_values[i - 1:i + 2])

                # Keep endpoints as they are
                smoothed[0] = window_values[0]
                smoothed[-1] = window_values[-1]

                # Update signal with smoothed values
                denoised_signal[window] = smoothed

    # Final progress update
    if progress_callback:
        progress_callback(100)

    return denoised_signal


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


def center_signal(signal):
    """
    Center a signal by subtracting its mean.

    Parameters:
    -----------
    signal : array
        Signal to center

    Returns:
    --------
    centered_signal : array
        Signal centered around zero
    """
    if signal is None or len(signal) == 0:
        return signal

    return signal - np.mean(signal)


def process_data(time, analog_1, analog_2, digital_1, digital_2,
                 low_cutoff=0.01,
                 high_cutoff=1.0,
                 downsample_factor=20,
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

    # Apply artifact correction to the main signal
    if artifact_mask is not None and np.any(artifact_mask):
        processed_signal = remove_artifacts_fast(processed_signal, artifact_mask)

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

    # Calculate dF/F as percentage for the processed signal
    # First get baseline (mean of first 10% of recording, or first 10 seconds, whichever is less)
    baseline_idx = min(int(len(processed_signal) * 0.1), int(10 * fs))
    if baseline_idx < 10:  # Ensure at least 10 points for baseline
        baseline_idx = min(10, len(processed_signal))

    baseline = np.mean(processed_signal[:baseline_idx])
    if baseline != 0:  # Avoid division by zero
        # Convert to percentage change
        processed_signal = 100 * (processed_signal - baseline) / baseline

    # Center the raw signals around zero
    downsampled_raw_analog_1 = center_signal(downsampled_raw_analog_1)
    if downsampled_raw_analog_2 is not None:
        downsampled_raw_analog_2 = center_signal(downsampled_raw_analog_2)

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


def get_font_size_for_resolution():
    """Calculate appropriate font sizes based on screen resolution"""
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the temporary window

        # Get screen width
        screen_width = root.winfo_screenwidth()

        # Determine base font size based on screen width
        if screen_width >= 3840:  # 4K or higher
            base_size = 16
            button_size = 14
            title_size = 18
        elif screen_width >= 2560:  # 1440p
            base_size = 14
            button_size = 12
            title_size = 16
        elif screen_width >= 1920:  # 1080p
            base_size = 12
            button_size = 10
            title_size = 14
        else:  # Lower resolution
            base_size = 10
            button_size = 9
            title_size = 12

        root.destroy()

        return {
            'base': base_size,
            'button': button_size,
            'title': title_size,
            'slider': int(base_size * 0.9)
        }
    except:
        # Default sizes if there's an error
        return {
            'base': 12,
            'button': 10,
            'title': 14,
            'slider': 11
        }


class PhotometryViewer:
    def __init__(self, root, file_path=None):
        self.root = root
        self.file_path = file_path

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

        # Initialize variables
        self.header = None
        self.data_bytes = None
        self.time = None
        self.analog_1 = None
        self.analog_2 = None
        self.digital_1 = None
        self.digital_2 = None
        self.raw_analog_1 = None
        self.raw_analog_2 = None

        # Initialize processing parameters with default downsample=20
        self.low_cutoff = 0.01
        self.high_cutoff = 1.0
        self.downsample_factor = 20  # Changed default to 20
        self.artifact_threshold = 3.0

        # Denoise thread control
        self.denoise_thread = None
        self.cancel_denoise_event = threading.Event()
        self.denoise_progress_queue = queue.Queue()

        # Setup proper closing behavior
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Setup the menu bar
        self.create_menu()

        # If a file path was provided, load it
        if file_path:
            self.load_file(file_path)

    def create_menu(self):
        # Create menu bar
        menubar = Menu(self.root)
        self.root.config(menu=menubar)

        # Create File menu
        file_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open...", command=self.open_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        # Create Tools menu
        tools_menu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Advanced Denoising...", command=self.run_advanced_denoising)

    def open_file(self):
        # Show file dialog
        file_path = filedialog.askopenfilename(
            title="Select PPD File",
            filetypes=[("PPD files", "*.ppd"), ("All files", "*.*")]
        )

        if file_path:
            self.load_file(file_path)

    def load_file(self, file_path):
        """Load and process a new file"""
        self.file_path = file_path

        # Clean up any existing figure
        if hasattr(self, 'fig'):
            plt.close(self.fig)

        # Clear any existing widgets
        if hasattr(self, 'frame') and self.frame:
            self.frame.destroy()
        if hasattr(self, 'slider_frame') and self.slider_frame:
            self.slider_frame.destroy()

        # Read and parse data
        self.header, self.data_bytes = read_ppd_file(file_path)
        self.time, self.analog_1, self.analog_2, self.digital_1, self.digital_2 = parse_ppd_data(
            self.header, self.data_bytes)

        if self.time is None or self.analog_1 is None:
            print("Error: Failed to load or parse data")
            messagebox.showerror("Error", "Failed to load or parse data")
            return

        # Store original data
        self.raw_analog_1 = self.analog_1.copy()
        self.raw_analog_2 = self.analog_2.copy() if self.analog_2 is not None else None

        # Process data with initial parameters (using default downsample=20)
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

        # Update the window title to include the file name
        import os
        file_name = os.path.basename(file_path)
        self.root.title(f"Photometry Signal Viewer - {file_name}")

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

        self.ax1.set_ylabel('% ΔF/F', fontsize=self.font_sizes['base'])
        self.ax1.set_title('Processed Photometry Signal', fontsize=self.font_sizes['title'])
        self.ax1.grid(True, alpha=0.3)
        self.ax1.tick_params(labelsize=self.font_sizes['base'])

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

            self.ax2.legend(loc='upper right', fontsize=self.font_sizes['base'])

        self.ax2.set_ylabel('mV', fontsize=self.font_sizes['base'])
        self.ax2.set_title('Raw Analog Signals', fontsize=self.font_sizes['title'])
        self.ax2.grid(True, alpha=0.3)
        self.ax2.tick_params(labelsize=self.font_sizes['base'])

        # Panel 3: Combined Digital signals
        self.ax3 = self.fig.add_subplot(gs[2], sharex=self.ax1)
        self.digital_lines = []

        if self.processed_digital_1 is not None:
            line, = self.ax3.plot(self.time / 60, self.processed_digital_1, 'b-', lw=1, label='Digital 1')
            self.digital_lines.append(line)

        if self.processed_digital_2 is not None:
            line, = self.ax3.plot(self.time / 60, self.processed_digital_2, 'r-', lw=1, label='Digital 2')
            self.digital_lines.append(line)

        self.ax3.set_ylabel('TTL', fontsize=self.font_sizes['base'])
        self.ax3.set_ylim(-0.1, 1.1)
        self.ax3.set_xlabel('Time (minutes)', fontsize=self.font_sizes['base'])
        self.ax3.tick_params(labelsize=self.font_sizes['base'])
        if self.digital_lines:
            self.ax3.legend(loc='upper right', fontsize=self.font_sizes['base'])

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
            # If denoising is running, cancel it
            if self.denoise_thread and self.denoise_thread.is_alive():
                self.cancel_denoise_event.set()
                self.denoise_thread.join(timeout=1.0)

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

        font_size = self.font_sizes['base']
        slider_font_size = self.font_sizes['slider']

        # Low cutoff slider - CHANGED RANGE to 0.0001 to 0.01
        tk.Label(self.slider_frame, text="Low cutoff (Hz):", font=('Arial', font_size)).grid(
            row=0, column=0, sticky="w", padx=10, pady=10)
        self.low_slider = tk.Scale(self.slider_frame, from_=0.0001, to=0.01,
                                   resolution=0.0001, orient=tk.HORIZONTAL,
                                   length=800, width=25, font=('Arial', slider_font_size),
                                   command=self.update_filter)
        self.low_slider.set(self.low_cutoff)
        self.low_slider.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # High cutoff slider
        tk.Label(self.slider_frame, text="High cutoff (Hz):", font=('Arial', font_size)).grid(
            row=1, column=0, sticky="w", padx=10, pady=10)
        self.high_slider = tk.Scale(self.slider_frame, from_=0.1, to=5.0,
                                    resolution=0.1, orient=tk.HORIZONTAL,
                                    length=800, width=25, font=('Arial', slider_font_size),
                                    command=self.update_filter)
        self.high_slider.set(self.high_cutoff)
        self.high_slider.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Downsample slider
        tk.Label(self.slider_frame, text="Downsample factor:", font=('Arial', font_size)).grid(
            row=2, column=0, sticky="w", padx=10, pady=10)
        self.downsample_slider = tk.Scale(self.slider_frame, from_=1, to=100,
                                          resolution=1, orient=tk.HORIZONTAL,
                                          length=800, width=25, font=('Arial', slider_font_size),
                                          command=self.update_filter)
        self.downsample_slider.set(self.downsample_factor)
        self.downsample_slider.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        # Artifact threshold slider
        tk.Label(self.slider_frame, text="Artifact threshold:", font=('Arial', font_size)).grid(
            row=3, column=0, sticky="w", padx=10, pady=10)
        self.artifact_slider = tk.Scale(self.slider_frame, from_=1.0, to=10.0,
                                        resolution=0.1, orient=tk.HORIZONTAL,
                                        length=800, width=25, font=('Arial', slider_font_size),
                                        command=self.update_filter)
        self.artifact_slider.set(self.artifact_threshold)
        self.artifact_slider.grid(row=3, column=1, padx=10, pady=10, sticky="ew")

        # Add advanced denoising button
        denoising_frame = tk.Frame(self.slider_frame)
        denoising_frame.grid(row=4, column=0, columnspan=2, pady=10, sticky="ew")

        self.denoise_button = tk.Button(
            denoising_frame,
            text="Run Advanced Denoising",
            font=('Arial', self.font_sizes['title'], 'bold'),
            bg='lightgreen',
            height=2,
            command=self.run_advanced_denoising
        )
        self.denoise_button.pack(side=tk.LEFT, padx=10)

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

        button_font = ('Arial', self.font_sizes['button'])
        title_font = ('Arial', self.font_sizes['title'], 'bold')

        # Use a more compact layout for the navigation controls
        # Row 1: Zoom and Pan controls
        tk.Label(nav_frame, text="Navigation:", font=title_font).grid(
            row=0, column=0, padx=10, pady=5, sticky="w")

        # X-axis zoom buttons
        tk.Button(nav_frame, text="Zoom In (X)", font=button_font, width=10,
                  command=lambda: self.zoom_x(in_out="in")).grid(
            row=0, column=1, padx=5, pady=5)

        tk.Button(nav_frame, text="Zoom Out (X)", font=button_font, width=10,
                  command=lambda: self.zoom_x(in_out="out")).grid(
            row=0, column=2, padx=5, pady=5)

        # Y-axis zoom buttons (for all panels)
        tk.Button(nav_frame, text="Zoom In (Y)", font=button_font, width=10,
                  command=lambda: self.zoom_y_all("in")).grid(
            row=0, column=3, padx=5, pady=5)

        tk.Button(nav_frame, text="Zoom Out (Y)", font=button_font, width=10,
                  command=lambda: self.zoom_y_all("out")).grid(
            row=0, column=4, padx=5, pady=5)

        # Pan buttons
        tk.Button(nav_frame, text="← Pan Left", font=button_font, width=10,
                  command=lambda: self.pan_x(direction="left")).grid(
            row=0, column=5, padx=5, pady=5)

        tk.Button(nav_frame, text="Pan Right →", font=button_font, width=10,
                  command=lambda: self.pan_x(direction="right")).grid(
            row=0, column=6, padx=5, pady=5)

        # Reset view button
        tk.Button(nav_frame, text="Reset View", font=title_font, width=15,
                  command=self.reset_view, bg='lightblue').grid(
            row=0, column=7, padx=10, pady=5)

        # Add a file open button
        tk.Button(nav_frame, text="Open File", font=title_font, width=15,
                  command=self.open_file, bg='lightyellow').grid(
            row=0, column=8, padx=10, pady=5)

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
        raw_data = [self.downsampled_raw_analog_1]
        if self.downsampled_raw_analog_2 is not None:
            raw_data.append(self.downsampled_raw_analog_2)

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

    def run_advanced_denoising(self):
        """Run the advanced denoising process with progress tracking"""
        if self.artifact_mask is None or not np.any(self.artifact_mask):
            messagebox.showinfo("Denoising", "No artifacts detected to denoise!")
            return

        if self.denoise_thread and self.denoise_thread.is_alive():
            messagebox.showinfo("Denoising", "Denoising is already running!")
            return

        # Create a dialog for the denoising process
        self.denoise_dialog = tk.Toplevel(self.root)
        self.denoise_dialog.title("Advanced Denoising")
        self.denoise_dialog.geometry("500x150")
        self.denoise_dialog.transient(self.root)
        self.denoise_dialog.grab_set()

        # Make it non-resizable
        self.denoise_dialog.resizable(False, False)

        # Add a label
        label = tk.Label(self.denoise_dialog,
                         text="Running advanced denoising...",
                         font=('Arial', self.font_sizes['base'], 'bold'))
        label.pack(pady=10)

        # Add a progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.denoise_dialog,
                                            variable=self.progress_var,
                                            maximum=100,
                                            length=400)
        self.progress_bar.pack(pady=10)

        # Add a cancel button
        self.cancel_button = tk.Button(self.denoise_dialog,
                                       text="Cancel",
                                       font=('Arial', self.font_sizes['base']),
                                       command=self.cancel_denoising)
        self.cancel_button.pack(pady=10)

        # Reset cancellation event
        self.cancel_denoise_event.clear()

        # Clear queue
        while not self.denoise_progress_queue.empty():
            try:
                self.denoise_progress_queue.get_nowait()
            except queue.Empty:
                break

        # Start the denoising thread
        self.denoise_thread = threading.Thread(
            target=self.denoise_worker,
            args=(self.processed_signal, self.artifact_mask, self.processed_time)
        )
        self.denoise_thread.daemon = True
        self.denoise_thread.start()

        # Start checking the queue for progress updates
        self.root.after(100, self.check_denoise_progress)

    def denoise_worker(self, signal, artifact_mask, time):
        """Worker function that runs in a separate thread to perform denoising"""
        try:
            # Run the denoising function
            denoised_signal = advanced_denoise_signal(
                signal,
                artifact_mask,
                time,
                progress_callback=lambda p: self.denoise_progress_queue.put(p),
                cancel_event=self.cancel_denoise_event
            )

            # If process was not cancelled, send the result
            if not self.cancel_denoise_event.is_set():
                self.denoise_progress_queue.put(('result', denoised_signal))
        except Exception as e:
            # If an error occurred, send it to the main thread
            self.denoise_progress_queue.put(('error', str(e)))

    def check_denoise_progress(self):
        """Check the progress queue and update the UI"""
        try:
            # Check if denoising is still running
            if not self.denoise_thread.is_alive() and self.denoise_dialog:
                # Thread has finished, but we might still have messages in the queue
                while not self.denoise_progress_queue.empty():
                    try:
                        message = self.denoise_progress_queue.get_nowait()
                        if isinstance(message, tuple) and message[0] == 'result':
                            # We have a result, update the signal
                            self.processed_signal = message[1]
                            self.line.set_ydata(self.processed_signal)
                            self.canvas.draw_idle()
                        elif isinstance(message, tuple) and message[0] == 'error':
                            # Show error message
                            messagebox.showerror("Denoising Error", message[1])
                    except queue.Empty:
                        break

                # Close the dialog
                self.denoise_dialog.destroy()
                self.denoise_dialog = None
                return

            # Process any pending progress updates
            while not self.denoise_progress_queue.empty():
                try:
                    message = self.denoise_progress_queue.get_nowait()
                    if isinstance(message, (int, float)):
                        # Update progress
                        self.progress_var.set(message)
                    elif isinstance(message, tuple) and message[0] == 'result':
                        # We have a result, update the signal
                        self.processed_signal = message[1]
                        self.line.set_ydata(self.processed_signal)
                        self.canvas.draw_idle()
                    elif isinstance(message, tuple) and message[0] == 'error':
                        # Show error message
                        messagebox.showerror("Denoising Error", message[1])
                except queue.Empty:
                    break

            # Schedule the next check
            self.root.after(100, self.check_denoise_progress)

        except Exception as e:
            print(f"Error in check_denoise_progress: {e}")
            # Make sure dialog is closed if there's an error
            if hasattr(self, 'denoise_dialog') and self.denoise_dialog:
                self.denoise_dialog.destroy()
                self.denoise_dialog = None

    def cancel_denoising(self):
        """Cancel the denoising process"""
        if self.denoise_thread and self.denoise_thread.is_alive():
            # Set the cancel event
            self.cancel_denoise_event.set()

            # Update the cancel button
            self.cancel_button.config(text="Cancelling...", state=tk.DISABLED)


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

        # Create viewer application without initially loading a file
        app = PhotometryViewer(root)

        # Open file dialog
        app.open_file()

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