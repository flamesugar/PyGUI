import json
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
import time as time_module
# Add missing imports for sparse matrix operations used in ALS baseline correction
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve
from scipy.signal import butter, filtfilt, resample, savgol_filter
from scipy.stats import linregress
from scipy.ndimage import median_filter
# Add a more robust implementation of matplotlib 2D panning
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def read_ppd_file(file_path):
    """Reads a .ppd file and returns the header and data.

    Parameters
    ----------
    file_path : str
        Path to the PPD file

    Returns
    -------
    tuple
        (header, data_bytes) tuple containing the header dictionary and data bytes
    """
    try:
        with open(file_path, 'rb') as f:
            # First two bytes are the header length in little-endian
            header_len_bytes = f.read(2)
            header_len = int.from_bytes(header_len_bytes, byteorder='little')

            # Read the header
            header_bytes = f.read(header_len)
            header_str = header_bytes.decode('utf-8')

            # Parse the header JSON
            import json
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
    """
    Parses the data bytes from a .ppd file and returns the analog and digital signals.

    Parameters
    ----------
    header : dict
        Header dictionary containing metadata
    data_bytes : bytes
        Raw data bytes from the PPD file

    Returns
    -------
    tuple
        (time, analog_1, analog_2, digital_1, digital_2) tuple containing the parsed signals
    """
    try:
        if header is None or data_bytes is None or len(data_bytes) == 0:
            # Return dummy data for testing or if file couldn't be read properly
            print("Using dummy data for visualization")
            dummy_size = 1000
            return np.linspace(0, 100, dummy_size), np.random.randn(dummy_size), np.random.randn(dummy_size), \
                np.zeros(dummy_size), np.zeros(dummy_size)

        # Get sampling rate from header
        sampling_rate = header.get('sampling_rate', 1000)  # Default to 1000 Hz

        # Get volts per division if available
        volts_per_division = header.get('volts_per_division', [1.0, 1.0])  # Default to 1.0 if not specified

        # Parse data bytes - original format is unsigned 16-bit integers in little-endian
        data = np.frombuffer(data_bytes, dtype=np.dtype('<u2'))

        # Safety check for empty data
        if len(data) == 0:
            print("Warning: No data found in file")
            dummy_size = 1000
            return np.linspace(0, 100, dummy_size), np.random.randn(dummy_size), np.random.randn(dummy_size), \
                np.zeros(dummy_size), np.zeros(dummy_size)

        # If the data is very large, downsample immediately to prevent memory issues
        max_points = 500000  # Reasonable maximum points to process
        if len(data) > max_points * 2:  # Account for two channels
            print(f"Data is very large ({len(data)} points), downsampling to prevent memory issues")
            downsample_factor = len(data) // (max_points * 2)
            data = data[::downsample_factor]
            print(f"Downsampled to {len(data)} points")

        # Extract analog and digital signals
        # The last bit is the digital signal, the rest is the analog value
        analog = data >> 1  # Shift right to remove digital bit
        digital = data & 1  # Mask to get only the last bit

        # Separate channels - even indices are channel 1, odd indices are channel 2
        analog_1 = analog[0::2]
        analog_2 = analog[1::2]
        digital_1 = digital[0::2]
        digital_2 = digital[1::2]

        # Apply volts per division scaling if available
        if len(volts_per_division) >= 2:
            analog_1 = analog_1 * volts_per_division[0]
            analog_2 = analog_2 * volts_per_division[1]

        # Create time array
        time = np.arange(len(analog_1)) / sampling_rate  # Time in seconds

        print(f"Parsed data: {len(time)} samples, sampling rate: {sampling_rate} Hz")
        return time, analog_1, analog_2, digital_1, digital_2
    except Exception as e:
        print(f"Error parsing PPD data: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return dummy data so the visualization still works
        dummy_size = 1000
        return np.linspace(0, 100, dummy_size), np.random.randn(dummy_size), np.random.randn(dummy_size), \
            np.zeros(dummy_size), np.zeros(dummy_size)


def butter_filter(data, cutoff, fs, order=5):
    """
    Applies a Butterworth filter to the data.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def correct_405(signal, control):
    """
    Corrects the signal using the 405 control signal.
    """
    # Linear regression
    slope, intercept, r_value, p_value, std_err = linregress(control, signal)
    corrected_signal = signal - (slope * control + intercept)
    return corrected_signal


def calculate_dff(signal):
    """
    Calculates dF/F (delta F over F).
    """
    F0 = np.mean(signal)  # Baseline fluorescence
    dF = signal - F0
    dff = dF / F0
    return dff


def remove_noise(signal, threshold=5):
    """
    Detects and removes sudden noise using a median filter.
    """
    filtered_signal = median_filter(signal, size=3)
    diff = np.abs(signal - filtered_signal)
    spike_indices = np.where(diff > threshold)[0]
    signal[spike_indices] = filtered_signal[spike_indices]
    return signal


def identify_signal_quality(signal, time, window_size=5, threshold=5):
    """
    Identifies good and bad signal regions based on STD and baseline.
    window_size: time window in minutes
    threshold: factor to multiply the baseline STD
    """
    sampling_rate = len(signal) / (time[-1] / 60)  # samples per minute
    window_samples = int(window_size * sampling_rate)

    good_signal = np.ones(len(signal), dtype=bool)

    for i in range(0, len(signal), window_samples):
        window = signal[i:i + window_samples]
        if len(window) < window_samples:  # Handle the last window
            break
        baseline = np.mean(window)
        std = np.std(window)

        if std > threshold * baseline:
            good_signal[i:i + window_samples] = False  # Mark as bad signal

    return good_signal


def correct_baseline_drift(signal, window_size=90):  # Adjusted window size
    """
    Corrects baseline drift using a moving average.
    """
    # Calculate moving average
    window = np.ones(window_size) / window_size
    baseline = np.convolve(signal, window, mode='same')

    # Subtract baseline from signal
    corrected_signal = signal - 0.7 * baseline

    return corrected_signal


def process_data(time, analog_1, analog_2, digital_1, digital_2,
                 low_cutoff=0.01,
                 high_cutoff=1.0,
                 drift_correction=5.0):
    """Processes the data by applying filters, correcting baseline drift, and calculating dF/F."""
    # Compute sampling rate
    dt = np.diff(time).mean()
    fs = 1 / dt

    # Make a copy of the signal for processing
    processed_signal = analog_1.copy()

    # Apply filters if needed
    if low_cutoff > 0 or high_cutoff > 0:
        nyquist = fs / 2
        if low_cutoff > 0 and low_cutoff < nyquist:
            b, a = butter(2, low_cutoff / nyquist, 'high')
            processed_signal = filtfilt(b, a, processed_signal)
        if high_cutoff > 0 and high_cutoff < nyquist:
            b, a = butter(2, high_cutoff / nyquist, 'low')
            processed_signal = filtfilt(b, a, processed_signal)

    # Apply drift correction if needed
    if drift_correction > 0:
        window_size = int(drift_correction * fs)
        if window_size > 2 and window_size < len(processed_signal) // 2:
            window = np.ones(window_size) / window_size
            baseline = np.convolve(processed_signal, window, mode='same')
            processed_signal = processed_signal - baseline

    # Return the processed data
    return time, processed_signal, digital_1, digital_2


def als_baseline(y, lam=1e5, p=0.01, niter=10):
    """
    Asymmetric Least Squares Smoothing Baseline Correction

    This algorithm estimates the baseline of a signal by iteratively
    smoothing the signal with different weights for points above and
    below the smoothed curve.

    Parameters
    ----------
    y : array_like
        Input signal
    lam : float, optional
        Smoothness parameter. Higher values make smoother baselines
    p : float, optional
        Asymmetry parameter. Should be between 0 and 1.
        The closer to 0, the more asymmetric (stronger baseline correction).
    niter : int, optional
        Number of iterations

    Returns
    -------
    array_like
        Estimated baseline
    """
    L = len(y)
    # Second derivative approximation (for penalty matrix)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))

    # Penalty matrix
    P = lam * D.dot(D.transpose())

    # Initialize weights for asymmetric least squares
    w = np.ones(L)

    # Iterative process
    for i in range(niter):
        # Current weighted system
        W = diags(w, 0, shape=(L, L))

        # Left side of the equation
        Z = W + P

        # Solve linear system for the baseline
        z = spsolve(Z, w * y)

        # Update weights based on residuals
        w = p * (y > z) + (1 - p) * (y <= z)

    return z  # Return the baseline


def plot_interactive(file_path, processor_func=None):
    """Create an interactive plot for the given PPD file."""
    # Read the PPD file
    header, data_bytes = read_ppd_file(file_path)
    time, analog_1, analog_2, digital_1, digital_2 = parse_ppd_data(header, data_bytes)

    # Store the raw data for reprocessing
    raw_analog_1 = analog_1.copy()

    # Process the data initially with default parameters
    processed_time, processed_signal, processed_digital_1, processed_digital_2 = process_data(
        time, analog_1, analog_2, digital_1, digital_2
    )

    # Create the figure
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 0.5])

    # Main signal plot
    ax1 = fig.add_subplot(gs[0])
    line, = ax1.plot(processed_time / 60, processed_signal, 'g-', lw=1)
    ax1.set_ylabel('\u0394F/F')
    ax1.set_title('Photometry Signal')
    ax1.grid(True, alpha=0.3)

    # Digital signals plots
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    if processed_digital_1 is not None:
        digital_line, = ax2.plot(processed_time / 60, processed_digital_1, 'b-', lw=1)
        ax2.set_ylabel('Digital 1')
        ax2.set_ylim(-0.1, 1.1)

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    if processed_digital_2 is not None:
        digital_line2, = ax3.plot(processed_time / 60, processed_digital_2, 'r-', lw=1)
        ax3.set_ylabel('Digital 2')
        ax3.set_ylim(-0.1, 1.1)
    ax3.set_xlabel('Time (minutes)')

    # Add sliders for parameter adjustment
    ax_low = plt.axes([0.25, 0.01, 0.65, 0.02])
    ax_high = plt.axes([0.25, 0.04, 0.65, 0.02])
    ax_drift = plt.axes([0.25, 0.07, 0.65, 0.02])

    slider_low = Slider(ax_low, 'Low cutoff (Hz)', 0.001, 0.5, valinit=0.01)
    slider_high = Slider(ax_high, 'High cutoff (Hz)', 0.1, 5.0, valinit=1.0)
    slider_drift = Slider(ax_drift, 'Drift correction', 0.0, 20.0, valinit=5.0)

    # Function to update the plot when sliders change
    def update(val):
        low = slider_low.val
        high = slider_high.val
        drift = slider_drift.val

        print(f"Processing with: low={low}, high={high}, drift={drift}")

        # Process with new parameters
        new_time, new_signal, new_digital1, new_digital2 = process_data(
            time, raw_analog_1, analog_2, digital_1, digital_2,
            low_cutoff=low, high_cutoff=high, drift_correction=drift
        )

        # Update plots
        line.set_ydata(new_signal)
        if processed_digital_1 is not None:
            digital_line.set_ydata(new_digital1)
        if processed_digital_2 is not None:
            digital_line2.set_ydata(new_digital2)

        # Redraw
        ax1.relim()
        ax1.autoscale_view()
        fig.canvas.draw()

    # Connect sliders to update function
    slider_low.on_changed(update)
    slider_high.on_changed(update)
    slider_drift.on_changed(update)

    plt.tight_layout()

    return fig


def open_file_dialog():
    """
    Opens a file dialog and returns the selected file path.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[("PPD Files", "*.ppd")])
    return file_path


def display_ppd_file(file_path, filter_params=None):
    """Load and display PPD file data with interactive controls.

    Parameters
    ----------
    file_path : str
        Path to the PPD file to display
    filter_params : dict, optional
        Filter parameters to apply to the signal
    """
    # Load PPD file data
    header, data = read_ppd_file(file_path)

    if data is None:
        print(f"Error: Could not load data from {file_path}")
        return

    # Parse the data
    time, analog_1, analog_2, digital_1, digital_2 = parse_ppd_data(header, data)

    # Downsample very large datasets for faster processing
    if len(time) > 10000:
        print(f"Downsampling from {len(time)} to 10000 points")
        new_len = min(len(time), 10000)  # Limit to 10,000 points
        time = resample(time, new_len)
        analog_1 = resample(analog_1, new_len)
        analog_2 = resample(analog_2, new_len)
        digital_1 = resample(digital_1, new_len)
        digital_2 = resample(digital_2, new_len)
        raw_analog_1 = analog_1.copy()  # Store the raw analog signal
    else:
        raw_analog_1 = analog_1.copy()

    # Initial processing with default parameters
    time, analog_1, digital_1, digital_2 = process_data(time, analog_1, analog_2, digital_1, digital_2)

    # Define a function to reprocess data with new parameters
    def reprocess_with_params(time_arg, analog_1_arg, analog_2_arg, digital_1_arg, digital_2_arg, low_cutoff=0.01,
                              high_cutoff=1.0, drift_correction=5.0):
        print(f"Reprocessing with params: low={low_cutoff}, high={high_cutoff}, drift={drift_correction}")
        # Reprocess with new parameters but keep the original time and raw data
        return process_data(time_arg, analog_1_arg, analog_2_arg, digital_1_arg, digital_2_arg,
                            low_cutoff=low_cutoff,
                            high_cutoff=high_cutoff,
                            drift_correction=drift_correction)

    # Create filter parameters dictionary
    filter_params = {
        'low_cutoff': 0.01,
        'high_cutoff': 1.0,
        'drift_correction': 5.0
    }

    # Display the data with the interactive visualization
    plot_interactive(file_path, reprocess_with_params)


def main():
    """Main function to start the application."""
    try:
        # Start by showing a file selection dialog or create dummy data
        from tkinter import Tk, filedialog
        import os
        import sys

        # Create a root window but hide it
        root = Tk()
        root.withdraw()  # Hide the root window

        # Show the file selection dialog without the info message
        file_path = filedialog.askopenfilename(
            title="Select PPD File",
            filetypes=[("PPD files", "*.ppd"), ("All files", "*.*")]
        )

        # If a file was selected, open it, otherwise use dummy data
        if file_path and os.path.exists(file_path):
            # Keep plt.ion() and plt.show() here to make sure the window stays open
            plt.ion()  # Turn on interactive mode
            display_ppd_file(file_path)
            plt.show(block=True)  # Block execution until window is closed
        else:
            print("No file selected. Using dummy data.")

            # Create dummy data for demonstration if needed
            def create_dummy_data():
                print("Creating dummy data for demonstration")
                # Create some dummy time and signal data
                num_points = 5000
                dummy_time = np.linspace(0, 100, num_points)  # 100 seconds of data
                # Create a dummy signal with some features
                dummy_analog_1 = np.sin(dummy_time * 0.5) + 0.2 * np.random.randn(num_points)
                dummy_analog_2 = np.cos(dummy_time * 0.3) + 0.1 * np.random.randn(num_points)
                dummy_digital_1 = np.zeros(num_points)
                dummy_digital_1[np.arange(500, num_points, 1000)] = 1  # Add some pulses
                dummy_digital_2 = np.zeros(num_points)

                # Create filter parameters dictionary
                filter_params = {
                    'low_cutoff': 0.01,
                    'high_cutoff': 1.0,
                    'drift_correction': 5.0
                }

                # Define a simple reprocessing function for the dummy data
                def dummy_reprocess_func(time_arg, analog_1_arg, analog_2_arg, digital_1_arg, digital_2_arg,
                                         low_cutoff=0.01, high_cutoff=1.0, drift_correction=5.0):
                    # Just return the data with a bit of filtering to simulate processing
                    filtered = butter_filter(analog_1_arg, low_cutoff, 1000, order=2)
                    return time_arg, filtered, digital_1_arg, digital_2_arg

                # Display the dummy data
                return plot_interactive("dummy.ppd", dummy_reprocess_func)

            plt.ion()  # Turn on interactive mode
            create_dummy_data()
            plt.show(block=True)  # Block execution until window is closed
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
