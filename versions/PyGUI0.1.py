import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.signal import filtfilt, butter
import time as time_module
from matplotlib.backend_bases import MouseButton


def read_ppd_file(file_path):
    """
    Reads a .ppd file and returns the header and data.
    """
    try:
        with open(file_path, 'rb') as f:
            # Try the first format parsing method
            try:
                # Read file format version (first 2 bytes as int16)
                header_size = int.from_bytes(f.read(2), 'little')
                header_bytes = f.read(header_size)

                # Try different encodings for the header
                try:
                    header_str = header_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        header_str = header_bytes.decode('latin-1')
                    except:
                        header_str = header_bytes.decode('ascii', errors='replace')

                header = json.loads(header_str)

                # The rest of the file contains data
                data_bytes = f.read()
                print(f"Successfully read file: {len(data_bytes)} bytes of data")
                return header, data_bytes

            except Exception as format1_error:
                # If first format fails, try second format
                f.seek(0)  # Reset file pointer to beginning

                # Try the second format (4-byte int32 version followed by 4-byte int32 header length)
                version = int.from_bytes(f.read(4), 'little')
                header_len = int.from_bytes(f.read(4), 'little')

                header_bytes = f.read(header_len)
                # Try different encodings
                try:
                    header_str = header_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        header_str = header_bytes.decode('latin-1')
                    except:
                        header_str = header_bytes.decode('ascii', errors='replace')

                header = json.loads(header_str)

                # The rest of the file contains data
                data_bytes = f.read()
                print(f"Successfully read file (second format): {len(data_bytes)} bytes of data")
                return header, data_bytes

    except Exception as e:
        print(f"Error reading PPD file: {str(e)}")
        # Return dummy header and data for demonstration
        dummy_size = 1000
        dummy_header = {"sampleCount": dummy_size, "hasAnalog1": True, "hasDigital1": True}
        dummy_time = np.linspace(0, 100, dummy_size)
        dummy_analog = np.sin(dummy_time / 5) + 0.1 * np.random.randn(dummy_size)
        dummy_digital = np.zeros(dummy_size)

        # Pack dummy data into bytes
        dummy_data = np.concatenate([
            dummy_time.astype(np.float64),
            dummy_analog.astype(np.float64),
            dummy_digital.astype(np.uint8)
        ])

        return dummy_header, dummy_data.tobytes()


def parse_ppd_data(header, data_bytes):
    """
    Parse the binary data into analog and digital signals.

    Parameters
    -----------
    header : dict
        Header information from PPD file
    data_bytes : bytes
        Binary data from PPD file

    Returns
    --------
    time : array
        Time in seconds
    analog_1 : array
        Analog channel 1 (470nm signal)
    analog_2 : array
        Analog channel 2 (405nm signal)
    digital_1 : array
        Digital channel 1
    digital_2 : array
        Digital channel 2
    """
    try:
        if header is None or data_bytes is None:
            print("Error: Invalid header or data_bytes")
            return None, None, None, None, None

        # Extract relevant information from header
        samples = header.get('samples', 0)
        channels = header.get('channels', 0)
        sample_rate = header.get('samplerate', 0)

        if samples <= 0 or channels <= 0 or sample_rate <= 0:
            # Try alternative parsing for older file formats
            print("Trying alternative data format parsing...")

            # Check if this is standard PPD data
            if len(data_bytes) > 0:
                # If missing metadata, try to infer from data size
                bytes_per_sample = 2  # Assuming 16-bit samples

                # Infer channels (typically 4 for fiber photometry)
                if channels <= 0:
                    channels = 4

                # Calculate samples based on data size and channels
                samples = len(data_bytes) // (bytes_per_sample * channels)

                # Use default sample rate for fiber photometry if not specified
                if sample_rate <= 0:
                    sample_rate = 1000  # Default to 1000 Hz common for fiber photometry

        # Generate time array based on sample count and rate
        if samples > 0 and sample_rate > 0:
            time = np.linspace(0, samples / sample_rate, samples, endpoint=False)
        else:
            print("Warning: Could not determine valid time array parameters")
            time = np.arange(len(data_bytes) // 8)  # Create a default time array

        # For debugging
        print(f"Sample count: {samples}, Sample rate: {sample_rate} Hz")

        # Check if we're handling dummy data
        if header.get('sampleCount', 0) == 1000 and 'hasAnalog1' in header and header['hasAnalog1']:
            # For dummy data return predefined signals
            dummy_size = 1000
            time = np.linspace(0, 100, dummy_size)
            analog_1 = np.sin(time / 5) + 0.1 * np.random.randn(dummy_size)
            digital_1 = np.zeros(dummy_size, dtype=np.uint8)
            return time, analog_1, None, digital_1, None

        # Try to parse from standard PPD format
        sample_count = header.get('sampleCount', 0)
        if sample_count > 0:
            # Standard format parsing
            offset = 0

            # Get time values
            time = np.frombuffer(data_bytes[offset:offset + 8 * sample_count], dtype=np.float64)
            offset += 8 * sample_count

            # Get analog signals
            analog_1 = None
            analog_2 = None
            if header.get('hasAnalog1', False):
                analog_1 = np.frombuffer(data_bytes[offset:offset + 8 * sample_count], dtype=np.float64)
                offset += 8 * sample_count
            if header.get('hasAnalog2', False):
                analog_2 = np.frombuffer(data_bytes[offset:offset + 8 * sample_count], dtype=np.float64)
                offset += 8 * sample_count

            # Get digital signals
            digital_1 = None
            digital_2 = None
            if header.get('hasDigital1', False):
                digital_1 = np.frombuffer(data_bytes[offset:offset + sample_count], dtype=np.uint8)
                offset += sample_count
            if header.get('hasDigital2', False):
                digital_2 = np.frombuffer(data_bytes[offset:offset + sample_count], dtype=np.uint8)

            return time, analog_1, analog_2, digital_1, digital_2

        # If standard format fails, try alternative format parsing (data without explicit sample count)
        else:
            print("Trying alternative data format parsing...")
            # Determine data structure from size
            total_bytes = len(data_bytes)

            # Assume the data is interleaved (time, analog, time, analog, ...)
            # First, determine what signals we have
            has_analog = True  # Assume at minimum we have analog signal
            has_digital = header.get('hasDigitalChannel', False)

            # Determine record size based on contents
            record_size = 8  # At minimum 8 bytes for timestamp
            if has_analog:
                record_size += 8  # 8 more bytes for analog value
            if has_digital:
                record_size += 1  # 1 more byte for digital value

            # Calculate number of records
            n_records = total_bytes // record_size

            # Now extract the data
            time = np.zeros(n_records)
            analog_1 = np.zeros(n_records) if has_analog else None
            digital_1 = np.zeros(n_records, dtype=np.uint8) if has_digital else None

            for i in range(n_records):
                offset = i * record_size
                time[i] = np.frombuffer(data_bytes[offset:offset + 8], dtype=np.float64)[0]
                offset += 8

                if has_analog:
                    analog_1[i] = np.frombuffer(data_bytes[offset:offset + 8], dtype=np.float64)[0]
                    offset += 8

                if has_digital:
                    digital_1[i] = data_bytes[offset]

            return time, analog_1, None, digital_1, None

    except Exception as e:
        print(f"Error parsing data: {str(e)}")
        # Return dummy data as fallback
        dummy_size = 1000
        time = np.linspace(0, 100, dummy_size)
        analog_1 = np.sin(time / 5) + 0.1 * np.random.randn(dummy_size)
        digital_1 = np.zeros(dummy_size, dtype=np.uint8)
        return time, analog_1, None, digital_1, None


def butter_filter(data, cutoff, fs, order=5):
    """
    Applies a Butterworth filter to the data.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    y = filtfilt(b, a, data)
    return y


def correct_405(signal_470, control_405):
    """
    Performs motion correction for fiber photometry data using the isosbestic control signal.
    Uses linear regression to scale the control signal to match the calcium-dependent signal.

    Parameters
    ----------
    signal_470 : array
        The 470nm calcium-dependent signal
    control_405 : array
        The 405nm isosbestic control signal

    Returns
    -------
    corrected_signal : array
        Motion-corrected 470nm signal
    """
    # Safety checks
    if signal_470 is None or control_405 is None:
        print("Warning: Missing signal for motion correction")
        return signal_470

    if len(signal_470) != len(control_405):
        print(f"Warning: Signal length mismatch in motion correction: {len(signal_470)} vs {len(control_405)}")
        # Use the shorter length to avoid index errors
        min_length = min(len(signal_470), len(control_405))
        signal_470 = signal_470[:min_length]
        control_405 = control_405[:min_length]

    try:
        # Linear regression to match control signal to calcium-dependent signal
        X = control_405.reshape(-1, 1)
        y = signal_470

        # Add a constant term to the predictor for intercept
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        # Calculate coefficients (beta = [intercept, slope])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        # Scale and fit the 405 signal to the 470 signal
        fitted_control = beta[0] + beta[1] * control_405

        # Subtract the fitted control signal from the calcium signal
        motion_corrected = signal_470 - fitted_control

        print(f"Motion correction applied: Scale factor = {beta[1]:.4f}, Offset = {beta[0]:.4f}")

        return motion_corrected
    except Exception as e:
        print(f"Error in motion correction: {e}")
        return signal_470  # Return original signal on error


def calculate_dff(signal):
    """
    Calculates dF/F (delta F over F).
    """
    # Use a robust method to calculate baseline
    F0 = np.percentile(signal, 10)  # 10th percentile as baseline

    # Ensure F0 is not zero to avoid division by zero
    if abs(F0) < 1e-10:
        F0 = np.mean(np.abs(signal))
        if abs(F0) < 1e-10:
            F0 = 1.0  # Safe fallback

    # Calculate dF/F with proper clipping to avoid overflow
    dFF = (signal - F0) / F0 * 100  # Convert to percentage

    # Clip extreme values that might cause overflow
    dFF = np.clip(dFF, -1000, 1000)

    # Replace any invalid values with zeros
    dFF[~np.isfinite(dFF)] = 0

    return dFF


def remove_noise(signal, threshold=5):
    """
    Detects and removes sudden noise using a median filter.
    """
    from scipy.signal import medfilt

    # Apply median filter to get smoothed signal
    smoothed = medfilt(signal, kernel_size=5)

    # Find extreme deviations
    diffs = np.abs(signal - smoothed)
    std_diff = np.std(diffs)
    mask = diffs > threshold * std_diff

    # Replace noisy points with median values
    cleaned = signal.copy()
    cleaned[mask] = smoothed[mask]

    return cleaned


def identify_signal_quality(signal, time, window_size=5, threshold=5):
    """
    Identifies good and bad signal regions based on STD and baseline.
    window_size: time window in minutes
    threshold: factor to multiply the baseline STD
    """
    # Convert window size from minutes to samples
    dt = np.mean(np.diff(time))
    window_samples = int(window_size * 60 / dt)

    # Calculate rolling STD and baseline
    rolling_std = []
    n_windows = len(signal) // window_samples

    for i in range(n_windows):
        start = i * window_samples
        end = (i + 1) * window_samples
        window_std = np.std(signal[start:end])
        rolling_std.extend([window_std] * window_samples)

    # Fill any remaining samples
    remaining = len(signal) - len(rolling_std)
    if remaining > 0:
        rolling_std.extend([rolling_std[-1]] * remaining)

    rolling_std = np.array(rolling_std)

    # Calculate baseline STD (10th percentile of all window STDs)
    baseline_std = np.percentile(rolling_std, 10)

    # Identify bad regions (where STD is too high)
    is_bad = rolling_std > baseline_std * threshold

    return is_bad


def correct_baseline_drift(signal, window_size=90):
    """
    Corrects baseline drift using a moving average.
    """
    # Use convolution for moving average
    if window_size <= 0:
        return signal  # No correction

    kernel_size = window_size
    kernel = np.ones(kernel_size) / kernel_size

    # Pad the signal to minimize edge effects
    padded_signal = np.pad(signal, (kernel_size // 2, kernel_size // 2), mode='reflect')

    # Compute the moving average
    baseline = np.convolve(padded_signal, kernel, mode='valid')

    # Subtract the baseline and add back the mean
    corrected_signal = signal - baseline + np.mean(signal)

    return corrected_signal


def process_data(time, analog_1, analog_2, digital_1, digital_2,
                 low_cutoff=0.01,
                 high_cutoff=1.0,
                 drift_correction=10):
    """
    Processes fiber photometry data following the standard workflow:
    1. Filter both 470nm and 405nm signals
    2. Motion correction using 405nm reference
    3. Calculate dF/F

    Parameters
    -----------
    time : array
        Time in seconds
    analog_1 : array
        470nm signal (calcium-dependent)
    analog_2 : array
        405nm reference signal (isosbestic control)
    low_cutoff : float
        Low cutoff frequency for bandpass filter (Hz)
    high_cutoff : float
        High cutoff frequency for bandpass filter (Hz)
    drift_correction : float
        Length of the running-average filter to remove drift, in seconds
    """
    # Safety checks for input data
    if analog_1 is None or len(analog_1) == 0:
        print("Warning: No 470nm signal data to process")
        return time, np.zeros_like(time), digital_1, digital_2

    # Print what channels are available
    print("Processing fiber photometry data:")
    print(f"  - 470nm signal (calcium-dependent): {'Available' if analog_1 is not None else 'Not available'}")
    print(f"  - 405nm signal (isosbestic control): {'Available' if analog_2 is not None else 'Not available'}")
    print(f"  - Digital signals: {'Available' if digital_1 is not None or digital_2 is not None else 'Not available'}")

    # Compute sampling rate
    if len(time) > 1:
        dt = np.diff(time).mean()
        if dt > 0:
            fs = 1 / dt
            print(f"Sampling rate: {fs:.2f} Hz")
        else:
            # Default to 1000 Hz if time data is problematic
            fs = 1000
            print("Warning: Invalid time data, using default sampling rate of 1000 Hz")
    else:
        fs = 1000  # Default
        print("Warning: Not enough time points, using default sampling rate of 1000 Hz")

    # Ensure we have a valid sampling rate
    if fs <= 0 or np.isnan(fs) or np.isinf(fs):
        fs = 1000  # Default to 1000 Hz which is common for fiber photometry
        print("Warning: Invalid sampling rate detected, using default 1000 Hz")

    print(f"Processing data with sampling rate: {fs:.2f} Hz")
    print(f"Filter settings: low_cutoff={low_cutoff} Hz, high_cutoff={high_cutoff} Hz")

    # Store original digital signals - do not process them
    original_digital_1 = digital_1.copy() if digital_1 is not None else None
    original_digital_2 = digital_2.copy() if digital_2 is not None else None

    # Make copies of the analog signals for processing
    signal_470nm = analog_1.copy()
    control_405nm = analog_2.copy() if analog_2 is not None else None

    # Clean both signals: remove extreme outliers
    # Calculate robust statistics for outlier detection
    try:
        if signal_470nm is not None:
            median_val = np.median(signal_470nm)
            mad = np.median(np.abs(signal_470nm - median_val))
            outlier_mask = np.abs(signal_470nm - median_val) > 10 * mad
            if np.any(outlier_mask):
                print(f"Removing {np.sum(outlier_mask)} extreme outliers from 470nm signal")
                signal_470nm[outlier_mask] = median_val

        if control_405nm is not None:
            median_val = np.median(control_405nm)
            mad = np.median(np.abs(control_405nm - median_val))
            outlier_mask = np.abs(control_405nm - median_val) > 10 * mad
            if np.any(outlier_mask):
                print(f"Removing {np.sum(outlier_mask)} extreme outliers from 405nm signal")
                control_405nm[outlier_mask] = median_val
    except Exception as e:
        print(f"Warning: Error in outlier removal: {e}")

    # Pad the signal to reduce edge effects
    try:
        pad_ratio = 3
        padded_length = int(pad_ratio * len(signal_470nm))
        padded_signal = np.pad(signal_470nm, (padded_length, padded_length), mode='reflect')
    except Exception as e:
        print(f"Warning: Error in signal padding: {e}")
        padded_signal = signal_470nm  # Fall back to non-padded signal

    # Apply filters - make sure cutoff frequencies are in valid range
    filtered_470nm = signal_470nm.copy()
    filtered_405nm = control_405nm.copy() if control_405nm is not None else None

    try:
        # Normalize cutoff frequencies to the Nyquist frequency
        nyquist = fs / 2

        # Check for valid values and scale appropriately
        if nyquist <= 0:
            print("Warning: Invalid Nyquist frequency, skipping filtering")
        else:
            normalized_low_cutoff = low_cutoff / nyquist
            normalized_high_cutoff = high_cutoff / nyquist

            # Clamp the normalized cutoff frequency to a valid range (0 < Wn < 1)
            normalized_low_cutoff = min(0.9, max(0.001, normalized_low_cutoff))
            normalized_high_cutoff = min(0.95, max(0.01, normalized_high_cutoff))

            # Make sure high cutoff is greater than low cutoff
            if normalized_high_cutoff <= normalized_low_cutoff:
                normalized_high_cutoff = min(0.95, normalized_low_cutoff * 1.5)

            # Create bandpass filter with normalized cutoffs
            if low_cutoff > 0 and high_cutoff < nyquist:
                print(f"Applying bandpass filter ({low_cutoff:.4f}-{high_cutoff:.4f} Hz)")
                b_band, a_band = butter(3, [normalized_low_cutoff, normalized_high_cutoff], 'bandpass')

                # Apply filter to 470nm signal
                filtered_470nm = filtfilt(b_band, a_band, signal_470nm)

                # Apply same filter to 405nm signal if available
                if filtered_405nm is not None:
                    filtered_405nm = filtfilt(b_band, a_band, control_405nm)

            # Create notch filter at 60 Hz to remove line noise, if sampling rate is high enough
            if fs > 120:  # Nyquist frequency must be above 60 Hz
                notch_freq = 60 / nyquist
                if 0 < notch_freq < 1:  # Check if normalized frequency is valid
                    print("Applying 60 Hz notch filter")
                    b_notch, a_notch = butter(3, [max(0.01, notch_freq - 0.02), min(0.99, notch_freq + 0.02)],
                                              'bandstop')

                    # Apply notch filter to both signals
                    filtered_470nm = filtfilt(b_notch, a_notch, filtered_470nm)
                    if filtered_405nm is not None:
                        filtered_405nm = filtfilt(b_notch, a_notch, filtered_405nm)
    except Exception as e:
        print(f"Warning: Error in filtering: {e}")
        # Fall back to unfiltered signals
        filtered_470nm = signal_470nm
        filtered_405nm = control_405nm

    # Compute running average over drift_correction seconds for baseline removal
    try:
        if drift_correction > 0:
            # Calculate window size in samples
            window_size = int(drift_correction * fs)
            if window_size > 1:
                print(f"Applying baseline drift correction (window: {drift_correction} s)")
                filtered_470nm = correct_baseline_drift(filtered_470nm, window_size)
                if filtered_405nm is not None:
                    filtered_405nm = correct_baseline_drift(filtered_405nm, window_size)
    except Exception as e:
        print(f"Warning: Error in drift correction: {e}")

    # Motion correction using 405nm reference signal
    try:
        if filtered_405nm is not None:
            print("Performing motion correction using 405nm reference signal (isosbestic control)")
            motion_corrected = correct_405(filtered_470nm, filtered_405nm)
            print("Motion correction complete")
        else:
            print("Skipping motion correction - no 405nm reference signal available")
            motion_corrected = filtered_470nm
    except Exception as e:
        print(f"Warning: Error in motion correction: {e}")
        motion_corrected = filtered_470nm

    # Calculate dF/F (percent change)
    try:
        print("Calculating dF/F (percent change from baseline)")
        dFF = calculate_dff(motion_corrected)
        print("dF/F calculation complete")
    except Exception as e:
        print(f"Warning: Error in dF/F calculation: {e}")
        dFF = motion_corrected  # Fall back to motion-corrected signal

    return time, dFF, original_digital_1, original_digital_2


def plot_data(time, signal, raw_analog_1, digital_1=None, digital_2=None, filter_params=None, processor_func=None):
    """
    Plots the signal with zooming and panning.
    If digital data is provided, stimulation timepoints are indicated with markers.

    Parameters
    -----------
    time : array
        Time in seconds
    signal : array
        Signal to plot (e.g., dF/F)
    raw_analog_1 : array
        Raw analog signal for reference
    digital_1 : array, optional
        Digital signal (e.g., stimulus timing)
    digital_2 : array, optional
        Second digital signal
    filter_params : dict, optional
        Dictionary of filter parameters for interactive adjustment
    processor_func : function, optional
        Function to process data with new parameters
    """
    # Convert time to minutes for plotting
    time_min = time / 60

    # Create figure with subplots (one for signal, one for digital)
    fig = plt.figure(figsize=(12, 8))

    # Create axis for the main signal
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

    # Create axis for digital signals (if available)
    ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, sharex=ax1)

    # Create axis for raw signal
    ax3 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, sharex=ax1)

    # Hide x-axis of top plots (use only bottom plot x-axis)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # Plot the main signal
    ax1.plot(time_min, signal, 'g-', linewidth=1)
    ax1.set_ylabel('dF/F (%)')
    ax1.set_title('Fiber Photometry Data')

    # Calculate automatic y limits based on signal statistics
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    if np.isfinite(signal_mean) and np.isfinite(signal_std) and signal_std > 0:
        y_min = signal_mean - 5 * signal_std
        y_max = signal_mean + 5 * signal_std
        ax1.set_ylim(y_min, y_max)

    # Plot raw signal in the bottom subplot
    ax3.plot(time_min, raw_analog_1, 'b-', linewidth=1)
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Raw Signal')

    # Calculate automatic y limits for raw signal
    raw_mean = np.mean(raw_analog_1)
    raw_std = np.std(raw_analog_1)
    if np.isfinite(raw_mean) and np.isfinite(raw_std) and raw_std > 0:
        raw_min = raw_mean - 3 * raw_std
        raw_max = raw_mean + 3 * raw_std
        ax3.set_ylim(raw_min, raw_max)

    # Plot digital signals if available
    if digital_1 is not None:
        # Clear second axis
        ax2.clear()

        # Plot digital signal 1
        ax2.plot(time_min, digital_1, 'r-', linewidth=2)
        ax2.set_ylabel('TTL')
        ax2.set_ylim(-0.1, 1.1)  # Binary signals are 0 or 1

        # Add markers at the onset of TTL pulses to the main plot
        stim_onsets = np.where(np.diff(digital_1) > 0.5)[0]
        for onset in stim_onsets:
            ax1.plot(time_min[onset], signal[onset], 'r*', markersize=8)

    # Share x-axis for all subplots
    ax2.sharex(ax1)
    ax3.sharex(ax1)

    # Set up interactive zoom and pan
    # Variables to store the starting point of a pan
    pan_start_x = None
    pan_start_y = None
    mouse_pressed = False
    last_update_time = 0  # For frame rate limiting

    # Button dimensions and positions
    button_width = 0.15
    button_height = 0.04
    button_spacing = 0.02

    # Add 'Reset Zoom' button
    reset_zoom_ax = plt.axes([0.01, 0.01, button_width, button_height])
    reset_zoom_button = Button(reset_zoom_ax, 'Reset Zoom')

    # Add 'Save PDF' button
    save_pdf_ax = plt.axes([0.01 + button_width + button_spacing, 0.01, button_width, button_height])
    save_pdf_button = Button(save_pdf_ax, 'Save PDF')

    # Add 'Open File' button
    open_file_ax = plt.axes([0.01 + 2 * (button_width + button_spacing), 0.01, button_width, button_height])
    open_file_button = Button(open_file_ax, 'Open File')

    # Function to handle mouse press
    def on_press(event):
        nonlocal pan_start_x, pan_start_y, mouse_pressed

        # Only handle events in the plot area
        if event.inaxes not in [ax1, ax2, ax3]:
            return

        # Store the starting position for panning
        if event.button == MouseButton.LEFT:
            pan_start_x = event.xdata
            pan_start_y = event.ydata
            mouse_pressed = True

    # Function to handle mouse release
    def on_release(event):
        nonlocal mouse_pressed
        mouse_pressed = False

    # Function to handle mouse motion
    def on_motion(event):
        nonlocal pan_start_x, pan_start_y, mouse_pressed, last_update_time

        # Only process if mouse is pressed for panning
        if not mouse_pressed or pan_start_x is None or pan_start_y is None:
            return

        # Only handle events in the plot area
        if event.inaxes not in [ax1, ax2, ax3]:
            return

        # Get the current time for frame rate limiting
        current_time = time_module.time()

        # Limit update rate to every 20ms (50 Hz) for smoother panning
        if current_time - last_update_time < 0.02:
            return

        # Calculate the panning distance in data coordinates
        dx = pan_start_x - event.xdata

        # Apply the panning to all plots (with linked x-axis)
        xlim = ax1.get_xlim()
        ax1.set_xlim(xlim[0] + dx, xlim[1] + dx)

        # If panning in the main plot, also adjust y-axis
        if event.inaxes == ax1:
            dy = pan_start_y - event.ydata
            ylim = ax1.get_ylim()
            ax1.set_ylim(ylim[0] + dy, ylim[1] + dy)
        elif event.inaxes == ax3:
            # Adjust y-axis for raw signal plot
            dy = pan_start_y - event.ydata
            ylim = ax3.get_ylim()
            ax3.set_ylim(ylim[0] + dy, ylim[1] + dy)

        # Store the current position as the new start position
        pan_start_x = event.xdata
        pan_start_y = event.ydata

        # Update last update time for frame rate limiting
        last_update_time = current_time

        # Draw the changes
        fig.canvas.draw_idle()

    # Add mouse wheel zoom functionality - Y-axis scaling only
    def on_scroll(event):
        if event.inaxes != ax1 and event.inaxes != ax3:
            return

        # Check if we have valid data coordinates
        if event.xdata is None or event.ydata is None:
            return

        # Calculate zoom factors
        base_scale = 1.1  # Zoom factor
        if event.button == 'up':  # Zoom in
            scale_factor = 1.0 / base_scale
        else:  # Zoom out
            scale_factor = base_scale

        # CTRL + mouse wheel scales X-axis
        if event.key == 'control':
            # Get the current x limits
            x_lim = ax1.get_xlim()

            # Calculate new limits keeping mouse position fixed (x-axis only)
            x_left = event.xdata - (event.xdata - x_lim[0]) * scale_factor
            x_right = event.xdata + (x_lim[1] - event.xdata) * scale_factor

            # Set new x-limits on all plots
            ax1.set_xlim(x_left, x_right)
            # Other axes will follow via sharex
        else:
            # Normal mouse wheel scales Y-axis
            if event.inaxes == ax1:
                # Get the current y limits
                y_lim = ax1.get_ylim()

                # Calculate new limits keeping mouse position fixed (y-axis only)
                y_bottom = event.ydata - (event.ydata - y_lim[0]) * scale_factor
                y_top = event.ydata + (y_lim[1] - event.ydata) * scale_factor

                # Set new y-limits for main plot
                ax1.set_ylim(y_bottom, y_top)

            elif event.inaxes == ax3:
                # Scale y-axis for raw signal plot
                y_lim_raw = ax3.get_ylim()

                # Calculate new limits keeping mouse position fixed
                y_bottom = event.ydata - (event.ydata - y_lim_raw[0]) * scale_factor
                y_top = event.ydata + (y_lim_raw[1] - event.ydata) * scale_factor

                ax3.set_ylim(y_bottom, y_top)

        fig.canvas.draw_idle()

    # Function to handle zoom reset
    def reset_zoom_function(event):
        ax1.set_xlim(time_min[0], time_min[-1])

        # Calculate autoscale based on mean signal standard deviation (5 times)
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        ax1.set_ylim(signal_mean - 5 * signal_std, signal_mean + 5 * signal_std)

        # Reset raw signal plot to original scale
        raw_min = np.min(raw_analog_1)
        raw_max = np.max(raw_analog_1)
        raw_range = raw_max - raw_min
        ax3.set_ylim(raw_min - 0.1 * raw_range, raw_max + 0.1 * raw_range)

        # Ensure TTL plot is correctly scaled
        ax2.set_ylim(-0.1, 1.1)

        fig.canvas.draw_idle()

    # Function to save as PDF
    def save_pdf(event):
        file_path = filedialog.asksaveasfilename(
            defaultextension='.pdf',
            filetypes=[('PDF files', '*.pdf')],
            title='Save as PDF'
        )
        if file_path:
            fig.savefig(file_path, format='pdf', dpi=300, bbox_inches='tight')
            print(f"Saved to {file_path}")

    # Function to handle opening a new file
    def open_new_file(event):
        # Create and run a new instance of the application without closing the current one
        plt.figure()  # Create a temporary figure to prevent closing the current one

        # Start a new thread to run main() so it doesn't block or close current window
        import threading
        def run_main():
            main()

        thread = threading.Thread(target=run_main)
        thread.daemon = True  # Make thread terminate when main program exits
        thread.start()

    # Connect event handlers
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    # Connect button click handlers
    reset_zoom_button.on_clicked(reset_zoom_function)
    save_pdf_button.on_clicked(save_pdf)
    open_file_button.on_clicked(open_new_file)

    # Function to reprocess data with new filter parameters
    def reprocess_data(event):
        if filter_params is not None:
            # Get slider values
            low_cutoff = low_slider.val
            high_cutoff = high_slider.val
            drift_corr = drift_slider.val

            # Close the current plot
            plt.close(fig)

            # Reprocess with new parameters
            main(low_cutoff=low_cutoff, high_cutoff=high_cutoff, drift_correction=drift_corr)

    # Apply filters button
    apply_filter_ax = plt.axes([0.65, 0.01, button_width, 0.04])
    apply_filter_button = Button(apply_filter_ax, 'Apply Filters')
    apply_filter_button.on_clicked(reprocess_data)

    # Add filter parameter sliders directly on the main figure
    # Low cutoff frequency slider (bandpass lower bound)
    low_slider_ax = plt.axes([0.2, 0.17, 0.65, 0.03])
    low_slider = Slider(
        ax=low_slider_ax,
        label='Low Cutoff (Hz)',
        valmin=0.001,
        valmax=0.5,
        valinit=filter_params.get('low_cutoff', 0.01) if filter_params else 0.01,
        valfmt='%.3f',
    )

    # High cutoff frequency slider (bandpass higher bound)
    high_slider_ax = plt.axes([0.2, 0.13, 0.65, 0.03])
    high_slider = Slider(
        ax=high_slider_ax,
        label='High Cutoff (Hz)',
        valmin=0.05,
        valmax=5.0,
        valinit=filter_params.get('high_cutoff', 1.0) if filter_params else 1.0,
        valfmt='%.2f',
    )

    # Drift correction strength slider
    drift_slider_ax = plt.axes([0.2, 0.09, 0.65, 0.03])
    drift_slider = Slider(
        ax=drift_slider_ax,
        label='Drift Correction (s)',
        valmin=0.0,
        valmax=10.0,
        valinit=filter_params.get('drift_correction', 10.0) if filter_params else 10.0,
        valfmt='%.1f',
    )

    # Function to update plot with new filter settings
    def update_plot(event=None):
        if processor_func is not None:
            # Get current slider values
            low_cutoff = low_slider.val
            high_cutoff = high_slider.val
            drift_corr = drift_slider.val

            # Store current view limits to restore after update
            current_xlim = ax1.get_xlim()
            current_ylim = ax1.get_ylim()

            # Reprocess the data with new parameters
            _, new_signal, _, _ = processor_func(low_cutoff, high_cutoff, drift_corr)

            # Clear and redraw the signal plot
            ax1.clear()
            ax1.plot(time / 60, new_signal, color='green', linewidth=1)

            # Restore the view limits
            ax1.set_xlim(current_xlim)
            ax1.set_ylim(current_ylim)

            # Add back the title and labels
            ax1.set_ylabel('dF/F (%)')
            ax1.set_title('Fiber Photometry Data')

            # Draw the changes
            fig.canvas.draw_idle()

    # Connect the sliders to the update function
    low_slider.on_changed(update_plot)
    high_slider.on_changed(update_plot)
    drift_slider.on_changed(update_plot)

    # Store the filter parameter sliders in the filter_params dictionary
    filter_params['low_cutoff_slider'] = low_slider
    filter_params['high_cutoff_slider'] = high_slider
    filter_params['drift_correction_slider'] = drift_slider

    # Set up layout
    plt.tight_layout()
    plt.show()


def open_file_dialog():
    """
    Opens a file dialog and returns the selected file path.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[("PPD Files", "*.ppd")])
    return file_path


def main(low_cutoff=0.01, high_cutoff=1.0, drift_correction=10):
    file_path = open_file_dialog()
    if file_path:
        header, data_bytes = read_ppd_file(file_path)
        if header:
            # Parse the data
            time, analog_1, analog_2, digital_1, digital_2 = parse_ppd_data(header, data_bytes)

            # Check if time array is valid
            if time is None or len(time) == 0:
                print("Error: Could not read valid time data")
                return

            # Print basic file info
            print(f"File loaded: {file_path}")
            print(f"Samples: {len(time)}")

            # Calculate and display sampling rate
            if len(time) > 1:
                # Calculate sample rate from time differences
                dt = np.diff(time).mean()
                if dt > 0:
                    sample_rate = 1.0 / dt
                    print(f"Sample rate: {sample_rate:.2f} Hz")
                else:
                    # Try to get from header if time differences are invalid
                    sample_rate = header.get('samplerate', 0)
                    if sample_rate > 0:
                        print(f"Sample rate from header: {sample_rate:.2f} Hz")
                    else:
                        # Last resort: estimate from data size
                        sample_rate = 1000  # Default to 1000 Hz for fiber photometry
                        print(f"Sample rate estimate: {sample_rate:.2f} Hz (default)")

                # Calculate and display duration
                duration_seconds = time[-1] - time[0]
                if duration_seconds > 0:
                    print(f"Duration: {duration_seconds / 60:.2f} minutes ({duration_seconds:.2f} seconds)")
                else:
                    print("Warning: Invalid time range")
            else:
                print("Warning: Not enough time points to calculate sample rate")
                sample_rate = 1000  # Default

            # Store raw signals for reference
            raw_470nm = analog_1.copy() if analog_1 is not None else None
            raw_405nm = analog_2.copy() if analog_2 is not None else None

            # Label the signals
            print("Signal identification:")
            if raw_470nm is not None:
                print("  - analog_1: 470nm calcium-dependent signal")
            if raw_405nm is not None:
                print("  - analog_2: 405nm isosbestic control signal")
            if digital_1 is not None:
                print("  - digital_1: TTL input channel 1")
            if digital_2 is not None:
                print("  - digital_2: TTL input channel 2")

            # Initial processing with default parameters
            time, processed_signal, digital_1, digital_2 = process_data(
                time, raw_470nm, raw_405nm, digital_1, digital_2,
                low_cutoff, high_cutoff, drift_correction
            )

            # Define a function to reprocess data with new parameters
            def reprocess_with_params(low_cutoff, high_cutoff, drift_corr):
                """
                Reprocesses the signal with new filter parameters.
                Returns the processed signal for plotting.
                """
                # Use raw signals for reprocessing
                return process_data(
                    time,
                    raw_470nm,  # 470nm signal
                    raw_405nm,  # 405nm reference
                    digital_1.copy() if digital_1 is not None else None,
                    digital_2.copy() if digital_2 is not None else None,
                    low_cutoff=low_cutoff,
                    high_cutoff=high_cutoff,
                    drift_correction=drift_corr
                )

            # Create filter parameters dictionary
            filter_params = {
                'low_cutoff': low_cutoff,
                'high_cutoff': high_cutoff,
                'drift_correction': drift_correction
            }

            # Plot the data - display both the processed signal and raw signals
            plot_data(time, processed_signal, raw_470nm, digital_1, digital_2, filter_params, reprocess_with_params)
        else:
            print("Error reading file.")
    else:
        print("No file selected.")


if __name__ == "__main__":
    main()