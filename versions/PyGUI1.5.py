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
import numpy.polynomial.polynomial as poly
import os

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


def advanced_denoise_signal(signal, artifact_mask, time, control_signal=None, progress_callback=None, cancel_event=None,
                            aggressive_mode=False, max_gap_to_remove=None):
    """
    Enhanced denoising with options for aggressive noise removal and segment deletion.

    Parameters:
    -----------
    signal : array
        Signal to clean (typically 470nm channel)
    artifact_mask : array of bool
        Boolean mask where True indicates artifacts
    time : array
        Time array for interpolation
    control_signal : array, optional
        Control signal (405nm isosbestic) for reference
    progress_callback : callable, optional
        Function to call with progress updates (0-100)
    cancel_event : threading.Event, optional
        Event to check for cancellation requests
    aggressive_mode : bool, optional
        Whether to use more aggressive denoising methods
    max_gap_to_remove : float, optional
        Maximum time gap (in seconds) to completely remove and replace with interpolation

    Returns:
    --------
    denoised_signal : array
        The cleaned signal
    """
    if signal is None or artifact_mask is None or not np.any(artifact_mask):
        return signal.copy()

    # Make a copy of the signal
    denoised_signal = signal.copy()

    # Find artifact segments
    artifact_indices = np.where(artifact_mask)[0]
    segments = []
    current_segment = [artifact_indices[0]]

    for i in range(1, len(artifact_indices)):
        if cancel_event and cancel_event.is_set():
            return signal.copy()

        if artifact_indices[i] == artifact_indices[i - 1] + 1:
            current_segment.append(artifact_indices[i])
        else:
            segments.append(current_segment)
            current_segment = [artifact_indices[i]]

    if current_segment:
        segments.append(current_segment)

    if progress_callback:
        progress_callback(10)

    # Get valid (non-artifact) indices for interpolation
    valid_indices = np.where(~artifact_mask)[0]
    if len(valid_indices) < 2:
        return denoised_signal

    # Identify large segments for complete removal if max_gap_to_remove is specified
    segments_to_remove = []
    if max_gap_to_remove is not None:
        dt = np.mean(np.diff(time))  # Average time step
        max_points = int(max_gap_to_remove / dt)  # Convert time to points

        for segment in segments:
            if len(segment) > max_points:
                segments_to_remove.append(segment)

    # Create interpolation function for valid points
    try:
        interp_func = interp1d(time[valid_indices], signal[valid_indices],
                               kind='cubic', bounds_error=False, fill_value='extrapolate')
    except Exception as e:
        print(f"Interpolation error: {e}")
        try:
            interp_func = interp1d(time[valid_indices], signal[valid_indices],
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
        except Exception as e:
            print(f"Linear interpolation also failed: {e}")
            return denoised_signal

    # Process regular segments based on their length
    segments_processed = 0

    # First handle segments marked for complete removal
    for segment in segments_to_remove:
        if cancel_event and cancel_event.is_set():
            return signal.copy()

        # Get clean data from before and after the segment
        margin = min(100, len(signal) // 20)  # Use 5% of signal or 100 points

        # Find valid points before and after
        start_idx = segment[0]
        end_idx = segment[-1]

        # Find valid points before the segment
        before_points = []
        idx = start_idx - 1
        count = 0
        while idx >= 0 and count < margin:
            if not artifact_mask[idx]:
                before_points.append(idx)
                count += 1
            idx -= 1

        # Find valid points after the segment
        after_points = []
        idx = end_idx + 1
        count = 0
        while idx < len(signal) and count < margin:
            if not artifact_mask[idx]:
                after_points.append(idx)
                count += 1
            idx += 1

        # Use direct interpolation for complete replacement
        if before_points and after_points:
            # Complete replacement using surrounding clean data
            x_interp = np.concatenate([time[before_points], time[after_points]])
            y_interp = np.concatenate([signal[before_points], signal[after_points]])

            try:
                # Create local interpolation for this segment
                local_interp = interp1d(x_interp, y_interp, kind='cubic', bounds_error=False, fill_value='extrapolate')
                # Apply to the segment
                denoised_signal[segment] = local_interp(time[segment])
            except Exception as e:
                print(f"Local interpolation failed: {e}")
                # Fall back to global interpolation
                denoised_signal[segment] = interp_func(time[segment])
        else:
            # Fall back to global interpolation if we don't have enough clean points
            denoised_signal[segment] = interp_func(time[segment])

        segments_processed += 1
        if progress_callback:
            progress = 10 + int(40 * segments_processed / len(segments))
            progress_callback(progress)

    # Process remaining segments
    remaining_segments = [s for s in segments if s not in segments_to_remove]

    for segment in remaining_segments:
        if cancel_event and cancel_event.is_set():
            return signal.copy()

        segment_len = len(segment)

        # Short segments: Use cubic interpolation
        if segment_len <= 3:
            for idx in segment:
                denoised_signal[idx] = interp_func(time[idx])

        # Medium segments: Use blend of interpolation and median filtering
        elif segment_len <= 10 or not aggressive_mode:
            # Get interpolated values
            interp_values = interp_func(time[segment])

            # Get median filtered values
            window_size = min(11, len(signal) // 100)
            if window_size % 2 == 0:
                window_size += 1
            median_values = median_filter(signal, size=window_size)[segment]

            # Blend interpolation with median filtering
            # More aggressive for longer segments
            blend_factor = 0.8 if aggressive_mode else 0.7
            blended_values = blend_factor * interp_values + (1 - blend_factor) * median_values

            # Apply blended values
            denoised_signal[segment] = blended_values

        # Longer segments with aggressive mode: Use control signal if available
        elif aggressive_mode and control_signal is not None:
            try:
                # If we have the control signal, use it for isosbestic correction
                # This is especially useful for fiber photometry where 405nm is a reference

                # Find valid segments of data before and after the noise
                pre_segment_start = max(0, segment[0] - 100)
                pre_segment_end = segment[0]
                post_segment_start = segment[-1] + 1
                post_segment_end = min(len(signal), segment[-1] + 101)

                # Get scaling factors from clean periods
                pre_segment = np.arange(pre_segment_start, pre_segment_end)
                post_segment = np.arange(post_segment_start, post_segment_end)

                # Only use valid (non-artifact) points
                pre_valid = pre_segment[~artifact_mask[pre_segment]]
                post_valid = post_segment[~artifact_mask[post_segment]]

                valid_segments = np.concatenate([pre_valid, post_valid])

                if len(valid_segments) > 10:  # Need enough valid points
                    # Calculate scaling between signal and control
                    # Linear regression: signal = a * control + b
                    valid_signal = signal[valid_segments]
                    valid_control = control_signal[valid_segments]

                    # Linear regression
                    slope, intercept, _, _, _ = linregress(valid_control, valid_signal)

                    # Apply correction: replace with predicted signal from control
                    predicted_signal = slope * control_signal[segment] + intercept

                    # Blend with interpolation for smoothness
                    interp_values = interp_func(time[segment])
                    blend_ratio = 0.7  # 70% control-based, 30% interpolation
                    denoised_signal[segment] = blend_ratio * predicted_signal + (1 - blend_ratio) * interp_values
                else:
                    # Fall back to Savitzky-Golay if not enough valid points
                    interp_values = interp_func(time[segment])
                    denoised_signal[segment] = interp_values
            except Exception as e:
                print(f"Control-based correction failed: {e}")
                # Fall back to interpolation
                interp_values = interp_func(time[segment])
                denoised_signal[segment] = interp_values

        # Longer segments without control signal: Use Savitzky-Golay
        else:
            try:
                window_size = min(segment_len + 1, 51)
                if window_size % 2 == 0:  # Must be odd
                    window_size -= 1

                poly_order = min(3, window_size - 2)

                # Get expanded segment with buffer
                buffer = window_size // 2
                start = max(0, segment[0] - buffer)
                end = min(len(signal), segment[-1] + buffer + 1)
                expanded_segment = np.arange(start, end)

                if len(expanded_segment) > window_size:
                    sg_filtered = savgol_filter(signal[expanded_segment], window_size, poly_order)

                    for i, idx in enumerate(segment):
                        if idx in expanded_segment:
                            pos = np.where(expanded_segment == idx)[0][0]
                            denoised_signal[idx] = sg_filtered[pos]
            except Exception as e:
                print(f"Savitzky-Golay filtering failed: {e}")
                # Fall back to interpolation
                denoised_signal[segment] = interp_func(time[segment])

        segments_processed += 1
        if progress_callback:
            progress = 50 + int(40 * segments_processed / len(remaining_segments))
            progress_callback(min(90, progress))

    # Final smoothing at segment boundaries to avoid discontinuities
    boundaries = []
    for segment in segments:
        if segment[0] > 0:
            boundaries.append(segment[0] - 1)
        if segment[-1] < len(signal) - 1:
            boundaries.append(segment[-1] + 1)

    for boundary in boundaries:
        window_size = 5
        start = max(0, boundary - window_size // 2)
        end = min(len(signal), boundary + window_size // 2 + 1)

        if end - start >= 3:
            window = np.arange(start, end)
            window_values = denoised_signal[window]

            # Apply 3-point moving average
            smoothed = np.zeros_like(window_values)
            for i in range(1, len(window_values) - 1):
                smoothed[i] = np.mean(window_values[i - 1:i + 2])

            smoothed[0] = window_values[0]
            smoothed[-1] = window_values[-1]

            denoised_signal[window] = smoothed

    if progress_callback:
        progress_callback(100)

    return denoised_signal


def fit_drift_curve(time, signal, poly_degree=2):
    """
    Fit a polynomial curve to the signal to model baseline drift.

    Parameters:
    -----------
    time : array
        Time array
    signal : array
        Signal to fit
    poly_degree : int
        Degree of polynomial to fit

    Returns:
    --------
    fitted_curve : array
        The fitted polynomial curve
    coeffs : array
        Polynomial coefficients
    """
    # Normalize time to improve numerical stability
    time_norm = (time - time[0]) / (time[-1] - time[0])

    # Fit polynomial to signal
    coeffs = poly.polyfit(time_norm, signal, poly_degree)

    # Generate fitted curve
    fitted_curve = poly.polyval(time_norm, coeffs)

    return fitted_curve, coeffs


def correct_drift(signal, fitted_curve):
    """
    Correct baseline drift by subtracting the fitted curve.

    Parameters:
    -----------
    signal : array
        Signal to correct
    fitted_curve : array
        Fitted drift curve

    Returns:
    --------
    corrected_signal : array
        Signal with drift removed
    """
    # Subtract fitted curve but preserve the mean to avoid shifting the signal
    signal_mean = np.mean(signal)
    corrected_signal = signal - fitted_curve + signal_mean

    return corrected_signal


def butter_filter(data, cutoff, fs, order=5):
    """Applies a Butterworth filter to the data with reduced edge effects."""
    if data is None or len(data) == 0 or cutoff <= 0:
        return data  # Return original data if cutoff is 0 or negative

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
                 low_cutoff=0.001,
                 high_cutoff=1.0,
                 downsample_factor=50,
                 artifact_threshold=3.0,
                 drift_correction=True,
                 drift_degree=2,
                 external_control=None,
                 edge_protection=True):  # Add edge protection parameter
    """Processes the data by applying filters and downsampling with edge protection."""
    # Compute sampling rate
    dt = np.diff(time).mean()
    fs = 1 / dt

    # Make a copy of the signals for processing
    processed_signal = analog_1.copy()
    processed_analog_2 = analog_2.copy() if analog_2 is not None else None

    # Use external control if provided and needs to be resized to match current signal
    if external_control is not None:
        # Check if external control has a different size than the current signal
        if len(external_control) != len(processed_signal):
            print(
                f"External control size ({len(external_control)}) doesn't match signal size ({len(processed_signal)}). Resizing...")

            # Resize external control to match current signal using interpolation
            orig_indices = np.linspace(0, 1, len(external_control))
            new_indices = np.linspace(0, 1, len(processed_signal))
            control_for_artifacts = np.interp(new_indices, orig_indices, external_control)

            print(f"Resized external control from {len(external_control)} to {len(control_for_artifacts)} points")
        else:
            control_for_artifacts = external_control
    else:
        # Use the file's own control channel
        control_for_artifacts = processed_analog_2

    # Also make downsampled copies of raw data
    downsampled_raw_analog_1 = analog_1.copy()
    downsampled_raw_analog_2 = analog_2.copy() if analog_2 is not None else None

    try:
        # Apply high-pass filter only if low_cutoff > 0
        if low_cutoff > 0 and low_cutoff < fs / 2:
            # Enhanced edge protection for high-pass filtering
            if edge_protection:
                # Determine edge size - larger for better protection
                edge_size = int(min(len(processed_signal) * 0.3, 30 * fs))  # 30 seconds or 30% of signal

                # Calculate tapering window for smooth transition
                taper = np.ones(len(processed_signal))
                taper_window = np.hanning(edge_size * 2)
                taper[:edge_size] = taper_window[:edge_size]

                # Create expanded signal with reflection
                reflected_start = processed_signal[edge_size:0:-1]
                expanded_signal = np.concatenate((reflected_start, processed_signal))

                # Apply filter to expanded signal
                b, a = butter(2, low_cutoff / (fs / 2), 'high')
                filtered_expanded = filtfilt(b, a, expanded_signal)

                # Extract the properly filtered part
                processed_signal = filtered_expanded[len(reflected_start):]

                # Apply same process to the control channel if available
                if processed_analog_2 is not None:
                    reflected_start_control = processed_analog_2[edge_size:0:-1]
                    expanded_control = np.concatenate((reflected_start_control, processed_analog_2))
                    filtered_expanded_control = filtfilt(b, a, expanded_control)
                    processed_analog_2 = filtered_expanded_control[len(reflected_start_control):]
            else:
                # Standard filtering without edge protection
                b, a = butter(2, low_cutoff / (fs / 2), 'high')
                processed_signal = filtfilt(b, a, processed_signal)
                if processed_analog_2 is not None:
                    processed_analog_2 = filtfilt(b, a, processed_analog_2)

        # Apply low-pass filter (high_cutoff)
        if high_cutoff > 0 and high_cutoff < fs / 2:
            b, a = butter(2, high_cutoff / (fs / 2), 'low')
            processed_signal = filtfilt(b, a, processed_signal)
            if processed_analog_2 is not None:
                processed_analog_2 = filtfilt(b, a, processed_analog_2)

        # Detect artifacts in the control channel (if available)
        artifact_mask = None
        if control_for_artifacts is not None:
            artifact_mask = detect_artifacts(control_for_artifacts, threshold=artifact_threshold)
            # Double-check mask size matches current signal
            if np.size(artifact_mask) != len(processed_signal):
                print(
                    f"Warning: artifact mask size mismatch! Expected {len(processed_signal)}, got {np.size(artifact_mask)}")
                # Create a safe mask with the correct size
                artifact_mask = np.zeros(len(processed_signal), dtype=bool)

        # Apply artifact correction to the main signal
        if artifact_mask is not None and np.any(artifact_mask):
            processed_signal = remove_artifacts_fast(processed_signal, artifact_mask)

        # Special handling for start of recording (first 10% or first 10 seconds)
        start_idx = min(int(len(processed_signal) * 0.1), int(10 * fs))
        if edge_protection and start_idx > 0:
            # Calculate median of the signal after initial noisy period
            stable_median = np.median(processed_signal[start_idx:start_idx * 2])

            # Calculate smooth transition weight from 0 to 1
            weight = np.linspace(0, 1, start_idx) ** 2  # Squared for more aggressive early correction

            # Create linear trend from stable_median to the actual value at start_idx
            target_val = processed_signal[start_idx]
            linear_trend = np.linspace(stable_median, target_val, start_idx)

            # Blend the linear trend with actual signal using increasing weight
            for i in range(start_idx):
                processed_signal[i] = (1 - weight[i]) * linear_trend[i] + weight[i] * processed_signal[i]

        # Fit drift curve and store it for visualization
        drift_curve = None
        if drift_correction:
            try:
                # Skip the first 10% for drift fitting if edge protection is enabled
                if edge_protection:
                    start_fit_idx = start_idx
                    fit_time = time[start_fit_idx:]
                    fit_signal = processed_signal[start_fit_idx:]

                    # Fit drift on stable portion
                    drift_curve_stable, coeffs = fit_drift_curve(fit_time, fit_signal, poly_degree=drift_degree)

                    # Generate full curve based on the coefficients
                    time_norm = (time - time[0]) / (time[-1] - time[0])
                    drift_curve = poly.polyval(time_norm, coeffs)
                else:
                    drift_curve, _ = fit_drift_curve(time, processed_signal, poly_degree=drift_degree)

                # Correct drift
                processed_signal = correct_drift(processed_signal, drift_curve)
            except Exception as e:
                print(f"Error in drift correction: {e}")
                drift_curve = None

        # Calculate dF/F as percentage for the processed signal
        # Use stable region for baseline (skip first 10% if noisy)
        if edge_protection:
            baseline_idx = start_idx
        else:
            baseline_idx = min(int(len(processed_signal) * 0.1), int(10 * fs))

        if baseline_idx < 10:  # Ensure at least 10 points for baseline
            baseline_idx = min(10, len(processed_signal))

        # Use median for more robust baseline estimation when edge effects are present
        if edge_protection:
            baseline = np.median(processed_signal[baseline_idx:baseline_idx * 2])
        else:
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

            # Downsample drift curve if it exists
            if drift_curve is not None:
                _, drift_curve = downsample_data(time, drift_curve, int(downsample_factor))
        else:
            processed_time = time
            downsampled_time = time

        # DO NOT downsample digital signals
        processed_digital_1 = digital_1
        processed_digital_2 = digital_2

    except Exception as e:
        print(f"Error in process_data: {e}")
        import traceback
        traceback.print_exc()
        # Return sensible defaults in case of error
        processed_time = time
        processed_digital_1 = digital_1
        processed_digital_2 = digital_2
        drift_curve = None
        # Make sure all arrays are initialized
        if 'processed_signal' not in locals():
            processed_signal = analog_1.copy()
        if 'processed_analog_2' not in locals():
            processed_analog_2 = analog_2.copy() if analog_2 is not None else None
        if 'downsampled_raw_analog_1' not in locals():
            downsampled_raw_analog_1 = analog_1.copy()
        if 'downsampled_raw_analog_2' not in locals():
            downsampled_raw_analog_2 = analog_2.copy() if analog_2 is not None else None
        if 'artifact_mask' not in locals():
            artifact_mask = None

    # Return the processed data, downsampled raw data, and artifact mask
    return processed_time, processed_signal, processed_analog_2, downsampled_raw_analog_1, downsampled_raw_analog_2, processed_digital_1, processed_digital_2, artifact_mask, drift_curve

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
        file_menu.add_command(label="Open Primary File...", command=self.open_file)
        file_menu.add_command(label="Open Secondary File...", command=self.open_secondary_file)
        file_menu.add_command(label="Clear Secondary File", command=self.clear_secondary_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

    def open_file(self):
        # Show file dialog
        file_path = filedialog.askopenfilename(
            title="Select Primary PPD File",
            filetypes=[("PPD files", "*.ppd"), ("All files", "*.*")]
        )

        if file_path:
            self.load_file(file_path)

    def open_secondary_file(self):
        """Open a secondary file to display alongside the primary file"""
        if not self.file_path:
            messagebox.showinfo("Error", "Please load a primary file first")
            return

        # Show file dialog
        file_path = filedialog.askopenfilename(
            title="Select Secondary PPD File",
            filetypes=[("PPD files", "*.ppd"), ("All files", "*.*")]
        )

        if file_path:
            self.load_secondary_file(file_path)

    def clear_secondary_file(self):
        """Clear the secondary file from the display"""
        if not hasattr(self, 'secondary_line') or self.secondary_line is None:
            return  # No secondary file to clear

        # Remove secondary lines from plots
        if self.secondary_line:
            self.secondary_line.remove()
            self.secondary_line = None

        if self.secondary_raw_line:
            self.secondary_raw_line.remove()
            self.secondary_raw_line = None

        if self.secondary_raw_line2:
            self.secondary_raw_line2.remove()
            self.secondary_raw_line2 = None

        if self.secondary_drift_line:
            self.secondary_drift_line.remove()
            self.secondary_drift_line = None

        if self.secondary_artifact_markers_processed:
            self.secondary_artifact_markers_processed.set_data([], [])

        if self.secondary_artifact_markers_raw:
            self.secondary_artifact_markers_raw.set_data([], [])

        # Clear secondary digital lines
        for line in self.secondary_digital_lines:
            line.remove()
        self.secondary_digital_lines = []

        # Clear secondary file variables
        self.secondary_file_path = None
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

        # Update legends
        self.ax1_legend = self.create_checkbox_legend(self.ax1)
        self.ax2_legend = self.create_checkbox_legend(self.ax2)
        if self.digital_lines:
            self.ax3_legend = self.create_checkbox_legend(self.ax3)

        # Redraw canvas
        self.canvas.draw_idle()

        # Update window title
        self.update_window_title()

    def load_file(self, file_path):
        """Load and process a new primary file"""
        self.file_path = file_path

        # Clean up any existing figure
        if self.fig:
            plt.close(self.fig)
            self.fig = None

        # Clear all existing UI elements
        self.clear_ui_elements()

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

        # Process data with initial parameters
        result = process_data(
            self.time, self.analog_1, self.analog_2, self.digital_1, self.digital_2,
            self.low_cutoff, self.high_cutoff, self.downsample_factor,
            self.artifact_threshold, self.drift_correction, self.drift_degree
        )

        # Unpack results
        self.processed_time, self.processed_signal, self.processed_analog_2, \
            self.downsampled_raw_analog_1, self.downsampled_raw_analog_2, \
            self.processed_digital_1, self.processed_digital_2, self.artifact_mask, \
            self.drift_curve = result

        # Create the GUI
        self.create_gui()

        # Update the window title to include the file name
        self.update_window_title()

    def load_secondary_file(self, file_path):
        """Load and process a secondary file to display alongside the primary file"""
        if not self.file_path or not self.fig:
            messagebox.showinfo("Error", "Please load a primary file first")
            return

        self.secondary_file_path = file_path

        # Read and parse secondary data
        self.secondary_header, self.secondary_data_bytes = read_ppd_file(file_path)
        self.secondary_time, self.secondary_analog_1, self.secondary_analog_2, \
            self.secondary_digital_1, self.secondary_digital_2 = parse_ppd_data(
            self.secondary_header, self.secondary_data_bytes)

        if self.secondary_time is None or self.secondary_analog_1 is None:
            print("Error: Failed to load or parse secondary data")
            messagebox.showerror("Error", "Failed to load or parse secondary data")
            return

        # Store original secondary data
        self.secondary_raw_analog_1 = self.secondary_analog_1.copy()
        self.secondary_raw_analog_2 = self.secondary_analog_2.copy() if self.secondary_analog_2 is not None else None

        # Process secondary data with the same parameters as primary
        result = process_data(
            self.secondary_time, self.secondary_analog_1, self.secondary_analog_2,
            self.secondary_digital_1, self.secondary_digital_2,
            self.low_cutoff, self.high_cutoff, self.downsample_factor,
            self.artifact_threshold, self.drift_correction, self.drift_degree
        )

        # Unpack secondary results
        self.secondary_processed_time, self.secondary_processed_signal, self.secondary_processed_analog_2, \
            self.secondary_downsampled_raw_analog_1, self.secondary_downsampled_raw_analog_2, \
            self.secondary_processed_digital_1, self.secondary_processed_digital_2, \
            self.secondary_artifact_mask, self.secondary_drift_curve = result

        # Add secondary data to existing plots
        self.add_secondary_data_to_plots()

        # Update window title
        self.update_window_title()

    def determine_better_control_channel(self):
        """
        Determines which file has the more realistic isosbestic control channel (analog_2).
        A realistic control channel should have natural fluctuations rather than just flat noise.

        Returns:
        --------
        tuple: (better_control, primary_has_better_control)
            better_control: The better control signal
            primary_has_better_control: Boolean indicating if primary file has better control
        """
        # Check if shared control is disabled
        if hasattr(self, 'shared_isosbestic_var') and not self.shared_isosbestic_var.get():
            # Clear indicator text
            if hasattr(self, 'control_indicator'):
                self.control_indicator.config(text="Each file using its own control")
            return None, True  # Return None to use each file's own control

        # Check if we have both primary and secondary data
        if not hasattr(self, 'secondary_downsampled_raw_analog_2') or self.secondary_downsampled_raw_analog_2 is None:
            if hasattr(self, 'control_indicator'):
                self.control_indicator.config(text="Using primary file control (only one available)")
            return self.downsampled_raw_analog_2, True

        if self.downsampled_raw_analog_2 is None:
            if hasattr(self, 'control_indicator'):
                self.control_indicator.config(text="Using secondary file control (only one available)")
            return self.secondary_downsampled_raw_analog_2, False

        # Calculate metrics to determine which control is better

        # 1. Variance - flat signals will have very low variance
        primary_variance = np.var(self.downsampled_raw_analog_2)
        secondary_variance = np.var(self.secondary_downsampled_raw_analog_2)

        # 2. Dynamic range - difference between min and max (after filtering outliers)
        def filtered_range(signal):
            # Use percentiles to avoid extreme outliers
            low = np.percentile(signal, 5)
            high = np.percentile(signal, 95)
            return high - low

        primary_range = filtered_range(self.downsampled_raw_analog_2)
        secondary_range = filtered_range(self.secondary_downsampled_raw_analog_2)

        # 3. Check for autocorrelation - natural signals have some autocorrelation
        def autocorrelation_strength(signal):
            # Calculate autocorrelation for lag-1
            if len(signal) < 3:
                return 0
            # Simplified autocorrelation calculation
            lag_1_corr = np.corrcoef(signal[:-1], signal[1:])[0, 1]
            return abs(lag_1_corr)  # Use absolute value as we care about strength not direction

        primary_autocorr = autocorrelation_strength(self.downsampled_raw_analog_2)
        secondary_autocorr = autocorrelation_strength(self.secondary_downsampled_raw_analog_2)

        # Combine metrics - weight each factor
        primary_score = (primary_variance * 0.4 +
                         primary_range * 0.3 +
                         primary_autocorr * 0.3)

        secondary_score = (secondary_variance * 0.4 +
                           secondary_range * 0.3 +
                           secondary_autocorr * 0.3)

        # Determine which is better
        primary_is_better = primary_score > secondary_score

        # Update the visual indicator with detailed info
        if hasattr(self, 'control_indicator'):
            primary_region = self.get_region_label_from_filename(self.file_path)
            secondary_region = self.get_region_label_from_filename(self.secondary_file_path)

            if primary_is_better:
                self.control_indicator.config(
                    text=f"Using {primary_region} control (score: {primary_score:.2f} vs {secondary_score:.2f})")
            else:
                self.control_indicator.config(
                    text=f"Using {secondary_region} control (score: {secondary_score:.2f} vs {primary_score:.2f})")

        # Log the decision
        print(f"Control channel assessment:")
        print(
            f"  Primary: var={primary_variance:.6f}, range={primary_range:.6f}, autocorr={primary_autocorr:.6f}, score={primary_score:.6f}")
        print(
            f"  Secondary: var={secondary_variance:.6f}, range={secondary_range:.6f}, autocorr={secondary_autocorr:.6f}, score={secondary_score:.6f}")
        print(f"  Selected: {'Primary' if primary_is_better else 'Secondary'} for isosbestic control")

        return (self.downsampled_raw_analog_2 if primary_is_better else self.secondary_downsampled_raw_analog_2,
                primary_is_better)

    def update_window_title(self):
        """Update the window title based on loaded files"""
        if not self.file_path:
            self.root.title("Photometry Signal Viewer")
            return

        primary_filename = os.path.basename(self.file_path)

        if self.secondary_file_path:
            secondary_filename = os.path.basename(self.secondary_file_path)
            self.root.title(f"Photometry Signal Viewer - Primary: {primary_filename}, Secondary: {secondary_filename}")
        else:
            self.root.title(f"Photometry Signal Viewer - {primary_filename}")

    def add_secondary_data_to_plots(self):
        """Add the secondary file data to the existing plots"""
        if not hasattr(self, 'ax1') or not self.secondary_processed_signal is not None:
            return

        # Extract region identifier from filename for legend labels
        region_label = self.get_region_label_from_filename(self.secondary_file_path)

        # Add to processed signal plot (Panel 1)
        self.secondary_line, = self.ax1.plot(
            self.secondary_processed_time / 60,
            self.secondary_processed_signal,
            'r-', lw=1,
            label=f'{region_label} Signal'
        )

        # Add secondary drift curve if available
        if self.secondary_drift_curve is not None and self.drift_correction:
            # Convert to dF/F scale for visualization
            baseline_idx = min(int(len(self.secondary_drift_curve) * 0.1), 10)
            baseline = np.mean(self.secondary_drift_curve[:baseline_idx])
            if baseline != 0:
                df_drift_curve = 100 * (self.secondary_drift_curve - baseline) / baseline
            else:
                df_drift_curve = self.secondary_drift_curve

            self.secondary_drift_line, = self.ax1.plot(
                self.secondary_processed_time / 60,
                df_drift_curve,
                'r--', lw=1, alpha=0.5,
                label=f'{region_label} Drift'
            )

        # Add artifact markers for secondary signal
        self.secondary_artifact_markers_processed, = self.ax1.plot(
            [], [], 'mo', ms=4, alpha=0.7,
            label=f'{region_label} Artifacts'
        )

        # Add to raw signals plot (Panel 2)
        self.secondary_raw_line, = self.ax2.plot(
            self.secondary_processed_time / 60,
            self.secondary_downsampled_raw_analog_1,
            'r-', lw=1,
            label=f'{region_label} Analog 1'
        )

        if self.secondary_downsampled_raw_analog_2 is not None:
            self.secondary_raw_line2, = self.ax2.plot(
                self.secondary_processed_time / 60,
                self.secondary_downsampled_raw_analog_2,
                'm-', lw=1,
                label=f'{region_label} Analog 2'
            )

            # Add artifact markers on the secondary 405nm channel
            self.secondary_artifact_markers_raw, = self.ax2.plot(
                [], [], 'mo', ms=4, alpha=0.7,
                label=f'{region_label} Artifacts'
            )

        # Add to digital signals plot (Panel 3)
        if self.secondary_processed_digital_1 is not None:
            line, = self.ax3.plot(
                self.secondary_time / 60,
                self.secondary_processed_digital_1,
                'c-', lw=1,
                label=f'{region_label} Digital 1'
            )
            self.secondary_digital_lines.append(line)

        if self.secondary_processed_digital_2 is not None:
            line, = self.ax3.plot(
                self.secondary_time / 60,
                self.secondary_processed_digital_2,
                'm-', lw=1,
                label=f'{region_label} Digital 2'
            )
            self.secondary_digital_lines.append(line)

        # Update artifact markers
        self.update_secondary_artifact_markers()

        # Update legends with new items
        self.ax1_legend = self.create_checkbox_legend(self.ax1)
        self.ax2_legend = self.create_checkbox_legend(self.ax2)
        if self.digital_lines or self.secondary_digital_lines:
            self.ax3_legend = self.create_checkbox_legend(self.ax3)

        # Redraw canvas
        self.canvas.draw_idle()

    def get_region_label_from_filename(self, filepath):
        """Extract region label (PVN or SON) from filename or return a default"""
        if filepath is None:
            return "Unknown"

        filename = os.path.basename(filepath).upper()

        if "PVN" in filename:
            return "PVN"
        elif "SON" in filename:
            return "SON"
        else:
            # Extract just the filename without extension for a more generic label
            return os.path.splitext(os.path.basename(filepath))[0]

    def update_secondary_artifact_markers(self):
        """Update the artifact markers for the secondary signal"""
        if (self.secondary_artifact_mask is None or
                not np.any(self.secondary_artifact_mask) or
                len(self.secondary_artifact_mask) != len(self.secondary_processed_time)):
            # No artifacts detected or size mismatch, clear the markers
            if hasattr(self, 'secondary_artifact_markers_processed'):
                self.secondary_artifact_markers_processed.set_data([], [])
            if hasattr(self, 'secondary_artifact_markers_raw'):
                self.secondary_artifact_markers_raw.set_data([], [])
            return

        # Get the time points where artifacts occur (in minutes)
        artifact_times = self.secondary_processed_time[self.secondary_artifact_mask] / 60

        # Update markers on the processed signal
        if hasattr(self, 'secondary_artifact_markers_processed'):
            artifact_values_processed = self.secondary_processed_signal[self.secondary_artifact_mask]
            self.secondary_artifact_markers_processed.set_data(artifact_times, artifact_values_processed)

        # Update markers on the raw signal (405nm channel)
        if hasattr(self, 'secondary_artifact_markers_raw') and self.secondary_downsampled_raw_analog_2 is not None:
            artifact_values_raw = self.secondary_downsampled_raw_analog_2[self.secondary_artifact_mask]
            self.secondary_artifact_markers_raw.set_data(artifact_times, artifact_values_raw)
    def clear_ui_elements(self):
        """Clear all UI elements to avoid duplication"""
        # Destroy all frames if they exist
        if self.main_frame:
            self.main_frame.destroy()
            self.main_frame = None

        self.nav_frame = None
        self.frame = None
        self.slider_frame = None
        self.ax1_legend = None
        self.ax2_legend = None
        self.ax3_legend = None

    def create_checkbox_legend(self, ax):
        """Create a custom legend with checkboxes for the given axis"""
        # Get all lines and their labels from the axis
        lines = [line for line in ax.get_lines() if line.get_label()[0] != '_']
        labels = [line.get_label() for line in lines]

        # Create legend with custom handler
        leg = ax.legend(loc='upper right', fontsize=self.font_sizes['base'])

        # Make legend interactive
        leg.set_draggable(True)

        # Store line visibility state
        for line in lines:
            if line not in self.line_visibility:
                self.line_visibility[line] = True

        # Add checkbox functionality
        for i, legline in enumerate(leg.get_lines()):
            legline.set_picker(5)  # Enable picking on the legend line
            legline.line = lines[i]  # Store reference to the actual line

        # Connect the pick event
        self.fig.canvas.mpl_connect('pick_event', self.on_legend_pick)

        return leg

    def on_legend_pick(self, event):
        """Handle legend picking to toggle line visibility"""
        # Get all legend lines
        all_legends = []
        if hasattr(self, 'ax1_legend') and self.ax1_legend:
            all_legends.extend(self.ax1_legend.get_lines())
        if hasattr(self, 'ax2_legend') and self.ax2_legend:
            all_legends.extend(self.ax2_legend.get_lines())
        if hasattr(self, 'ax3_legend') and self.ax3_legend:
            all_legends.extend(self.ax3_legend.get_lines())

        # Check if the picked object is a legend line
        if event.artist in all_legends:
            # Get the original line this legend item represents
            line = event.artist.line

            # Toggle visibility
            visible = not line.get_visible()
            line.set_visible(visible)

            # Change the alpha of the legend markers
            event.artist.set_alpha(1.0 if visible else 0.2)

            # Store the new state
            self.line_visibility[line] = visible

            # Redraw the canvas
            self.canvas.draw_idle()

    def create_gui(self):
        # Create a main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Add navigation buttons at the top
        self.nav_frame = tk.Frame(self.main_frame)
        self.nav_frame.pack(fill=tk.X, padx=20, pady=10)
        self.add_navigation_buttons()

        # Create a frame for the matplotlib figure
        self.frame = tk.Frame(self.main_frame)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1])

        # Panel 1: Processed signal plot with artifact markers and drift curve
        self.ax1 = self.fig.add_subplot(gs[0])

        # Get region label for primary file
        primary_region = self.get_region_label_from_filename(self.file_path)

        self.line, = self.ax1.plot(
            self.processed_time / 60,
            self.processed_signal,
            'g-', lw=1,
            label=f'{primary_region} Signal'
        )

        # Add drift curve if available
        if self.drift_curve is not None and self.drift_correction:
            # Convert drift curve to dF/F scale for visualization
            baseline_idx = min(int(len(self.drift_curve) * 0.1), 10)
            baseline = np.mean(self.drift_curve[:baseline_idx])
            df_drift_curve = 100 * (self.drift_curve - baseline) / baseline if baseline != 0 else self.drift_curve
            self.drift_line, = self.ax1.plot(
                self.processed_time / 60,
                df_drift_curve,
                'r--', lw=1, alpha=0.5,
                label=f'{primary_region} Drift'
            )

        # Initialize artifact markers (will be updated later)
        self.artifact_markers_processed, = self.ax1.plot(
            [], [], 'ro', ms=4, alpha=0.7,
            label=f'{primary_region} Artifacts'
        )

        self.ax1.set_ylabel('% ΔF/F', fontsize=self.font_sizes['base'])
        self.ax1.set_title('Processed Photometry Signal', fontsize=self.font_sizes['title'])
        self.ax1.grid(True, alpha=0.3)
        self.ax1.tick_params(labelsize=self.font_sizes['base'])

        # Create custom legend with checkboxes
        self.ax1_legend = self.create_checkbox_legend(self.ax1)

        # Auto-scale y-axis with asymmetric scaling (5*STD positive, 2*STD negative)
        self.update_y_limits()

        # Panel 2: Raw signals plot (both analog channels) with artifact markers
        self.ax2 = self.fig.add_subplot(gs[1], sharex=self.ax1)
        self.raw_line, = self.ax2.plot(
            self.processed_time / 60,
            self.downsampled_raw_analog_1,
            'b-', lw=1,
            label=f'{primary_region} Analog 1'
        )

        if self.downsampled_raw_analog_2 is not None:
            self.raw_line2, = self.ax2.plot(
                self.processed_time / 60,
                self.downsampled_raw_analog_2,
                'g-', lw=1,
                label=f'{primary_region} Analog 2'
            )

            # Add artifact markers on the 405nm channel
            self.artifact_markers_raw, = self.ax2.plot(
                [], [], 'ro', ms=4, alpha=0.7,
                label=f'{primary_region} Artifacts'
            )

        # Create custom legend with checkboxes for raw signals
        self.ax2_legend = self.create_checkbox_legend(self.ax2)

        self.ax2.set_ylabel('mV', fontsize=self.font_sizes['base'])
        self.ax2.set_title('Raw Analog Signals', fontsize=self.font_sizes['title'])
        self.ax2.grid(True, alpha=0.3)
        self.ax2.tick_params(labelsize=self.font_sizes['base'])

        # Panel 3: Combined Digital signals
        self.ax3 = self.fig.add_subplot(gs[2], sharex=self.ax1)
        self.digital_lines = []

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

        # Create custom legend with checkboxes for digital signals
        if self.digital_lines:
            self.ax3_legend = self.create_checkbox_legend(self.ax3)

        self.ax3.set_ylabel('TTL', fontsize=self.font_sizes['base'])
        self.ax3.set_ylim(-0.1, 1.1)
        self.ax3.set_xlabel('Time (minutes)', fontsize=self.font_sizes['base'])
        self.ax3.tick_params(labelsize=self.font_sizes['base'])

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
        """Create sliders with organized grouping"""
        # Use a notebook with tabs for better organization
        self.control_notebook = ttk.Notebook(self.slider_frame)
        self.control_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create tabs for different groups of controls
        self.signal_processing_tab = ttk.Frame(self.control_notebook)
        self.artifact_tab = ttk.Frame(self.control_notebook)
        self.advanced_denoising_tab = ttk.Frame(self.control_notebook)

        # Add tabs to notebook
        self.control_notebook.add(self.signal_processing_tab, text="Signal Processing")
        self.control_notebook.add(self.artifact_tab, text="Artifact Detection")
        self.control_notebook.add(self.advanced_denoising_tab, text="Advanced Denoising")

        # Configure grid for each tab
        for tab in [self.signal_processing_tab, self.artifact_tab, self.advanced_denoising_tab]:
            tab.columnconfigure(1, weight=1)  # Make slider column expandable

        font_size = self.font_sizes['base']
        slider_font_size = self.font_sizes['slider']

        # ------------ TAB 1: Signal Processing ------------ #
        # Low cutoff slider
        tk.Label(self.signal_processing_tab, text="Low cutoff (Hz):", font=('Arial', font_size)).grid(
            row=0, column=0, sticky="w", padx=10, pady=10)
        self.low_slider = tk.Scale(self.signal_processing_tab, from_=0.0, to=0.01,
                                   resolution=0.0001, orient=tk.HORIZONTAL,
                                   length=800, width=25, font=('Arial', slider_font_size),
                                   command=self.update_filter)
        self.low_slider.set(self.low_cutoff)
        self.low_slider.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # High cutoff slider
        tk.Label(self.signal_processing_tab, text="High cutoff (Hz):", font=('Arial', font_size)).grid(
            row=1, column=0, sticky="w", padx=10, pady=10)
        self.high_slider = tk.Scale(self.signal_processing_tab, from_=0.1, to=5.0,
                                    resolution=0.1, orient=tk.HORIZONTAL,
                                    length=800, width=25, font=('Arial', slider_font_size),
                                    command=self.update_filter)
        self.high_slider.set(self.high_cutoff)
        self.high_slider.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Downsample slider
        tk.Label(self.signal_processing_tab, text="Downsample factor:", font=('Arial', font_size)).grid(
            row=2, column=0, sticky="w", padx=10, pady=10)
        self.downsample_slider = tk.Scale(self.signal_processing_tab, from_=1, to=100,
                                          resolution=1, orient=tk.HORIZONTAL,
                                          length=800, width=25, font=('Arial', slider_font_size),
                                          command=self.update_filter)
        self.downsample_slider.set(self.downsample_factor)
        self.downsample_slider.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        # Drift correction settings
        drift_frame = tk.LabelFrame(self.signal_processing_tab, text="Drift Correction",
                                    font=('Arial', font_size, 'bold'))
        drift_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        self.drift_var = tk.BooleanVar(value=self.drift_correction)
        self.drift_check = tk.Checkbutton(
            drift_frame, text="Enable Drift Correction",
            variable=self.drift_var, font=('Arial', font_size),
            command=self.update_filter)
        self.drift_check.pack(side=tk.LEFT, padx=10, pady=5)

        # Drift polynomial degree
        tk.Label(drift_frame, text="Polynomial Degree:", font=('Arial', font_size)).pack(side=tk.LEFT, padx=10)
        self.poly_degree_var = tk.IntVar(value=self.drift_degree)
        for i in range(1, 5):  # Degrees 1-4
            tk.Radiobutton(drift_frame, text=str(i), variable=self.poly_degree_var,
                           value=i, font=('Arial', font_size), command=self.update_filter).pack(side=tk.LEFT, padx=5)

        # ------------ TAB 2: Artifact Detection ------------ #
        # Artifact threshold slider with more room
        tk.Label(self.artifact_tab, text="Artifact threshold:", font=('Arial', font_size)).grid(
            row=0, column=0, sticky="w", padx=10, pady=10)
        self.artifact_slider = tk.Scale(self.artifact_tab, from_=1.0, to=10.0,
                                        resolution=0.1, orient=tk.HORIZONTAL,
                                        length=800, width=25, font=('Arial', slider_font_size),
                                        command=self.update_filter)
        self.artifact_slider.set(self.artifact_threshold)
        self.artifact_slider.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # Add description for artifact threshold
        artifact_desc = tk.Label(self.artifact_tab,
                                 text="Lower values detect more artifacts. Higher values are less sensitive.",
                                 font=('Arial', font_size - 1), fg='gray')
        artifact_desc.grid(row=1, column=0, columnspan=2, sticky="w", padx=10)

        # Button to show artifacts
        highlight_btn = tk.Button(self.artifact_tab, text="Highlight Artifacts",
                                  font=('Arial', font_size),
                                  command=self.highlight_artifacts)
        highlight_btn.grid(row=2, column=0, columnspan=2, pady=10)

        # Shared isosbestic control option
        self.shared_control_frame = tk.Frame(self.artifact_tab)
        self.shared_control_frame.grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=10)

        self.shared_isosbestic_var = tk.BooleanVar(value=True)
        self.shared_isosbestic_check = tk.Checkbutton(
            self.shared_control_frame,
            text="Use best isosbestic channel for both files",
            variable=self.shared_isosbestic_var,
            font=('Arial', font_size),
            command=self.update_filter)
        self.shared_isosbestic_check.pack(side=tk.LEFT, padx=5)

        # Add indicator for which control is being used
        self.control_indicator = tk.Label(
            self.shared_control_frame,
            text="",
            font=('Arial', font_size - 1),
            fg="blue")
        self.control_indicator.pack(side=tk.LEFT, padx=10)

        # Status indicator for processing
        self.status_label = tk.Label(self.nav_frame, text="Ready", fg="green",
                                     font=('Arial', self.font_sizes['base']))
        self.status_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=10)

        # ------------ TAB 3: Advanced Denoising ------------ #
        # Aggressive mode option
        self.aggressive_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.advanced_denoising_tab,
                       text="Aggressive Mode (stronger correction for large artifacts)",
                       variable=self.aggressive_var,
                       font=('Arial', font_size)).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        # Complete removal option
        self.remove_var = tk.BooleanVar(value=True)
        remove_frame = tk.Frame(self.advanced_denoising_tab)
        remove_frame.grid(row=1, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        tk.Checkbutton(remove_frame,
                       text="Remove large gaps completely: ",
                       variable=self.remove_var,
                       font=('Arial', font_size)).pack(side='left')

        self.max_gap_var = tk.DoubleVar(value=1.0)
        tk.Spinbox(remove_frame,
                   from_=0.1, to=10.0, increment=0.1,
                   textvariable=self.max_gap_var,
                   width=5).pack(side='left', padx=5)

        tk.Label(remove_frame,
                 text="seconds",
                 font=('Arial', font_size)).pack(side='left')

        # Use control signal option
        self.control_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.advanced_denoising_tab,
                       text="Use isosbestic (405nm) channel for correction",
                       variable=self.control_var,
                       font=('Arial', font_size)).grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        # Explanation text
        explanation = tk.Label(self.advanced_denoising_tab,
                               text="Advanced denoising applies stronger noise removal algorithms\n"
                                    "to continuous artifact segments. It can completely replace large\n"
                                    "noise segments with interpolated data based on clean signal regions.",
                               font=('Arial', font_size - 1), fg='gray', justify='left')
        explanation.grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=10)

        # Add run denoising button
        self.denoise_button = tk.Button(
            self.advanced_denoising_tab,
            text="Run Advanced Denoising",
            font=('Arial', self.font_sizes['button'], 'bold'),
            bg='lightgreen',
            height=2,
            command=self.run_advanced_denoising
        )
        self.denoise_button.grid(row=4, column=0, columnspan=2, pady=10)

        # Add progress bar (hidden initially)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.advanced_denoising_tab,
            variable=self.progress_var,
            maximum=100,
            length=400)
        self.progress_bar.grid(row=5, column=0, columnspan=2, pady=10)
        self.progress_bar.grid_remove()  # Hide initially

        # Edge protection for noisy beginnings
        self.edge_protection_var = tk.BooleanVar(value=True)
        edge_frame = tk.Frame(self.signal_processing_tab)
        edge_frame.grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=10)

        tk.Checkbutton(edge_frame,
                       text="Enable edge protection (reduces distortion at beginning of recording)",
                       variable=self.edge_protection_var,
                       font=('Arial', font_size),
                       command=self.update_filter).pack(side=tk.LEFT, padx=5)

        # Cancel button (hidden initially)
        self.cancel_denoise_button = tk.Button(
            self.advanced_denoising_tab,
            text="Cancel Denoising",
            font=('Arial', self.font_sizes['base']),
            command=self.cancel_denoising
        )
        self.cancel_denoise_button.grid(row=6, column=0, columnspan=2, pady=5)
        self.cancel_denoise_button.grid_remove()  # Hide initially

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

    def highlight_artifacts(self):
        """Temporarily highlight all detected artifacts"""
        primary_has_artifacts = self.artifact_mask is not None and np.any(self.artifact_mask)
        secondary_has_artifacts = hasattr(self,
                                          'secondary_artifact_mask') and self.secondary_artifact_mask is not None and np.any(
            self.secondary_artifact_mask)

        if not primary_has_artifacts and not secondary_has_artifacts:
            messagebox.showinfo("Artifacts", "No artifacts detected!")
            return

        # Count artifacts
        primary_count = np.sum(self.artifact_mask) if primary_has_artifacts else 0
        secondary_count = np.sum(self.secondary_artifact_mask) if secondary_has_artifacts else 0

        # Temporarily increase the size and opacity of artifact markers for primary signal
        if primary_has_artifacts:
            orig_size = self.artifact_markers_processed.get_markersize()
            orig_alpha = self.artifact_markers_processed.get_alpha()

            self.artifact_markers_processed.set_markersize(8)
            self.artifact_markers_processed.set_alpha(1.0)

            if hasattr(self, 'artifact_markers_raw'):
                self.artifact_markers_raw.set_markersize(8)
                self.artifact_markers_raw.set_alpha(1.0)

        # Temporarily increase the size and opacity of artifact markers for secondary signal
        if secondary_has_artifacts and hasattr(self, 'secondary_artifact_markers_processed'):
            secondary_orig_size = self.secondary_artifact_markers_processed.get_markersize()
            secondary_orig_alpha = self.secondary_artifact_markers_processed.get_alpha()

            self.secondary_artifact_markers_processed.set_markersize(8)
            self.secondary_artifact_markers_processed.set_alpha(1.0)

            if hasattr(self, 'secondary_artifact_markers_raw'):
                self.secondary_artifact_markers_raw.set_markersize(8)
                self.secondary_artifact_markers_raw.set_alpha(1.0)

        self.canvas.draw_idle()

        # Show info about artifacts
        primary_label = self.get_region_label_from_filename(self.file_path)
        secondary_label = self.get_region_label_from_filename(
            self.secondary_file_path) if self.secondary_file_path else ""

        message = f"{primary_count} artifacts detected in {primary_label}.\n"
        if secondary_has_artifacts:
            message += f"{secondary_count} artifacts detected in {secondary_label}.\n"
        message += f"\nCurrent threshold: {self.artifact_threshold}\n\n"
        message += "Artifacts are shown as red dots on both plots."

        messagebox.showinfo("Artifacts", message)

        # Restore original appearance for primary signal
        if primary_has_artifacts:
            self.artifact_markers_processed.set_markersize(orig_size)
            self.artifact_markers_processed.set_alpha(orig_alpha)

            if hasattr(self, 'artifact_markers_raw'):
                self.artifact_markers_raw.set_markersize(orig_size)
                self.artifact_markers_raw.set_alpha(orig_alpha)

        # Restore original appearance for secondary signal
        if secondary_has_artifacts and hasattr(self, 'secondary_artifact_markers_processed'):
            self.secondary_artifact_markers_processed.set_markersize(secondary_orig_size)
            self.secondary_artifact_markers_processed.set_alpha(secondary_orig_alpha)

            if hasattr(self, 'secondary_artifact_markers_raw'):
                self.secondary_artifact_markers_raw.set_markersize(secondary_orig_size)
                self.secondary_artifact_markers_raw.set_alpha(secondary_orig_alpha)

        self.canvas.draw_idle()

    def add_navigation_buttons(self):
        button_font = ('Arial', self.font_sizes['button'])
        title_font = ('Arial', self.font_sizes['title'], 'bold')

        # Use a more compact layout for the navigation controls
        # Row 1: Zoom and Pan controls
        tk.Label(self.nav_frame, text="Navigation:", font=title_font).grid(
            row=0, column=0, padx=10, pady=5, sticky="w")

        # X-axis zoom buttons
        tk.Button(self.nav_frame, text="Zoom In (X)", font=button_font, width=10,
                  command=lambda: self.zoom_x(in_out="in")).grid(
            row=0, column=1, padx=5, pady=5)

        tk.Button(self.nav_frame, text="Zoom Out (X)", font=button_font, width=10,
                  command=lambda: self.zoom_x(in_out="out")).grid(
            row=0, column=2, padx=5, pady=5)

        # Y-axis zoom buttons (for all panels)
        tk.Button(self.nav_frame, text="Zoom In (Y)", font=button_font, width=10,
                  command=lambda: self.zoom_y_all("in")).grid(
            row=0, column=3, padx=5, pady=5)

        tk.Button(self.nav_frame, text="Zoom Out (Y)", font=button_font, width=10,
                  command=lambda: self.zoom_y_all("out")).grid(
            row=0, column=4, padx=5, pady=5)

        # Pan buttons
        tk.Button(self.nav_frame, text="← Pan Left", font=button_font, width=10,
                  command=lambda: self.pan_x(direction="left")).grid(
            row=0, column=5, padx=5, pady=5)

        tk.Button(self.nav_frame, text="Pan Right →", font=button_font, width=10,
                  command=lambda: self.pan_x(direction="right")).grid(
            row=0, column=6, padx=5, pady=5)

        # Reset view button
        tk.Button(self.nav_frame, text="Reset View", font=title_font, width=15,
                  command=self.reset_view, bg='lightblue').grid(
            row=0, column=7, padx=10, pady=5)

        # Add file buttons
        file_frame = tk.Frame(self.nav_frame)
        file_frame.grid(row=0, column=8, padx=10, pady=5)

        tk.Button(file_frame, text="Primary File", font=title_font, width=15,
                  command=self.open_file, bg='lightyellow').pack(side=tk.LEFT, padx=5)

        tk.Button(file_frame, text="Secondary File", font=title_font, width=15,
                  command=self.open_secondary_file, bg='lightgreen').pack(side=tk.LEFT, padx=5)

        tk.Button(file_frame, text="Clear Secondary", font=button_font, width=15,
                  command=self.clear_secondary_file).pack(side=tk.LEFT, padx=5)

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
        # Determine overall time range for both primary and secondary
        min_time = self.time[0] / 60
        max_time = self.time[-1] / 60

        if hasattr(self, 'secondary_time') and self.secondary_time is not None:
            min_time = min(min_time, self.secondary_time[0] / 60)
            max_time = max(max_time, self.secondary_time[-1] / 60)

        # Reset x-axis to show all data
        self.ax1.set_xlim(min_time, max_time)

        # Reset y-axis for processed signal
        # Include both primary and secondary signals for auto-scaling
        signals = [self.processed_signal]
        if hasattr(self, 'secondary_processed_signal') and self.secondary_processed_signal is not None:
            signals.append(self.secondary_processed_signal)

        # Calculate overall statistics for y-axis scaling
        combined_signal = np.concatenate(signals)
        mean_val = np.mean(combined_signal)
        std_val = np.std(combined_signal)
        self.ax1.set_ylim(mean_val - 2 * std_val, mean_val + 5 * std_val)

        # Reset y-axis for raw signal
        raw_data = [self.downsampled_raw_analog_1]
        if self.downsampled_raw_analog_2 is not None:
            raw_data.append(self.downsampled_raw_analog_2)

        # Include secondary raw data if available
        if hasattr(self, 'secondary_downsampled_raw_analog_1') and self.secondary_downsampled_raw_analog_1 is not None:
            raw_data.append(self.secondary_downsampled_raw_analog_1)
            if hasattr(self,
                       'secondary_downsampled_raw_analog_2') and self.secondary_downsampled_raw_analog_2 is not None:
                raw_data.append(self.secondary_downsampled_raw_analog_2)

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
        try:
            # Show processing indicator
            self.root.config(cursor="watch")  # Change cursor to hourglass
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Processing...", fg="blue")
            self.root.update_idletasks()  # Force UI update

            # Save current axis limits before updating
            x_lim = self.ax1.get_xlim()
            y1_lim = self.ax1.get_ylim()
            y2_lim = self.ax2.get_ylim()

            # Get updated parameter values
            self.low_cutoff = self.low_slider.get()
            self.high_cutoff = self.high_slider.get()
            self.downsample_factor = int(self.downsample_slider.get())
            self.artifact_threshold = self.artifact_slider.get()
            self.drift_correction = self.drift_var.get()
            self.drift_degree = self.poly_degree_var.get()

            print(f"Processing with: low={self.low_cutoff}, high={self.high_cutoff}, "
                  f"downsample={self.downsample_factor}, artifact_threshold={self.artifact_threshold}, "
                  f"drift_correction={self.drift_correction}, drift_degree={self.drift_degree}")

            # If both files are loaded, determine which has the better control channel
            external_control = None
            secondary_should_use_primary_control = False

            if hasattr(self, 'secondary_raw_analog_2') and self.secondary_raw_analog_2 is not None:
                # Get the better control channel and whether it's from primary file
                external_control, primary_has_better_control = self.determine_better_control_channel()

                # If secondary file should use primary's control, note this for later
                secondary_should_use_primary_control = primary_has_better_control

                # If primary should use secondary's control, set it now
                if not primary_has_better_control:
                    external_control_for_primary = self.secondary_downsampled_raw_analog_2
                else:
                    external_control_for_primary = None
            else:
                external_control_for_primary = None
                primary_has_better_control = True

            # Reprocess primary file with new parameters
            result = process_data(
                self.time, self.raw_analog_1, self.raw_analog_2, self.digital_1, self.digital_2,
                low_cutoff=self.low_cutoff, high_cutoff=self.high_cutoff,
                downsample_factor=self.downsample_factor, artifact_threshold=self.artifact_threshold,
                drift_correction=self.drift_correction, drift_degree=self.drift_degree,
                external_control=external_control_for_primary,
                edge_protection=self.edge_protection_var.get()  # Add this line
            )

            # Unpack results
            self.processed_time, self.processed_signal, self.processed_analog_2, \
                self.downsampled_raw_analog_1, self.downsampled_raw_analog_2, \
                self.processed_digital_1, self.processed_digital_2, self.artifact_mask, \
                self.drift_curve = result

            # Update artifact markers
            self.update_artifact_markers()

            # Update processed signal plot
            self.line.set_xdata(self.processed_time / 60)
            self.line.set_ydata(self.processed_signal)

            # Update drift curve if available
            # First remove existing drift line if it exists
            if hasattr(self, 'drift_line'):
                try:
                    self.drift_line.remove()  # Correct way to remove a line
                except Exception as e:
                    print(f"Error removing drift line: {e}")
                finally:
                    # Ensure the attribute is deleted even if removal fails
                    if hasattr(self, 'drift_line'):
                        delattr(self, 'drift_line')

            # Add new drift line if drift correction is enabled
            if self.drift_curve is not None and self.drift_correction:
                try:
                    # Convert drift curve to dF/F scale for visualization
                    baseline_idx = min(int(len(self.drift_curve) * 0.1), 10)
                    baseline = np.mean(self.drift_curve[:baseline_idx])
                    if baseline != 0:
                        df_drift_curve = 100 * (self.drift_curve - baseline) / baseline
                    else:
                        df_drift_curve = self.drift_curve

                    # Add the drift line
                    primary_region = self.get_region_label_from_filename(self.file_path)
                    self.drift_line, = self.ax1.plot(
                        self.processed_time / 60,
                        df_drift_curve,
                        'r--', lw=1, alpha=0.5,
                        label=f'{primary_region} Drift'
                    )
                except Exception as e:
                    print(f"Error adding primary drift line: {e}")

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

            # Process secondary data with the same parameters if available
            if self.secondary_file_path and self.secondary_raw_analog_1 is not None:
                # Determine which control to use for secondary file
                secondary_external_control = self.downsampled_raw_analog_2 if secondary_should_use_primary_control else None

                # Reprocess secondary file
                secondary_result = process_data(
                    self.secondary_time, self.secondary_raw_analog_1, self.secondary_raw_analog_2,
                    self.secondary_digital_1, self.secondary_digital_2,
                    low_cutoff=self.low_cutoff, high_cutoff=self.high_cutoff,
                    downsample_factor=self.downsample_factor, artifact_threshold=self.artifact_threshold,
                    drift_correction=self.drift_correction, drift_degree=self.drift_degree,
                    external_control=secondary_external_control,
                    edge_protection=self.edge_protection_var.get()  # Add this line
                )

                # Unpack secondary results
                self.secondary_processed_time, self.secondary_processed_signal, self.secondary_processed_analog_2, \
                    self.secondary_downsampled_raw_analog_1, self.secondary_downsampled_raw_analog_2, \
                    self.secondary_processed_digital_1, self.secondary_processed_digital_2, \
                    self.secondary_artifact_mask, self.secondary_drift_curve = secondary_result

                # Update secondary artifact markers
                self.update_secondary_artifact_markers()

                # Update secondary processed signal
                if hasattr(self, 'secondary_line') and self.secondary_line is not None:
                    self.secondary_line.set_xdata(self.secondary_processed_time / 60)
                    self.secondary_line.set_ydata(self.secondary_processed_signal)

                # Update secondary drift curve
                if hasattr(self, 'secondary_drift_line'):
                    try:
                        self.secondary_drift_line.remove()
                    except Exception as e:
                        print(f"Error removing secondary drift line: {e}")
                    finally:
                        if hasattr(self, 'secondary_drift_line'):
                            delattr(self, 'secondary_drift_line')

                # Add new secondary drift line if drift correction is enabled
                if self.secondary_drift_curve is not None and self.drift_correction:
                    try:
                        # Convert drift curve to dF/F scale for visualization
                        baseline_idx = min(int(len(self.secondary_drift_curve) * 0.1), 10)
                        baseline = np.mean(self.secondary_drift_curve[:baseline_idx])
                        if baseline != 0:
                            df_drift_curve = 100 * (self.secondary_drift_curve - baseline) / baseline
                        else:
                            df_drift_curve = self.secondary_drift_curve

                        # Add the drift line
                        secondary_region = self.get_region_label_from_filename(self.secondary_file_path)
                        self.secondary_drift_line, = self.ax1.plot(
                            self.secondary_processed_time / 60,
                            df_drift_curve,
                            'm--', lw=1, alpha=0.5,
                            label=f'{secondary_region} Drift'
                        )
                    except Exception as e:
                        print(f"Error adding secondary drift line: {e}")

                # Update secondary raw signals
                if hasattr(self, 'secondary_raw_line') and self.secondary_raw_line is not None:
                    self.secondary_raw_line.set_xdata(self.secondary_processed_time / 60)
                    self.secondary_raw_line.set_ydata(self.secondary_downsampled_raw_analog_1)

                if hasattr(self, 'secondary_raw_line2') and self.secondary_raw_line2 is not None:
                    self.secondary_raw_line2.set_xdata(self.secondary_processed_time / 60)
                    self.secondary_raw_line2.set_ydata(self.secondary_downsampled_raw_analog_2)

                # Update secondary digital signals
                if len(self.secondary_digital_lines) > 0 and self.secondary_processed_digital_1 is not None:
                    self.secondary_digital_lines[0].set_xdata(self.secondary_time / 60)
                    self.secondary_digital_lines[0].set_ydata(self.secondary_processed_digital_1)
                if len(self.secondary_digital_lines) > 1 and self.secondary_processed_digital_2 is not None:
                    self.secondary_digital_lines[1].set_xdata(self.secondary_time / 60)
                    self.secondary_digital_lines[1].set_ydata(self.secondary_processed_digital_2)

                # Highlight the channel being used as control if shared control is enabled
                if hasattr(self, 'shared_isosbestic_var') and self.shared_isosbestic_var.get():
                    if hasattr(self, 'raw_line2') and hasattr(self, 'secondary_raw_line2'):
                        if secondary_should_use_primary_control:
                            # Primary control is being used for both
                            self.raw_line2.set_color('darkgreen')  # Make primary control more visible
                            self.raw_line2.set_linewidth(2)
                            if self.secondary_raw_line2 is not None:
                                self.secondary_raw_line2.set_color('lightgray')  # Fade secondary control
                                self.secondary_raw_line2.set_linewidth(1)
                        elif not primary_has_better_control:
                            # Secondary control is being used for both
                            if self.secondary_raw_line2 is not None:
                                self.secondary_raw_line2.set_color('darkgreen')  # Make secondary control more visible
                                self.secondary_raw_line2.set_linewidth(2)
                            self.raw_line2.set_color('lightgray')  # Fade primary control
                            self.raw_line2.set_linewidth(1)
                else:
                    # Reset colors if shared control is disabled
                    if hasattr(self, 'raw_line2'):
                        self.raw_line2.set_color('g')  # Reset to default color
                        self.raw_line2.set_linewidth(1)
                    if hasattr(self, 'secondary_raw_line2') and self.secondary_raw_line2 is not None:
                        self.secondary_raw_line2.set_color('g')  # Reset to default color
                        self.secondary_raw_line2.set_linewidth(1)

            # Update legends to reflect changes
            self.ax1_legend = self.create_checkbox_legend(self.ax1)
            self.ax2_legend = self.create_checkbox_legend(self.ax2)
            if self.digital_lines or (hasattr(self, 'secondary_digital_lines') and self.secondary_digital_lines):
                self.ax3_legend = self.create_checkbox_legend(self.ax3)

            # Restore previous axis limits
            self.ax1.set_xlim(x_lim)
            self.ax1.set_ylim(y1_lim)
            self.ax2.set_ylim(y2_lim)

            # Redraw
            self.canvas.draw_idle()

            # Update status when done
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Ready", fg="green")
            self.root.config(cursor="")  # Restore cursor

        except Exception as e:
            print(f"Error in update_filter: {e}")
            import traceback
            traceback.print_exc()

            # Show error in status
            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"Error: {str(e)}", fg="red")
            self.root.config(cursor="")  # Restore cursor

    def update_y_limits(self):
        """Update y-axis limits for processed signal with asymmetric scaling.
        Positive side: 5*STD, Negative side: 2*STD"""
        # Get all available signals
        signals = [self.processed_signal]
        if hasattr(self, 'secondary_processed_signal') and self.secondary_processed_signal is not None:
            signals.append(self.secondary_processed_signal)

        # Calculate combined statistics
        combined_signal = np.concatenate(signals)
        mean_val = np.mean(combined_signal)
        std_val = np.std(combined_signal)

        # Apply asymmetric scaling
        self.ax1.set_ylim(mean_val - 2 * std_val, mean_val + 5 * std_val)

    def run_advanced_denoising(self):
        """Run the advanced denoising process using the integrated controls"""
        # Choose which file to denoise
        if hasattr(self, 'secondary_processed_signal') and self.secondary_processed_signal is not None:
            # Ask user which file to denoise
            dialog = tk.Toplevel(self.root)
            dialog.title("Choose File")
            dialog.geometry("300x150")
            dialog.transient(self.root)
            dialog.grab_set()

            tk.Label(dialog, text="Which file would you like to denoise?",
                     font=('Arial', self.font_sizes['base'])).pack(pady=10)

            primary_region = self.get_region_label_from_filename(self.file_path)
            secondary_region = self.get_region_label_from_filename(self.secondary_file_path)

            choice_var = tk.StringVar(value="primary")

            tk.Radiobutton(dialog, text=f"Primary File ({primary_region})",
                           variable=choice_var, value="primary",
                           font=('Arial', self.font_sizes['base'])).pack(anchor="w", padx=20, pady=5)

            tk.Radiobutton(dialog, text=f"Secondary File ({secondary_region})",
                           variable=choice_var, value="secondary",
                           font=('Arial', self.font_sizes['base'])).pack(anchor="w", padx=20, pady=5)

            def on_denoising_choice():
                self.denoising_choice = choice_var.get()
                dialog.destroy()
                self.perform_denoising(self.denoising_choice)

            tk.Button(dialog, text="Start Denoising",
                      font=('Arial', self.font_sizes['base']),
                      command=on_denoising_choice).pack(pady=10)

            # Wait for user response
            self.root.wait_window(dialog)
        else:
            # Only primary file is available
            self.perform_denoising("primary")

    def perform_denoising(self, file_choice):
        """Execute denoising on the selected file"""
        # First determine if we need to use a control from another file
        better_control, primary_has_better_control = self.determine_better_control_channel() if hasattr(self,
                                                                                                        'secondary_downsampled_raw_analog_2') else (
        None, True)

        # Set up variables based on which file is being denoised
        if file_choice == "primary":
            if self.artifact_mask is None or not np.any(self.artifact_mask):
                messagebox.showinfo("Denoising", "No artifacts detected in primary file!")
                return

            signal = self.processed_signal
            artifact_mask = self.artifact_mask
            time = self.processed_time

            # Use the best control channel
            control_signal = better_control if not primary_has_better_control else self.downsampled_raw_analog_2

            region_label = self.get_region_label_from_filename(self.file_path)
            line = self.line
        else:  # secondary
            if self.secondary_artifact_mask is None or not np.any(self.secondary_artifact_mask):
                messagebox.showinfo("Denoising", "No artifacts detected in secondary file!")
                return

            signal = self.secondary_processed_signal
            artifact_mask = self.secondary_artifact_mask
            time = self.secondary_processed_time

            # Use the best control channel
            control_signal = self.downsampled_raw_analog_2 if primary_has_better_control else self.secondary_downsampled_raw_analog_2

            region_label = self.get_region_label_from_filename(self.secondary_file_path)
            line = self.secondary_line

        if self.denoise_thread and self.denoise_thread.is_alive():
            messagebox.showinfo("Denoising", "Denoising is already running!")
            return

        # Get options from the UI
        aggressive_mode = self.aggressive_var.get()
        use_control = self.control_var.get()
        max_gap = self.max_gap_var.get() if self.remove_var.get() else None
        if not use_control:
            control_signal = None

        # Show progress bar and cancel button
        self.progress_bar.grid()
        self.cancel_denoise_button.grid()

        # Disable run button
        self.denoise_button.config(state=tk.DISABLED, text=f"Denoising {region_label}...")

        # Reset cancellation event
        self.cancel_denoise_event.clear()

        # Clear queue
        while not self.denoise_progress_queue.empty():
            try:
                self.denoise_progress_queue.get_nowait()
            except queue.Empty:
                break

        # Start the denoising thread with options
        self.denoise_thread = threading.Thread(
            target=self.denoise_worker,
            args=(signal, artifact_mask, time, control_signal, aggressive_mode, max_gap, file_choice, line)
        )
        self.denoise_thread.daemon = True
        self.denoise_thread.start()

        # Start checking the queue for progress updates
        self.root.after(100, self.check_denoise_progress)

    def denoise_worker(self, signal, artifact_mask, time, control_signal=None,
                       aggressive_mode=False, max_gap=None, file_choice="primary", line=None):
        """Worker function that runs in a separate thread to perform denoising"""
        try:
            # Run the enhanced denoising function with additional parameters
            denoised_signal = advanced_denoise_signal(
                signal,
                artifact_mask,
                time,
                control_signal=control_signal,
                progress_callback=lambda p: self.denoise_progress_queue.put(p),
                cancel_event=self.cancel_denoise_event,
                aggressive_mode=aggressive_mode,
                max_gap_to_remove=max_gap
            )

            # If process was not cancelled, send the result
            if not self.cancel_denoise_event.is_set():
                self.denoise_progress_queue.put(('result', denoised_signal, file_choice, line))
        except Exception as e:
            # If an error occurred, send it to the main thread
            self.denoise_progress_queue.put(('error', str(e)))
            print(f"Denoising error: {e}")
            import traceback
            traceback.print_exc()

    def check_denoise_progress(self):
        """Check the progress queue and update the UI"""
        try:
            # Check if denoising is still running
            if not self.denoise_thread.is_alive():
                # Thread has finished, but we might still have messages in the queue
                while not self.denoise_progress_queue.empty():
                    try:
                        message = self.denoise_progress_queue.get_nowait()
                        if isinstance(message, tuple) and message[0] == 'result':
                            # We have a result, update the appropriate signal
                            denoised_signal = message[1]
                            file_choice = message[2]
                            line = message[3]

                            if file_choice == "primary":
                                self.processed_signal = denoised_signal
                                if line:
                                    line.set_ydata(denoised_signal)
                            else:  # secondary
                                self.secondary_processed_signal = denoised_signal
                                if line:
                                    line.set_ydata(denoised_signal)

                            self.canvas.draw_idle()
                        elif isinstance(message, tuple) and message[0] == 'error':
                            # Show error message
                            messagebox.showerror("Denoising Error", message[1])
                    except queue.Empty:
                        break

                # Hide progress UI and restore button
                self.progress_bar.grid_remove()
                self.cancel_denoise_button.grid_remove()
                self.denoise_button.config(state=tk.NORMAL, text="Run Advanced Denoising")
                return

            # Process any pending progress updates
            while not self.denoise_progress_queue.empty():
                try:
                    message = self.denoise_progress_queue.get_nowait()
                    if isinstance(message, (int, float)):
                        # Update progress
                        self.progress_var.set(message)
                    elif isinstance(message, tuple) and message[0] == 'result':
                        # We have a result, update the appropriate signal
                        denoised_signal = message[1]
                        file_choice = message[2]
                        line = message[3]

                        if file_choice == "primary":
                            self.processed_signal = denoised_signal
                            if line:
                                line.set_ydata(denoised_signal)
                        else:  # secondary
                            self.secondary_processed_signal = denoised_signal
                            if line:
                                line.set_ydata(denoised_signal)

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
            # Make sure UI is restored if there's an error
            if hasattr(self, 'denoise_button'):
                self.denoise_button.config(state=tk.NORMAL, text="Run Advanced Denoising")
            if hasattr(self, 'progress_bar'):
                self.progress_bar.grid_remove()
            if hasattr(self, 'cancel_denoise_button'):
                self.cancel_denoise_button.grid_remove()

    def cancel_denoising(self):
        """Cancel the denoising process"""
        if self.denoise_thread and self.denoise_thread.is_alive():
            # Set the cancel event
            self.cancel_denoise_event.set()

            # Update the cancel button
            self.cancel_denoise_button.config(text="Cancelling...", state=tk.DISABLED)


def main():
    """Main function to start the application."""
    try:
        print("Starting application...")

        # Create the root window
        root = tk.Tk()
        print("Created root window")

        root.title("Photometry Signal Viewer")
        root.geometry("1400x1000")  # Larger window size

        # Create viewer application without initially loading a file
        app = PhotometryViewer(root)
        print("Created PhotometryViewer")

        # Make the open_file function optional
        def delayed_file_open():
            try:
                app.open_file()
            except Exception as e:
                print(f"Error opening file: {e}")
                import traceback
                traceback.print_exc()

        # Schedule file dialog to open after GUI appears
        root.after(100, delayed_file_open)

        # Start main loop
        print("Starting mainloop...")
        root.mainloop()
        print("Mainloop ended")

    except Exception as e:
        import traceback
        print(f"Error in main function: {e}")
        traceback.print_exc()
        input("Press Enter to exit...")
        sys.exit(1)

    print("Application ending normally")


if __name__ == "__main__":
     main()