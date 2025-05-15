import json
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, Menu, ttk
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
from scipy.integrate import trapezoid
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


def find_peaks_valleys(signal, time, prominence=1.0, width=None, distance=None, threshold=None):
    """
    Find peaks and valleys in the signal using scipy's find_peaks function.

    Parameters:
    -----------
    signal : array
        Signal to analyze
    time : array
        Corresponding time points
    prominence : float
        Minimum peak prominence (vertical distance to neighboring valleys)
    width : float or None
        Minimum peak width
    distance : float or None
        Minimum horizontal distance between peaks (in seconds)
    threshold : float or None
        Minimum absolute height of peaks

    Returns:
    --------
    peak_data : dict
        Dictionary containing peak indices, heights, and properties
    valley_data : dict
        Dictionary containing valley indices, depths, and properties
    """
    from scipy.signal import find_peaks

    # Find peaks
    peak_indices, peak_props = find_peaks(
        signal,
        prominence=prominence,
        width=width,
        distance=distance,
        height=threshold
    )

    # Find valleys (by inverting the signal and finding peaks)
    valley_indices, valley_props = find_peaks(
        -signal,
        prominence=prominence,
        width=width,
        distance=distance,
        height=None if threshold is None else -threshold
    )

    # Calculate additional peak metrics
    peak_heights = signal[peak_indices]
    peak_times = time[peak_indices]

    # Calculate valley metrics
    valley_depths = signal[valley_indices]
    valley_times = time[valley_indices]

    # Convert prominence values for valleys (they're calculated on inverted signal)
    if 'prominences' in valley_props:
        valley_props['prominences'] = valley_props['prominences'].copy()

    return {
        'indices': peak_indices,
        'times': peak_times,
        'heights': peak_heights,
        'properties': peak_props
    }, {
        'indices': valley_indices,
        'times': valley_times,
        'depths': valley_depths,
        'properties': valley_props
    }


def calculate_peak_metrics(peak_data, valley_data, signal, time):
    """
    Calculate additional metrics for detected peaks.

    Parameters:
    -----------
    peak_data : dict
        Dictionary containing peak information
    valley_data : dict
        Dictionary containing valley information
    signal : array
        The signal array
    time : array
        The time array

    Returns:
    --------
    peak_metrics : dict
        Dictionary with additional peak metrics
    """
    # Debug to track execution
    print(f"Calculating metrics for {len(peak_data['indices'])} peaks...")

    # Safety checks
    if peak_data is None or valley_data is None:
        print("Peak metrics: Missing peak or valley data")
        return None

    peak_indices = peak_data['indices']
    valley_indices = valley_data['indices']

    if len(peak_indices) == 0:
        print("Peak metrics: No peaks found")
        return None
    if len(valley_indices) == 0:
        print("Peak metrics: No valleys found")
        return None

    # Initialize metrics dictionary
    metrics = {
        'area_under_curve': [],
        'full_width_half_max': [],
        'rise_time': [],
        'decay_time': [],
        'preceding_valley': [],
        'following_valley': []
    }

    # For each peak, find the closest valleys before and after
    for peak_idx in peak_indices:
        try:
            # Find preceding valley
            preceding_valleys = valley_indices[valley_indices < peak_idx]
            preceding_valley_idx = preceding_valleys[-1] if len(preceding_valleys) > 0 else 0

            # Find following valley
            following_valleys = valley_indices[valley_indices > peak_idx]
            following_valley_idx = following_valleys[0] if len(following_valleys) > 0 else len(signal) - 1

            # Store the valley indices
            metrics['preceding_valley'].append(preceding_valley_idx)
            metrics['following_valley'].append(following_valley_idx)

            # Calculate peak width at half max
            peak_height = signal[peak_idx]
            base_level = min(signal[preceding_valley_idx], signal[following_valley_idx])
            half_height = base_level + (peak_height - base_level) / 2

            # Find crossing points
            left_segment = signal[preceding_valley_idx:peak_idx + 1]
            right_segment = signal[peak_idx:following_valley_idx + 1]

            # Default values in case calculation fails
            fwhm = 0.0
            area = 0.0
            rise_time = 0.0
            decay_time = 0.0

            # Find where signal crosses half height
            left_crosses = np.where(left_segment >= half_height)[0]
            right_crosses = np.where(right_segment <= half_height)[0]

            if len(left_crosses) > 0 and len(right_crosses) > 0:
                left_cross_idx = preceding_valley_idx + left_crosses[0]
                right_cross_idx = peak_idx + right_crosses[-1] if len(right_crosses) > 0 else following_valley_idx

                # Calculate full width at half max in seconds
                fwhm = time[right_cross_idx] - time[left_cross_idx]

                # Calculate rise and decay times
                rise_time = time[peak_idx] - time[left_cross_idx]
                decay_time = time[right_cross_idx] - time[peak_idx]

                # Calculate area under the curve (simple trapezoidal integration)
                peak_segment = signal[preceding_valley_idx:following_valley_idx + 1]
                peak_times = time[preceding_valley_idx:following_valley_idx + 1]

                # Use trapezoid instead of deprecated trapz
                try:
                    from scipy.integrate import trapezoid
                    area = trapezoid(peak_segment - base_level, peak_times)
                except ImportError:
                    # Fall back to numpy trapz if scipy.integrate.trapezoid is not available
                    area = np.trapz(peak_segment - base_level, peak_times)

                # Ensure positive area
                area = max(0, area)

            # Add calculated metrics
            metrics['full_width_half_max'].append(fwhm)
            metrics['rise_time'].append(rise_time)
            metrics['decay_time'].append(decay_time)
            metrics['area_under_curve'].append(area)

        except Exception as e:
            print(f"Error calculating metrics for peak at index {peak_idx}: {e}")
            # Add default values if calculation fails
            metrics['full_width_half_max'].append(0.0)
            metrics['rise_time'].append(0.0)
            metrics['decay_time'].append(0.0)
            metrics['area_under_curve'].append(0.0)
            # Make sure valley indices are still added even if calculation fails
            if 'preceding_valley' not in metrics:
                metrics['preceding_valley'].append(0)
            if 'following_valley' not in metrics:
                metrics['following_valley'].append(0)

    # Debug output
    print(f"Finished peak metrics calculation. Example width: {metrics['full_width_half_max'][:5]}")
    return metrics


def adaptive_motion_correction(signal_470, control_405, time, artifact_threshold=3.0):
    """
    Sophisticated adaptive motion correction that targets whole sections of movement
    """
    # Create a copy for the corrected signal
    corrected_signal = signal_470.copy()

    # 1. Identify motion periods more comprehensively
    # Use control signal variability as motion indicator
    control_median = np.median(control_405)
    control_mad = np.median(np.abs(control_405 - control_median))
    base_artifact_mask = np.abs(control_405 - control_median) > artifact_threshold * control_mad * 1.4826

    # 2. Expand artifact regions to include entire movement episodes
    motion_mask = np.zeros_like(base_artifact_mask)
    min_region_size = min(int(len(time) * 0.01), 50)  # Minimum 1% of recording or 50 points

    # Find connected regions and expand them
    in_region = False
    region_start = 0

    for i in range(len(base_artifact_mask)):
        if base_artifact_mask[i] and not in_region:
            # Start of a new motion region
            in_region = True
            region_start = max(0, i - min_region_size // 2)
        elif not base_artifact_mask[i] and in_region:
            # End of motion region
            region_end = min(len(base_artifact_mask) - 1, i + min_region_size // 2)
            # Mark the entire expanded region
            motion_mask[region_start:region_end + 1] = True
            in_region = False

    # Handle case if we end inside a region
    if in_region:
        region_end = len(base_artifact_mask) - 1
        motion_mask[region_start:region_end + 1] = True

    # 3. Process each motion region and clean region separately
    regions = []
    in_motion = False
    start_idx = 0

    for i in range(1, len(motion_mask)):
        if motion_mask[i] != motion_mask[i - 1]:
            regions.append({
                'start': start_idx,
                'end': i - 1,
                'is_motion': motion_mask[i - 1]
            })
            start_idx = i

    # Add the final region
    regions.append({
        'start': start_idx,
        'end': len(motion_mask) - 1,
        'is_motion': motion_mask[-1]
    })

    # 4. Process each region
    for region in regions:
        start, end, is_motion = region['start'], region['end'], region['is_motion']
        seg_time = time[start:end + 1]
        seg_470 = signal_470[start:end + 1]
        seg_405 = control_405[start:end + 1]

        if is_motion:
            # This is a motion region - more aggressive correction
            # Find values at boundaries for anchoring
            pre_val = signal_470[max(0, start - 3):max(1, start)]
            post_val = signal_470[min(end + 1, len(signal_470) - 1):min(end + 4, len(signal_470))]

            pre_mean = np.mean(pre_val) if len(pre_val) > 0 else seg_470[0]
            post_mean = np.mean(post_val) if len(post_val) > 0 else seg_470[-1]

            # For long segments, use local regression if possible
            if len(seg_470) > 20 and np.std(seg_405) > 0:
                # Calculate local regression between 405 and 470
                slope, intercept, r_value, p_value, std_err = linregress(seg_405, seg_470)

                if r_value ** 2 > 0.3:  # Only use regression if correlation is substantial
                    # Predict and correct the motion component
                    motion_component = slope * seg_405 + intercept

                    # Calculate local baseline trend (connecting pre and post values)
                    t = np.linspace(0, 1, len(seg_470))
                    baseline = pre_mean + t * (post_mean - pre_mean)

                    # Calculate signal fluctuations with motion removed
                    fluctuations = seg_470 - motion_component
                    # Center fluctuations and add back expected baseline trend
                    corrected = fluctuations - np.mean(fluctuations) + baseline

                    # Blend boundaries for smooth transitions
                    blend_window = min(5, len(seg_470) // 10)
                    weights = np.ones_like(seg_470)
                    if blend_window > 0:
                        weights[:blend_window] = np.linspace(0, 1, blend_window)
                        weights[-blend_window:] = np.linspace(1, 0, blend_window)

                    # Apply correction with blended boundaries
                    corrected_signal[start:end + 1] = weights * corrected + (1 - weights) * seg_470
                    continue

            # If regression isn't viable or segment is short, use trend-based correction
            # Simple but effective approach: replace with a trend line plus filtered noise
            t = np.linspace(0, 1, len(seg_470))
            trend = pre_mean + t * (post_mean - pre_mean)

            # Add physiological-like fluctuations (filtered random noise scaled to signal amplitude)
            if len(trend) > 10:
                # Start with small random fluctuations
                rng = np.random.RandomState(42)  # Fixed seed for reproducibility
                noise_amp = min(np.std(seg_470) * 0.2, abs(post_mean - pre_mean) * 0.1)
                noise = rng.normal(0, noise_amp, len(trend))

                # Filter noise to physiological frequencies
                window = min(11, len(noise) // 5 * 2 + 1)  # Must be odd
                if window > 3:
                    filtered_noise = savgol_filter(noise, window, 3)
                    trend = trend + filtered_noise

            # Apply with boundary blending
            blend_window = min(5, len(seg_470) // 10)
            weights = np.ones_like(seg_470)
            if blend_window > 0:
                weights[:blend_window] = np.linspace(0, 1, blend_window)
                weights[-blend_window:] = np.linspace(1, 0, blend_window)

            corrected_signal[start:end + 1] = weights * trend + (1 - weights) * seg_470

        else:
            # For non-motion regions, apply gentler correction to preserve signal
            if len(seg_470) > 10 and np.std(seg_405) > 0:
                slope, intercept, r_value, p_value, std_err = linregress(seg_405, seg_470)

                # Only apply gentle correction if there's meaningful correlation
                if r_value ** 2 > 0.2:
                    motion_component = slope * seg_405 + intercept
                    # Preserve mean of original signal
                    motion_free = seg_470 - (motion_component - np.mean(motion_component))

                    # Apply mild correction (weighted by correlation strength)
                    blend = min(0.5, r_value ** 2)  # Cap correction at 50%
                    corrected_signal[start:end + 1] = blend * motion_free + (1 - blend) * seg_470

    return corrected_signal

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


def comprehensive_photometry_processing(signal_470, control_405, time, sampling_rate=300.0):
    """
    Complete fiber photometry signal processing pipeline following published methods.

    Parameters:
    -----------
    signal_470 : array
        The calcium-dependent signal (470nm channel)
    control_405 : array
        The control signal (405nm channel)
    time : array
        Time array in seconds
    sampling_rate : float
        Sampling rate in Hz

    Returns:
    --------
    processed_signal : dict
        Dictionary containing all processed signals and parameters
    """
    # Create result dictionary to store all processed signals
    result = {
        'time': time,
        'raw_470': signal_470,
        'raw_405': control_405
    }

    # Step 15: Low-pass filter (10 Hz Butterworth)
    def butterworth_lowpass(signal, cutoff=10.0, fs=sampling_rate, order=2):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        sos = butter(order, normal_cutoff, btype='low', analog=False, output='sos')
        return sosfiltfilt(sos, signal)

    # Apply low-pass filter to both signals
    filtered_470 = butterworth_lowpass(signal_470)
    filtered_405 = butterworth_lowpass(control_405)

    result['filtered_470'] = filtered_470
    result['filtered_405'] = filtered_405

    # Step 16: Scale control signal to match main signal (empirically determined)
    # Calculate scaling using regression over stable regions
    # Find stable regions (exclude extremes in either channel)
    p5_470, p95_470 = np.percentile(filtered_470, [5, 95])
    p5_405, p95_405 = np.percentile(filtered_405, [5, 95])

    stable_mask = ((filtered_470 > p5_470) & (filtered_470 < p95_470) &
                   (filtered_405 > p5_405) & (filtered_405 < p95_405))

    # Calculate scaling using regression
    if np.sum(stable_mask) > 10:
        slope, intercept, r_value, p_value, std_err = linregress(
            filtered_405[stable_mask], filtered_470[stable_mask]
        )
        print(
            f"Control scaling: {filtered_470.mean():.4f} = {slope:.4f} × {filtered_405.mean():.4f} + {intercept:.4f}, R²={r_value ** 2:.3f}")
        scaled_405 = slope * filtered_405 + intercept
    else:
        # Fallback to standard scaling if not enough stable points
        scale_factor = np.std(filtered_470) / np.std(filtered_405)
        offset = np.mean(filtered_470) - scale_factor * np.mean(filtered_405)
        scaled_405 = scale_factor * filtered_405 + offset
        print(f"Using standard scaling: factor={scale_factor:.4f}, offset={offset:.4f}")

    result['scaled_405'] = scaled_405

    # Motion corrected signal (control subtracted)
    motion_corrected = filtered_470 - scaled_405 + np.mean(filtered_470)
    result['motion_corrected'] = motion_corrected

    # Step 17: Remove remaining artifacts (fast symmetric transients)
    # Calculate signal derivative to find abrupt changes
    derivative = np.diff(motion_corrected, prepend=motion_corrected[0])

    # Identify extreme derivatives (abrupt changes)
    derivative_threshold = np.std(derivative) * 4  # 4 standard deviations
    artifact_mask = np.abs(derivative) > derivative_threshold

    # Expand mask to include surrounding points
    expanded_mask = np.zeros_like(artifact_mask)
    for i in range(len(artifact_mask)):
        if artifact_mask[i]:
            # Mark 5 points before and after
            start = max(0, i - 5)
            end = min(len(artifact_mask), i + 6)
            expanded_mask[start:end] = True

    # Replace artifacts with interpolated values
    artifact_cleaned = motion_corrected.copy()
    if np.any(expanded_mask):
        # Use cubic interpolation where possible
        non_artifact_indices = np.where(~expanded_mask)[0]
        if len(non_artifact_indices) > 3:
            artifact_indices = np.where(expanded_mask)[0]

            # Create interpolation function
            interp_func = interp1d(
                time[non_artifact_indices],
                motion_corrected[non_artifact_indices],
                kind='cubic', bounds_error=False, fill_value='extrapolate'
            )

            # Replace artifact points
            artifact_cleaned[artifact_indices] = interp_func(time[artifact_indices])

    result['artifact_cleaned'] = artifact_cleaned

    # Step 18: Detrend to remove photobleaching
    # Use polynomial fit on whole recording
    time_normalized = (time - time[0]) / (time[-1] - time[0])
    poly_degree = 3  # Cubic polynomial as shown in the paper
    poly_coeffs = poly.polyfit(time_normalized, artifact_cleaned, poly_degree)
    bleaching_trend = poly.polyval(time_normalized, poly_coeffs)

    # Subtract trend
    detrended = artifact_cleaned - bleaching_trend + np.mean(artifact_cleaned)
    result['detrended'] = detrended
    result['bleaching_trend'] = bleaching_trend

    # Step 19: Smooth signal with Savitzky-Golay filter
    # Window size should be odd and approximately 0.5-1 second of data
    window_size = int(sampling_rate * 0.5) // 2 * 2 + 1  # Ensure odd
    poly_order = 3

    # Apply smoothing
    smoothed = savgol_filter(detrended, window_size, poly_order)
    result['smoothed'] = smoothed

    # Step 20: Calculate ΔF/F
    # Use first 10% of recording (or 60 seconds as noted) for F0
    f0_duration = min(time[-1] * 0.1, 60)  # Use 10% of recording or 60s
    f0_indices = time <= f0_duration

    if np.any(f0_indices):
        # Calculate baseline as 10th percentile of baseline period
        # (more robust to transients than mean)
        f0 = np.percentile(smoothed[f0_indices], 10)
    else:
        # Fallback if no suitable baseline period
        f0 = np.percentile(smoothed, 10)

    # Calculate ΔF/F
    dff = (smoothed - f0) / f0 * 100  # Convert to percent
    result['dff'] = dff

    # Step 21: Segment data into 1-minute bins for analysis
    bin_duration = 60  # seconds
    num_bins = int(np.ceil(time[-1] / bin_duration))

    bins = []
    for i in range(num_bins):
        bin_start = i * bin_duration
        bin_end = (i + 1) * bin_duration

        bin_mask = (time >= bin_start) & (time < bin_end)
        if np.any(bin_mask):
            bin_data = {
                'time': time[bin_mask],
                'dff': dff[bin_mask],
                'start_time': bin_start,
                'end_time': bin_end
            }

            # Step 22: Calculate area under curve and other metrics
            # Detect transients above threshold
            threshold_0_5 = 0.5  # 0.5% dF/F threshold
            threshold_1_0 = 1.0  # 1.0% dF/F threshold

            # Calculate metrics
            bin_data['above_0_5'] = bin_data['dff'] > threshold_0_5
            bin_data['above_1_0'] = bin_data['dff'] > threshold_1_0

            # Time spent above thresholds (%)
            bin_data['time_above_0_5'] = np.mean(bin_data['above_0_5']) * 100
            bin_data['time_above_1_0'] = np.mean(bin_data['above_1_0']) * 100

            # Area under curve (only positive parts, as in paper)
            positive_dff = np.maximum(bin_data['dff'], 0)
            bin_data['auc'] = np.trapz(positive_dff, bin_data['time'])

            bins.append(bin_data)

    result['bins'] = bins

    # Calculate overall metrics
    result['metrics'] = {
        'mean_dff': np.mean(dff),
        'max_dff': np.max(dff),
        'std_dff': np.std(dff),
        'time_above_0_5': np.mean(dff > 0.5) * 100,
        'time_above_1_0': np.mean(dff > 1.0) * 100,
        'total_auc': np.trapz(np.maximum(dff, 0), time)
    }

    return result

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
                 edge_protection=True):
    """Processes the data by applying filters and downsampling with edge protection."""
    # Check for null or empty inputs
    if time is None or len(time) == 0:
        print("Error: Time array is None or empty")
        return None, None, None, None, None, None, None, None, None
    # Compute sampling rate
    dt = np.diff(time).mean()
    fs = 1 / dt

    # Make copies for processing
    processed_signal = analog_1.copy()
    processed_analog_2 = analog_2.copy() if analog_2 is not None else None

    # NEW: Make a copy for processing without isosbestic control
    # Clone the main signal to ensure it receives the same filtering
    processed_signal_no_control = analog_1.copy()

    # IMPORTANT: Calculate baseline from the RAW signal BEFORE filtering
    # This ensures our baseline doesn't shift with filter changes
    # Define a consistent stable region for baseline calculation (20-40% of signal)
    stable_start = int(len(analog_1) * 0.2)  # 20% in
    stable_end = int(len(analog_1) * 0.4)  # 40% in
    raw_baseline = np.median(analog_1[stable_start:stable_end])

    print(f"Using stable baseline region from {stable_start}-{stable_end}, baseline value: {raw_baseline}")

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
            # Create SOS filter for better numerical stability
            sos = butter(2, low_cutoff / (fs / 2), 'high', output='sos')

            # Enhanced edge protection for high-pass filtering
            if edge_protection:
                # For high-pass filtering with edge protection
                pad_size = min(len(processed_signal), int(30 * fs))  # 30 seconds or full signal

                # Process main signal
                padded_signal = np.pad(processed_signal, pad_size, mode='reflect')
                filtered_padded = sosfiltfilt(sos, padded_signal)
                processed_signal = filtered_padded[pad_size:pad_size + len(processed_signal)]

                # Also process the no-control signal with the same filters
                padded_nc = np.pad(processed_signal_no_control, pad_size, mode='reflect')
                filtered_padded_nc = sosfiltfilt(sos, padded_nc)
                processed_signal_no_control = filtered_padded_nc[pad_size:pad_size + len(processed_signal_no_control)]

                # Apply same process to the control channel if available
                if processed_analog_2 is not None:
                    padded_control = np.pad(processed_analog_2, pad_size, mode='reflect')
                    filtered_control = sosfiltfilt(sos, padded_control)
                    processed_analog_2 = filtered_control[pad_size:pad_size + len(processed_analog_2)]
            else:
                # Standard filtering without edge protection
                processed_signal = sosfiltfilt(sos, processed_signal)
                processed_signal_no_control = sosfiltfilt(sos, processed_signal_no_control)  # Filter no-control signal
                if processed_analog_2 is not None:
                    processed_analog_2 = sosfiltfilt(sos, processed_analog_2)

        # Apply low-pass filter (high_cutoff)
        if high_cutoff > 0 and high_cutoff < fs / 2:
            # Use SOS for low-pass filter too
            sos = butter(2, high_cutoff / (fs / 2), 'low', output='sos')
            processed_signal = sosfiltfilt(sos, processed_signal)
            processed_signal_no_control = sosfiltfilt(sos, processed_signal_no_control)  # Filter no-control signal
            if processed_analog_2 is not None:
                processed_analog_2 = sosfiltfilt(sos, processed_analog_2)

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

            # Store the unprocessed signal for comparison
            processed_signal_no_control = processed_signal.copy()

            # Apply our refined motion correction approach
            if processed_analog_2 is not None:
                print("Applying adaptive motion correction...")
                try:
                    processed_signal = adaptive_motion_correction(
                        processed_signal,
                        processed_analog_2,
                        time,
                        artifact_threshold=artifact_threshold
                    )
                except Exception as e:
                    print(f"Error in motion correction: {e}, falling back to simpler method")
                    if artifact_mask is not None and np.any(artifact_mask):
                        processed_signal = remove_artifacts_fast(processed_signal, artifact_mask)
            else:
                # Fall back to simple correction if no control signal
                if artifact_mask is not None and np.any(artifact_mask):
                    processed_signal = remove_artifacts_fast(processed_signal, artifact_mask)

        # Special handling for start of recording (first 10% or first 10 seconds)
        start_idx = min(int(len(processed_signal) * 0.1), int(10 * fs))
        if edge_protection and start_idx > 0:
            # Apply edge protection to main signal
            stable_median = np.median(processed_signal[start_idx:start_idx * 2])
            weight = np.linspace(0, 1, start_idx) ** 2
            target_val = processed_signal[start_idx]
            linear_trend = np.linspace(stable_median, target_val, start_idx)
            for i in range(start_idx):
                processed_signal[i] = (1 - weight[i]) * linear_trend[i] + weight[i] * processed_signal[i]

            # Also apply similar edge protection to no-control signal
            nc_stable_median = np.median(processed_signal_no_control[start_idx:start_idx * 2])
            nc_target_val = processed_signal_no_control[start_idx]
            nc_linear_trend = np.linspace(nc_stable_median, nc_target_val, start_idx)
            for i in range(start_idx):
                processed_signal_no_control[i] = (1 - weight[i]) * nc_linear_trend[i] + weight[i] * \
                                                 processed_signal_no_control[i]

        # NEW: SMART BASELINE SELECTION
        # Find the most stable region of the signal by using a sliding window approach
        window_size = min(int(len(processed_signal) * 0.2), int(20 * fs))  # 20% of signal or 20 seconds
        min_variance = float('inf')
        best_start_idx = 0

        # Skip the first 10% that may contain noise or blanked regions
        search_start = max(start_idx, int(len(processed_signal) * 0.1))

        # Search through the signal to find the most stable window (lowest variance)
        step_size = max(1, window_size // 10)  # Use 10% of window as step size
        for i in range(search_start, len(processed_signal) - window_size, step_size):
            window = processed_signal[i:i + window_size]

            # Skip windows with artifacts if we have an artifact mask
            if artifact_mask is not None and np.any(artifact_mask[i:i + window_size]):
                continue

            window_var = np.var(window)
            if window_var < min_variance:
                min_variance = window_var
                best_start_idx = i

        # Use the most stable region for baseline calculation
        baseline_region = processed_signal[best_start_idx:best_start_idx + window_size]

        # Use 10th percentile of this region as baseline to prevent negative dF/F
        # This is more robust than mean/median for signals with transients
        smart_baseline = np.percentile(baseline_region, 10)

        print(f"Smart baseline selected from region {best_start_idx}-{best_start_idx + window_size}")
        print(f"Smart baseline value: {smart_baseline:.4f} (10th percentile of most stable region)")

        # Fit drift curve and store it for visualization
        drift_curve = None
        if drift_correction:
            try:
                # Create a mask to exclude artifacts and blanked regions from the fit
                valid_mask = np.ones_like(processed_signal, dtype=bool)

                # Exclude first 10% or blanked regions
                valid_mask[:start_idx] = False

                # Exclude artifacts if we have an artifact mask
                if artifact_mask is not None:
                    valid_mask[artifact_mask] = False

                # Get valid time and signal points for fitting
                valid_time = time[valid_mask]
                valid_signal = processed_signal[valid_mask]

                if len(valid_time) > 100:  # Ensure enough points for fitting
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
                    # Apply same drift correction to no-control signal

                if drift_curve is not None:
                        processed_signal_no_control = correct_drift(processed_signal_no_control, drift_curve)
            except Exception as e:
                print(f"Error in drift correction: {e}")
                drift_curve = None

        # Calculate dF/F using the smart baseline
        if smart_baseline != 0:
            # Ensure positive dF/F values
            min_signal = min(np.min(processed_signal), np.min(processed_signal_no_control))

            if smart_baseline > min_signal:
                adjustment_factor = 0.95
                adjusted_baseline = min_signal * adjustment_factor
                print(f"Adjusted baseline from {smart_baseline:.4f} to {adjusted_baseline:.4f} to ensure positive dF/F")
                smart_baseline = adjusted_baseline

            # Calculate dF/F with adjusted baseline for both signals
            processed_signal = 100 * (processed_signal - smart_baseline) / abs(smart_baseline)
            processed_signal_no_control = 100 * (processed_signal_no_control - smart_baseline) / abs(smart_baseline)

            # Add offset to center around a nicer value
            mean_value = np.mean(processed_signal)
            if mean_value < 0:
                offset = abs(mean_value) + 100
                processed_signal += offset
                processed_signal_no_control += offset
                print(f"Added offset of {offset:.2f}% to center signals")
        else:
            print("Warning: Smart baseline is zero, cannot calculate dF/F")

        # Center both signals to have the same mean
        mean_control = np.mean(processed_signal)
        mean_no_control = np.mean(processed_signal_no_control)

        # Calculate offset to align the two signals
        mean_offset = mean_control - mean_no_control

        # Apply offset to no-control signal to align with control signal
        processed_signal_no_control += mean_offset

        print(f"Aligned no-control signal by adding offset of {mean_offset:.2f}%")

        # Center the raw signals around zero
        downsampled_raw_analog_1 = center_signal(downsampled_raw_analog_1)
        if downsampled_raw_analog_2 is not None:
            downsampled_raw_analog_2 = center_signal(downsampled_raw_analog_2)

        # Apply downsampling if needed
        if downsample_factor > 1:
            processed_time, processed_signal = downsample_data(time, processed_signal, int(downsample_factor))
            processed_time, processed_signal_no_control = downsample_data(time, processed_signal_no_control,
                                                                          int(downsample_factor))

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

    # Ensure all arrays have consistent lengths before returning
    print(f"Final processed array lengths: time={len(processed_time)}, signal={len(processed_signal)}")
    min_len = min(len(processed_time), len(processed_signal))
    processed_time = processed_time[:min_len]
    processed_signal = processed_signal[:min_len]
    if processed_analog_2 is not None:
        processed_analog_2 = processed_analog_2[:min_len]
    if artifact_mask is not None and len(artifact_mask) > min_len:
        artifact_mask = artifact_mask[:min_len]
    if drift_curve is not None and len(drift_curve) > min_len:
        drift_curve = drift_curve[:min_len]

    # Return the processed data, downsampled raw data, and artifact mask
    return processed_time, processed_signal, processed_signal_no_control, processed_analog_2, downsampled_raw_analog_1, downsampled_raw_analog_2, processed_digital_1, processed_digital_2, artifact_mask, drift_curve



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

        # Setup proper closing behavior
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Setup the menu bar
        self.create_menu()

        # If a file path was provided, load it
        if file_path:
            self.load_file(file_path)

        def setup_keyboard_shortcuts(self):
            """Setup keyboard shortcuts for common operations"""
            self.root.bind('<Home>', lambda event: self.reset_view())
            self.root.bind('<Control-0>', lambda event: self.reset_view())  # Ctrl+0 also resets view

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

        # Unpack results - with the new signal added
        self.processed_time, self.processed_signal, self.processed_signal_no_control, \
            self.processed_analog_2, self.downsampled_raw_analog_1, self.downsampled_raw_analog_2, \
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
        self.secondary_processed_time, self.secondary_processed_signal, self.secondary_processed_signal_no_control, \
            self.secondary_processed_analog_2, self.secondary_downsampled_raw_analog_1, self.secondary_downsampled_raw_analog_2, \
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


    def toggle_feature_visualization(self):
        """Toggle display of peak features (width, area)"""
        if self.feature_viz_var.get():
            self.visualize_peak_features()
        else:
            # Clear feature annotations
            for annotation in self.peak_feature_annotations:
                annotation.remove() if hasattr(annotation, 'remove') else None
            self.peak_feature_annotations = []
            self.canvas.draw_idle()

    def toggle_manual_peak_mode(self):
        """Toggle manual peak selection mode"""
        self.manual_peak_mode = not self.manual_peak_mode
        self.manual_valley_mode = False  # Turn off valley mode if it was on

        if self.manual_peak_mode:
            # Use the correct button reference
            self.manual_peak_btn.config(text="Cancel Peak Selection", bg='salmon')
            self.manual_valley_btn.config(state=tk.DISABLED)
            self.connect_selection_events("peak")

            # Show instructions
            if hasattr(self, 'status_label'):
                self.status_label.config(
                    text="Click on the signal to add a peak. Right-click a peak to remove it.",
                    fg="blue"
                )
        else:
            # Disable selection mode
            self.manual_peak_btn.config(text="Manual Peak Selection", bg='lightblue')
            self.manual_valley_btn.config(state=tk.NORMAL)
            self.disconnect_selection_events()

            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Manual peak selection mode disabled", fg="green")

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
        # Add the no-control version for secondary signal
        self.secondary_line_no_control, = self.ax1.plot(
            self.secondary_processed_time / 60,
            self.secondary_processed_signal_no_control,
            color='lightsalmon',  # Light red/orange color
            linestyle='--',
            lw=1.5,
            alpha=0.7,
            label=f'{region_label} No-Control'
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
        # Check if secondary_artifact_markers_processed exists before trying to set data
        if not hasattr(self,
                       'secondary_artifact_markers_processed') or self.secondary_artifact_markers_processed is None:
            return

        if (self.secondary_artifact_mask is None or
                not np.any(self.secondary_artifact_mask) or
                len(self.secondary_artifact_mask) != len(self.secondary_processed_time)):
            # No artifacts detected or size mismatch, clear the markers
            self.secondary_artifact_markers_processed.set_data([], [])
            if hasattr(self, 'secondary_artifact_markers_raw') and self.secondary_artifact_markers_raw is not None:
                self.secondary_artifact_markers_raw.set_data([], [])
            return

        # Get the time points where artifacts occur (in minutes)
        artifact_times = self.secondary_processed_time[self.secondary_artifact_mask] / 60

        # Update markers on the processed signal
        artifact_values_processed = self.secondary_processed_signal[self.secondary_artifact_mask]
        self.secondary_artifact_markers_processed.set_data(artifact_times, artifact_values_processed)

        # Update markers on the raw signal (405nm channel)
        if hasattr(self,
                   'secondary_artifact_markers_raw') and self.secondary_artifact_markers_raw is not None and self.secondary_downsampled_raw_analog_2 is not None:
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

    # Inside the PhotometryViewer class
    def analyze_peak_intervals(self):
        """Analyze intervals between consecutive peaks in the integrated plot frame"""
        # Determine which peaks to analyze based on signal selection
        if self.signal_selection_var.get() == "Secondary" and hasattr(self,
                                                                      'secondary_peaks') and self.secondary_peaks is not None:
            times = self.secondary_peaks['times']
            region = "SON"
        else:
            if self.peaks is None or len(self.peaks.get('times', [])) == 0:
                messagebox.showinfo("Analysis Error", "No peaks available to analyze")
                return
            times = self.peaks['times']
            region = "PVN"

        # Calculate intervals
        interval_stats = self.calculate_intervals(times)

        if interval_stats is None:
            messagebox.showinfo("Analysis Error", "Need at least two peaks to calculate intervals")
            return

        # Show the plot frame
        self.peak_interval_plot_frame.grid()

        # Clear previous plot
        self.peak_interval_fig.clear()

        # Create plot with two subplots
        ax1 = self.peak_interval_fig.add_subplot(211)
        ax2 = self.peak_interval_fig.add_subplot(212)

        # Histogram of intervals
        ax1.hist(interval_stats['intervals'], bins=20, alpha=0.7, color='blue')
        ax1.set_title(f"{region} Inter-Peak Intervals")
        ax1.set_xlabel("Interval (seconds)")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)

        # Plot intervals over time to show rhythm patterns
        x = np.arange(len(interval_stats['intervals']))
        ax2.plot(x, interval_stats['intervals'], 'o-', color='green')
        ax2.set_title("Interval Pattern")
        ax2.set_xlabel("Peak Number")
        ax2.set_ylabel("Interval to Next Peak (seconds)")
        ax2.grid(True, alpha=0.3)

        # Add a horizontal line for mean interval
        ax2.axhline(interval_stats['mean'], color='red', linestyle='--', label=f"Mean: {interval_stats['mean']:.2f}s")
        ax2.legend()

        # Update canvas
        self.peak_interval_fig.tight_layout()
        self.peak_interval_canvas.draw()

        # Add statistics text
        if hasattr(self, 'peak_stats_label'):
            self.peak_stats_label.destroy()

        stats_text = (
            f"Number of Intervals: {interval_stats['count']}\n"
            f"Mean Interval: {interval_stats['mean']:.3f} seconds\n"
            f"Median Interval: {interval_stats['median']:.3f} seconds\n"
            f"Std Dev: {interval_stats['std']:.3f} seconds\n"
            f"Range: {interval_stats['min']:.3f} - {interval_stats['max']:.3f} seconds"
        )

        self.peak_stats_label = tk.Label(self.peak_interval_plot_frame, text=stats_text,
                                         font=('Arial', self.font_sizes['base']),
                                         justify=tk.LEFT)
        self.peak_stats_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        # Add export button
        if hasattr(self, 'peak_export_interval_btn'):
            self.peak_export_interval_btn.destroy()

        self.peak_export_interval_btn = tk.Button(
            self.peak_interval_plot_frame,
            text="Export Interval Data",
            font=('Arial', self.font_sizes['button']),
            command=lambda: self.export_interval_data(interval_stats, region, "peak")
        )
        self.peak_export_interval_btn.pack(side=tk.BOTTOM, padx=10, pady=5)

    def create_peak_detection_tab(self):
        """Create the peak detection tab with controls"""
        # Create the tab
        self.peak_tab = ttk.Frame(self.control_notebook)
        self.control_notebook.add(self.peak_tab, text="Peak Detection")

        # Add these two lines to define the font sizes
        font_size = self.font_sizes['base']
        slider_font_size = self.font_sizes['slider']

        # Configure grid layout
        self.peak_tab.columnconfigure(1, weight=1)

        # --- Peak Parameter Sliders ---
        tk.Label(self.peak_tab, text="Peak Prominence:", font=('Arial', font_size)).grid(
            row=0, column=0, sticky="w", padx=10, pady=5)
        self.peak_prominence_slider = tk.Scale(self.peak_tab, from_=0.1, to=20.0,
                                               resolution=0.1, orient=tk.HORIZONTAL,
                                               length=800, width=20, font=('Arial', slider_font_size))
        self.peak_prominence_slider.set(10.0)  # Higher default value
        self.peak_prominence_slider.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        # Width slider
        tk.Label(self.peak_tab, text="Min Peak Width (s):", font=('Arial', font_size)).grid(
            row=1, column=0, sticky="w", padx=10, pady=5)
        self.peak_width_slider = tk.Scale(self.peak_tab, from_=0.0, to=10.0,
                                          resolution=0.1, orient=tk.HORIZONTAL,
                                          length=800, width=20, font=('Arial', slider_font_size))
        self.peak_width_slider.set(2.0)
        self.peak_width_slider.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # Distance slider
        tk.Label(self.peak_tab, text="Min Peak Distance (s):", font=('Arial', font_size)).grid(
            row=2, column=0, sticky="w", padx=10, pady=5)
        self.peak_distance_slider = tk.Scale(self.peak_tab, from_=0.0, to=30.0,
                                             resolution=0.5, orient=tk.HORIZONTAL,
                                             length=800, width=20, font=('Arial', slider_font_size))
        self.peak_distance_slider.set(5.0)
        self.peak_distance_slider.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        # Threshold slider
        tk.Label(self.peak_tab, text="Peak Threshold:", font=('Arial', font_size)).grid(
            row=3, column=0, sticky="w", padx=10, pady=5)
        self.peak_threshold_slider = tk.Scale(self.peak_tab, from_=0.0, to=20.0,
                                              resolution=0.5, orient=tk.HORIZONTAL,
                                              length=800, width=20, font=('Arial', slider_font_size))
        self.peak_threshold_slider.set(5.0)
        self.peak_threshold_slider.grid(row=3, column=1, padx=10, pady=5, sticky="ew")

        # --- Signal Selection Frame --- (row 4)
        signal_frame = tk.LabelFrame(self.peak_tab, text="Signal Selection", font=('Arial', font_size, 'bold'))
        signal_frame.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        # Initialize signal selection variable
        self.signal_selection_var = tk.StringVar(value="Primary")

        # Create radio buttons for signal selection
        signal_options_frame = tk.Frame(signal_frame)
        signal_options_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Radiobutton(signal_options_frame,
                       text="PVN (Primary)",
                       variable=self.signal_selection_var,
                       value="Primary",
                       font=('Arial', font_size)).pack(side=tk.LEFT, padx=20)

        tk.Radiobutton(signal_options_frame,
                       text="SON (Secondary)",
                       variable=self.signal_selection_var,
                       value="Secondary",
                       font=('Arial', font_size)).pack(side=tk.LEFT, padx=20)

        tk.Radiobutton(signal_options_frame,
                       text="Both",
                       variable=self.signal_selection_var,
                       value="Both",
                       font=('Arial', font_size)).pack(side=tk.LEFT, padx=20)

        # --- Action Buttons --- (row 5)
        btn_frame = tk.Frame(self.peak_tab)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=5, sticky="w")

        # Detect peaks button
        self.run_peaks_btn = tk.Button(
            btn_frame,
            text="Detect Peaks",
            font=('Arial', self.font_sizes['button']),
            bg='lightgreen',
            command=self.run_peak_detection_from_tab
        )
        self.run_peaks_btn.pack(side=tk.LEFT, padx=10)

        # Clear peaks button
        self.clear_peaks_btn = tk.Button(
            btn_frame,
            text="Clear Peaks",
            font=('Arial', self.font_sizes['button']),
            command=self.clear_peaks
        )
        self.clear_peaks_btn.pack(side=tk.LEFT, padx=10)

        # Manual peak selection button
        self.manual_peak_btn = tk.Button(
            btn_frame,
            text="Manual Peak Selection",
            font=('Arial', self.font_sizes['button']),
            bg='lightblue',
            command=self.toggle_manual_peak_mode
        )
        self.manual_peak_btn.pack(side=tk.LEFT, padx=10)

        # --- Display Options --- (row 6)
        options_frame = tk.Frame(self.peak_tab)
        options_frame.grid(row=6, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        # Show peaks checkbox
        self.show_peaks_var = tk.BooleanVar(value=True)
        self.show_peaks_check = tk.Checkbutton(
            options_frame,
            text="Show Peaks",
            variable=self.show_peaks_var,
            font=('Arial', font_size),
            command=self.update_peak_visibility
        )
        self.show_peaks_check.pack(side=tk.LEFT, padx=5)

        # Show peak labels checkbox
        self.show_peak_labels_var = tk.BooleanVar(value=False)
        self.show_peak_labels_check = tk.Checkbutton(
            options_frame,
            text="Show Labels",
            variable=self.show_peak_labels_var,
            font=('Arial', font_size),
            command=self.update_peak_visibility
        )
        self.show_peak_labels_check.pack(side=tk.LEFT, padx=20)

        # Add feature visualization checkbox
        self.peak_feature_viz_var = tk.BooleanVar(value=False)
        self.peak_feature_viz_check = tk.Checkbutton(
            options_frame,
            text="Show Peak Features",
            variable=self.peak_feature_viz_var,
            font=('Arial', font_size),
            command=self.toggle_peak_feature_visualization
        )
        self.peak_feature_viz_check.pack(side=tk.LEFT, padx=20)

        # --- Peak Metrics Display --- (row 7)
        metrics_frame = tk.LabelFrame(self.peak_tab, text="Peak Metrics", font=('Arial', font_size, 'bold'))
        metrics_frame.grid(row=7, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        columns = ("Index", "Time (s)", "Height (%)", "Width (s)", "Area", "Rise (s)", "Decay (s)")
        self.peak_metrics_tree = ttk.Treeview(metrics_frame, columns=columns, show="headings", height=5)
        for col in columns:
            self.peak_metrics_tree.heading(col, text=col)
            self.peak_metrics_tree.column(col, width=90, anchor='center')
        scrollbar = ttk.Scrollbar(metrics_frame, orient="vertical", command=self.peak_metrics_tree.yview)
        self.peak_metrics_tree.configure(yscrollcommand=scrollbar.set)
        self.peak_metrics_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        # --- Export Button --- (row 8)
        export_frame = tk.Frame(self.peak_tab)
        export_frame.grid(row=8, column=0, columnspan=2, pady=5, sticky="w")
        self.export_peaks_btn = tk.Button(
            export_frame,
            text="Export Peak Data",
            font=('Arial', self.font_sizes['button']),
            command=self.export_peak_data
        )
        self.export_peaks_btn.pack(side=tk.LEFT, padx=10)

        # Add peak-specific initialize variables
        if not hasattr(self, 'peaks'): self.peaks = None
        if not hasattr(self, 'peak_metrics'): self.peak_metrics = None
        if not hasattr(self, 'peak_annotations'): self.peak_annotations = []
        if not hasattr(self, 'peak_lines'): self.peak_lines = []
        if not hasattr(self, 'peak_feature_annotations'): self.peak_feature_annotations = []
        if not hasattr(self, 'secondary_peaks'): self.secondary_peaks = None
        if not hasattr(self, 'secondary_peak_metrics'): self.secondary_peak_metrics = None

        # IPI Analysis Frame - INDENT THIS BLOCK
        ipi_frame = tk.LabelFrame(self.peak_tab, text="Inter-Peak Interval Analysis", font=('Arial', font_size, 'bold'))
        ipi_frame.grid(row=9, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        # Add button to generate IPI analysis
        self.ipi_btn = tk.Button(
            ipi_frame,
            text="Analyze Intervals",
            font=('Arial', self.font_sizes['button']),
            bg='lightblue',
            command=self.analyze_peak_intervals
        )
        self.ipi_btn.pack(side=tk.LEFT, padx=10, pady=5)

        # Add plot frame for interval analysis
        self.peak_interval_plot_frame = tk.Frame(self.peak_tab)
        self.peak_interval_plot_frame.grid(row=10, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        self.peak_interval_plot_frame.grid_remove()  # Hide initially

        # Create matplotlib figure and canvas for interval plot
        self.peak_interval_fig = plt.Figure(figsize=(8, 6))
        self.peak_interval_canvas = FigureCanvasTkAgg(self.peak_interval_fig, master=self.peak_interval_plot_frame)
        self.peak_interval_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        self.peak_interval_toolbar = NavigationToolbar2Tk(self.peak_interval_canvas, self.peak_interval_plot_frame)
        self.peak_interval_toolbar.update()

    # Inside the PhotometryViewer class

    # 2. Create a similar modified valley detection tab with higher default values
    def create_valley_detection_tab(self):
        """Create the valley detection tab with controls"""
        # Create the tab
        self.valley_tab = ttk.Frame(self.control_notebook)
        self.control_notebook.add(self.valley_tab, text="Valley Detection")

        # Add these two lines to define the font sizes
        font_size = self.font_sizes['base']
        slider_font_size = self.font_sizes['slider']

        # Configure grid layout
        self.valley_tab.columnconfigure(1, weight=1)

        # --- Valley Parameter Sliders ---
        tk.Label(self.valley_tab, text="Valley Prominence:", font=('Arial', font_size)).grid(
            row=0, column=0, sticky="w", padx=10, pady=5)
        self.valley_prominence_slider = tk.Scale(self.valley_tab, from_=0.1, to=20.0,
                                                 resolution=0.1, orient=tk.HORIZONTAL,
                                                 length=800, width=20, font=('Arial', slider_font_size))
        self.valley_prominence_slider.set(8.0)  # Higher default value
        self.valley_prominence_slider.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        # Width slider
        tk.Label(self.valley_tab, text="Min Valley Width (s):", font=('Arial', font_size)).grid(
            row=1, column=0, sticky="w", padx=10, pady=5)
        self.valley_width_slider = tk.Scale(self.valley_tab, from_=0.0, to=10.0,
                                            resolution=0.1, orient=tk.HORIZONTAL,
                                            length=800, width=20, font=('Arial', slider_font_size))
        self.valley_width_slider.set(2.0)
        self.valley_width_slider.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # Distance slider
        tk.Label(self.valley_tab, text="Min Valley Distance (s):", font=('Arial', font_size)).grid(
            row=2, column=0, sticky="w", padx=10, pady=5)
        self.valley_distance_slider = tk.Scale(self.valley_tab, from_=0.0, to=30.0,
                                               resolution=0.5, orient=tk.HORIZONTAL,
                                               length=800, width=20, font=('Arial', slider_font_size))
        self.valley_distance_slider.set(5.0)
        self.valley_distance_slider.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        # --- Signal Selection Frame --- (row 3)
        signal_frame = tk.LabelFrame(self.valley_tab, text="Signal Selection", font=('Arial', font_size, 'bold'))
        signal_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky="ew")

        # Create separate valley signal selection variable
        self.valley_signal_selection_var = tk.StringVar(value="Primary")

        # Create radio buttons for signal selection
        signal_options_frame = tk.Frame(signal_frame)
        signal_options_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Radiobutton(signal_options_frame,
                       text="PVN (Primary)",
                       variable=self.valley_signal_selection_var,
                       value="Primary",
                       font=('Arial', font_size)).pack(side=tk.LEFT, padx=20)

        tk.Radiobutton(signal_options_frame,
                       text="SON (Secondary)",
                       variable=self.valley_signal_selection_var,
                       value="Secondary",
                       font=('Arial', font_size)).pack(side=tk.LEFT, padx=20)

        tk.Radiobutton(signal_options_frame,
                       text="Both",
                       variable=self.valley_signal_selection_var,
                       value="Both",
                       font=('Arial', font_size)).pack(side=tk.LEFT, padx=20)

        # --- Action Buttons --- (row 4)
        btn_frame = tk.Frame(self.valley_tab)
        btn_frame.grid(row=4, column=0, columnspan=2, pady=5, sticky="w")

        # Detect valleys button
        self.run_valleys_btn = tk.Button(
            btn_frame,
            text="Detect Valleys",
            font=('Arial', self.font_sizes['button']),
            bg='lightgreen',
            command=self.run_valley_detection_from_tab
        )
        self.run_valleys_btn.pack(side=tk.LEFT, padx=10)

        # Clear valleys button
        self.clear_valleys_btn = tk.Button(
            btn_frame,
            text="Clear Valleys",
            font=('Arial', self.font_sizes['button']),
            command=self.clear_valleys
        )
        self.clear_valleys_btn.pack(side=tk.LEFT, padx=10)

        # Manual valley selection button
        self.manual_valley_btn = tk.Button(
            btn_frame,
            text="Manual Valley Selection",
            font=('Arial', self.font_sizes['button']),
            bg='lightblue',
            command=self.toggle_manual_valley_mode
        )
        self.manual_valley_btn.pack(side=tk.LEFT, padx=10)

        # --- Display Options --- (row 5)
        options_frame = tk.Frame(self.valley_tab)
        options_frame.grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=5)

        # Show valleys checkbox
        self.show_valleys_var = tk.BooleanVar(value=True)
        self.show_valleys_check = tk.Checkbutton(
            options_frame,
            text="Show Valleys",
            variable=self.show_valleys_var,
            font=('Arial', font_size),
            command=self.update_valley_visibility
        )
        self.show_valleys_check.pack(side=tk.LEFT, padx=5)

        # Show valley labels checkbox
        self.show_valley_labels_var = tk.BooleanVar(value=False)
        self.show_valley_labels_check = tk.Checkbutton(
            options_frame,
            text="Show Labels",
            variable=self.show_valley_labels_var,
            font=('Arial', font_size),
            command=self.update_valley_visibility
        )
        self.show_valley_labels_check.pack(side=tk.LEFT, padx=20)

        # IPI Analysis Frame
        ivi_frame = tk.LabelFrame(self.valley_tab, text="Inter-Valley Interval Analysis",
                                  font=('Arial', font_size, 'bold'))
        ivi_frame.grid(row=9, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        # Add button to generate IVI analysis
        self.ivi_btn = tk.Button(
            ivi_frame,
            text="Analyze Valley Intervals",
            font=('Arial', self.font_sizes['button']),
            bg='lightblue',
            command=self.analyze_valley_intervals  # Updated to use the correct method
        )
        self.ivi_btn.pack(side=tk.LEFT, padx=10, pady=5)

        # Add plot frame for valley interval analysis
        self.valley_interval_plot_frame = tk.Frame(self.valley_tab)
        self.valley_interval_plot_frame.grid(row=10, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        self.valley_interval_plot_frame.grid_remove()  # Hide initially

        # Create matplotlib figure and canvas for interval plot
        self.valley_interval_fig = plt.Figure(figsize=(8, 6))
        self.valley_interval_canvas = FigureCanvasTkAgg(self.valley_interval_fig,
                                                        master=self.valley_interval_plot_frame)
        self.valley_interval_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        self.valley_interval_toolbar = NavigationToolbar2Tk(self.valley_interval_canvas,
                                                            self.valley_interval_plot_frame)
        self.valley_interval_toolbar.update()

        # --- Valley Metrics Display --- (row 6)
        metrics_frame = tk.LabelFrame(self.valley_tab, text="Valley Metrics", font=('Arial', font_size, 'bold'))
        metrics_frame.grid(row=6, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        valley_columns = ("Index", "Time (s)", "Depth (%)", "Width (s)", "Area Above")
        self.valley_metrics_tree = ttk.Treeview(metrics_frame, columns=valley_columns, show="headings", height=5)
        for col in valley_columns:
            self.valley_metrics_tree.heading(col, text=col)
            self.valley_metrics_tree.column(col, width=90, anchor='center')
        scrollbar = ttk.Scrollbar(metrics_frame, orient="vertical", command=self.valley_metrics_tree.yview)
        self.valley_metrics_tree.configure(yscrollcommand=scrollbar.set)
        self.valley_metrics_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

        # --- Export Button --- (row 7)
        export_frame = tk.Frame(self.valley_tab)
        export_frame.grid(row=7, column=0, columnspan=2, pady=5, sticky="w")
        self.export_valleys_btn = tk.Button(
            export_frame,
            text="Export Valley Data",
            font=('Arial', self.font_sizes['button']),
            command=self.export_valley_data
        )
        self.export_valleys_btn.pack(side=tk.LEFT, padx=10)

        # Initialize valley-specific variables
        if not hasattr(self, 'valleys'): self.valleys = None
        if not hasattr(self, 'valley_metrics'): self.valley_metrics = None
        if not hasattr(self, 'valley_annotations'): self.valley_annotations = []
        if not hasattr(self, 'valley_lines'): self.valley_lines = []
        if not hasattr(self, 'secondary_valleys'): self.secondary_valleys = None
        if not hasattr(self, 'secondary_valley_metrics'): self.secondary_valley_metrics = None

    def analyze_valley_intervals(self):
        """Analyze intervals between consecutive valleys in the integrated plot frame"""
        # Determine which valleys to analyze based on signal selection
        if self.valley_signal_selection_var.get() == "Secondary" and hasattr(self,
                                                                             'secondary_valleys') and self.secondary_valleys is not None:
            times = self.secondary_valleys['times']
            region = "SON"
        else:
            if self.valleys is None or len(self.valleys.get('times', [])) == 0:
                messagebox.showinfo("Analysis Error", "No valleys available to analyze")
                return
            times = self.valleys['times']
            region = "PVN"

        # Calculate intervals
        interval_stats = self.calculate_intervals(times)

        if interval_stats is None:
            messagebox.showinfo("Analysis Error", "Need at least two valleys to calculate intervals")
            return

        # Show the plot frame
        self.valley_interval_plot_frame.grid()

        # Clear previous plot
        self.valley_interval_fig.clear()

        # Create plot with two subplots
        ax1 = self.valley_interval_fig.add_subplot(211)
        ax2 = self.valley_interval_fig.add_subplot(212)

        # Histogram of intervals
        ax1.hist(interval_stats['intervals'], bins=20, alpha=0.7, color='purple')
        ax1.set_title(f"{region} Inter-Valley Intervals")
        ax1.set_xlabel("Interval (seconds)")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)

        # Plot intervals over time to show rhythm patterns
        x = np.arange(len(interval_stats['intervals']))
        ax2.plot(x, interval_stats['intervals'], 'o-', color='green')
        ax2.set_title("Interval Pattern")
        ax2.set_xlabel("Valley Number")
        ax2.set_ylabel("Interval to Next Valley (seconds)")
        ax2.grid(True, alpha=0.3)

        # Add a horizontal line for mean interval
        ax2.axhline(interval_stats['mean'], color='red', linestyle='--', label=f"Mean: {interval_stats['mean']:.2f}s")
        ax2.legend()

        # Update canvas
        self.valley_interval_fig.tight_layout()
        self.valley_interval_canvas.draw()

        # Add statistics text
        if hasattr(self, 'valley_stats_label'):
            self.valley_stats_label.destroy()

        stats_text = (
            f"Number of Intervals: {interval_stats['count']}\n"
            f"Mean Interval: {interval_stats['mean']:.3f} seconds\n"
            f"Median Interval: {interval_stats['median']:.3f} seconds\n"
            f"Std Dev: {interval_stats['std']:.3f} seconds\n"
            f"Range: {interval_stats['min']:.3f} - {interval_stats['max']:.3f} seconds"
        )

        self.valley_stats_label = tk.Label(self.valley_interval_plot_frame, text=stats_text,
                                           font=('Arial', self.font_sizes['base']),
                                           justify=tk.LEFT)
        self.valley_stats_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        # Add export button
        if hasattr(self, 'valley_export_interval_btn'):
            self.valley_export_interval_btn.destroy()

        self.valley_export_interval_btn = tk.Button(
            self.valley_interval_plot_frame,
            text="Export Interval Data",
            font=('Arial', self.font_sizes['button']),
            command=lambda: self.export_interval_data(interval_stats, region, "valley")
        )
        self.valley_export_interval_btn.pack(side=tk.BOTTOM, padx=10, pady=5)

    def detect_peaks_valleys(self):
        """Detect peaks and valleys in the currently selected signal"""
        # Determine which signal to analyze
        if self.signal_selection_var.get() == "Secondary" and hasattr(self, 'secondary_processed_signal'):
            signal = self.secondary_processed_signal
            time = self.secondary_processed_time
        else:
            signal = self.processed_signal
            time = self.processed_time

        # Get detection parameters
        prominence = self.prominence_slider.get()

        # Convert width and distance from seconds to sample points
        dt = np.mean(np.diff(time))
        width = None if self.width_slider.get() <= 0 else self.width_slider.get() / dt
        distance = None if self.distance_slider.get() <= 0 else int(self.distance_slider.get() / dt)

        # Get threshold (or None if set to 0)
        threshold = None if self.threshold_slider.get() <= 0 else self.threshold_slider.get()

        # Find peaks and valleys
        self.peaks, self.valleys = find_peaks_valleys(
            signal, time, prominence=prominence, width=width,
            distance=distance, threshold=threshold
        )

        # Calculate additional metrics
        self.peak_metrics = calculate_peak_metrics(self.peaks, self.valleys, signal, time)

        # Update the display
        self.update_peak_display()
        self.update_metrics_display()

        # Show status
        if hasattr(self, 'status_label'):
            self.status_label.config(
                text=f"Detected {len(self.peaks['indices'])} peaks and {len(self.valleys['indices'])} valleys",
                fg="green"
            )

    def export_interval_data(self, interval_stats, region, event_type):
        """Export interval data to CSV file"""
        if interval_stats is None:
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title=f"Export {region} {event_type.capitalize()} Interval Data"
        )

        if not file_path:
            return  # User canceled

        try:
            import csv

            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                writer.writerow(["Interval Number", "Duration (seconds)"])

                # Write interval data
                for i, interval in enumerate(interval_stats['intervals']):
                    writer.writerow([i + 1, f"{interval:.4f}"])

                # Add summary statistics
                writer.writerow([])
                writer.writerow(["Summary Statistics"])
                writer.writerow(["Mean Interval (s)", f"{interval_stats['mean']:.4f}"])
                writer.writerow(["Median Interval (s)", f"{interval_stats['median']:.4f}"])
                writer.writerow(["Std Dev (s)", f"{interval_stats['std']:.4f}"])
                writer.writerow(["Min Interval (s)", f"{interval_stats['min']:.4f}"])
                writer.writerow(["Max Interval (s)", f"{interval_stats['max']:.4f}"])
                writer.writerow(["Number of Intervals", interval_stats['count']])

            messagebox.showinfo("Export Complete",
                                f"{region} {event_type.capitalize()} interval data successfully exported to {file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting interval data: {str(e)}")

    def update_peak_detection(self, _=None):
        """Update peak detection with current parameters"""
        if self.peaks is not None:
            self.detect_peaks_valleys()

    def clear_peaks(self):
        """Clear all detected peaks and valleys"""
        self.peaks = None
        self.valleys = None
        self.peak_metrics = None

        # Clear displayed annotations
        self.clear_peak_annotations()

        # Clear metrics display
        self.metrics_tree.delete(*self.metrics_tree.get_children())

        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Peaks and valleys cleared", fg="green")

    def clear_peak_annotations(self):
        """Clear peak and valley annotations from the plot"""
        # Remove peak markers
        for annotation in self.peak_annotations:
            annotation.remove() if hasattr(annotation, 'remove') else None
        self.peak_annotations = []

        # Remove valley markers
        for annotation in self.valley_annotations:
            annotation.remove() if hasattr(annotation, 'remove') else None
        self.valley_annotations = []

        # Remove peak and valley lines
        for line in self.peak_lines + self.valley_lines:
            line.remove() if hasattr(line, 'remove') else None
        self.peak_lines = []
        self.valley_lines = []

        # Redraw canvas
        self.canvas.draw_idle()

    def update_peak_display(self):
        """Update the display of peaks on the plot."""
        # Clear previous peak annotations
        for annotation in self.peak_annotations:
            if hasattr(annotation, 'remove'):
                annotation.remove()
        self.peak_annotations = []

        for line in self.peak_lines:
            if hasattr(line, 'remove'):
                line.remove()
        self.peak_lines = []

        # Clear peak feature annotations
        for annotation in self.peak_feature_annotations:
            if hasattr(annotation, 'remove'):
                annotation.remove()
        self.peak_feature_annotations = []

        # Exit if nothing to show or peaks are hidden
        if not self.show_peaks_var.get():
            self.canvas.draw_idle()
            return

        # Display primary peaks if available and selected
        if self.peaks is not None and self.signal_selection_var.get() in ["Primary", "Both"]:
            self._display_peaks_on_signal(
                self.peaks,
                self.processed_signal,
                self.processed_time,
                'red',  # color
                primary=True
            )

        # Display secondary peaks if available and selected
        if hasattr(self, 'secondary_peaks') and self.secondary_peaks is not None and \
                self.signal_selection_var.get() in ["Secondary", "Both"]:

            # Check if secondary signal is available
            if hasattr(self, 'secondary_processed_signal') and self.secondary_processed_signal is not None:
                self._display_peaks_on_signal(
                    self.secondary_peaks,
                    self.secondary_processed_signal,
                    self.secondary_processed_time,
                    'orange',  # different color
                    primary=False
                )

        # Show peak features if enabled
        if self.peak_feature_viz_var.get():
            self.visualize_peak_features()

        # Redraw
        self.canvas.draw_idle()

    def _display_peaks_on_signal(self, peaks, signal, time_array, color, primary=True):
        """Helper method to display peaks on a specific signal"""
        if peaks is None or 'times' not in peaks or len(peaks['times']) == 0:
            return

        # Convert peak times to minutes for plotting
        peak_times_min = peaks['times'] / 60

        # Plot peak markers with appropriate color and label
        signal_label = "PVN" if primary else "SON"
        peaks_scatter = self.ax1.scatter(
            peak_times_min,
            peaks['heights'],
            marker='^',
            color=color,
            s=100,
            zorder=10,
            picker=5,
            label=f"{signal_label} Peaks"
        )
        self.peak_annotations.append(peaks_scatter)

        # Add peak labels if enabled
        if self.show_peak_labels_var.get():
            for i, (t, h) in enumerate(zip(peak_times_min, peaks['heights'])):
                prefix = "P" if primary else "SP"  # P for Primary/PVN, SP for Secondary/SON
                label = self.ax1.annotate(
                    f"{prefix}{i + 1}",
                    (t, h),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                )
                self.peak_annotations.append(label)

        # Add vertical lines for peaks
        for i, t in enumerate(peak_times_min):
            line = self.ax1.axvline(
                t,
                color=color,
                linestyle='--',
                alpha=0.3,
                zorder=5
            )
            self.peak_lines.append(line)

    def _display_valleys_on_signal(self, valleys, signal, time_array, color, primary=True):
        """Helper method to display valleys on a specific signal"""
        if valleys is None or 'times' not in valleys or len(valleys['times']) == 0:
            return

        # Convert valley times to minutes for plotting
        valley_times_min = valleys['times'] / 60

        # Plot valley markers with appropriate color and label
        signal_label = "PVN" if primary else "SON"
        valleys_scatter = self.ax1.scatter(
            valley_times_min,
            valleys['depths'],
            marker='v',
            color=color,
            s=100,
            zorder=10,
            picker=5,
            label=f"{signal_label} Valleys"
        )
        self.valley_annotations.append(valleys_scatter)

        # Add valley labels if enabled
        if self.show_valley_labels_var.get():
            for i, (t, d) in enumerate(zip(valley_times_min, valleys['depths'])):
                prefix = "V" if primary else "SV"  # V for Primary/PVN, SV for Secondary/SON
                label = self.ax1.annotate(
                    f"{prefix}{i + 1}",
                    (t, d),
                    xytext=(0, -15),
                    textcoords="offset points",
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                )
                self.valley_annotations.append(label)

        # Add vertical lines for valleys
        for i, t in enumerate(valley_times_min):
            line = self.ax1.axvline(
                t,
                color=color,
                linestyle=':',
                alpha=0.4,
                zorder=5
            )
            self.valley_lines.append(line)

    def update_peak_visibility(self):
        """Update visibility of peaks and valleys based on checkbox settings"""
        if self.peaks is not None or self.valleys is not None:
            self.update_peak_display()

    def update_metrics_display(self):
        """Update the peak metrics tree view"""
        # Clear existing data
        if not hasattr(self, 'peak_metrics_tree'):
            print("Peak metrics tree not found")
            return

        self.peak_metrics_tree.delete(*self.peak_metrics_tree.get_children())

        # Debug output
        if self.peaks is None:
            print("No peaks detected")
            return
        if self.peak_metrics is None:
            print("No peak metrics calculated")
            return

        print(f"Displaying metrics for {len(self.peaks['indices'])} peaks")

        # Add data to tree
        for i, (idx, time_s, height) in enumerate(zip(
                self.peaks['indices'], self.peaks['times'], self.peaks['heights'])):

            # Safely get metrics
            fwhm = self.peak_metrics['full_width_half_max'][i] if i < len(
                self.peak_metrics['full_width_half_max']) else 0.0
            area = self.peak_metrics['area_under_curve'][i] if i < len(self.peak_metrics['area_under_curve']) else 0.0
            rise = self.peak_metrics['rise_time'][i] if i < len(self.peak_metrics['rise_time']) else 0.0
            decay = self.peak_metrics['decay_time'][i] if i < len(self.peak_metrics['decay_time']) else 0.0

            # Format values - use "0.00" instead of "N/A" to match valley display
            values = (
                f"P{i + 1}",
                f"{time_s:.2f}",
                f"{height:.2f}",
                f"{fwhm:.2f}",
                f"{area:.2f}",
                f"{rise:.2f}",
                f"{decay:.2f}"
            )

            try:
                self.peak_metrics_tree.insert("", "end", values=values)
            except Exception as e:
                print(f"Error inserting peak metric row: {e}, values: {values}")

    def _add_peak_metrics_to_tree(self, peaks, metrics, prefix=""):
        """Helper to add peak metrics to the tree with appropriate prefix"""
        for i, (idx, time_val, height) in enumerate(zip(
                peaks['indices'], peaks['times'], peaks['heights'])):
            # Get metrics for this peak safely
            fwhm = metrics['full_width_half_max'][i] if i < len(metrics['full_width_half_max']) else np.nan
            area = metrics['area_under_curve'][i] if i < len(metrics['area_under_curve']) else np.nan
            rise = metrics['rise_time'][i] if i < len(metrics['rise_time']) else np.nan
            decay = metrics['decay_time'][i] if i < len(metrics['decay_time']) else np.nan

            # Format values
            values = (
                f"{prefix}{i + 1}",  # Add prefix for Primary/Secondary distinction
                f"{time_val:.2f}",
                f"{height:.2f}",
                f"{fwhm:.2f}" if not np.isnan(fwhm) else "N/A",
                f"{area:.2f}" if not np.isnan(area) else "N/A",
                f"{rise:.2f}" if not np.isnan(rise) else "N/A",
                f"{decay:.2f}" if not np.isnan(decay) else "N/A"
            )

            # Add to tree
            self.peak_metrics_tree.insert("", "end", values=values)

    def export_peak_data(self):
        """Export peak data to CSV file"""
        if self.peaks is None or self.peak_metrics is None:
            messagebox.showinfo("Export Error", "No peak data available to export")
            return

        # Get file path
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Peak Data"
        )

        if not file_path:
            return  # User canceled

        try:
            import csv

            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write header
                writer.writerow(["Peak #", "Time (s)", "Height (%)", "Width (s)",
                                 "Area", "Rise Time (s)", "Decay Time (s)",
                                 "Preceding Valley Index", "Following Valley Index"])

                # Write data for each peak
                for i, (idx, time, height) in enumerate(zip(
                        self.peaks['indices'],
                        self.peaks['times'],
                        self.peaks['heights'])):

                    # Get metrics for this peak
                    if i < len(self.peak_metrics['full_width_half_max']):
                        fwhm = self.peak_metrics['full_width_half_max'][i]
                        area = self.peak_metrics['area_under_curve'][i]
                        rise = self.peak_metrics['rise_time'][i]
                        decay = self.peak_metrics['decay_time'][i]
                        preceding = self.peak_metrics['preceding_valley'][i]
                        following = self.peak_metrics['following_valley'][i]
                    else:
                        fwhm = area = rise = decay = preceding = following = np.nan

                    # Handle NaN values
                    fwhm_str = f"{fwhm:.4f}" if not np.isnan(fwhm) else "N/A"
                    area_str = f"{area:.4f}" if not np.isnan(area) else "N/A"
                    rise_str = f"{rise:.4f}" if not np.isnan(rise) else "N/A"
                    decay_str = f"{decay:.4f}" if not np.isnan(decay) else "N/A"

                    # Write row
                    writer.writerow([
                        i + 1,
                        f"{time:.4f}",
                        f"{height:.4f}",
                        fwhm_str,
                        area_str,
                        rise_str,
                        decay_str,
                        preceding,
                        following
                    ])

                # Add summary statistics
                writer.writerow([])
                writer.writerow(["Summary Statistics"])

                # Calculate average metrics (excluding NaN values)
                avg_height = np.nanmean(self.peaks['heights'])
                avg_width = np.nanmean(self.peak_metrics['full_width_half_max'])
                avg_area = np.nanmean(self.peak_metrics['area_under_curve'])
                avg_rise = np.nanmean(self.peak_metrics['rise_time'])
                avg_decay = np.nanmean(self.peak_metrics['decay_time'])

                writer.writerow(["Average Height (%)", f"{avg_height:.4f}"])
                writer.writerow(["Average Width (s)", f"{avg_width:.4f}"])
                writer.writerow(["Average Area", f"{avg_area:.4f}"])
                writer.writerow(["Average Rise Time (s)", f"{avg_rise:.4f}"])
                writer.writerow(["Average Decay Time (s)", f"{avg_decay:.4f}"])
                writer.writerow(["Total Peaks", len(self.peaks['indices'])])
                writer.writerow(["Total Valleys", len(self.valleys['indices'])])

            messagebox.showinfo("Export Complete", f"Peak data successfully exported to {file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting peak data: {str(e)}")

    def toggle_manual_valley_mode(self):
        """Toggle manual valley selection mode"""
        self.manual_valley_mode = not self.manual_valley_mode
        self.manual_peak_mode = False  # Turn off peak mode if it was on

        if self.manual_valley_mode:
            # Use correct button reference
            self.manual_valley_btn.config(text="Cancel Valley Selection", bg='salmon')
            self.manual_peak_btn.config(state=tk.DISABLED)
            self.connect_selection_events("valley")

            # Show instructions
            if hasattr(self, 'status_label'):
                self.status_label.config(
                    text="Click on the signal to add a valley. Right-click a valley to remove it.",
                    fg="blue"
                )
        else:
            # Disable selection mode
            self.manual_valley_btn.config(text="Manual Valley Selection", bg='lightblue')
            self.manual_peak_btn.config(state=tk.NORMAL)
            self.disconnect_selection_events()

            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Manual valley selection mode disabled", fg="green")

    def connect_selection_events(self, mode):
        """Connect mouse events for manual peak/valley selection"""
        self.canvas.get_tk_widget().config(cursor="crosshair")

        # Connect click event for adding peaks/valleys
        self.selection_cid = self.canvas.mpl_connect('button_press_event',
                                                     lambda event: self.on_selection_click(event, mode))

        # Connect pick event for removing peaks/valleys
        self.pick_cid = self.canvas.mpl_connect('pick_event',
                                                lambda event: self.on_pick_event(event, mode))

    def disconnect_selection_events(self):
        """Disconnect manual selection events"""
        self.canvas.get_tk_widget().config(cursor="")

        if hasattr(self, 'selection_cid') and self.selection_cid is not None:
            self.canvas.mpl_disconnect(self.selection_cid)
            self.selection_cid = None

        if hasattr(self, 'pick_cid') and self.pick_cid is not None:
            self.canvas.mpl_disconnect(self.pick_cid)
            self.pick_cid = None

    def on_selection_click(self, event, mode):
        """Handle mouse click for adding peaks or valleys"""
        # Only handle left click in the main plot
        if event.button != 1 or event.inaxes != self.ax1:
            return

        # Get coordinates in data space
        x_min = event.xdata  # Time in minutes
        y = event.ydata  # Signal value
        x_sec = x_min * 60  # Convert to seconds

        # Determine which signal we're working with
        if self.signal_selection_var.get() == "Secondary" and hasattr(self, 'secondary_processed_signal'):
            signal = self.secondary_processed_signal
            time = self.secondary_processed_time
        else:
            signal = self.processed_signal
            time = self.processed_time

        # Find the closest time index
        idx = np.argmin(np.abs(time - x_sec))

        # Initialize peaks/valleys if not already done
        if self.peaks is None or self.valleys is None:
            # Initialize with empty arrays
            self.peaks = {
                'indices': np.array([], dtype=int),
                'times': np.array([]),
                'heights': np.array([]),
                'properties': {}
            }
            self.valleys = {
                'indices': np.array([], dtype=int),
                'times': np.array([]),
                'depths': np.array([]),
                'properties': {}
            }
            self.peak_metrics = {
                'area_under_curve': [],
                'full_width_half_max': [],
                'rise_time': [],
                'decay_time': [],
                'preceding_valley': [],
                'following_valley': []
            }

        # Add the new peak or valley
        if mode == "peak":
            # Add the peak
            self.peaks['indices'] = np.append(self.peaks['indices'], idx)
            self.peaks['times'] = np.append(self.peaks['times'], time[idx])
            self.peaks['heights'] = np.append(self.peaks['heights'], signal[idx])

            # Sort the peaks by time
            sort_indices = np.argsort(self.peaks['times'])
            self.peaks['indices'] = self.peaks['indices'][sort_indices]
            self.peaks['times'] = self.peaks['times'][sort_indices]
            self.peaks['heights'] = self.peaks['heights'][sort_indices]

            # Update peaks display
            self.status_label.config(text=f"Added peak at {time[idx]:.2f}s", fg="green")
        else:  # valley
            # Add the valley
            self.valleys['indices'] = np.append(self.valleys['indices'], idx)
            self.valleys['times'] = np.append(self.valleys['times'], time[idx])
            self.valleys['depths'] = np.append(self.valleys['depths'], signal[idx])

            # Sort the valleys by time
            sort_indices = np.argsort(self.valleys['times'])
            self.valleys['indices'] = self.valleys['indices'][sort_indices]
            self.valleys['times'] = self.valleys['times'][sort_indices]
            self.valleys['depths'] = self.valleys['depths'][sort_indices]

            # Update valleys display
            self.status_label.config(text=f"Added valley at {time[idx]:.2f}s", fg="green")

        # Recalculate peak metrics
        if len(self.peaks['indices']) > 0 and len(self.valleys['indices']) > 0:
            self.peak_metrics = calculate_peak_metrics(self.peaks, self.valleys, signal, time)

        # Update the display
        self.update_peak_display()
        self.update_metrics_display()

    def on_pick_event(self, event, mode):
        """Handle picking events for removing peaks or valleys"""
        # Check if it was a right-click
        if not hasattr(event, 'mouseevent') or event.mouseevent.button != 3:
            return

        # Get the artist that was picked
        artist = event.artist

        # Make sure we have valid scatter data
        if not hasattr(artist, 'get_offsets'):
            return

        # Get indices of the picked points
        ind = event.ind
        if len(ind) == 0:
            return

        # Use first selected point
        point_idx = ind[0]

        # Get coordinates
        offsets = artist.get_offsets()
        if point_idx >= len(offsets):
            return

        # Get x-coordinate (time in minutes)
        x_min = offsets[point_idx, 0]
        x_sec = x_min * 60

        if mode == "peak" and artist in self.peak_annotations:
            # Find the closest peak
            if hasattr(self, 'peaks') and self.peaks is not None and len(self.peaks['times']) > 0:
                closest_idx = np.argmin(np.abs(self.peaks['times'] - x_sec))
                self.remove_peak(closest_idx)

        elif mode == "valley" and artist in self.valley_annotations:
            # Find the closest valley
            if hasattr(self, 'valleys') and self.valleys is not None and len(self.valleys['times']) > 0:
                closest_idx = np.argmin(np.abs(self.valleys['times'] - x_sec))
                self.remove_valley(closest_idx)

    def remove_peak(self, ind):
        """Remove a peak by index"""
        if self.peaks is None or ind >= len(self.peaks['indices']):
            return

        # Get the time of the peak to be removed
        peak_time = self.peaks['times'][ind]

        # Remove the peak
        self.peaks['indices'] = np.delete(self.peaks['indices'], ind)
        self.peaks['times'] = np.delete(self.peaks['times'], ind)
        self.peaks['heights'] = np.delete(self.peaks['heights'], ind)

        # Recalculate peak metrics if we still have peaks and valleys
        if len(self.peaks['indices']) > 0 and self.valleys is not None and len(self.valleys['indices']) > 0:
            # Determine which signal we're working with
            if self.signal_selection_var.get() == "Secondary" and hasattr(self, 'secondary_processed_signal'):
                signal = self.secondary_processed_signal
                time = self.secondary_processed_time
            else:
                signal = self.processed_signal
                time = self.processed_time

            self.peak_metrics = calculate_peak_metrics(self.peaks, self.valleys, signal, time)

        # Update the display
        self.update_peak_display()
        self.update_metrics_display()

        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Removed peak at {peak_time:.2f}s", fg="green")

    def remove_valley(self, ind):
        """Remove a valley by index"""
        if self.valleys is None or ind >= len(self.valleys['indices']):
            return

        # Get the time of the valley to be removed
        valley_time = self.valleys['times'][ind]

        # Remove the valley
        self.valleys['indices'] = np.delete(self.valleys['indices'], ind)
        self.valleys['times'] = np.delete(self.valleys['times'], ind)
        self.valleys['depths'] = np.delete(self.valleys['depths'], ind)

        # Recalculate peak metrics if we still have peaks and valleys
        if self.peaks is not None and len(self.peaks['indices']) > 0 and len(self.valleys['indices']) > 0:
            # Determine which signal we're working with
            if self.signal_selection_var.get() == "Secondary" and hasattr(self, 'secondary_processed_signal'):
                signal = self.secondary_processed_signal
                time = self.secondary_processed_time
            else:
                signal = self.processed_signal
                time = self.processed_time

            self.peak_metrics = calculate_peak_metrics(self.peaks, self.valleys, signal, time)

        # Update the display
        self.update_peak_display()
        self.update_metrics_display()

        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Removed valley at {valley_time:.2f}s", fg="green")

    def visualize_peak_features(self):
        """Visualize peak features like width and area"""
        if self.peaks is None or self.valleys is None or self.peak_metrics is None:
            return

        # Determine which signal to use
        if self.signal_selection_var.get() == "Secondary" and hasattr(self, 'secondary_processed_signal'):
            signal = self.secondary_processed_signal
            time = self.secondary_processed_time
        else:
            signal = self.processed_signal
            time = self.processed_time

        # Clear previous annotations
        for annotation in self.peak_feature_annotations:
            annotation.remove() if hasattr(annotation, 'remove') else None
        self.peak_feature_annotations = []

        # Add feature visualization for each peak
        for i, peak_idx in enumerate(self.peaks['indices']):
            # Get the peak time and height
            peak_time = self.peaks['times'][i]
            peak_height = self.peaks['heights'][i]

            # Get the preceding and following valleys
            preceding_idx = self.peak_metrics['preceding_valley'][i]
            following_idx = self.peak_metrics['following_valley'][i]

            # Skip if we don't have valid valleys
            if preceding_idx is None or following_idx is None:
                continue

            # Get valley times and depths
            preceding_time = time[preceding_idx]
            following_time = time[following_idx]
            preceding_depth = signal[preceding_idx]
            following_depth = signal[following_idx]

            # Determine base level (minimum of the two valleys)
            base_level = min(preceding_depth, following_depth)

            # Get half-max level
            half_max = base_level + (peak_height - base_level) / 2

            # Plot FWHM line
            fwhm = self.peak_metrics['full_width_half_max'][i]
            if not np.isnan(fwhm):
                # Calculate start and end times (in minutes for plotting)
                rise_time = self.peak_metrics['rise_time'][i]
                fwhm_start = (peak_time - rise_time) / 60
                fwhm_end = (peak_time + self.peak_metrics['decay_time'][i]) / 60

                # Plot horizontal line at half max
                fwhm_line = self.ax1.hlines(
                    half_max, fwhm_start, fwhm_end,
                    color='orange', linestyle='-', linewidth=2,
                    label="FWHM" if i == 0 else ""
                )
                self.peak_feature_annotations.append(fwhm_line)

                # Add width annotation
                width_annotation = self.ax1.annotate(
                    f"{fwhm:.2f}s",
                    ((fwhm_start + fwhm_end) / 2, half_max),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center',
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7)
                )
                self.peak_feature_annotations.append(width_annotation)

            # Shade area under the curve
            peak_segment = signal[preceding_idx:following_idx + 1]
            peak_times = time[preceding_idx:following_idx + 1] / 60  # Convert to minutes for plotting

            area_fill = self.ax1.fill_between(
                peak_times,
                base_level,
                signal[preceding_idx:following_idx + 1],
                alpha=0.2,
                color='green',
                label="Peak Area" if i == 0 else ""
            )
            self.peak_feature_annotations.append(area_fill)

        # Redraw canvas
        self.canvas.draw_idle()

    def create_gui(self):
        # Create a container frame for everything
        self.container_frame = tk.Frame(self.root)
        self.container_frame.pack(fill=tk.BOTH, expand=True)

        # Create a main control panel at the top that's always visible
        self.control_panel = tk.Frame(self.container_frame)
        self.control_panel.pack(side=tk.TOP, fill=tk.X)

        # Add navigation buttons and scrolling toggle to this always-visible panel
        self.add_navigation_buttons_to_panel(self.control_panel)

        # Add scrolling toggle button to the always-visible panel
        self.scroll_toggle_btn = tk.Button(
            self.control_panel,
            text="Toggle Scrolling",
            font=('Arial', self.font_sizes['button']),
            width=15,
            bg='lightblue',
            command=self.toggle_scrolling
        )
        self.scroll_toggle_btn.pack(side=tk.RIGHT, padx=10, pady=5)

        # Create scrollable area for the rest of the content
        self.scroll_frame = tk.Frame(self.container_frame)
        self.scroll_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Create a canvas for scrolling
        self.scroll_canvas = tk.Canvas(self.scroll_frame)

        # Add vertical scrollbar - CRITICAL: Pack this BEFORE the canvas
        self.v_scrollbar = ttk.Scrollbar(self.scroll_frame, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Configure canvas
        self.scroll_canvas.configure(yscrollcommand=self.v_scrollbar.set)
        self.scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create the main frame inside the canvas
        self.main_frame = tk.Frame(self.scroll_canvas)

        # Add the main frame to the canvas with explicit width
        self.canvas_window = self.scroll_canvas.create_window((0, 0), window=self.main_frame, anchor=tk.NW)

        # Scrolling is enabled by default
        self.scrolling_enabled = True

        # Create a frame for the matplotlib figure
        self.frame = tk.Frame(self.main_frame)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create matplotlib figure with adjusted size
        self.fig = plt.figure(figsize=(10, 8))

        # Create GridSpec for subplots
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1])

        # Panel 1: Processed signal plot with artifact markers and drift curve
        self.ax1 = self.fig.add_subplot(gs[0])

        # Get region label for primary file
        primary_region = self.get_region_label_from_filename(self.file_path)

        # Plot the main processed signal
        self.line, = self.ax1.plot(
            self.processed_time / 60,
            self.processed_signal,
            'g-', lw=1,
            label=f'{primary_region} Signal'
        )

        # Add the no-control version as a dashed line
        self.line_no_control, = self.ax1.plot(
            self.processed_time / 60,
            self.processed_signal_no_control,
            color='lightgreen',  # Use a lighter green for contrast
            linestyle='--',
            lw=1.5,
            alpha=0.9,
            label=f'{primary_region} No-Control'
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

        # Create matplotlib canvas - ONLY ONCE!
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.toolbar.update()

        # Create slider frame with more space - ONLY ONCE!
        self.slider_frame = tk.Frame(self.main_frame)
        self.slider_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Create sliders with larger size - ONLY ONCE!
        self.create_sliders()

        # Connect mouse wheel events for zoom (keep this functionality)
        self.connect_zoom_events()

        # Adjust layout
        self.fig.tight_layout()

        # Force an update to calculate proper sizes
        self.root.update_idletasks()

        # Configure initial scroll region with FIXED height
        self.on_frame_configure()

        # Bind events for proper scrolling updates
        self.main_frame.bind("<Configure>", self.on_frame_configure)
        self.scroll_canvas.bind("<Configure>", self.on_canvas_configure)

        # Bind mousewheel scrolling
        self.bind_mousewheel()

    def add_navigation_buttons_to_panel(self, panel):
        """Add navigation buttons to the specified panel"""
        # Get screen width for responsive sizing
        screen_width = self.root.winfo_screenwidth()

        # Adjust button width based on screen size
        btn_width = max(8, min(10, int(screen_width / 180)))
        button_font = ('Arial', self.font_sizes['button'])

        # Create a frame for navigation buttons
        nav_buttons = tk.Frame(panel)
        nav_buttons.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)

        # Add label
        tk.Label(nav_buttons, text="Navigation:", font=('Arial', self.font_sizes['base'], 'bold')).pack(side=tk.LEFT,
                                                                                                        padx=5)

        # X-axis zoom buttons
        tk.Button(nav_buttons, text="Zoom In (X)", font=button_font, width=btn_width,
                  command=lambda: self.zoom_x(in_out="in")).pack(side=tk.LEFT, padx=2)

        tk.Button(nav_buttons, text="Zoom Out (X)", font=button_font, width=btn_width,
                  command=lambda: self.zoom_x(in_out="out")).pack(side=tk.LEFT, padx=2)

        # Y-axis zoom buttons
        tk.Button(nav_buttons, text="Zoom In (Y)", font=button_font, width=btn_width,
                  command=lambda: self.zoom_y_all("in")).pack(side=tk.LEFT, padx=2)

        tk.Button(nav_buttons, text="Zoom Out (Y)", font=button_font, width=btn_width,
                  command=lambda: self.zoom_y_all("out")).pack(side=tk.LEFT, padx=2)

        # Pan buttons
        tk.Button(nav_buttons, text="← Pan", font=button_font, width=btn_width - 2,
                  command=lambda: self.pan_x(direction="left")).pack(side=tk.LEFT, padx=2)

        tk.Button(nav_buttons, text="Pan →", font=button_font, width=btn_width - 2,
                  command=lambda: self.pan_x(direction="right")).pack(side=tk.LEFT, padx=2)

        # Reset view button
        tk.Button(nav_buttons, text="Reset View", font=button_font, width=btn_width,
                  command=self.reset_view, bg='lightblue').pack(side=tk.LEFT, padx=5)

        # File buttons in a separate frame
        file_buttons = tk.Frame(panel)
        file_buttons.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)

        tk.Button(file_buttons, text="Primary File", font=button_font, width=btn_width,
                  command=self.open_file, bg='lightyellow').pack(side=tk.LEFT, padx=2)

        tk.Button(file_buttons, text="Secondary File", font=button_font, width=btn_width,
                  command=self.open_secondary_file, bg='lightgreen').pack(side=tk.LEFT, padx=2)

        tk.Button(file_buttons, text="Clear Secondary", font=button_font, width=btn_width,
                  command=self.clear_secondary_file).pack(side=tk.LEFT, padx=2)

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

    def toggle_no_control_line(self):
        """Toggle visibility of the no-control signal line"""
        if hasattr(self, 'line_no_control') and self.line_no_control is not None:
            visible = self.show_no_control_var.get()
            self.line_no_control.set_visible(visible)

            # Do the same for secondary if it exists
            if hasattr(self, 'secondary_line_no_control') and self.secondary_line_no_control is not None:
                self.secondary_line_no_control.set_visible(visible)

            # Update the legend
            self.ax1_legend = self.create_checkbox_legend(self.ax1)

            # Redraw
            self.canvas.draw_idle()

    def adjust_nocontrol_offset(self, _=None):
        """Adjust the vertical position of the no-control signal"""
        if hasattr(self, 'line_no_control') and self.line_no_control is not None:
            offset = self.nocontrol_offset_var.get()

            # Get the base no-control signal
            if hasattr(self, 'processed_signal_no_control'):
                # Apply offset to the displayed data
                adjusted_data = self.processed_signal_no_control + offset
                self.line_no_control.set_ydata(adjusted_data)

            # Do the same for secondary if it exists
            if hasattr(self, 'secondary_line_no_control') and self.secondary_line_no_control is not None:
                if hasattr(self, 'secondary_processed_signal_no_control'):
                    adjusted_data = self.secondary_processed_signal_no_control + offset
                    self.secondary_line_no_control.set_ydata(adjusted_data)

            self.canvas.draw_idle()

    def create_sliders(self):
        """Create sliders with organized grouping that adapts to screen size"""
        # Get screen dimensions for responsive sizing
        screen_width = self.root.winfo_screenwidth()

        # Calculate appropriate slider length based on screen width
        slider_length = min(800, int(screen_width * 0.6))

        # Create control notebook
        self.control_notebook = ttk.Notebook(self.slider_frame)
        self.control_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create combined tab for signal processing
        self.combined_processing_tab = ttk.Frame(self.control_notebook)
        self.control_notebook.add(self.combined_processing_tab, text="Signal Processing")

        # Now add detection tabs
        self.create_peak_detection_tab()
        self.create_valley_detection_tab()
        self.create_psth_analysis_tab()

        # Configure grid for combined tab
        self.combined_processing_tab.columnconfigure(1, weight=1)  # Make slider column expandable

        font_size = self.font_sizes['base']
        slider_font_size = self.font_sizes['slider']

        # Use a throttled update function for sliders
        self.update_timer = None

        # Create section frames for better organization
        basic_filter_frame = tk.LabelFrame(self.combined_processing_tab, text="Basic Filtering",
                                           font=('Arial', font_size, 'bold'))
        basic_filter_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)

        # Low cutoff slider
        tk.Label(basic_filter_frame, text="Low cutoff (Hz):", font=('Arial', font_size)).grid(
            row=0, column=0, sticky="w", padx=10, pady=10)
        self.low_slider = tk.Scale(basic_filter_frame, from_=0.0, to=0.01,
                                   resolution=0.0001, orient=tk.HORIZONTAL,
                                   length=slider_length, width=25, font=('Arial', slider_font_size),
                                   command=self.update_filter)
        self.low_slider.set(self.low_cutoff)
        self.low_slider.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        # High cutoff slider
        tk.Label(basic_filter_frame, text="High cutoff (Hz):", font=('Arial', font_size)).grid(
            row=1, column=0, sticky="w", padx=10, pady=10)
        self.high_slider = tk.Scale(basic_filter_frame, from_=0.1, to=5.0,
                                    resolution=0.1, orient=tk.HORIZONTAL,
                                    length=slider_length, width=25, font=('Arial', slider_font_size),
                                    command=self.update_filter)
        self.high_slider.set(self.high_cutoff)
        self.high_slider.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        # Downsample slider
        tk.Label(basic_filter_frame, text="Downsample factor:", font=('Arial', font_size)).grid(
            row=2, column=0, sticky="w", padx=10, pady=10)
        self.downsample_slider = tk.Scale(basic_filter_frame, from_=1, to=100,
                                          resolution=1, orient=tk.HORIZONTAL,
                                          length=slider_length, width=25, font=('Arial', slider_font_size),
                                          command=self.update_filter)
        self.downsample_slider.set(self.downsample_factor)
        self.downsample_slider.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        # Drift correction settings
        drift_frame = tk.LabelFrame(basic_filter_frame, text="Drift Correction",
                                    font=('Arial', font_size))
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

        # Edge protection for noisy beginnings
        self.edge_protection_var = tk.BooleanVar(value=True)
        edge_frame = tk.Frame(basic_filter_frame)
        edge_frame.grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=10)

        tk.Checkbutton(edge_frame,
                       text="Enable edge protection (reduces distortion at beginning of recording)",
                       variable=self.edge_protection_var,
                       font=('Arial', font_size),
                       command=self.update_filter).pack(side=tk.LEFT, padx=5)

        # ------------ Manual Blanking Section ------------ #
        manual_blanking_frame = tk.LabelFrame(self.combined_processing_tab, text="Manual Noise Blanking",
                                              font=('Arial', font_size, 'bold'))
        manual_blanking_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        self.manual_blanking_frame = manual_blanking_frame  # Store reference for later use

        tk.Label(manual_blanking_frame,
                 text="Click and drag on the signal plot to select a time window to blank out.",
                 font=('Arial', font_size - 1)).pack(padx=10, pady=5)

        btn_frame = tk.Frame(manual_blanking_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.manual_blank_btn = tk.Button(
            btn_frame,
            text="Enable Selection Mode",
            font=('Arial', self.font_sizes['base']),
            bg='lightblue',
            command=self.toggle_blanking_mode
        )
        self.manual_blank_btn.pack(side=tk.LEFT, padx=5)

        self.apply_blanking_btn = tk.Button(
            btn_frame,
            text="Apply Blanking",
            font=('Arial', self.font_sizes['base']),
            state=tk.DISABLED,
            command=self.apply_blanking
        )
        self.apply_blanking_btn.pack(side=tk.LEFT, padx=5)

        self.clear_selection_btn = tk.Button(
            btn_frame,
            text="Clear Selection",
            font=('Arial', self.font_sizes['base']),
            state=tk.DISABLED,
            command=self.clear_blanking_selection
        )
        self.clear_selection_btn.pack(side=tk.LEFT, padx=5)

        self.cancel_all_blanking_btn = tk.Button(
            btn_frame,
            text="Cancel All Blanking",
            font=('Arial', self.font_sizes['base']),
            bg='salmon',
            command=self.cancel_blanking
        )
        self.cancel_all_blanking_btn.pack(side=tk.LEFT, padx=5)

        # Status indicator for processing
        self.status_label = tk.Label(self.nav_frame, text="Ready", fg="green",
                                     font=('Arial', self.font_sizes['base']))
        self.status_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=10)

        # Initialize blanking variables
        self.blanking_mode = False
        self.blanking_selection = None
        self.blanking_rectangle = None
        self.blanking_cid = None
        self.blanking_start = None
        self.blanking_file = "primary"  # Default to primary file

    def throttled_update(self, _=None):
        """Throttle updates to prevent UI freezing with rapid slider movements"""
        # Cancel any pending update
        if self.update_timer is not None:
            self.root.after_cancel(self.update_timer)

        # Schedule new update after a short delay (200ms)
        self.update_timer = self.root.after(200, self.update_filter)

    # Add method to PhotometryViewer class
    def run_comprehensive_processing(self):
        """Run comprehensive signal processing pipeline"""
        if not hasattr(self, 'downsampled_raw_analog_1') or self.downsampled_raw_analog_1 is None:
            messagebox.showinfo("Error", "No data loaded")
            return

        # Process the primary signal
        result = comprehensive_photometry_processing(
            self.downsampled_raw_analog_1,
            self.downsampled_raw_analog_2,
            self.processed_time,
            sampling_rate=1.0 / (self.processed_time[1] - self.processed_time[0])
        )

        # Store results
        self.processed_results = result

        # Update the display with processed signal
        self.processed_signal = result['dff']
        self.line.set_ydata(self.processed_signal)

        # Add option to display various processing stages
        # (You might want to create a dropdown or radio buttons for this)

        # Update the plot
        self.canvas.draw_idle()

        # Show summary statistics
        metrics = result['metrics']
        message = (f"Processing complete!\n\n"
                   f"Mean ΔF/F: {metrics['mean_dff']:.2f}%\n"
                   f"Max ΔF/F: {metrics['max_dff']:.2f}%\n"
                   f"Time above 0.5%: {metrics['time_above_0_5']:.2f}%\n"
                   f"Time above 1.0%: {metrics['time_above_1_0']:.2f}%\n"
                   f"Total AUC: {metrics['total_auc']:.2f}")

        messagebox.showinfo("Processing Results", message)

    def toggle_blanking_mode(self):
        """Toggle between normal and blanking selection modes"""
        self.blanking_mode = not self.blanking_mode

        if self.blanking_mode:
            # Enable selection mode
            self.manual_blank_btn.config(text="Disable Selection Mode", bg='salmon')

            # Create a radio button group for file selection
            if hasattr(self, 'secondary_processed_signal') and self.secondary_processed_signal is not None:
                selection_frame = tk.Toplevel(self.root)
                selection_frame.title("Select File to Blank")
                selection_frame.geometry("300x150")
                selection_frame.transient(self.root)
                selection_frame.grab_set()

                primary_label = self.get_region_label_from_filename(self.file_path)
                secondary_label = self.get_region_label_from_filename(self.secondary_file_path)

                tk.Label(selection_frame,
                         text="Which signal do you want to blank?",
                         font=('Arial', self.font_sizes['base'])).pack(pady=10)

                file_var = tk.StringVar(value="primary")

                tk.Radiobutton(selection_frame,
                               text=f"Primary ({primary_label})",
                               variable=file_var,
                               value="primary",
                               font=('Arial', self.font_sizes['base'])).pack(anchor="w", padx=20)

                tk.Radiobutton(selection_frame,
                               text=f"Secondary ({secondary_label})",
                               variable=file_var,
                               value="secondary",
                               font=('Arial', self.font_sizes['base'])).pack(anchor="w", padx=20)

                def select_file():
                    self.blanking_file = file_var.get()
                    selection_frame.destroy()
                    self.connect_blanking_events()

                tk.Button(selection_frame,
                          text="Select",
                          command=select_file,
                          font=('Arial', self.font_sizes['base'])).pack(pady=10)
            else:
                # Only primary file available
                self.blanking_file = "primary"
                self.connect_blanking_events()
        else:
            # Disable selection mode
            self.manual_blank_btn.config(text="Enable Selection Mode", bg='lightblue')
            self.disconnect_blanking_events()

    def connect_blanking_events(self):
        """Connect mouse events for blanking selection"""
        self.canvas.get_tk_widget().config(cursor="crosshair")
        self.blanking_cid = self.canvas.mpl_connect('button_press_event', self.on_blanking_press)

        # Show instructions
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Click and drag to select area on the {self.blanking_file} signal to blank",
                                     fg="blue")

    def disconnect_blanking_events(self):
        """Disconnect mouse events for blanking selection"""
        self.canvas.get_tk_widget().config(cursor="")
        if self.blanking_cid is not None:
            self.canvas.mpl_disconnect(self.blanking_cid)
            self.blanking_cid = None

        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Ready", fg="green")

    def on_blanking_press(self, event):
        """Handle mouse press for blanking selection"""
        if not event.inaxes == self.ax1:
            return

        # Store start point
        self.blanking_start = (event.xdata, event.ydata)

        # Connect motion and release events
        self.motion_cid = self.canvas.mpl_connect('motion_notify_event', self.on_blanking_motion)
        self.release_cid = self.canvas.mpl_connect('button_release_event', self.on_blanking_release)

    def on_blanking_motion(self, event):
        """Handle mouse motion for blanking selection"""
        if not event.inaxes == self.ax1 or self.blanking_start is None:
            return

        # Remove previous rectangle if it exists
        if self.blanking_rectangle is not None:
            self.blanking_rectangle.remove()

        # Draw new rectangle
        x_start, y_start = self.blanking_start
        width = event.xdata - x_start
        height = self.ax1.get_ylim()[1] - self.ax1.get_ylim()[0]
        y_bottom = self.ax1.get_ylim()[0]

        self.blanking_rectangle = plt.Rectangle(
            (x_start, y_bottom), width, height,
            edgecolor='r', facecolor='r', alpha=0.3, zorder=1000)
        self.ax1.add_patch(self.blanking_rectangle)

        self.canvas.draw_idle()

    def on_blanking_release(self, event):
        """Handle mouse release for blanking selection"""
        if not event.inaxes == self.ax1 or self.blanking_start is None:
            return

        # Disconnect motion and release events
        self.canvas.mpl_disconnect(self.motion_cid)
        self.canvas.mpl_disconnect(self.release_cid)

        # Store selection
        x_start, _ = self.blanking_start
        x_end = event.xdata

        # Make sure start is before end
        if x_start > x_end:
            x_start, x_end = x_end, x_start

        # Convert to time in seconds
        t_start = x_start * 60  # Convert from minutes to seconds
        t_end = x_end * 60

        self.blanking_selection = (t_start, t_end)

        # Enable apply button
        self.apply_blanking_btn.config(state=tk.NORMAL)
        self.clear_selection_btn.config(state=tk.NORMAL)

        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.config(
                text=f"Selected time window: {t_start:.2f}s to {t_end:.2f}s in {self.blanking_file} signal. Click 'Apply Blanking' to blank out.",
                fg="blue")

    def clear_blanking_selection(self):
        """Clear the current blanking selection"""
        if self.blanking_rectangle is not None:
            self.blanking_rectangle.remove()
            self.blanking_rectangle = None

        self.blanking_selection = None
        self.blanking_start = None

        self.apply_blanking_btn.config(state=tk.DISABLED)
        self.clear_selection_btn.config(state=tk.DISABLED)

        self.canvas.draw_idle()

        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Selection cleared. Click and drag to select a new area.", fg="blue")

    def apply_blanking(self):
        """
        Apply blanking to the selected time range with smooth transitions
        without adding visual markers that block the signal
        """
        if self.blanking_selection is None:
            return

        t_start, t_end = self.blanking_selection

        # Get the appropriate signal and time arrays based on which file is being edited
        if self.blanking_file == "primary":
            # Get arrays
            time_array = self.processed_time
            signal_array = self.processed_signal
            line = self.line
            self.primary_has_blanking = True  # Set the flag to track blanking
        else:  # secondary
            time_array = self.secondary_processed_time
            signal_array = self.secondary_processed_signal
            line = self.secondary_line
            self.secondary_has_blanking = True  # Set the flag to track blanking

        # Find indices within the time range
        mask = (time_array >= t_start) & (time_array <= t_end)
        if not np.any(mask):
            return  # No data in selected range

        idx_start = np.where(mask)[0][0]
        idx_end = np.where(mask)[0][-1]

        # Calculate a reasonable transition width (in indices)
        # Add safety check for very short selections
        blank_length = idx_end - idx_start
        trans_width = min(100, max(10, int(blank_length * 0.1)))  # 10% of blanked region

        # Get indices for expanded context
        pre_idx = max(0, idx_start - trans_width)
        post_idx = min(len(signal_array) - 1, idx_end + trans_width)

        # Get values at context edges for interpolation reference
        pre_val = np.median(signal_array[pre_idx:idx_start])
        post_val = np.median(signal_array[idx_end:post_idx])

        # Create a copy of the signal for modification
        modified_signal = signal_array.copy()

        # Create smooth transition (linear interpolation)
        interp_range = np.arange(idx_start, idx_end + 1)
        interp_vals = np.linspace(pre_val, post_val, len(interp_range))
        modified_signal[idx_start:idx_end + 1] = interp_vals

        # Apply additional smoothing to the transitions
        # Pre-transition smoothing
        if idx_start > 0:
            # Create a smooth transition from original to interpolated
            blend_range = np.arange(pre_idx, idx_start)
            if len(blend_range) > 0:
                weights = np.linspace(1, 0, len(blend_range)) ** 2  # Squared for more natural transition
                orig_vals = signal_array[blend_range]
                target_vals = np.linspace(orig_vals[0], pre_val, len(blend_range))
                modified_signal[blend_range] = weights * orig_vals + (1 - weights) * target_vals

        # Post-transition smoothing
        if idx_end < len(signal_array) - 1:
            # Create a smooth transition from interpolated to original
            blend_range = np.arange(idx_end + 1, post_idx + 1)
            if len(blend_range) > 0:
                weights = np.linspace(0, 1, len(blend_range)) ** 2  # Squared for more natural transition
                orig_vals = signal_array[blend_range]
                target_vals = np.linspace(post_val, orig_vals[-1], len(blend_range))
                modified_signal[blend_range] = weights * orig_vals + (1 - weights) * target_vals

        # Store the blanking region info for potential reapplication
        if not hasattr(self, 'blanking_regions'):
            self.blanking_regions = []

        self.blanking_regions.append({
            'file': self.blanking_file,
            'start_time': t_start,
            'end_time': t_end,
            'start_idx': idx_start,
            'end_idx': idx_end,
            'pre_idx': pre_idx,
            'post_idx': post_idx,
            'pre_val': pre_val,
            'post_val': post_val
        })

        # Update the signal
        if self.blanking_file == "primary":
            self.processed_signal = modified_signal
            line.set_ydata(modified_signal)
            self.primary_has_blanking = True  # Ensure flag is set
        else:  # secondary
            self.secondary_processed_signal = modified_signal
            line.set_ydata(modified_signal)
            self.secondary_has_blanking = True  # Ensure flag is set

        # Clear blanking selection rectangle
        if self.blanking_rectangle is not None:
            self.blanking_rectangle.remove()
            self.blanking_rectangle = None

        # Update the canvas
        self.canvas.draw_idle()

        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.config(
                text=f"Applied smooth blanking from {t_start:.2f}s to {t_end:.2f}s in {self.blanking_file} file.",
                fg="green")

        # Reset UI elements after blanking
        self.clear_blanking_selection()

    def reapply_all_blanking(self):
        """Reapply all stored blanking regions to the signals without visual markers"""
        if not hasattr(self, 'blanking_regions') or not self.blanking_regions:
            return  # No blanking regions to reapply

        # Process each saved blanking region
        for region in self.blanking_regions:
            file_type = region['file']
            t_start = region['start_time']
            t_end = region['end_time']

            # Determine which signal to modify
            if file_type == "primary":
                time_array = self.processed_time
                signal_array = self.processed_signal
                line = self.line
                self.primary_has_blanking = True
            else:  # secondary
                time_array = self.secondary_processed_time
                signal_array = self.secondary_processed_signal
                line = self.secondary_line
                self.secondary_has_blanking = True

            # Find indices within the time range (might have changed after reprocessing)
            mask = (time_array >= t_start) & (time_array <= t_end)
            if not np.any(mask):
                print(f"Warning: Blanking region {t_start}-{t_end}s no longer exists in the data")
                continue  # Skip if time range is no longer valid

            idx_start = np.where(mask)[0][0]
            idx_end = np.where(mask)[0][-1]

            # Calculate transition width
            blank_length = idx_end - idx_start
            trans_width = min(100, max(10, int(blank_length * 0.1)))

            # Get context indices
            pre_idx = max(0, idx_start - trans_width)
            post_idx = min(len(signal_array) - 1, idx_end + trans_width)

            # Determine interpolation values
            pre_val = np.median(signal_array[pre_idx:idx_start])
            post_val = np.median(signal_array[idx_end:post_idx])

            # Create a copy of the signal for modification
            modified_signal = signal_array.copy()

            # Apply linear interpolation to blanked region
            interp_range = np.arange(idx_start, idx_end + 1)
            interp_vals = np.linspace(pre_val, post_val, len(interp_range))
            modified_signal[idx_start:idx_end + 1] = interp_vals

            # Apply pre-transition smoothing
            if idx_start > 0:
                blend_range = np.arange(pre_idx, idx_start)
                if len(blend_range) > 0:
                    weights = np.linspace(1, 0, len(blend_range)) ** 2
                    orig_vals = signal_array[blend_range]
                    target_vals = np.linspace(orig_vals[0], pre_val, len(blend_range))
                    modified_signal[blend_range] = weights * orig_vals + (1 - weights) * target_vals

            # Apply post-transition smoothing
            if idx_end < len(signal_array) - 1:
                blend_range = np.arange(idx_end + 1, post_idx + 1)
                if len(blend_range) > 0:
                    weights = np.linspace(0, 1, len(blend_range)) ** 2
                    orig_vals = signal_array[blend_range]
                    target_vals = np.linspace(post_val, orig_vals[-1], len(blend_range))
                    modified_signal[blend_range] = weights * orig_vals + (1 - weights) * target_vals

            # Update the signal
            if file_type == "primary":
                self.processed_signal = modified_signal
                self.line.set_ydata(modified_signal)
            else:  # secondary
                self.secondary_processed_signal = modified_signal
                self.secondary_line.set_ydata(modified_signal)

        # Redraw the canvas
        self.canvas.draw_idle()

        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.config(
                text=f"Reapplied {len(self.blanking_regions)} blanking region(s)",
                fg="green")

    def cancel_blanking(self):
        """Cancel all blanking applied to both primary and secondary signals
        and reprocess the signals from raw data"""

        # Check if there are any blanking regions to cancel
        if not hasattr(self, 'blanking_regions') or not self.blanking_regions:
            if hasattr(self, 'status_label'):
                self.status_label.config(text="No blanking to cancel", fg="blue")
            return

        # Store the number of blanking regions for the status message
        num_regions = len(self.blanking_regions)

        # Clear blanking regions
        self.blanking_regions = []

        # Reset blanking flags
        self.primary_has_blanking = False
        self.secondary_has_blanking = False

        # Remove blanking indicators from plot
        for patch in list(self.ax1.patches):
            if hasattr(patch, 'is_blank_marker'):
                patch.remove()

        # Reprocess data from raw inputs with current filtering parameters
        # This effectively undoes all blanking
        result = process_data(
            self.time, self.raw_analog_1, self.raw_analog_2, self.digital_1, self.digital_2,
            low_cutoff=self.low_cutoff, high_cutoff=self.high_cutoff,
            downsample_factor=self.downsample_factor, artifact_threshold=self.artifact_threshold,
            drift_correction=self.drift_correction, drift_degree=self.drift_degree,
            edge_protection=self.edge_protection_var.get()
        )

        # Unpack results
        self.processed_time, self.processed_signal, self.processed_signal_no_control, \
            self.processed_analog_2, self.downsampled_raw_analog_1, self.downsampled_raw_analog_2, \
            self.processed_digital_1, self.processed_digital_2, self.artifact_mask, \
            self.drift_curve = result

        # If secondary file is loaded, reprocess it too
        if hasattr(self, 'secondary_file_path') and self.secondary_file_path:
            secondary_result = process_data(
                self.secondary_time, self.secondary_raw_analog_1, self.secondary_raw_analog_2,
                self.secondary_digital_1, self.secondary_digital_2,
                low_cutoff=self.low_cutoff, high_cutoff=self.high_cutoff,
                downsample_factor=self.downsample_factor, artifact_threshold=self.artifact_threshold,
                drift_correction=self.drift_correction, drift_degree=self.drift_degree,
                edge_protection=self.edge_protection_var.get()
            )

            # Unpack secondary results
            self.secondary_processed_time, self.secondary_processed_signal, self.secondary_processed_signal_no_control, \
                self.secondary_processed_analog_2, self.secondary_downsampled_raw_analog_1, self.secondary_downsampled_raw_analog_2, \
                self.secondary_processed_digital_1, self.secondary_processed_digital_2, self.secondary_artifact_mask, \
                self.secondary_drift_curve = secondary_result

        # Update plot data
        self.line.set_xdata(self.processed_time / 60)
        self.line.set_ydata(self.processed_signal)

        # Update the no-control signal plot
        if hasattr(self, 'line_no_control') and self.line_no_control is not None:
            self.line_no_control.set_xdata(self.processed_time / 60)
            self.line_no_control.set_ydata(self.processed_signal_no_control)

        # Update secondary line if exists
        if hasattr(self, 'secondary_line') and self.secondary_line is not None:
            self.secondary_line.set_xdata(self.secondary_processed_time / 60)
            self.secondary_line.set_ydata(self.secondary_processed_signal)

            if hasattr(self, 'secondary_line_no_control') and self.secondary_line_no_control is not None:
                self.secondary_line_no_control.set_xdata(self.secondary_processed_time / 60)
                self.secondary_line_no_control.set_ydata(self.secondary_processed_signal_no_control)

        # Redraw canvas
        self.canvas.draw_idle()

        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.config(
                text=f"Cancelled {num_regions} blanking region(s) and restored original signal",
                fg="green"
            )

    def cancel_blanking_btn_frame(self):
        """Add Cancel All Blanking button to the manual blanking section"""
        if not hasattr(self, 'manual_blanking_frame'):
            return

        cancel_frame = tk.Frame(self.manual_blanking_frame)
        cancel_frame.pack(fill=tk.X, padx=10, pady=5)

        self.cancel_blanking_btn = tk.Button(
            cancel_frame,
            text="Remove All Blanking",
            font=('Arial', self.font_sizes['button']),
            bg='lightcoral',
            command=self.cancel_blanking
        )
        self.cancel_blanking_btn.pack(side=tk.LEFT, padx=5)

        # Add description
        tk.Label(
            cancel_frame,
            text="Removes all blanking and restores original signal",
            font=('Arial', self.font_sizes['base'] - 1),
            fg='gray'
        ).pack(side=tk.LEFT, padx=5)

    def savgol_baseline(signal, window_length=151, polyorder=1):
        """Apply Savitzky-Golay filtering for baseline estimation"""
        from scipy.signal import savgol_filter

        baseline = savgol_filter(signal, window_length, polyorder)
        return baseline


    def mark_blanked_region(self, x_start, x_end):
        """Mark a blanked region with a hatched pattern"""
        # Remove previous blank markers that overlap
        for patch in self.ax1.patches:
            if hasattr(patch, 'is_blank_marker'):
                patch_x, patch_width = patch.get_x(), patch.get_width()
                # If there's any overlap, remove the old patch
                if not (patch_x + patch_width < x_start or patch_x > x_end):
                    patch.remove()

        # Create new patch with different style than flattened regions
        ylim = self.ax1.get_ylim()
        height = ylim[1] - ylim[0]

        # Choose color based on which file is being marked
        if self.blanking_file == "primary":
            color = 'gray'
        else:  # secondary
            color = 'pink'

        patch = plt.Rectangle(
            (x_start, ylim[0]), x_end - x_start, height,
            facecolor=color, alpha=0.3, hatch='///', zorder=-90)
        patch.is_blank_marker = True
        self.ax1.add_patch(patch)

    # Add this method:
    def reset_denoising(self):
        """Reset denoising and blanking and reprocess signals with current filter settings"""
        self.denoising_applied = False
        self.primary_has_blanking = False
        self.secondary_has_blanking = False

        # Clear blanking regions
        if hasattr(self, 'blanking_regions'):
            self.blanking_regions = []

        # Remove any marked regions
        for patch in self.ax1.patches:
            if hasattr(patch, 'is_flattened_marker') or hasattr(patch, 'is_blank_marker'):
                patch.remove()

        # Update filter to reprocess signals with preserve_blanking=False
        self.update_filter(preserve_blanking=False)

        # Show confirmation
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Denoising and manual blanking reset", fg="green")

    def smooth_blanking(signal, time, blank_start_idx, blank_end_idx, transition_width=0.1):
        """
        Apply gradient-based interpolation for blanking with smooth transitions.

        Parameters:
        -----------
        signal : array
            Original signal array
        time : array
            Time points array
        blank_start_idx : int
            Start index of region to blank
        blank_end_idx : int
            End index of region to blank
        transition_width : float
            Width of transition as a fraction of blanked region (0-1)

        Returns:
        --------
        blanked_signal : array
            Signal with smooth blanking applied
        """
        blanked_signal = signal.copy()

        # Get values at boundaries for interpolation
        if blank_start_idx > 0 and blank_end_idx < len(signal) - 1:
            pre_blank_val = signal[blank_start_idx - 1]
            post_blank_val = signal[blank_end_idx + 1]

            # Calculate transition zone size (in indices)
            region_size = blank_end_idx - blank_start_idx
            trans_size = max(3, int(region_size * transition_width))

            # Create pre-transition zone (ramping down)
            pre_trans_start = max(0, blank_start_idx - trans_size)
            if pre_trans_start < blank_start_idx:
                # Create weights from 1→0 for transition
                weights = np.linspace(1, 0, blank_start_idx - pre_trans_start)
                # Get target line from data to interpolated value
                target_line = np.linspace(signal[pre_trans_start], pre_blank_val, blank_start_idx - pre_trans_start)
                # Blend original with target line using weights
                blanked_signal[pre_trans_start:blank_start_idx] = weights * signal[pre_trans_start:blank_start_idx] + (
                            1 - weights) * target_line

            # Create post-transition zone (ramping up)
            post_trans_end = min(len(signal), blank_end_idx + trans_size)
            if blank_end_idx < post_trans_end:
                # Create weights from 0→1 for transition
                weights = np.linspace(0, 1, post_trans_end - blank_end_idx)
                # Get target line from interpolated value to data
                target_line = np.linspace(post_blank_val, signal[post_trans_end - 1], post_trans_end - blank_end_idx)
                # Blend target line with original using weights
                blanked_signal[blank_end_idx:post_trans_end] = weights * signal[blank_end_idx:post_trans_end] + (
                            1 - weights) * target_line

            # Linear interpolation for blanked region
            x_interp = np.array([blank_start_idx - 1, blank_end_idx + 1])
            y_interp = np.array([pre_blank_val, post_blank_val])
            blanked_region = np.interp(
                np.arange(blank_start_idx, blank_end_idx + 1),
                x_interp, y_interp
            )
            blanked_signal[blank_start_idx:blank_end_idx + 1] = blanked_region

        return blanked_signal

    def spline_blanking(signal, time, blank_start_idx, blank_end_idx, context_width=0.5):
        """
        Apply cubic spline interpolation for blanking with natural curves.

        Parameters:
        -----------
        signal : array
            Original signal array
        time : array
            Time points array
        blank_start_idx : int
            Start index of region to blank
        blank_end_idx : int
            End index of region to blank
        context_width : float
            Width of context as a fraction of blanked region (0-1)

        Returns:
        --------
        blanked_signal : array
            Signal with spline blanking applied
        """
        from scipy.interpolate import CubicSpline

        blanked_signal = signal.copy()

        # Calculate context region size
        blank_length = blank_end_idx - blank_start_idx
        context_size = max(5, int(blank_length * context_width))

        # Get context points before and after the blanked region
        pre_context_start = max(0, blank_start_idx - context_size)
        post_context_end = min(len(signal), blank_end_idx + context_size)

        # Extract context points for spline fitting
        x_context = np.concatenate([
            np.arange(pre_context_start, blank_start_idx),
            np.arange(blank_end_idx + 1, post_context_end)
        ])
        y_context = np.concatenate([
            signal[pre_context_start:blank_start_idx],
            signal[blank_end_idx + 1:post_context_end]
        ])

        # Ensure we have enough points for cubic spline
        if len(x_context) >= 4:
            # Create cubic spline interpolation
            cs = CubicSpline(x_context, y_context)

            # Generate interpolated values for blanked region
            x_blanked = np.arange(blank_start_idx, blank_end_idx + 1)
            blanked_signal[blank_start_idx:blank_end_idx + 1] = cs(x_blanked)
        else:
            # Fall back to linear interpolation if not enough points
            if blank_start_idx > 0 and blank_end_idx < len(signal) - 1:
                x_interp = np.array([blank_start_idx - 1, blank_end_idx + 1])
                y_interp = np.array([signal[blank_start_idx - 1], signal[blank_end_idx + 1]])
                blanked_region = np.interp(
                    np.arange(blank_start_idx, blank_end_idx + 1),
                    x_interp, y_interp
                )
                blanked_signal[blank_start_idx:blank_end_idx + 1] = blanked_region

        return blanked_signal

    def segment_based_blanking(signal, time, blank_regions):
        """
        Apply segment-based analysis inspired by GuPPy.

        Parameters:
        -----------
        signal : array
            Original signal array
        time : array
            Time points array
        blank_regions : list of tuples
            List of (start_idx, end_idx) regions to blank

        Returns:
        --------
        processed_signal : array
            Final processed signal
        segments : list of dict
            List of valid segments with their metadata
        """
        # Sort and merge overlapping regions
        blank_regions.sort()
        merged_regions = []
        for region in blank_regions:
            if not merged_regions or region[0] > merged_regions[-1][1]:
                merged_regions.append(region)
            else:
                merged_regions[-1] = (merged_regions[-1][0], max(merged_regions[-1][1], region[1]))

        # Create mask for valid data (non-blanked)
        valid_mask = np.ones(len(signal), dtype=bool)
        for start_idx, end_idx in merged_regions:
            valid_mask[start_idx:end_idx + 1] = False

        # Find contiguous valid segments
        segments = []
        in_segment = False
        segment_start = 0

        for i in range(len(valid_mask)):
            if valid_mask[i] and not in_segment:
                # Start of new segment
                segment_start = i
                in_segment = True
            elif not valid_mask[i] and in_segment:
                # End of segment
                segments.append({
                    'start_idx': segment_start,
                    'end_idx': i - 1,
                    'time': time[segment_start:i],
                    'signal': signal[segment_start:i],
                    'duration': time[i - 1] - time[segment_start]
                })
                in_segment = False

        # Handle last segment if we ended in a valid region
        if in_segment:
            segments.append({
                'start_idx': segment_start,
                'end_idx': len(signal) - 1,
                'time': time[segment_start:],
                'signal': signal[segment_start:],
                'duration': time[-1] - time[segment_start]
            })

        # Process each segment independently
        # (this is where you would apply segment-specific processing if needed)

        # Recombine segments with interpolation between them
        processed_signal = np.zeros_like(signal)

        # Fill valid segments with their processed data
        for segment in segments:
            processed_signal[segment['start_idx']:segment['end_idx'] + 1] = segment['signal']

        # Interpolate between segments
        for i in range(len(segments) - 1):
            current_end = segments[i]['end_idx']
            next_start = segments[i + 1]['start_idx']

            if next_start - current_end > 1:  # If there's a gap
                # Get boundary values
                left_val = processed_signal[current_end]
                right_val = processed_signal[next_start]

                # Linear interpolation
                gap_indices = np.arange(current_end + 1, next_start)
                weights = np.linspace(0, 1, len(gap_indices))

                # Create smooth transition
                processed_signal[gap_indices] = (1 - weights) * left_val + weights * right_val

        return processed_signal, segments

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
        # Get screen width for responsive sizing
        screen_width = self.root.winfo_screenwidth()

        # Adjust button width based on screen size
        btn_width = max(8, min(12, int(screen_width / 160)))

        button_font = ('Arial', self.font_sizes['button'])
        title_font = ('Arial', self.font_sizes['title'], 'bold')

        # Use Grid layout manager with weight for responsive behavior
        for i in range(8):  # Assuming 8 columns in your grid
            self.nav_frame.columnconfigure(i, weight=1)

        # Row 1: Zoom and Pan controls
        tk.Label(self.nav_frame, text="Navigation:", font=title_font).grid(
            row=0, column=0, padx=10, pady=5, sticky="w")

        # X-axis zoom buttons
        tk.Button(self.nav_frame, text="Zoom In (X)", font=button_font, width=btn_width,
                  command=lambda: self.zoom_x(in_out="in")).grid(
            row=0, column=1, padx=5, pady=5, sticky="ew")

        tk.Button(self.nav_frame, text="Zoom Out (X)", font=button_font, width=btn_width,
                  command=lambda: self.zoom_x(in_out="out")).grid(
            row=0, column=2, padx=5, pady=5)

        # Y-axis zoom buttons (for all panels)
        tk.Button(self.nav_frame, text="Zoom In (Y)", font=button_font, width=btn_width,
                  command=lambda: self.zoom_y_all("in")).grid(
            row=0, column=3, padx=5, pady=5)

        tk.Button(self.nav_frame, text="Zoom Out (Y)", font=button_font, width=btn_width,
                  command=lambda: self.zoom_y_all("out")).grid(
            row=0, column=4, padx=5, pady=5)

        # Pan buttons
        tk.Button(self.nav_frame, text="← Pan Left", font=button_font, width=btn_width,
                  command=lambda: self.pan_x(direction="left")).grid(
            row=0, column=5, padx=5, pady=5)

        tk.Button(self.nav_frame, text="Pan Right →", font=button_font, width=btn_width,
                  command=lambda: self.pan_x(direction="right")).grid(
            row=0, column=6, padx=5, pady=5)

        # Reset view button (store reference to allow visual feedback during zooming)
        self.reset_view_button = tk.Button(
            self.nav_frame,
            text="Reset View",
            font=title_font,
            width=15,
            command=self.reset_view,
            bg='lightblue'
        )
        self.reset_view_button.grid(row=0, column=7, padx=10, pady=5)

        # Add file buttons
        file_frame = tk.Frame(self.nav_frame)
        file_frame.grid(row=0, column=8, padx=10, pady=5)

        tk.Button(file_frame, text="Primary File", font=title_font, width=15,
                  command=self.open_file, bg='lightyellow').pack(side=tk.LEFT, padx=5)

        tk.Button(file_frame, text="Secondary File", font=title_font, width=15,
                  command=self.open_secondary_file, bg='lightgreen').pack(side=tk.LEFT, padx=5)

        tk.Button(file_frame, text="Clear Secondary", font=button_font, width=15,
                  command=self.clear_secondary_file).pack(side=tk.LEFT, padx=5)
        # Add scrolling toggle button (add this at the end)
        self.scroll_toggle_btn = tk.Button(
            self.nav_frame,
            text="Enable Scrolling",
            font=('Arial', self.font_sizes['button']),
            width=15,
            bg='lightblue',
            command=self.toggle_scrolling
        )
        self.scroll_toggle_btn.grid(row=0, column=9, padx=10, pady=5)  # Adjust column as needed

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
        # Clear denoising flag
        self.denoising_applied = False

        # Remove any marked regions
        for patch in self.ax1.patches:
            if hasattr(patch, 'is_flattened_marker'):
                patch.remove()
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
        """Connect mouse wheel events for zooming plots"""
        # Store the connection ID so we can disconnect later
        self.zoom_id = self.canvas.mpl_connect('scroll_event', self.on_scroll)

        # Also connect key events for Ctrl detection
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
        - Ctrl+scroll: zoom x-axis of all plots with enhanced control
        """
        # Check if the event occurred within an axis
        if event.inaxes is None:
            return

        if self.ctrl_pressed:
            # Ctrl+scroll: zoom x-axis (all axes)
            # Adjust this factor to control zoom speed (higher = faster zoom)
            base_scale = 1.2  # Reduced from 1.3 for finer control

            # Get current x limits
            xlim = self.ax1.get_xlim()  # Use ax1 as the reference

            # Use mouse x position as the zoom center point
            xdata = event.xdata

            # If mouse is outside the plot, use center of view as focal point
            if xdata is None:
                xdata = (xlim[0] + xlim[1]) / 2

            # Calculate new limits
            if event.button == 'up':  # Zoom in
                new_xlim = [xdata - (xdata - xlim[0]) / base_scale,
                            xdata + (xlim[1] - xdata) / base_scale]
            else:  # Zoom out
                new_xlim = [xdata - (xdata - xlim[0]) * base_scale,
                            xdata + (xlim[1] - xdata) * base_scale]

                # If zooming out would make the view larger than the full dataset, limit it
                if hasattr(self, 'time') and self.time is not None:
                    # Calculate full time range in minutes
                    full_range = (self.time[-1] - self.time[0]) / 60

                    # Don't zoom out beyond data boundaries plus small margins
                    margin = full_range * 0.05  # 5% margin
                    min_time = self.time[0] / 60 - margin
                    max_time = self.time[-1] / 60 + margin

                    # Modify new limits if they exceed data boundaries
                    if new_xlim[0] < min_time:
                        new_xlim[0] = min_time
                    if new_xlim[1] > max_time:
                        new_xlim[1] = max_time

            # Apply new limits to all axes
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.set_xlim(new_xlim)

            # Add visual feedback for zoom level
            # Update status bar if it exists
            if hasattr(self, 'status_label'):
                minutes_shown = new_xlim[1] - new_xlim[0]
                if minutes_shown < 60:
                    self.status_label.config(text=f"X Scale: {minutes_shown:.1f} minutes shown", fg="blue")
                else:
                    hours = minutes_shown / 60
                    self.status_label.config(text=f"X Scale: {hours:.1f} hours shown", fg="blue")

            # Optional: Highlight zoom reset button when zoomed in
            if hasattr(self, 'reset_view_button'):
                full_range = (self.time[-1] - self.time[0]) / 60
                current_range = new_xlim[1] - new_xlim[0]
                if current_range < full_range * 0.95:  # If zoomed in more than 5%
                    self.reset_view_button.config(bg='lightblue', fg='darkblue', relief=tk.RAISED)
                else:
                    self.reset_view_button.config(bg='lightblue', fg='black', relief=tk.FLAT)
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

            # Apply new limits to the appropriate panel
            if event.inaxes == self.ax1:
                self.ax1.set_ylim(new_ylim)
            elif event.inaxes == self.ax2:
                self.ax2.set_ylim(new_ylim)

        # Redraw the figure
        self.canvas.draw_idle()

    def update_filter(self, _=None, preserve_blanking=True):
        """Update filter parameters and reprocess data while preserving manual blanking."""
        try:
            # Show processing indicator
            self.root.config(cursor="watch")
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Processing...", fg="blue")
            self.root.update_idletasks()  # Force UI update

            # Save current axis limits to restore them after processing
            x_lim = self.ax1.get_xlim()
            y1_lim = self.ax1.get_ylim()
            y2_lim = self.ax2.get_ylim()

            # Store blanked signal if blanking was applied
            primary_blanked = None
            secondary_blanked = None
            primary_has_blanking = hasattr(self, 'primary_has_blanking') and self.primary_has_blanking
            secondary_has_blanking = hasattr(self, 'secondary_has_blanking') and self.secondary_has_blanking

            # Store blanking regions for later reapplication
            blanking_regions_backup = []
            if preserve_blanking and hasattr(self, 'blanking_regions'):
                blanking_regions_backup = self.blanking_regions.copy()

            # Get updated parameter values from UI controls
            self.low_cutoff = self.low_slider.get()
            self.high_cutoff = self.high_slider.get()
            self.downsample_factor = int(self.downsample_slider.get())
            self.artifact_threshold = self.artifact_slider.get()
            self.drift_correction = self.drift_var.get()
            self.drift_degree = self.poly_degree_var.get()

            # Process data with new parameters
            result = process_data(
                self.time, self.raw_analog_1, self.raw_analog_2, self.digital_1, self.digital_2,
                low_cutoff=self.low_cutoff, high_cutoff=self.high_cutoff,
                downsample_factor=self.downsample_factor, artifact_threshold=self.artifact_threshold,
                drift_correction=self.drift_correction, drift_degree=self.drift_degree,
                edge_protection=self.edge_protection_var.get()
            )

            # Unpack results
            self.processed_time, self.processed_signal, self.processed_signal_no_control, \
                self.processed_analog_2, self.downsampled_raw_analog_1, self.downsampled_raw_analog_2, \
                self.processed_digital_1, self.processed_digital_2, self.artifact_mask, \
                self.drift_curve = result

            # Process secondary file with the same parameters if it exists
            if hasattr(self, 'secondary_file_path') and self.secondary_file_path:
                secondary_result = process_data(
                    self.secondary_time, self.secondary_analog_1, self.secondary_analog_2,
                    self.secondary_digital_1, self.secondary_digital_2,
                    low_cutoff=self.low_cutoff, high_cutoff=self.high_cutoff,
                    downsample_factor=self.downsample_factor, artifact_threshold=self.artifact_threshold,
                    drift_correction=self.drift_correction, drift_degree=self.drift_degree,
                    edge_protection=self.edge_protection_var.get()
                )

                # Unpack secondary results
                self.secondary_processed_time, self.secondary_processed_signal, self.secondary_processed_signal_no_control, \
                    self.secondary_processed_analog_2, self.secondary_downsampled_raw_analog_1, self.secondary_downsampled_raw_analog_2, \
                    self.secondary_processed_digital_1, self.secondary_processed_digital_2, self.secondary_artifact_mask, \
                    self.secondary_drift_curve = secondary_result

            # Reapply all blanking if needed
            if preserve_blanking and blanking_regions_backup:
                self.blanking_regions = blanking_regions_backup
                self.reapply_all_blanking()  # This will handle all blanking regions

            # Update plot data
            self.line.set_xdata(self.processed_time / 60)
            self.line.set_ydata(self.processed_signal)

            # Update the no-control signal plot
            if hasattr(self, 'line_no_control') and self.line_no_control is not None:
                self.line_no_control.set_xdata(self.processed_time / 60)
                self.line_no_control.set_ydata(self.processed_signal_no_control)

            # Update raw signal plot
            self.raw_line.set_xdata(self.processed_time / 60)
            self.raw_line.set_ydata(self.downsampled_raw_analog_1)

            if hasattr(self, 'raw_line2') and self.downsampled_raw_analog_2 is not None:
                self.raw_line2.set_xdata(self.processed_time / 60)
                self.raw_line2.set_ydata(self.downsampled_raw_analog_2)

            # Also update secondary line if exists
            if hasattr(self, 'secondary_line') and self.secondary_line is not None:
                self.secondary_line.set_xdata(self.secondary_processed_time / 60)
                self.secondary_line.set_ydata(self.secondary_processed_signal)

                if hasattr(self, 'secondary_line_no_control') and self.secondary_line_no_control is not None:
                    self.secondary_line_no_control.set_xdata(self.secondary_processed_time / 60)
                    self.secondary_line_no_control.set_ydata(self.secondary_processed_signal_no_control)

            # Update drift lines if they exist
            if hasattr(self, 'drift_line') and self.drift_line is not None and self.drift_curve is not None:
                self.drift_line.set_xdata(self.processed_time / 60)
                self.drift_line.set_ydata(self.drift_curve)

            if hasattr(self,
                       'secondary_drift_line') and self.secondary_drift_line is not None and self.secondary_drift_curve is not None:
                self.secondary_drift_line.set_xdata(self.secondary_processed_time / 60)
                self.secondary_drift_line.set_ydata(self.secondary_drift_curve)

            # Update artifact markers
            self.update_artifact_markers()
            if hasattr(self, 'secondary_processed_signal') and self.secondary_processed_signal is not None:
                # Only call this if we actually have a secondary signal
                if hasattr(self,
                           'secondary_artifact_markers_processed') and self.secondary_artifact_markers_processed is not None:
                    self.update_secondary_artifact_markers()

            # Restore axis limits
            self.ax1.set_xlim(x_lim)
            self.ax1.set_ylim(y1_lim)
            self.ax2.set_ylim(y2_lim)

            # CRITICAL: Redraw the canvas
            self.canvas.draw_idle()

            # Reset the cursor
            self.root.config(cursor="")
            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"Filters updated: low={self.low_cutoff}, high={self.high_cutoff}",
                                         fg="green")

        except Exception as e:
            print(f"Error in update_filter: {e}")
            import traceback
            traceback.print_exc()

            # Show error in status
            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"Error: {str(e)}", fg="red")
            self.root.config(cursor="")  # Restore cursor

    def update_y_limits(self):
        """Update y-axis limits for processed signal with asymmetric scaling."""
        # Get all available signals
        signals = [self.processed_signal]
        if hasattr(self, 'secondary_processed_signal') and self.secondary_processed_signal is not None:
            signals.append(self.secondary_processed_signal)

        # Calculate combined statistics
        combined_signal = np.concatenate(signals)
        mean_val = np.mean(combined_signal)
        std_val = np.std(combined_signal)

        # Get min and max for all signals
        min_val = min([np.min(signal) for signal in signals])
        max_val = max([np.max(signal) for signal in signals])

        # Apply asymmetric scaling, ensuring all data is visible
        lower_bound = min(mean_val - 2 * std_val, min_val - 0.1 * (max_val - min_val))
        upper_bound = max(mean_val + 5 * std_val, max_val + 0.1 * (max_val - min_val))

        self.ax1.set_ylim(lower_bound, upper_bound)

        # Function to detect flat regions
        def find_flat_regions(signal, threshold=1e-6):
            """Find regions where the signal is flat (standard deviation near zero)"""
            flat_regions = []
            in_flat_region = False
            flat_start = 0

            # Use a window to compute rolling standard deviation
            window_size = min(100, len(signal) // 100)
            if window_size < 3:
                return []  # Signal too short

            for i in range(len(signal) - window_size):
                std = np.std(signal[i:i + window_size])
                if std < threshold and not in_flat_region:
                    flat_start = i
                    in_flat_region = True
                elif std >= threshold and in_flat_region:
                    if i - flat_start > window_size:  # Only mark substantial regions
                        flat_regions.append((flat_start, i))
                    in_flat_region = False

            # Check if we ended in a flat region
            if in_flat_region and len(signal) - flat_start > window_size:
                flat_regions.append((flat_start, len(signal)))

            return flat_regions

        # Mark flat regions in primary signal
        primary_flats = find_flat_regions(self.processed_signal)
        for start, end in primary_flats:
            if start > 0:  # Only mark flattened beginnings
                patch = plt.Rectangle(
                    (self.processed_time[start] / 60, self.ax1.get_ylim()[0]),
                    (self.processed_time[end - 1] - self.processed_time[start]) / 60,
                    self.ax1.get_ylim()[1] - self.ax1.get_ylim()[0],
                    facecolor='lightgray', alpha=0.4, zorder=-100)
                patch.is_flattened_marker = True
                self.ax1.add_patch(patch)

        # Mark flat regions in secondary signal if available
        if hasattr(self, 'secondary_processed_signal') and self.secondary_processed_signal is not None:
            secondary_flats = find_flat_regions(self.secondary_processed_signal)
            for start, end in secondary_flats:
                if start > 0:  # Only mark flattened beginnings
                    patch = plt.Rectangle(
                        (self.secondary_processed_time[start] / 60, self.ax1.get_ylim()[0]),
                        (self.secondary_processed_time[end - 1] - self.secondary_processed_time[start]) / 60,
                        self.ax1.get_ylim()[1] - self.ax1.get_ylim()[0],
                        facecolor='mistyrose', alpha=0.4, zorder=-100)
                    patch.is_flattened_marker = True
                    self.ax1.add_patch(patch)

    def setup_window_resize_handler(self):
        """Setup handler for window resize events"""
        # Add this to your __init__ method
        self.root.bind("<Configure>", self.on_window_resize)
        self.resize_timer = None

    def on_window_resize(self, event):
        """Handle window resize events with debouncing"""
        # Only respond to root window resizing, not child widgets
        if event.widget != self.root:
            return

        # Cancel previous timer if it exists
        if self.resize_timer is not None:
            self.root.after_cancel(self.resize_timer)

        # Set new timer to update layout after resize is complete
        self.resize_timer = self.root.after(200, self.update_layout_after_resize)

    def update_layout_after_resize(self):
        """Update layout after window resize is complete"""
        if self.fig is not None:
            # Get new window dimensions
            window_width = self.root.winfo_width()
            window_height = self.root.winfo_height()

            # Adjust figure size
            self.fig.set_size_inches(window_width / 100, (window_height * 0.6) / 100)

            # Adjust slider length
            slider_length = min(800, int(window_width * 0.6))
            for slider in [self.low_slider, self.high_slider, self.downsample_slider, self.artifact_slider]:
                if hasattr(self, slider.__str__()) and slider is not None:
                    slider.config(length=slider_length)

            # Redraw canvas
            self.fig.tight_layout()
            self.canvas.draw_idle()

    def toggle_scrolling(self):
        """Toggle between plot zooming mode and GUI scrolling mode"""
        self.scrolling_enabled = not self.scrolling_enabled

        if self.scrolling_enabled:
            # Enable GUI scrolling mode
            self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.scroll_toggle_btn.config(text="Disable Scrolling (Enable Zoom)", bg='salmon')

            # Unbind zoom events from mousewheel
            if hasattr(self, 'zoom_id'):
                self.canvas.mpl_disconnect(self.zoom_id)
                self.zoom_id = None

            # Enable GUI scrolling with mousewheel
            self.bind_mousewheel()

        else:
            # Enable plot zooming mode
            self.v_scrollbar.pack_forget()
            self.scroll_toggle_btn.config(text="Enable Scrolling (Disable Zoom)", bg='lightblue')

            # Connect mousewheel to plot zooming
            self.connect_zoom_events()

            # Disable GUI scrolling
            self.unbind_mousewheel()

    def on_frame_configure(self, event=None):
        """Update the scrollregion when the frame changes size"""
        # Get the required size of the frame
        self.scroll_canvas.update_idletasks()
        self.main_frame.update_idletasks()

        # Get actual sizes
        width = self.main_frame.winfo_reqwidth()
        height = self.main_frame.winfo_reqheight()

        # Set a reasonable maximum height
        max_height = min(height + 100, 3000)  # Limit to 3000 pixels

        # Configure scroll region
        self.scroll_canvas.configure(scrollregion=(0, 0, width, max_height))

        # Make sure the frame fills the canvas width
        if event:
            self.scroll_canvas.itemconfig(self.canvas_window, width=width)

    def on_canvas_configure(self, event):
        """When canvas is resized, also resize the window inside it"""
        if event and hasattr(self, 'canvas_window'):
            # Get the current canvas width
            canvas_width = event.width

            # Update the internal frame width to match canvas
            self.scroll_canvas.itemconfig(self.canvas_window, width=canvas_width)

    def bind_mousewheel(self):
        """Bind mousewheel to GUI scrolling"""

        def _on_mousewheel(event):
            # Different handling for different OS
            if event.num == 4 or event.delta > 0:
                self.scroll_canvas.yview_scroll(-1, "units")
            elif event.num == 5 or event.delta < 0:
                self.scroll_canvas.yview_scroll(1, "units")

        # Unbind any previous bindings to avoid duplicates
        self.unbind_mousewheel()

        # Bind for different platforms
        if sys.platform.startswith('win'):
            self._wheel_binding = self.scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        elif sys.platform.startswith('darwin'):  # macOS
            self._wheel_binding = self.scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        else:  # Linux
            self._wheel_binding1 = self.scroll_canvas.bind_all("<Button-4>", _on_mousewheel)
            self._wheel_binding2 = self.scroll_canvas.bind_all("<Button-5>", _on_mousewheel)

        def unbind_mousewheel(self):
            """Unbind mousewheel from GUI scrolling"""
            if hasattr(self, '_wheel_binding'):
                self.scroll_canvas.unbind_all("<MouseWheel>")
                self._wheel_binding = None
            if hasattr(self, '_wheel_binding1'):
                self.scroll_canvas.unbind_all("<Button-4>")
                self._wheel_binding1 = None
            if hasattr(self, '_wheel_binding2'):
                self.scroll_canvas.unbind_all("<Button-5>")
                self._wheel_binding2 = None



    # --- New Methods for Valley Tab ---

    def run_valley_detection_from_tab(self, _=None):
        """Runs valley detection using parameters from the valley tab."""
        if self.processed_signal is None:
            return  # No data loaded

        # Get valley detection parameters
        prominence = self.valley_prominence_slider.get()
        width_val = self.valley_width_slider.get()
        distance_val = self.valley_distance_slider.get()

        # Process signals based on selection
        selection = self.valley_signal_selection_var.get()  # Use valley-specific variable

        # Process primary signal
        if selection in ["Primary", "Both"]:
            self._run_valley_detection_on_signal(
                self.processed_signal,
                self.processed_time,
                "primary",
                prominence=prominence,
                width_val=width_val,
                distance_val=distance_val
            )

        # Process secondary signal if available
        if selection in ["Secondary", "Both"] and hasattr(self,
                                                          'secondary_processed_signal') and self.secondary_processed_signal is not None:
            self._run_valley_detection_on_signal(
                self.secondary_processed_signal,
                self.secondary_processed_time,
                "secondary",
                prominence=prominence,
                width_val=width_val,
                distance_val=distance_val
            )
        elif selection in ["Secondary", "Both"]:
            if hasattr(self, 'status_label'):
                self.status_label.config(text="No secondary signal available", fg="red")

        # Update displays
        self.update_valley_display()
        self.update_valley_metrics_display()

    # 6. Helper method for running valley detection on a specific signal
    def _run_valley_detection_on_signal(self, signal, time_array, signal_type, prominence, width_val, distance_val):
        """Helper function to run valley detection on a specific signal"""
        if signal is None or time_array is None:
            return

        # Convert width and distance from seconds to sample points
        dt = np.mean(np.diff(time_array))
        width = None if width_val <= 0 else width_val / dt
        distance = None if distance_val <= 0 else int(distance_val / dt)

        # Find peaks and valleys - we need both for complete metrics
        peaks, valleys = find_peaks_valleys(
            signal, time_array,
            prominence=prominence, width=width, distance=distance
        )

        # Store the results in the appropriate variables
        if signal_type == "primary":
            # We need to have peaks to calculate valley metrics
            self.peaks = peaks
            self.valleys = valleys
            if len(peaks['indices']) > 0 and len(valleys['indices']) > 0:
                self.valley_metrics = self.calculate_valley_metrics(peaks, valleys, signal, time_array)
            else:
                self.valley_metrics = None
        else:  # "secondary"
            self.secondary_peaks = peaks
            self.secondary_valleys = valleys
            if len(peaks['indices']) > 0 and len(valleys['indices']) > 0:
                self.secondary_valley_metrics = self.calculate_valley_metrics(peaks, valleys, signal, time_array)
            else:
                self.secondary_valley_metrics = None

        # Status update
        if hasattr(self, 'status_label'):
            count = len(valleys['indices']) if valleys is not None else 0
            region = "PVN" if signal_type == "primary" else "SON"
            self.status_label.config(
                text=f"Detected {count} {region} valleys",
                fg="green"
            )

    def clear_valleys(self):
        """Clear detected valleys and their display/metrics."""
        self.valleys = None
        self.valley_metrics = None

        # Clear valley annotations from the plot
        for annotation in self.valley_annotations:
            annotation.remove() if hasattr(annotation, 'remove') else None
        self.valley_annotations = []
        for line in self.valley_lines:
            line.remove() if hasattr(line, 'remove') else None
        self.valley_lines = []

        # Clear valley metrics display
        if hasattr(self, 'valley_metrics_tree'):
            self.valley_metrics_tree.delete(*self.valley_metrics_tree.get_children())

        self.canvas.draw_idle()
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Valleys cleared", fg="green")

    def update_valley_display(self):
        """Update the display of valleys on the plot."""
        # Clear previous valley annotations
        for annotation in self.valley_annotations:
            if hasattr(annotation, 'remove'):
                annotation.remove()
        self.valley_annotations = []

        for line in self.valley_lines:
            if hasattr(line, 'remove'):
                line.remove()
        self.valley_lines = []

        # Exit if nothing to show or valleys are hidden
        if not self.show_valleys_var.get():
            self.canvas.draw_idle()
            return

        # Display primary valleys if available and selected
        if self.valleys is not None and self.valley_signal_selection_var.get() in ["Primary", "Both"]:
            self._display_valleys_on_signal(
                self.valleys,
                self.processed_signal,
                self.processed_time,
                'blue',  # color
                primary=True
            )

        # Display secondary valleys if available and selected
        if hasattr(self, 'secondary_valleys') and self.secondary_valleys is not None and \
                self.valley_signal_selection_var.get() in ["Secondary", "Both"]:

            # Check if secondary signal is available
            if hasattr(self, 'secondary_processed_signal') and self.secondary_processed_signal is not None:
                self._display_valleys_on_signal(
                    self.secondary_valleys,
                    self.secondary_processed_signal,
                    self.secondary_processed_time,
                    'purple',  # different color
                    primary=False
                )

        # Redraw
        self.canvas.draw_idle()

    def update_valley_visibility(self, _=None):
        """Update visibility of valleys based on checkbox settings in valley tab."""
        self.update_valley_display()

    # Inside PhotometryViewer class

    def calculate_valley_metrics(self, peak_data, valley_data, signal, time):
        """Calculate metrics for detected valleys (analogous to peaks)."""
        # --- ADDED CHECKS ---
        if valley_data is None or not valley_data.get('indices', np.array([])).size:
            print("Valley metrics: No valleys found.")
            return None  # No valleys to calculate metrics for
        if peak_data is None or not peak_data.get('indices', np.array([])).size:
            print("Valley metrics: No peaks found (needed for context). Returning None.")
            return None  # Need peaks for context
        # --- END ADDED CHECKS ---

        valley_indices = valley_data['indices']
        peak_indices = peak_data['indices']

        metrics = {
            'area_above_valley': [],
            'full_width_half_depth': [],  # Width relative to peaks
            'time_to_preceding_peak': [],
            'time_to_following_peak': [],
            'preceding_peak_idx': [],
            'following_peak_idx': []
        }

        print(f"Calculating metrics for {len(valley_indices)} valleys...")  # Debug print

        for i, valley_idx in enumerate(valley_indices):
            # Initialize metrics for this valley with NaN
            fwhd_val = np.nan
            area_val = np.nan
            time_to_prec = np.nan
            time_to_foll = np.nan
            prec_peak_idx = np.nan  # Use NaN instead of 0 for index if not found
            foll_peak_idx = np.nan  # Use NaN instead of len-1

            # Find surrounding peaks
            preceding_peaks_indices = np.where(peak_indices < valley_idx)[0]
            following_peaks_indices = np.where(peak_indices > valley_idx)[0]

            valid_prec_peak = False
            if preceding_peaks_indices.size > 0:
                prec_peak_idx = peak_indices[preceding_peaks_indices[-1]]
                time_to_prec = time[valley_idx] - time[prec_peak_idx]
                valid_prec_peak = True

            valid_foll_peak = False
            if following_peaks_indices.size > 0:
                foll_peak_idx = peak_indices[following_peaks_indices[0]]
                time_to_foll = time[foll_peak_idx] - time[valley_idx]
                valid_foll_peak = True

            metrics['preceding_peak_idx'].append(prec_peak_idx)
            metrics['following_peak_idx'].append(foll_peak_idx)
            metrics['time_to_preceding_peak'].append(time_to_prec)
            metrics['time_to_following_peak'].append(time_to_foll)

            # Only calculate width/area if surrounded by valid peaks
            if valid_prec_peak and valid_foll_peak:
                try:
                    valley_depth = signal[valley_idx]
                    # Ensure indices are valid before accessing signal
                    prec_peak_height = signal[int(prec_peak_idx)]
                    foll_peak_height = signal[int(foll_peak_idx)]

                    peak_level = np.mean([prec_peak_height, foll_peak_height])
                    half_depth_level = valley_depth + (peak_level - valley_depth) / 2

                    # Find width at half depth
                    left_segment_indices = np.arange(int(prec_peak_idx), valley_idx + 1)
                    right_segment_indices = np.arange(valley_idx, int(foll_peak_idx) + 1)

                    if left_segment_indices.size > 1 and right_segment_indices.size > 1:
                        left_segment = signal[left_segment_indices]
                        right_segment = signal[right_segment_indices]
                        time_left = time[left_segment_indices]
                        time_right = time[right_segment_indices]

                        # Interpolate to find exact crossing times
                        interp_left = interp1d(time_left, left_segment - half_depth_level, kind='linear',
                                               bounds_error=False, fill_value=np.nan)
                        interp_right = interp1d(time_right, right_segment - half_depth_level, kind='linear',
                                                bounds_error=False, fill_value=np.nan)

                        # Find roots (where signal - half_depth_level = 0)
                        # This requires a root-finding algorithm, which is complex.
                        # Let's stick to the simpler index-based method for now, but be aware of limitations.

                        # --- Using simpler index-based crossing ---
                        left_crosses = np.where(left_segment <= half_depth_level)[0]
                        right_crosses = np.where(right_segment <= half_depth_level)[0]

                        if left_crosses.size > 0 and right_crosses.size > 0:
                            left_cross_idx_rel = left_crosses[-1]  # Last point below half depth on the way down
                            right_cross_idx_rel = right_crosses[0]  # First point below half depth on the way up

                            left_cross_idx_abs = int(prec_peak_idx) + left_cross_idx_rel
                            right_cross_idx_abs = valley_idx + right_cross_idx_rel

                            fwhd_val = time[right_cross_idx_abs] - time[left_cross_idx_abs]
                        # --- End simpler method ---

                    # Calculate area above valley up to peak level
                    segment_indices = np.arange(int(prec_peak_idx), int(foll_peak_idx) + 1)
                    if segment_indices.size > 1:
                        segment_signal = signal[segment_indices]
                        segment_time = time[segment_indices]
                        area_val = trapezoid(np.maximum(0, peak_level - segment_signal), segment_time)

                except IndexError as ie:
                    print(f"IndexError calculating metrics for valley index {valley_idx}: {ie}")
                except Exception as e:
                    print(f"Error calculating metrics for valley index {valley_idx}: {e}")

            # Append calculated (or NaN) values
            metrics['full_width_half_depth'].append(fwhd_val)
            metrics['area_above_valley'].append(area_val)

        print(
            f"Finished valley metrics calculation. Example width: {metrics['full_width_half_depth'][:5]}")  # Debug print
        return metrics

    def update_valley_metrics_display(self):
        """Update the valley metrics tree view"""
        if not hasattr(self, 'valley_metrics_tree'):
            return

        self.valley_metrics_tree.delete(*self.valley_metrics_tree.get_children())

        # Determine which metrics to display based on selection
        if self.signal_selection_var.get() == "Primary" and self.valleys is not None and self.valley_metrics is not None:
            # Display primary valleys
            self._add_valley_metrics_to_tree(self.valleys, self.valley_metrics, "V")

        elif self.signal_selection_var.get() == "Secondary" and hasattr(self,
                                                                        'secondary_valleys') and self.secondary_valleys is not None:
            if hasattr(self, 'secondary_valley_metrics') and self.secondary_valley_metrics is not None:
                # Display secondary valleys
                self._add_valley_metrics_to_tree(self.secondary_valleys, self.secondary_valley_metrics, "SV")

        elif self.signal_selection_var.get() == "Both":
            # Display both, with primary first then secondary
            if self.valleys is not None and self.valley_metrics is not None:
                self._add_valley_metrics_to_tree(self.valleys, self.valley_metrics, "V")

            if hasattr(self, 'secondary_valleys') and self.secondary_valleys is not None:
                if hasattr(self, 'secondary_valley_metrics') and self.secondary_valley_metrics is not None:
                    self._add_valley_metrics_to_tree(self.secondary_valleys, self.secondary_valley_metrics, "SV")

    def _add_valley_metrics_to_tree(self, valleys, metrics, prefix=""):
        """Helper to add valley metrics to the tree with appropriate prefix"""
        if not valleys or not metrics:
            return

        for i, (idx, time_val, depth) in enumerate(zip(
                valleys['indices'], valleys['times'], valleys['depths'])):

            # Get metrics safely
            width = metrics.get('full_width_half_depth', [])[i] if i < len(
                metrics.get('full_width_half_depth', [])) else np.nan
            area = metrics.get('area_above_valley', [])[i] if i < len(metrics.get('area_above_valley', [])) else np.nan

            # Format values
            values = (
                f"{prefix}{i + 1}",  # Add prefix for Primary/Secondary distinction
                f"{time_val:.2f}",
                f"{depth:.2f}",
                f"{width:.2f}" if not np.isnan(width) else "N/A",
                f"{area:.2f}" if not np.isnan(area) else "N/A"
            )

            # Add to tree
            try:
                self.valley_metrics_tree.insert("", "end", values=values)
            except tk.TclError as e:
                print(f"Error inserting into valley tree: {e}. Values: {values}")


    def export_valley_data(self):
        """Export valley data to CSV file"""
        if self.valleys is None or self.valley_metrics is None:
            messagebox.showinfo("Export Error", "No valley data available to export")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Valley Data"
        )
        if not file_path: return

        try:
            import csv
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header based on valley_metrics keys and valley data
                header = ["Valley #", "Time (s)", "Depth (%)",
                          "Width@HalfDepth (s)", "Area Above Valley",
                          "Preceding Peak Idx", "Following Peak Idx",
                          "Time From Prec Peak (s)", "Time To Foll Peak (s)"]
                writer.writerow(header)

                # Write data for each valley
                for i, (idx, time_s, depth_val) in enumerate(zip(
                        self.valleys['indices'], self.valleys['times'], self.valleys['depths'])):
                    width = self.valley_metrics['full_width_half_depth'][i]
                    area = self.valley_metrics['area_above_valley'][i]
                    prec_peak = self.valley_metrics['preceding_peak_idx'][i]
                    foll_peak = self.valley_metrics['following_peak_idx'][i]
                    time_from_prec = self.valley_metrics['time_to_preceding_peak'][i]
                    time_to_foll = self.valley_metrics['time_to_following_peak'][i]

                    writer.writerow([
                        i + 1,
                        f"{time_s:.4f}",
                        f"{depth_val:.4f}",
                        f"{width:.4f}" if not np.isnan(width) else "N/A",
                        f"{area:.4f}" if not np.isnan(area) else "N/A",
                        prec_peak,
                        foll_peak,
                        f"{time_from_prec:.4f}" if not np.isnan(time_from_prec) else "N/A",
                        f"{time_to_foll:.4f}" if not np.isnan(time_to_foll) else "N/A"
                    ])

                # Add summary statistics if desired
                writer.writerow([])
                writer.writerow(["Summary Statistics"])
                avg_depth = np.nanmean(self.valleys['depths'])
                avg_width = np.nanmean(self.valley_metrics['full_width_half_depth'])
                avg_area = np.nanmean(self.valley_metrics['area_above_valley'])

                writer.writerow(["Average Depth (%)", f"{avg_depth:.4f}"])
                writer.writerow(["Average Width@HalfDepth (s)", f"{avg_width:.4f}"])
                writer.writerow(["Average Area Above", f"{avg_area:.4f}"])
                writer.writerow(["Total Valleys", len(self.valleys['indices'])])

            messagebox.showinfo("Export Complete", f"Valley data successfully exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting valley data: {str(e)}")

    # 4. Add helper method for running peak detection on a specific signal
    def _run_peak_detection_on_signal(self, signal, time_array, signal_type, prominence, width_val, distance_val,
                                      threshold):
        """Helper function to run peak detection on a specific signal"""
        print(f"Running peak detection on {signal_type} signal")

        if signal is None or time_array is None:
            print("Signal or time array is None")
            return

        # Convert width and distance from seconds to sample points
        dt = np.mean(np.diff(time_array))
        width = None if width_val <= 0 else width_val / dt
        distance = None if distance_val <= 0 else int(distance_val / dt)

        print(f"Converted parameters: width={width}, distance={distance}")

        # IMPORTANT: Use different prominence values for peaks and valleys
        # Peaks are usually more prominent than valleys
        print("Finding peaks and valleys...")
        peaks = find_peaks(signal, prominence=prominence, width=width,
                           distance=distance, height=threshold)[0]
        valleys = find_peaks(-signal, prominence=prominence / 2, width=width,
                             distance=distance)[0]

        # Convert to the expected format
        peak_data = {
            'indices': peaks,
            'times': time_array[peaks],
            'heights': signal[peaks],
            'properties': {}
        }

        valley_data = {
            'indices': valleys,
            'times': time_array[valleys],
            'depths': signal[valleys],
            'properties': {}
        }

        print(f"Detection results: {len(peak_data['indices'])} peaks, {len(valley_data['indices'])} valleys")

        # Store the results in the appropriate variables
        if signal_type == "primary":
            self.peaks = peak_data
            self.valleys = valley_data

            # CRITICAL CHANGE: Force peak metrics calculation even if valleys are limited
            print("Calculating peak metrics...")
            if len(peak_data['indices']) > 0:
                # If we don't have valleys, create artificial ones at the signal boundaries
                if len(valley_data['indices']) == 0:
                    print("No valleys detected - creating artificial valleys")
                    valley_data['indices'] = np.array([0, len(signal) - 1])
                    valley_data['times'] = np.array([time_array[0], time_array[-1]])
                    valley_data['depths'] = np.array([signal[0], signal[-1]])
                    self.valleys = valley_data

                # Now calculate metrics
                self.peak_metrics = self.simple_peak_metrics(peak_data, valley_data, signal, time_array)
                print(f"Metrics calculation result: {self.peak_metrics is not None}")
                if self.peak_metrics is not None:
                    print(f"Metrics keys: {list(self.peak_metrics.keys())}")
                    print(f"Number of metrics: {len(self.peak_metrics['full_width_half_max'])}")
            else:
                print("No peaks detected, cannot calculate metrics")
                self.peak_metrics = None
        else:  # "secondary"
            self.secondary_peaks = peaks
            self.secondary_valleys = valleys
            if len(peaks['indices']) > 0 and len(valleys['indices']) > 0:
                self.secondary_peak_metrics = calculate_peak_metrics(peaks, valleys, signal, time_array)
            else:
                self.secondary_peak_metrics = None

        # Status update
        if hasattr(self, 'status_label'):
            count = len(peak_data['indices']) if peak_data is not None else 0
            region = "PVN" if signal_type == "primary" else "SON"
            self.status_label.config(
                text=f"Detected {count} {region} peaks",
                fg="green"
            )

    def simple_peak_metrics(self, peak_data, valley_data, signal, time):
        """
        Simplified version of peak metrics calculation that's more robust
        to limited valley data.
        """
        print(f"Running simplified peak metrics calculation...")

        peak_indices = peak_data['indices']
        valley_indices = valley_data['indices']

        # Initialize metrics dictionary with empty lists
        metrics = {
            'area_under_curve': [],
            'full_width_half_max': [],
            'rise_time': [],
            'decay_time': [],
            'preceding_valley': [],
            'following_valley': []
        }

        # Calculate baseline as 10th percentile of signal
        baseline = np.percentile(signal, 10)
        print(f"Using baseline: {baseline}")

        for i, peak_idx in enumerate(peak_indices):
            try:
                # Find preceding valley (closest valley before peak)
                preceding_valleys = valley_indices[valley_indices < peak_idx]
                if len(preceding_valleys) > 0:
                    preceding_valley_idx = preceding_valleys[-1]
                else:
                    # Use signal start as fallback
                    preceding_valley_idx = 0

                # Find following valley (closest valley after peak)
                following_valleys = valley_indices[valley_indices > peak_idx]
                if len(following_valleys) > 0:
                    following_valley_idx = following_valleys[0]
                else:
                    # Use signal end as fallback
                    following_valley_idx = len(signal) - 1

                # Store valley indices
                metrics['preceding_valley'].append(preceding_valley_idx)
                metrics['following_valley'].append(following_valley_idx)

                # Calculate peak height and half-max
                peak_height = signal[peak_idx]
                half_max_height = baseline + (peak_height - baseline) / 2

                # Find width at half max (simplified approach)
                left_idx = peak_idx
                while left_idx > preceding_valley_idx and signal[left_idx] > half_max_height:
                    left_idx -= 1

                right_idx = peak_idx
                while right_idx < following_valley_idx and signal[right_idx] > half_max_height:
                    right_idx += 1

                # Calculate width, rise time, and decay time
                fwhm = time[right_idx] - time[left_idx]
                rise_time = time[peak_idx] - time[left_idx]
                decay_time = time[right_idx] - time[peak_idx]

                # Calculate area using trapezoidal rule
                peak_segment = signal[left_idx:right_idx + 1]
                peak_times = time[left_idx:right_idx + 1]
                area = np.trapz(peak_segment - baseline, peak_times)

                # Store metrics
                metrics['full_width_half_max'].append(fwhm)
                metrics['rise_time'].append(rise_time)
                metrics['decay_time'].append(decay_time)
                metrics['area_under_curve'].append(max(0, area))

            except Exception as e:
                print(f"Error calculating metrics for peak {i}: {e}")
                # Add default values
                metrics['full_width_half_max'].append(0.0)
                metrics['rise_time'].append(0.0)
                metrics['decay_time'].append(0.0)
                metrics['area_under_curve'].append(0.0)
                metrics['preceding_valley'].append(0)
                metrics['following_valley'].append(0)

        print(f"Completed metrics calculation for {len(peak_indices)} peaks")
        return metrics

    def run_peak_detection_from_tab(self, _=None):
        """Runs peak detection using parameters from the peak tab."""
        print("\n--- Starting Peak Detection ---")

        if self.processed_signal is None:
            print("No signal data loaded")
            return  # No data loaded

        # Get peak detection parameters
        prominence = self.peak_prominence_slider.get()
        width_val = self.peak_width_slider.get()
        distance_val = self.peak_distance_slider.get()
        threshold = self.peak_threshold_slider.get() if hasattr(self, 'peak_threshold_slider') else None

        print(
            f"Peak parameters: prominence={prominence}, width={width_val}, distance={distance_val}, threshold={threshold}")

        # Get selected signal
        selection = self.signal_selection_var.get()
        print(f"Signal selection: {selection}")

        # Process selected signals
        if selection in ["Primary", "Both"]:
            print("Processing primary signal...")
            self._run_peak_detection_on_signal(
                self.processed_signal,
                self.processed_time,
                "primary",
                prominence=prominence,
                width_val=width_val,
                distance_val=distance_val,
                threshold=threshold
            )

            # Debug
            if self.peaks is not None:
                print(f"Primary detection found {len(self.peaks['indices'])} peaks")
            else:
                print("Primary detection didn't set peaks")

            if self.peak_metrics is not None:
                print(f"Primary detection calculated metrics: {list(self.peak_metrics.keys())}")
            else:
                print("Primary detection didn't set metrics")

        # Update displays
        print("Updating displays...")
        self.update_peak_display()
        self.update_metrics_display()
        print("--- Peak Detection Completed ---\n")

    def update_peak_display(self):
        """Update the display of peaks on the plot."""
        # Clear previous peak annotations
        for annotation in self.peak_annotations:
            annotation.remove() if hasattr(annotation, 'remove') else None
        self.peak_annotations = []
        for line in self.peak_lines:
            line.remove() if hasattr(line, 'remove') else None
        self.peak_lines = []

        # Clear peak feature annotations
        for annotation in self.peak_feature_annotations:
            annotation.remove() if hasattr(annotation, 'remove') else None
        self.peak_feature_annotations = []

        # Exit if nothing to show
        if not self.show_peaks_var.get():
            self.canvas.draw_idle()
            return

        # Determine which signal and time arrays to use
        if self.signal_selection_var.get() == "Secondary" and hasattr(self, 'secondary_processed_signal'):
            signal = self.secondary_processed_signal
            time_array = self.secondary_processed_time
        else:
            signal = self.processed_signal
            time_array = self.processed_time

        # Plot peaks
        peak_times_min = self.peaks['times'] / 60
        peaks_scatter = self.ax1.scatter(
            peak_times_min, self.peaks['heights'],
            marker='^', color='red', s=100, zorder=10, picker=5
        )
        self.peak_annotations.append(peaks_scatter)  # Add to peak list

        # Add peak labels if enabled
        if self.show_peak_labels_var.get():  # Use peak-specific label var
            for i, (t, h) in enumerate(zip(peak_times_min, self.peaks['heights'])):
                label = self.ax1.annotate(
                    f"P{i + 1}", (t, h), xytext=(0, 10), textcoords="offset points",
                    ha='center', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                )
                self.peak_annotations.append(label)  # Add to peak list

        # Add vertical lines for peaks
        for i, t in enumerate(peak_times_min):
            line = self.ax1.axvline(t, color='red', linestyle='--', alpha=0.3, zorder=5)
            self.peak_lines.append(line)  # Add to peak list

        # Visualize peak features if enabled
        if self.peak_feature_viz_var.get():
            self.visualize_peak_features()  # Call the existing function

        self.canvas.draw_idle()

    def update_peak_visibility(self, _=None):
        """Update visibility of peaks based on checkbox settings in peak tab."""
        # This now only controls peaks
        self.update_peak_display()
        # We might want to call visualize_peak_features again if labels/features depend on peak visibility
        if self.peak_feature_viz_var.get():
            self.visualize_peak_features()

    def toggle_peak_feature_visualization(self, _=None):  # Make specific
        """Toggle display of peak features (width, area)"""
        # This controls peak features based on the peak tab checkbox
        if self.peak_feature_viz_var.get():
            self.visualize_peak_features()
        else:
            # Clear feature annotations
            for annotation in self.peak_feature_annotations:
                annotation.remove() if hasattr(annotation, 'remove') else None
            self.peak_feature_annotations = []
            self.canvas.draw_idle()

    def clear_peaks(self):
        """Clear detected peaks and their display/metrics."""
        self.peaks = None
        self.peak_metrics = None

        # Clear peak annotations from the plot
        for annotation in self.peak_annotations:
            annotation.remove() if hasattr(annotation, 'remove') else None
        self.peak_annotations = []
        for line in self.peak_lines:
            line.remove() if hasattr(line, 'remove') else None
        self.peak_lines = []
        # Clear peak feature annotations too
        for annotation in self.peak_feature_annotations:
            annotation.remove() if hasattr(annotation, 'remove') else None
        self.peak_feature_annotations = []

        # Clear peak metrics display
        if hasattr(self, 'peak_metrics_tree'):  # Check for peak tree
            self.peak_metrics_tree.delete(*self.peak_metrics_tree.get_children())

        self.canvas.draw_idle()
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Peaks cleared", fg="green")

    def update_metrics_display(self):
        """Update the peak metrics tree view"""
        # Clear existing data
        if not hasattr(self, 'peak_metrics_tree'):
            print("Peak metrics tree view not found!")
            return

        self.peak_metrics_tree.delete(*self.peak_metrics_tree.get_children())

        # Debug info
        print(f"Updating peak metrics display...")
        if self.peaks is None:
            print("No peaks data available")
            return

        print(f"Found {len(self.peaks['indices'])} peaks")
        if self.peak_metrics is None:
            print("No peak metrics calculated!")
            return

        print(f"Peak metrics keys: {list(self.peak_metrics.keys())}")
        print(f"Number of FWHM values: {len(self.peak_metrics['full_width_half_max'])}")

        # Add data to tree
        try:
            for i, (idx, time_val, height) in enumerate(zip(
                    self.peaks['indices'], self.peaks['times'], self.peaks['heights'])):

                # Get metrics with safety checks
                fwhm = "0.00"
                area = "0.00"
                rise = "0.00"
                decay = "0.00"

                if i < len(self.peak_metrics['full_width_half_max']):
                    fwhm = f"{self.peak_metrics['full_width_half_max'][i]:.2f}"

                if i < len(self.peak_metrics['area_under_curve']):
                    area = f"{self.peak_metrics['area_under_curve'][i]:.2f}"

                if i < len(self.peak_metrics['rise_time']):
                    rise = f"{self.peak_metrics['rise_time'][i]:.2f}"

                if i < len(self.peak_metrics['decay_time']):
                    decay = f"{self.peak_metrics['decay_time'][i]:.2f}"

                # Format the row values
                values = (
                    f"P{i + 1}",
                    f"{time_val:.2f}",
                    f"{height:.2f}",
                    fwhm,
                    area,
                    rise,
                    decay
                )

                # Insert into tree view
                self.peak_metrics_tree.insert("", "end", values=values)

            print(f"Inserted {i + 1} peak metrics rows into tree view")
        except Exception as e:
            print(f"Error displaying peak metrics: {e}")
            import traceback
            traceback.print_exc()

    # Modify manual mode toggles to disable the other tab's button
    def toggle_manual_peak_mode(self):
        """Toggle manual peak selection mode"""
        self.manual_peak_mode = not self.manual_peak_mode
        self.manual_valley_mode = False  # Turn off valley mode if it was on

        if self.manual_peak_mode:
            # Use the correct button reference
            self.manual_peak_btn.config(text="Cancel Peak Selection", bg='salmon')
            self.manual_valley_btn.config(state=tk.DISABLED)
            self.connect_selection_events("peak")

            # Show instructions
            if hasattr(self, 'status_label'):
                self.status_label.config(
                    text="Click on the signal to add a peak. Right-click a peak to remove it.",
                    fg="blue"
                )
        else:
            # Disable selection mode
            self.manual_peak_btn.config(text="Manual Peak Selection", bg='lightblue')
            self.manual_valley_btn.config(state=tk.NORMAL)
            self.disconnect_selection_events()

            # Update status
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Manual peak selection mode disabled", fg="green")

    # Modify on_selection_click and on_pick_event to check the mode
    def on_selection_click(self, event, mode):
        """Handle mouse click for adding peaks or valleys"""
        # Only handle left click in the main plot
        if event.button != 1 or event.inaxes != self.ax1:
            return

        # Get coordinates in data space
        x_min = event.xdata  # Time in minutes
        y = event.ydata  # Signal value
        x_sec = x_min * 60  # Convert to seconds

        # Determine which signal we're working with
        if self.valley_signal_selection_var.get() == "Secondary" and hasattr(self, 'secondary_processed_signal'):
            signal = self.secondary_processed_signal
            time = self.secondary_processed_time
        else:
            signal = self.processed_signal
            time = self.processed_time

        # Find the closest time index
        idx = np.argmin(np.abs(time - x_sec))

        # Add a minimum distance check to prevent selecting points too close to existing ones
        # This helps with the "jumpiness" by preventing selections too close to existing points
        min_distance_sec = 0.5  # Minimum 0.5 seconds between points

        # Initialize if needed
        if self.peaks is None:
            self.peaks = {'indices': np.array([], dtype=int), 'times': np.array([]),
                          'heights': np.array([]), 'properties': {}}
        if self.valleys is None:
            self.valleys = {'indices': np.array([], dtype=int), 'times': np.array([]),
                            'depths': np.array([]), 'properties': {}}

        # Check if we're too close to existing points
        too_close = False

        if mode == "peak" and len(self.peaks['times']) > 0:
            distances = np.abs(self.peaks['times'] - time[idx])
            if np.min(distances) < min_distance_sec:
                too_close = True

        if mode == "valley" and len(self.valleys['times']) > 0:
            distances = np.abs(self.valleys['times'] - time[idx])
            if np.min(distances) < min_distance_sec:
                too_close = True

        if too_close:
            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"Too close to existing point. Try a different location.", fg="red")
            return

        # Add the point based on mode
        if mode == "peak":
            # Add peak code
            self.peaks['indices'] = np.append(self.peaks['indices'], idx)
            self.peaks['times'] = np.append(self.peaks['times'], time[idx])
            self.peaks['heights'] = np.append(self.peaks['heights'], signal[idx])

            # Sort peaks by time
            sort_indices = np.argsort(self.peaks['times'])
            self.peaks['indices'] = self.peaks['indices'][sort_indices]
            self.peaks['times'] = self.peaks['times'][sort_indices]
            self.peaks['heights'] = self.peaks['heights'][sort_indices]

            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"Added peak at {time[idx]:.2f}s", fg="green")
        else:  # valley
            # Add valley code
            self.valleys['indices'] = np.append(self.valleys['indices'], idx)
            self.valleys['times'] = np.append(self.valleys['times'], time[idx])
            self.valleys['depths'] = np.append(self.valleys['depths'], signal[idx])

            # Sort valleys by time
            sort_indices = np.argsort(self.valleys['times'])
            self.valleys['indices'] = self.valleys['indices'][sort_indices]
            self.valleys['times'] = self.valleys['times'][sort_indices]
            self.valleys['depths'] = self.valleys['depths'][sort_indices]

            if hasattr(self, 'status_label'):
                self.status_label.config(text=f"Added valley at {time[idx]:.2f}s", fg="green")

        # Recalculate metrics if we have both peaks and valleys
        if len(self.peaks['indices']) > 0 and len(self.valleys['indices']) > 0:
            self.peak_metrics = calculate_peak_metrics(self.peaks, self.valleys, signal, time)
            self.valley_metrics = self.calculate_valley_metrics(self.peaks, self.valleys, signal, time)

        # Update displays
        if mode == "peak":
            self.update_peak_display()
            self.update_metrics_display()
        else:
            self.update_valley_display()
            self.update_valley_metrics_display()

    def on_pick_event(self, event, mode):
        """Handle picking events for removing peaks or valleys based on mode."""
        # Check right click
        if not hasattr(event, 'mouseevent') or event.mouseevent.button != 3:
            return

        artist = event.artist
        ind = event.ind[0]  # Index within the scatter plot data

        # Determine which signal we are working with based on dropdown
        if self.signal_selection_var.get() == "Secondary" and hasattr(self, 'secondary_processed_signal'):
            signal = self.secondary_processed_signal
            time_array = self.secondary_processed_time
        else:
            signal = self.processed_signal
            time_array = self.processed_time

        # If picking a peak scatter plot in peak mode
        if self.manual_peak_mode and hasattr(artist, 'get_offsets') and artist in self.peak_annotations:
            # Get the time corresponding to the picked point
            picked_time_min = artist.get_offsets()[ind][0]
            picked_time_sec = picked_time_min * 60
            # Find the closest peak in our data array
            if self.peaks and self.peaks['times'].size > 0:
                time_diffs = np.abs(self.peaks['times'] - picked_time_sec)
                closest_peak_idx = np.argmin(time_diffs)
                if time_diffs[closest_peak_idx] < 0.1:  # Allow small tolerance
                    self.remove_peak(closest_peak_idx)  # Remove using the index in the data array
                    # Recalculate metrics after removal
                    if len(self.peaks['indices']) > 0 and self.valleys is not None and len(self.valleys['indices']) > 0:
                        self.peak_metrics = calculate_peak_metrics(self.peaks, self.valleys, signal, time_array)
                        self.valley_metrics = self.calculate_valley_metrics(self.peaks, self.valleys, signal,
                                                                            time_array)
                        self.update_metrics_display()
                        self.update_valley_metrics_display()


        # If picking a valley scatter plot in valley mode
        elif self.manual_valley_mode and hasattr(artist, 'get_offsets') and artist in self.valley_annotations:
            picked_time_min = artist.get_offsets()[ind][0]
            picked_time_sec = picked_time_min * 60
            if self.valleys and self.valleys['times'].size > 0:
                time_diffs = np.abs(self.valleys['times'] - picked_time_sec)
                closest_valley_idx = np.argmin(time_diffs)
                if time_diffs[closest_valley_idx] < 0.1:
                    self.remove_valley(closest_valley_idx)  # Remove using the index in the data array
                    # Recalculate metrics after removal
                    if self.peaks is not None and len(self.peaks['indices']) > 0 and len(self.valleys['indices']) > 0:
                        self.peak_metrics = calculate_peak_metrics(self.peaks, self.valleys, signal, time_array)
                        self.valley_metrics = self.calculate_valley_metrics(self.peaks, self.valleys, signal,
                                                                            time_array)
                        self.update_metrics_display()
                        self.update_valley_metrics_display()

    def calculate_intervals(self, times):
        """Calculate intervals between consecutive events (peaks or valleys)"""
        if times is None or len(times) < 2:
            return None

        # Sort times to ensure chronological order
        sorted_times = np.sort(times)

        # Calculate intervals between consecutive events
        intervals = np.diff(sorted_times)

        # Basic statistics
        stats = {
            'mean': np.mean(intervals),
            'median': np.median(intervals),
            'std': np.std(intervals),
            'min': np.min(intervals),
            'max': np.max(intervals),
            'count': len(intervals),
            'intervals': intervals
        }

        return stats

    def create_psth_analysis_tab(self):
        """Create the PSTH analysis tab with controls and plot area"""
        # Create the tab
        self.psth_tab = ttk.Frame(self.control_notebook)
        self.control_notebook.add(self.psth_tab, text="PSTH Analysis")

        # Get font sizes
        font_size = self.font_sizes['base']
        slider_font_size = self.font_sizes['slider']

        # Configure grid layout
        self.psth_tab.columnconfigure(1, weight=1)

        # Control Panel Frame (left side)
        control_frame = tk.LabelFrame(self.psth_tab, text="PSTH Controls", font=('Arial', font_size, 'bold'))
        control_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nw")

        # Event Selection Frame
        event_frame = tk.LabelFrame(control_frame, text="Event Type", font=('Arial', font_size))
        event_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create event type radio buttons
        self.event_type_var = tk.StringVar(value="peaks")
        tk.Radiobutton(event_frame, text="Peaks",
                       variable=self.event_type_var,
                       value="peaks",
                       font=('Arial', font_size)).pack(side=tk.LEFT, padx=20)

        tk.Radiobutton(event_frame, text="Valleys",
                       variable=self.event_type_var,
                       value="valleys",
                       font=('Arial', font_size)).pack(side=tk.LEFT, padx=20)

        # Signal Selection Frame
        signal_frame = tk.LabelFrame(control_frame, text="Signal Selection", font=('Arial', font_size))
        signal_frame.pack(fill=tk.X, padx=5, pady=5)

        # Create radio buttons for signal selection
        self.psth_signal_var = tk.StringVar(value="Both")

        tk.Radiobutton(signal_frame,
                       text="PVN (Primary)",
                       variable=self.psth_signal_var,
                       value="Primary",
                       font=('Arial', font_size)).pack(anchor="w", padx=20)

        tk.Radiobutton(signal_frame,
                       text="SON (Secondary)",
                       variable=self.psth_signal_var,
                       value="Secondary",
                       font=('Arial', font_size)).pack(anchor="w", padx=20)

        tk.Radiobutton(signal_frame,
                       text="Both PVN & SON",
                       variable=self.psth_signal_var,
                       value="Both",
                       font=('Arial', font_size)).pack(anchor="w", padx=20)

        # Time Window Frame
        window_frame = tk.LabelFrame(control_frame, text="Time Window (seconds)", font=('Arial', font_size))
        window_frame.pack(fill=tk.X, padx=5, pady=5)

        # Before event
        tk.Label(window_frame, text="Before Event:", font=('Arial', font_size)).grid(row=0, column=0, sticky="w",
                                                                                     padx=5, pady=2)
        self.psth_before_var = tk.DoubleVar(value=60.0)
        tk.Spinbox(window_frame, from_=1, to=120, increment=1,
                   textvariable=self.psth_before_var, width=5).grid(row=0, column=1, sticky="w", padx=5, pady=2)

        # After event
        tk.Label(window_frame, text="After Event:", font=('Arial', font_size)).grid(row=1, column=0, sticky="w", padx=5,
                                                                                    pady=2)
        self.psth_after_var = tk.DoubleVar(value=60.0)
        tk.Spinbox(window_frame, from_=1, to=120, increment=1,
                   textvariable=self.psth_after_var, width=5).grid(row=1, column=1, sticky="w", padx=5, pady=2)

        # Plot Options Frame
        plot_options_frame = tk.LabelFrame(control_frame, text="Plot Options", font=('Arial', font_size))
        plot_options_frame.pack(fill=tk.X, padx=5, pady=5)

        # Show SEM
        self.show_sem_var = tk.BooleanVar(value=True)
        tk.Checkbutton(plot_options_frame, text="Show SEM Shading",
                       variable=self.show_sem_var,
                       font=('Arial', font_size)).pack(anchor="w", padx=5, pady=2)

        # Show Individual Traces
        self.show_individual_var = tk.BooleanVar(value=False)
        tk.Checkbutton(plot_options_frame, text="Show Individual Traces",
                       variable=self.show_individual_var,
                       font=('Arial', font_size)).pack(anchor="w", padx=5, pady=2)

        # Buttons Frame
        button_frame = tk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)

        # Plot Button
        self.plot_psth_btn = tk.Button(
            button_frame,
            text="Plot PSTH",
            font=('Arial', self.font_sizes['button']),
            bg='lightgreen',
            command=self.generate_psth_plot
        )
        self.plot_psth_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Export Button
        self.export_psth_btn = tk.Button(
            button_frame,
            text="Export Data",
            font=('Arial', self.font_sizes['button']),
            command=self.export_psth_data
        )
        self.export_psth_btn.pack(side=tk.LEFT, padx=5, pady=5)

        # Plot Frame (right side)
        plot_frame = tk.LabelFrame(self.psth_tab, text="PSTH Plot", font=('Arial', font_size, 'bold'))
        plot_frame.grid(row=0, column=1, padx=10, pady=5, sticky="nsew", rowspan=2)

        # Create matplotlib figure and canvas for PSTH plot
        self.psth_fig = plt.Figure(figsize=(10, 8))
        self.psth_canvas = FigureCanvasTkAgg(self.psth_fig, master=plot_frame)
        self.psth_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar
        self.psth_toolbar = NavigationToolbar2Tk(self.psth_canvas, plot_frame)
        self.psth_toolbar.update()

        # Status Frame
        status_frame = tk.Frame(self.psth_tab)
        status_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.psth_status = tk.Label(status_frame, text="Ready for PSTH analysis", fg="blue",
                                    font=('Arial', font_size - 1))
        self.psth_status.pack(fill=tk.X)

        # Initialize with empty plots
        self.setup_empty_psth_plots()

        # Initialize PSTH variables
        self.pvn_psth_data = None
        self.son_psth_data = None

    def setup_empty_psth_plots(self):
        """Setup the initial empty PSTH plot structure"""
        self.psth_fig.clear()

        # PSTH typically has 1 or 2 subplots (PVN/SON or both)
        if self.psth_signal_var.get() == "Both":
            self.psth_ax1 = self.psth_fig.add_subplot(211)  # PVN plot
            self.psth_ax2 = self.psth_fig.add_subplot(212)  # SON plot

            self.psth_ax1.set_title("PVN signal around events")
            self.psth_ax2.set_title("SON signal around events")
            self.psth_ax2.set_xlabel("Time (sec)")

            self.psth_ax1.set_ylabel("dFF%")
            self.psth_ax2.set_ylabel("dFF%")

        else:
            self.psth_ax1 = self.psth_fig.add_subplot(111)
            signal_name = "PVN" if self.psth_signal_var.get() == "Primary" else "SON"
            self.psth_ax1.set_title(f"{signal_name} signal around events")
            self.psth_ax1.set_xlabel("Time (sec)")
            self.psth_ax1.set_ylabel("dFF%")

        # Add vertical line at t=0 (event time)
        for ax in [getattr(self, x) for x in dir(self) if x.startswith('psth_ax')]:
            if hasattr(ax, 'axvline'):
                ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
                ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
                ax.grid(True, alpha=0.3)

        self.psth_fig.tight_layout()
        self.psth_canvas.draw_idle()

    def generate_psth_plot(self):
        """Generate the PSTH plot based on current settings"""
        # Get reference event selection
        ref_event_type = self.reference_event_var.get()

        # Map selection to actual data
        event_times = None
        event_description = ""

        if ref_event_type == "pvn_peaks":
            if self.peaks is None or len(self.peaks.get('times', [])) == 0:
                self.psth_status.config(text="No PVN peaks available. Run peak detection first.", fg="red")
                return
            event_times = self.peaks['times']
            event_description = "PVN peaks"

        elif ref_event_type == "pvn_valleys":
            if self.valleys is None or len(self.valleys.get('times', [])) == 0:
                self.psth_status.config(text="No PVN valleys available. Run valley detection first.", fg="red")
                return
            event_times = self.valleys['times']
            event_description = "PVN valleys"

        elif ref_event_type == "son_peaks":
            if not hasattr(self, 'secondary_peaks') or self.secondary_peaks is None or len(
                    self.secondary_peaks.get('times', [])) == 0:
                self.psth_status.config(text="No SON peaks available. Run peak detection first.", fg="red")
                return
            event_times = self.secondary_peaks['times']
            event_description = "SON peaks"

        elif ref_event_type == "son_valleys":
            if not hasattr(self, 'secondary_valleys') or self.secondary_valleys is None or len(
                    self.secondary_valleys.get('times', [])) == 0:
                self.psth_status.config(text="No SON valleys available. Run valley detection first.", fg="red")
                return
            event_times = self.secondary_valleys['times']
            event_description = "SON valleys"

        # Update status
        self.psth_status.config(text=f"Generating PSTH using {event_description} as reference points...", fg="blue")
        self.root.update_idletasks()

        # Get time window parameters
        before_event = self.psth_before_var.get()
        after_event = self.psth_after_var.get()

        # Check which signals to plot
        plot_pvn = self.plot_pvn_var.get()
        plot_son = self.plot_son_var.get()

        if not plot_pvn and not plot_son:
            self.psth_status.config(text="Please select at least one signal to plot", fg="red")
            return

        # Clear the figure
        self.psth_fig.clear()

        # Calculate PSTH data for selected signals using the same reference events
        pvn_psth_data = None
        son_psth_data = None

        if plot_pvn:
            pvn_psth_data = self.calculate_psth(
                self.processed_signal,
                self.processed_time,
                event_times,
                before_event,
                after_event
            )
            self.pvn_psth_data = pvn_psth_data

        if plot_son and hasattr(self, 'secondary_processed_signal') and self.secondary_processed_signal is not None:
            son_psth_data = self.calculate_psth(
                self.secondary_processed_signal,
                self.secondary_processed_time,
                event_times,
                before_event,
                after_event
            )
            self.son_psth_data = son_psth_data

        # Create plots based on which signals are selected
        if plot_pvn and plot_son:
            # Two plots
            self.psth_ax1 = self.psth_fig.add_subplot(211)  # PVN plot
            self.psth_ax2 = self.psth_fig.add_subplot(212)  # SON plot

            self.psth_ax1.set_title(f"PVN signal around {event_description} (n={len(event_times)})")
            self.psth_ax2.set_title(f"SON signal around {event_description} (n={len(event_times)})")

            # Plot PVN data
            if pvn_psth_data:
                self.plot_psth_data(self.psth_ax1, pvn_psth_data, 'blue')
            else:
                self.psth_ax1.text(0.5, 0.5, "No valid PVN data available",
                                   horizontalalignment='center', verticalalignment='center',
                                   transform=self.psth_ax1.transAxes)

            # Plot SON data
            if son_psth_data:
                self.plot_psth_data(self.psth_ax2, son_psth_data, 'red')
            else:
                self.psth_ax2.text(0.5, 0.5, "No valid SON data available",
                                   horizontalalignment='center', verticalalignment='center',
                                   transform=self.psth_ax2.transAxes)

            self.psth_ax2.set_xlabel("Time (sec)")
            self.psth_ax1.set_ylabel("dFF%")
            self.psth_ax2.set_ylabel("dFF%")

        elif plot_pvn:
            # Only PVN plot
            self.psth_ax1 = self.psth_fig.add_subplot(111)
            self.psth_ax1.set_title(f"PVN signal around {event_description} (n={len(event_times)})")

            if pvn_psth_data:
                self.plot_psth_data(self.psth_ax1, pvn_psth_data, 'blue')
            else:
                self.psth_ax1.text(0.5, 0.5, "No valid PVN data available",
                                   horizontalalignment='center', verticalalignment='center',
                                   transform=self.psth_ax1.transAxes)

            self.psth_ax1.set_xlabel("Time (sec)")
            self.psth_ax1.set_ylabel("dFF%")

        elif plot_son:
            # Only SON plot
            self.psth_ax1 = self.psth_fig.add_subplot(111)
            self.psth_ax1.set_title(f"SON signal around {event_description} (n={len(event_times)})")

            if son_psth_data:
                self.plot_psth_data(self.psth_ax1, son_psth_data, 'red')
            else:
                self.psth_ax1.text(0.5, 0.5, "No valid SON data available",
                                   horizontalalignment='center', verticalalignment='center',
                                   transform=self.psth_ax1.transAxes)

            self.psth_ax1.set_xlabel("Time (sec)")
            self.psth_ax1.set_ylabel("dFF%")

        self.psth_fig.tight_layout()
        self.psth_canvas.draw_idle()

        # Update status
        if (plot_pvn and pvn_psth_data) or (plot_son and son_psth_data):
            self.psth_status.config(text="PSTH plot generated successfully", fg="green")
        else:
            self.psth_status.config(text="Failed to generate valid PSTH data", fg="red")

    def calculate_psth(self, signal, time, event_times, before_event, after_event):
        """
        Calculate PSTH data for a signal around specified event times.
        Returns time series and aligned traces around events.
        """
        if len(event_times) == 0:
            return None

        # Calculate time step from signal
        dt = np.mean(np.diff(time))

        # Calculate sample points needed for time window
        before_points = int(before_event / dt)
        after_points = int(after_event / dt)

        # Create time series for alignment
        time_series = np.arange(-before_points, after_points + 1) * dt

        # Initialize array to store aligned traces
        aligned_traces = []

        # Process each event time
        for event_time in event_times:
            try:
                # Find the index closest to the event time
                event_idx = np.argmin(np.abs(time - event_time))

                # Check if we have enough points for window
                if event_idx >= before_points and event_idx + after_points < len(signal):
                    # Extract segment
                    segment = signal[event_idx - before_points: event_idx + after_points + 1]

                    # Verify segment has the right length and contains valid data
                    if len(segment) == len(time_series) and np.all(np.isfinite(segment)):
                        aligned_traces.append(segment)
            except Exception as e:
                print(f"Error processing event at {event_time}: {e}")
                continue

        # Convert to numpy array
        if aligned_traces:
            try:
                aligned_traces = np.array(aligned_traces)

                # Calculate mean and SEM
                mean_trace = np.nanmean(aligned_traces, axis=0)  # Use nanmean to handle NaN values
                sem_trace = np.nanstd(aligned_traces, axis=0) / np.sqrt(len(aligned_traces))

                # Replace any remaining NaN or inf values
                mean_trace = np.nan_to_num(mean_trace, nan=0.0, posinf=0.0, neginf=0.0)
                sem_trace = np.nan_to_num(sem_trace, nan=0.0, posinf=0.0, neginf=0.0)

                return {
                    'time_series': time_series,
                    'aligned_traces': aligned_traces,
                    'mean_trace': mean_trace,
                    'sem_trace': sem_trace,
                    'event_count': len(aligned_traces)
                }
            except Exception as e:
                print(f"Error calculating PSTH statistics: {e}")
                return None
        else:
            print("No valid aligned traces found")
            return None

    def plot_psth_data(self, ax, psth_data, color='blue'):
        """Plot PSTH data on the given axis with mean and SEM"""
        if psth_data is None or len(psth_data.get('aligned_traces', [])) == 0:
            ax.text(0.5, 0.5, "No valid data available",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            return

        # Plot mean trace
        ax.plot(psth_data['time_series'], psth_data['mean_trace'],
                color=color, linewidth=2, label='Mean')

        # Add SEM shading if enabled
        if self.show_sem_var.get():
            upper = psth_data['mean_trace'] + psth_data['sem_trace']
            lower = psth_data['mean_trace'] - psth_data['sem_trace']
            ax.fill_between(psth_data['time_series'], lower, upper,
                            color=color, alpha=0.2, label='SEM')

        # Add individual traces if enabled
        if self.show_individual_var.get() and len(psth_data['aligned_traces']) > 0:
            # Plot a subset of traces to avoid visual clutter
            max_traces = min(10, len(psth_data['aligned_traces']))
            indices = np.linspace(0, len(psth_data['aligned_traces']) - 1, max_traces, dtype=int)

            for i in indices:
                ax.plot(psth_data['time_series'], psth_data['aligned_traces'][i],
                        color=color, alpha=0.15, linewidth=0.5)

        # Add vertical line at t=0 (event time)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)

        # Set y limits with some padding, safely handling NaN/Inf values
        try:
            # Filter out NaN and Inf values for determining limits
            mean_filtered = psth_data['mean_trace'][np.isfinite(psth_data['mean_trace'])]
            sem_filtered = psth_data['sem_trace'][np.isfinite(psth_data['sem_trace'])]

            if len(mean_filtered) > 0 and len(sem_filtered) > 0:
                data_min = np.min(mean_filtered - 2 * sem_filtered)
                data_max = np.max(mean_filtered + 2 * sem_filtered)

                # Check if values are valid
                if np.isfinite(data_min) and np.isfinite(data_max):
                    padding = (data_max - data_min) * 0.1
                    ax.set_ylim(data_min - padding, data_max + padding)
                else:
                    # Use a fallback if values are not valid
                    ax.autoscale(axis='y')
            else:
                ax.autoscale(axis='y')
        except Exception as e:
            print(f"Error setting y-limits: {e}")
            # Fallback to auto-scaling
            ax.autoscale(axis='y')

        # Set x limits to match time window
        ax.set_xlim(psth_data['time_series'][0], psth_data['time_series'][-1])

    def export_psth_data(self):
        """Export PSTH data to CSV file"""
        # Check if there's data to export
        if self.pvn_psth_data is None and self.son_psth_data is None:
            messagebox.showinfo("Export Error", "No PSTH data available to export")
            return

        # Get file path for saving
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export PSTH Data"
        )

        if not file_path:
            return  # User canceled

        try:
            import csv

            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                # Write header based on available data
                header = ["Time (s)"]

                if self.pvn_psth_data is not None:
                    header.extend(["PVN Mean", "PVN SEM"])

                if self.son_psth_data is not None:
                    header.extend(["SON Mean", "SON SEM"])

                writer.writerow(header)

                # Determine which time series to use
                if self.pvn_psth_data is not None:
                    time_series = self.pvn_psth_data['time_series']
                else:
                    time_series = self.son_psth_data['time_series']

                # Write data rows
                for i, t in enumerate(time_series):
                    row = [f"{t:.3f}"]

                    if self.pvn_psth_data is not None:
                        row.extend([
                            f"{self.pvn_psth_data['mean_trace'][i]:.4f}",
                            f"{self.pvn_psth_data['sem_trace'][i]:.4f}"
                        ])

                    if self.son_psth_data is not None:
                        row.extend([
                            f"{self.son_psth_data['mean_trace'][i]:.4f}",
                            f"{self.son_psth_data['sem_trace'][i]:.4f}"
                        ])

                    writer.writerow(row)

                # Add metadata at the end
                writer.writerow([])
                writer.writerow(["Metadata"])

                if self.pvn_psth_data is not None:
                    writer.writerow(["PVN Events Count", self.pvn_psth_data['event_count']])

                if self.son_psth_data is not None:
                    writer.writerow(["SON Events Count", self.son_psth_data['event_count']])

                event_type = "Peaks" if self.event_type_var.get() == "peaks" else "Valleys"
                writer.writerow(["Event Type", event_type])
                writer.writerow(["Time Before Event (s)", self.psth_before_var.get()])
                writer.writerow(["Time After Event (s)", self.psth_after_var.get()])

            messagebox.showinfo("Export Complete", f"PSTH data successfully exported to {file_path}")

        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting PSTH data: {str(e)}")


def main():
    """Main function to start the application."""
    try:
        # Create the root window
        root = tk.Tk()

        # Get screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Set window size (80% of screen)
        window_width = min(int(screen_width * 0.8), 1400)
        window_height = min(int(screen_height * 0.8), 900)

        # Set minimum size to ensure controls are visible
        root.minsize(1000, 700)

        # Center the window
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2

        # Set geometry with exact positioning
        root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        root.title("Photometry Signal Viewer")

        # Create viewer application
        app = PhotometryViewer(root)

        # Optional file opening
        def delayed_file_open():
            try:
                app.open_file()
            except Exception as e:
                print(f"Error opening file: {e}")

        root.after(100, delayed_file_open)
        root.mainloop()

    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("Application ending normally")


if __name__ == "__main__":
    main()