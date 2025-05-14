"""
Signal processing module.
Provides filtering, normalization, baseline correction and other processing functions for photometry data.
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, savgol_filter, sosfilt, sosfiltfilt, zpk2sos
from scipy.sparse import diags, spdiags
from scipy.sparse.linalg import spsolve
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import numpy.polynomial.polynomial as poly
from typing import Dict, List, Tuple, Union, Optional, Any, Callable

# Try to import ruptures for changepoint detection, but handle if not available
try:
    import ruptures as rpt
except ImportError:
    rpt = None

def apply_filter(signal_data: np.ndarray, filter_type: str = 'lowpass', 
                cutoff_freq: Union[float, List[float]] = 2.0, 
                order: int = 3, fs: float = 100.0, 
                method: str = 'butterworth') -> np.ndarray:
    """
    Apply a filter to signal data.

    Args:
        signal_data (np.ndarray): Input signal data
        filter_type (str): Filter type, options: 'lowpass', 'highpass', 'bandpass', 'bandstop'
        cutoff_freq (float or List[float]): Cutoff frequency, for bandpass and bandstop provide [low, high]
        order (int): Filter order
        fs (float): Sampling rate (Hz)
        method (str): Filter design method, options: 'butterworth', 'chebyshev', 'bessel', 'elliptic', 'fir'

    Returns:
        np.ndarray: Filtered signal

    Raises:
        ValueError: If parameters are invalid
    """
    # Check input
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    
    # Normalize cutoff frequency
    if isinstance(cutoff_freq, (int, float)):
        cutoff_freq = float(cutoff_freq)
    
    # Ensure cutoff frequency is less than Nyquist frequency
    nyquist = 0.5 * fs
    if isinstance(cutoff_freq, list):
        if any(f >= nyquist for f in cutoff_freq):
            raise ValueError(f"Cutoff frequency must be less than Nyquist frequency ({nyquist} Hz)")
    else:
        if cutoff_freq >= nyquist:
            raise ValueError(f"Cutoff frequency must be less than Nyquist frequency ({nyquist} Hz)")
    
    # Choose appropriate design function for different filter methods
    if method.lower() == 'butterworth':
        design_func = signal.butter
    elif method.lower() == 'chebyshev':
        design_func = signal.cheby1
    elif method.lower() == 'bessel':
        design_func = signal.bessel
    elif method.lower() == 'elliptic':
        design_func = signal.ellip
    elif method.lower() == 'fir':
        # For FIR filters use firwin
        return _apply_fir_filter(signal_data, filter_type, cutoff_freq, order, fs)
    else:
        raise ValueError(f"Unsupported filter method: {method}")
    
    # Normalize cutoff frequency
    if isinstance(cutoff_freq, list):
        wn = [f / nyquist for f in cutoff_freq]
    else:
        wn = cutoff_freq / nyquist
    
    # Design filter
    try:
        if filter_type.lower() == 'bandpass' or filter_type.lower() == 'bandstop':
            if not isinstance(cutoff_freq, list) or len(cutoff_freq) != 2:
                raise ValueError(f"{filter_type} filter requires two cutoff frequency values [low, high]")
            
            b, a = design_func(order, wn, btype=filter_type.lower())
        else:
            b, a = design_func(order, wn, btype=filter_type.lower())
            
        # Apply filter (using filtfilt for zero-phase filtering)
        filtered_signal = signal.filtfilt(b, a, signal_data)
        
        return filtered_signal
    
    except Exception as e:
        raise ValueError(f"Filter application failed: {str(e)}")

def _apply_fir_filter(signal_data: np.ndarray, filter_type: str, 
                     cutoff_freq: Union[float, List[float]], 
                     order: int, fs: float) -> np.ndarray:
    """Apply FIR filter"""
    nyquist = 0.5 * fs
    
    # Normalize cutoff frequency
    if isinstance(cutoff_freq, list):
        wn = [f / nyquist for f in cutoff_freq]
    else:
        wn = cutoff_freq / nyquist
    
    # Design FIR filter coefficients
    if filter_type.lower() == 'lowpass':
        b = signal.firwin(order+1, wn, pass_zero=True)
    elif filter_type.lower() == 'highpass':
        b = signal.firwin(order+1, wn, pass_zero=False)
    elif filter_type.lower() == 'bandpass':
        if not isinstance(cutoff_freq, list) or len(cutoff_freq) != 2:
            raise ValueError("Bandpass filter requires two cutoff frequency values [low, high]")
        b = signal.firwin(order+1, wn, pass_zero=False)
    elif filter_type.lower() == 'bandstop':
        if not isinstance(cutoff_freq, list) or len(cutoff_freq) != 2:
            raise ValueError("Bandstop filter requires two cutoff frequency values [low, high]")
        b = signal.firwin(order+1, wn, pass_zero=True)
    else:
        raise ValueError(f"Unsupported filter type: {filter_type}")
    
    # Apply filter
    filtered_signal = signal.filtfilt(b, 1.0, signal_data)
    
    return filtered_signal

def butter_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 5):
    """
    Applies a Butterworth filter to the data with reduced edge effects.
    
    Args:
        data: Input signal data
        cutoff: Cutoff frequency
        fs: Sampling rate
        order: Filter order
    
    Returns:
        Filtered signal
    """
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

def smooth_signal(signal_data: np.ndarray, method: str = 'gaussian', 
                 window_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Smooth the signal data.

    Args:
        signal_data (np.ndarray): Input signal data
        method (str): Smoothing method, options: 'gaussian', 'moving_average', 'savgol', 'median', 'exponential'
        window_size (int): Window size (must be odd)
        sigma (float): Standard deviation for Gaussian smoothing (used only for Gaussian smoothing)

    Returns:
        np.ndarray: Smoothed signal
    """
    # Check input
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    
    # Ensure window size is odd
    if method != 'gaussian' and method != 'exponential' and window_size % 2 == 0:
        window_size += 1
    
    # Apply selected smoothing method
    if method.lower() == 'gaussian':
        # Gaussian smoothing
        return gaussian_filter1d(signal_data, sigma)
    
    elif method.lower() == 'moving_average':
        # Moving average
        window = np.ones(window_size) / window_size
        return np.convolve(signal_data, window, mode='same')
    
    elif method.lower() == 'savgol':
        # Savitzky-Golay filter
        poly_order = min(window_size - 1, 3)  # Ensure polynomial order is less than window size
        return signal.savgol_filter(signal_data, window_size, poly_order)
    
    elif method.lower() == 'median':
        # Median filter
        return signal.medfilt(signal_data, kernel_size=window_size)
    
    elif method.lower() == 'exponential':
        # Exponential moving average
        alpha = 2.0 / (window_size + 1)
        smoothed = np.zeros_like(signal_data)
        smoothed[0] = signal_data[0]
        for i in range(1, len(signal_data)):
            smoothed[i] = alpha * signal_data[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    else:
        raise ValueError(f"Unsupported smoothing method: {method}")

def normalize_signal(signal_data: np.ndarray, method: str = 'minmax', 
                    range_min: float = 0.0, range_max: float = 1.0,
                    percentile_min: float = 5.0, percentile_max: float = 95.0,
                    baseline_window: int = None) -> np.ndarray:
    """
    Normalize signal data.

    Args:
        signal_data (np.ndarray): Input signal data
        method (str): Normalization method, options: 'minmax', 'zscore', 'percentile', 'robust', 'baseline'
        range_min (float): Minimum value of normalization range (for minmax method)
        range_max (float): Maximum value of normalization range (for minmax method)
        percentile_min (float): Lower percentile (for percentile method)
        percentile_max (float): Upper percentile (for percentile method)
        baseline_window (int): Window size for baseline normalization

    Returns:
        np.ndarray: Normalized signal
    """
    # Check input
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    
    # Apply selected normalization method
    if method.lower() == 'minmax':
        # Min-max normalization
        min_val = np.min(signal_data)
        max_val = np.max(signal_data)
        
        if max_val == min_val:
            return np.full_like(signal_data, 0.5 * (range_min + range_max))
        
        normalized = range_min + (signal_data - min_val) * (range_max - range_min) / (max_val - min_val)
        return normalized
    
    elif method.lower() == 'zscore':
        # Z-score normalization (standardization)
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        
        if std == 0:
            return np.zeros_like(signal_data)
        
        return (signal_data - mean) / std
    
    elif method.lower() == 'percentile':
        # Percentile normalization
        p_min = np.percentile(signal_data, percentile_min)
        p_max = np.percentile(signal_data, percentile_max)
        
        if p_max == p_min:
            return np.full_like(signal_data, 0.5 * (range_min + range_max))
        
        normalized = range_min + (signal_data - p_min) * (range_max - range_min) / (p_max - p_min)
        
        # Clip values outside range
        normalized = np.clip(normalized, range_min, range_max)
        return normalized
    
    elif method.lower() == 'robust':
        # Robust normalization using median and IQR
        median = np.median(signal_data)
        q1 = np.percentile(signal_data, 25)
        q3 = np.percentile(signal_data, 75)
        iqr = q3 - q1
        
        if iqr == 0:
            return np.zeros_like(signal_data)
        
        return (signal_data - median) / iqr
    
    elif method.lower() == 'baseline':
        # Normalization relative to baseline (dF/F)
        if baseline_window is None or baseline_window <= 0:
            # Use first 10% of data as baseline
            baseline_window = int(0.1 * len(signal_data))
            baseline_window = max(baseline_window, 1)
        
        baseline = np.mean(signal_data[:baseline_window])
        
        if baseline == 0:
            # Avoid division by zero
            baseline = np.mean(signal_data)
            if baseline == 0:
                return np.zeros_like(signal_data)
        
        # Calculate relative change (dF/F)
        return (signal_data - baseline) / baseline
    
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

def correct_baseline(signal_data: np.ndarray, method: str = 'polynomial', 
                    window_size: int = 101, poly_order: int = 2,
                    percentile: float = 10.0, lambda_value: float = 1e5) -> np.ndarray:
    """
    Baseline correction.

    Args:
        signal_data (np.ndarray): Input signal data
        method (str): Correction method, options: 'polynomial', 'percentile', 'moving_average', 'savgol', 'asymmetric'
        window_size (int): Window size
        poly_order (int): Polynomial order (for polynomial and savgol methods)
        percentile (float): Percentile (for percentile method)
        lambda_value (float): Smoothing parameter (for asymmetric method)

    Returns:
        np.ndarray: Corrected signal
    """
    # Check input
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    
    # Apply selected baseline correction method
    if method.lower() == 'polynomial':
        # Polynomial fitting
        x = np.arange(len(signal_data))
        poly_coef = np.polyfit(x, signal_data, poly_order)
        baseline = np.polyval(poly_coef, x)
        corrected = signal_data - baseline
        return corrected
    
    elif method.lower() == 'percentile':
        # Moving percentile
        half_window = window_size // 2
        padded_signal = np.pad(signal_data, half_window, mode='reflect')
        baseline = np.zeros_like(signal_data)
        
        for i in range(len(signal_data)):
            window = padded_signal[i:i+window_size]
            baseline[i] = np.percentile(window, percentile)
        
        corrected = signal_data - baseline
        return corrected
    
    elif method.lower() == 'moving_average':
        # Moving average
        half_window = window_size // 2
        padded_signal = np.pad(signal_data, half_window, mode='reflect')
        window = np.ones(window_size) / window_size
        baseline = np.convolve(padded_signal, window, mode='valid')
        corrected = signal_data - baseline
        return corrected
    
    elif method.lower() == 'savgol':
        # Savitzky-Golay filter
        if window_size % 2 == 0:
            window_size += 1  # Ensure window size is odd
        baseline = signal.savgol_filter(signal_data, window_size, poly_order)
        corrected = signal_data - baseline
        return corrected
    
    elif method.lower() == 'asymmetric':
        # Asymmetric least squares (for fluorescence data baseline correction)
        # This is an improved version of the algorithm proposed by Eilers and Boelens
        baseline = _asymmetric_least_squares(signal_data, lambda_value=lambda_value, p=0.001)
        corrected = signal_data - baseline
        return corrected
    
    else:
        raise ValueError(f"Unsupported baseline correction method: {method}")

def _asymmetric_least_squares(signal_data: np.ndarray, lambda_value: float = 1e5, 
                            p: float = 0.001, n_iter: int = 10) -> np.ndarray:
    """
    Estimate baseline using asymmetric least squares.
    
    Reference: P. H. C. Eilers, H. F. M. Boelens, "Baseline Correction with Asymmetric 
    Least Squares Smoothing," 2005
    
    Args:
        signal_data: Input signal
        lambda_value: Smoothing parameter
        p: Asymmetry factor (0 < p < 1)
        n_iter: Number of iterations
    
    Returns:
        Estimated baseline
    """
    m = len(signal_data)
    D = np.diff(np.eye(m), 2)
    w = np.ones(m)
    
    for i in range(n_iter):
        W = np.diag(w)
        baseline = np.linalg.solve(W + lambda_value * D.T @ D, W @ signal_data)
        w = p * (signal_data > baseline) + (1-p) * (signal_data <= baseline)
    
    return baseline

def fit_drift_curve(time: np.ndarray, signal: np.ndarray, poly_degree: int = 2):
    """
    Fit a polynomial curve to the signal to model baseline drift.

    Args:
        time: Time array
        signal: Signal to fit
        poly_degree: Degree of polynomial to fit

    Returns:
        fitted_curve: The fitted polynomial curve
        coeffs: Polynomial coefficients
    """
    # Normalize time to improve numerical stability
    time_norm = (time - time[0]) / (time[-1] - time[0])

    # Fit polynomial to signal
    coeffs = poly.polyfit(time_norm, signal, poly_degree)

    # Generate fitted curve
    fitted_curve = poly.polyval(time_norm, coeffs)

    return fitted_curve, coeffs

def correct_drift(signal_data: np.ndarray, method: str = 'linear', 
                 window_size: int = None, poly_order: int = 1) -> np.ndarray:
    """
    Drift correction.

    Args:
        signal_data (np.ndarray): Input signal data
        method (str): Correction method, options: 'linear', 'polynomial', 'savgol', 'detrend', 'robust'
        window_size (int): Window size for drift estimation
        poly_order (int): Polynomial order (for polynomial method)

    Returns:
        np.ndarray: Corrected signal
    """
    # Check input
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    
    # Set default window size
    if window_size is None:
        window_size = max(int(len(signal_data) * 0.1), 1)  # Default to
        
    # Apply selected drift correction method
    if method.lower() == 'linear':
        # Linear drift correction
        x = np.arange(len(signal_data))
        
        # Select endpoints for linear fitting
        n_points = min(window_size, len(signal_data) // 4)
        start_indices = np.arange(n_points)
        end_indices = np.arange(len(signal_data) - n_points, len(signal_data))
        
        start_avg = np.mean(signal_data[start_indices])
        end_avg = np.mean(signal_data[end_indices])
        
        drift = start_avg + (end_avg - start_avg) * x / (len(signal_data) - 1)
        corrected = signal_data - drift + start_avg
        return corrected
    
    elif method.lower() == 'polynomial':
        # Polynomial drift correction
        x = np.arange(len(signal_data))
        
        # Select points for polynomial fitting
        if window_size >= len(signal_data):
            # If window is greater than or equal to signal length, use all points
            indices = np.arange(len(signal_data))
        else:
            # Evenly select points
            num_segments = poly_order + 1
            indices = []
            for i in range(num_segments):
                start = int(i * len(signal_data) / num_segments)
                end = int((i+1) * len(signal_data) / num_segments)
                segment_size = min(window_size, end - start)
                segment_indices = np.arange(start, start + segment_size)
                indices.extend(segment_indices)
            indices = np.array(indices)
            indices = indices[indices < len(signal_data)]
        
        # Fit polynomial
        poly_coef = np.polyfit(x[indices], signal_data[indices], poly_order)
        drift = np.polyval(poly_coef, x)
        
        # Maintain initial point value
        offset = drift[0]
        corrected = signal_data - drift + offset
        return corrected
    
    elif method.lower() == 'savgol':
        # Estimate drift using Savitzky-Golay filter
        if window_size % 2 == 0:
            window_size += 1  # Ensure window size is odd
        
        window_size = min(window_size, len(signal_data) - 1)
        if window_size < 3:
            window_size = 3  # Minimum window size
        
        poly_order = min(poly_order, window_size - 1)
        
        drift = signal.savgol_filter(signal_data, window_size, poly_order)
        corrected = signal_data - drift + drift[0]
        return corrected
    
    elif method.lower() == 'detrend':
        # Use scipy's detrend function
        return signal.detrend(signal_data, type='linear')
    
    elif method.lower() == 'robust':
        # Robust drift correction (using RANSAC regression)
        from sklearn.linear_model import RANSACRegressor
        
        x = np.arange(len(signal_data)).reshape(-1, 1)
        
        # Initialize RANSAC regressor
        ransac = RANSACRegressor(min_samples=0.1, max_trials=100)
        
        try:
            # Fit model
            ransac.fit(x, signal_data)
            drift = ransac.predict(x)
            
            # Maintain initial point value
            offset = drift[0]
            corrected = signal_data - drift + offset
            return corrected
        except:
            # If RANSAC fails, fall back to linear method
            return correct_drift(signal_data, method='linear', window_size=window_size)
    
    elif method.lower() == 'fitted_curve':
        # Correction using a fitted curve
        if isinstance(window_size, np.ndarray) and len(window_size) == len(signal_data):
            # In this case, window_size is actually the fitted curve
            fitted_curve = window_size
            # Subtract fitted curve but preserve the mean
            signal_mean = np.mean(signal_data)
            corrected = signal_data - fitted_curve + signal_mean
            return corrected
    
    else:
        raise ValueError(f"Unsupported drift correction method: {method}")

def auto_align_baselines(signal: np.ndarray, time: np.ndarray, threshold_pct: float = 3.0, 
                        window_size: int = None, smoothing_size: int = 101, 
                        transition_size: int = 100) -> Tuple[np.ndarray, List[int]]:
    """
    Automatically detect and align sudden changes in signal baseline.
    Optimized to detect jumps based on the derivative of the raw signal.

    Args:
        signal: Signal to process
        time: Corresponding time points
        threshold_pct: Percent change threshold to detect baseline shifts (default: 3.0%)
        window_size: Window size for shift grouping (default: auto based on signal length)
        smoothing_size: Window size for Savitzky-Golay filter if used for baseline estimation
        transition_size: Number of points for smooth transition at boundaries

    Returns:
        aligned_signal: Signal with baselines aligned
        shift_points: Indices where baseline shifts were detected
    """
    aligned_signal = signal.copy()

    if window_size is None:
        window_size = max(int(len(signal) * 0.03), 50)

    # Calculate derivative of the raw signal to find sudden changes
    signal_deriv = np.diff(signal, prepend=signal[0])
    
    deriv_abs = np.abs(signal_deriv)
    deriv_median = np.median(deriv_abs)
    deriv_mad = np.median(np.abs(deriv_abs - deriv_median))

    if deriv_mad > 1e-9:  # Avoid division by zero or very small MAD
        adaptive_threshold = deriv_median + 5.0 * deriv_mad * 1.4826  # Robust threshold
    else:  # Fallback if MAD is zero (e.g., flat signal derivative)
        adaptive_threshold = np.percentile(deriv_abs, 99) if len(deriv_abs) > 0 else 0.01

    signal_range = np.percentile(signal, 95) - np.percentile(signal, 5) if len(signal) > 10 else np.std(signal) if len(
        signal) > 1 else 1.0
    if signal_range < 1e-9: 
        signal_range = 1.0  # Avoid division by zero for flat signals

    min_threshold = signal_range * threshold_pct / 100.0
    final_threshold = max(adaptive_threshold, min_threshold)

    # Ensure final_threshold is a sensible positive value
    if final_threshold <= 1e-9:
        final_threshold = 0.01 * signal_range if signal_range > 1e-9 else 0.01

    print(
        f"Auto-align: Using raw signal derivative. Adaptive Threshold: {adaptive_threshold:.5f}, "
        f"Min Threshold: {min_threshold:.5f}, Final Threshold: {final_threshold:.5f}"
    )

    potential_shifts = np.where(deriv_abs > final_threshold)[0]
    shift_points = []
    if len(potential_shifts) > 0:
        current_group = [potential_shifts[0]]
        for i in range(1, len(potential_shifts)):
            if potential_shifts[i] - potential_shifts[i - 1] < window_size // 4:  # Group close points
                current_group.append(potential_shifts[i])
            else:
                shift_points.append(int(np.median(current_group)))  # Take median of group
                current_group = [potential_shifts[i]]
        if current_group:  # Add last group
            shift_points.append(int(np.median(current_group)))

    all_points = [0] + sorted(list(set(shift_points))) + [len(signal)]  # Ensure sorted unique points
    segments = []
    for i in range(len(all_points) - 1):
        if all_points[i] < all_points[i + 1]:  # Ensure start < end
            segments.append((all_points[i], all_points[i + 1]))

    print(f"Auto-align: Detected {len(shift_points)} baseline shifts at indices: {shift_points}")

    baselines = []
    for start, end in segments:
        segment_data = signal[start:end]  # Use original signal for baseline calculation
        if len(segment_data) > 0:
            baseline = np.percentile(segment_data, 10)  # Robust baseline for the segment
        else:
            baseline = aligned_signal[start - 1] if start > 0 else 0  # Fallback for empty segment
        baselines.append(baseline)

    if len(baselines) > 1:
        target_baseline = baselines[0]  # Align all segments to the baseline of the first segment

        for i in range(1, len(segments)):
            start, end = segments[i]
            original_segment_baseline = baselines[i]  # Baseline of this segment in the original signal

            # Calculate the offset needed to bring this segment's baseline to the target_baseline
            offset_needed = original_segment_baseline - target_baseline

            # Only correct if the shift is meaningful
            if abs(offset_needed) > min_threshold / 2:  # Using half of min_threshold for sensitivity
                print(f"Auto-align: Applying offset of {-offset_needed:.5f} to segment {i} ({start}-{end})")

                # Determine transition size for smoothing the correction
                trans_size = min(transition_size, (end - start) // 3 if (end - start) > 0 else 0)

                if trans_size > 0:
                    weights = np.linspace(0, 1, trans_size)  # Linear transition
                    # Apply transition at the beginning of the segment being shifted
                    aligned_signal[start: start + trans_size] -= offset_needed * weights
                    # Apply full correction for the rest of the segment
                    if start + trans_size < end:
                        aligned_signal[start + trans_size: end] -= offset_needed
                elif end > start:  # If no transition (or segment too small), apply full correction
                    aligned_signal[start:end] -= offset_needed

    return aligned_signal, shift_points

def advanced_baseline_alignment(signal: np.ndarray, time: np.ndarray, window_size: int = None,
                              segmentation_sensitivity: float = 1.0,
                              use_changepoint_detection: bool = True,
                              smooth_transitions: bool = True) -> Tuple[np.ndarray, List[int]]:
    """
    Advanced baseline alignment for complex signals with multiple shifts.
    Optimized to use raw signal for changepoint detection or derivative calculation.

    Args:
        signal: Signal to align
        time: Corresponding time points
        window_size: Size of window for transition smoothness (default: auto)
        segmentation_sensitivity: Controls sensitivity of shift detection (higher = more sensitive)
        use_changepoint_detection: Whether to use statistical changepoint detection
        smooth_transitions: Whether to apply smooth transitions at boundary points

    Returns:
        aligned_signal: Signal with aligned baselines
        changepoints: Indices where baseline shifts were detected
    """
    aligned_signal = signal.copy()
    changepoints = []

    if window_size is None:
        window_size = max(30, min(500, int(len(signal) * 0.02)))  # Default for transition smoothness

    # Use changepoint detection if available
    if use_changepoint_detection and rpt is not None:
        try:
            # Using raw signal for changepoint detection
            changepoint_input_signal = signal.copy()
            # Reshape for ruptures: (n_samples, n_features)
            algo = rpt.Binseg(model="l2").fit(changepoint_input_signal.reshape(-1, 1))

            # Adjust n_segments based on sensitivity
            n_segments_base = 5
            n_segments_calculated = int(n_segments_base * (1 + segmentation_sensitivity * 2))
            n_segments = max(2, min(n_segments_calculated, len(signal) // (
                window_size if window_size > 0 else 50)))
            n_bkps_to_predict = n_segments - 1

            if n_bkps_to_predict < 1:
                print(
                    "Advanced-align: Not enough data or too low sensitivity for multiple segments with ruptures. "
                    "Trying 1 breakpoint."
                )
                n_bkps_to_predict = 1

            if len(signal) > n_bkps_to_predict * 2:  # Ensure enough data points for ruptures
                detected_bkps = algo.predict(n_bkps=n_bkps_to_predict)
                changepoints = [bp for bp in detected_bkps if bp < len(signal)]  # Exclude endpoint if included
                print(
                    f"Advanced-align (ruptures): Predicted {len(changepoints)} changepoints with "
                    f"n_bkps={n_bkps_to_predict}, sensitivity={segmentation_sensitivity}"
                )
            else:
                print(f"Advanced-align (ruptures): Signal too short for {n_bkps_to_predict} breakpoints. Falling back.")
                use_changepoint_detection = False  # Fallback to derivative
                changepoints = []

        except Exception as e:
            print(
                f"Advanced-align: Changepoint detection with 'ruptures' failed: {e}. "
                f"Falling back to derivative method."
            )
            use_changepoint_detection = False  # Fallback if ruptures fails
            changepoints = []

    if not use_changepoint_detection or rpt is None:
        if rpt is None and use_changepoint_detection:  # Explicitly note if ruptures wasn't found
            print("Advanced-align: 'ruptures' package not found. Using derivative method.")

        # Fallback to derivative-based method, using raw signal
        jump_detection_signal = signal.copy()
        derivative = np.diff(jump_detection_signal, prepend=jump_detection_signal[0])
        abs_deriv = np.abs(derivative)
        deriv_median = np.median(abs_deriv)
        deriv_mad = np.median(np.abs(abs_deriv - deriv_median))

        # Adjust threshold: higher sensitivity means lower multiplier for MAD
        mad_multiplier = 5.0 / (segmentation_sensitivity + 1e-6)  # Add epsilon to avoid division by zero

        if deriv_mad > 1e-9:
            threshold = deriv_median + mad_multiplier * deriv_mad * 1.4826
        else:  # Fallback if MAD is zero
            threshold = np.percentile(abs_deriv, 100 - (5 / (segmentation_sensitivity + 1e-6))) if len(
                abs_deriv) > 0 else 0.01

        # Ensure threshold is meaningful relative to signal range
        signal_range_robust = np.percentile(signal, 95) - np.percentile(signal, 5) if len(signal) > 10 else np.std(
            signal) if len(signal) > 1 else 1.0
        if signal_range_robust < 1e-9: signal_range_robust = 1.0
        min_meaningful_jump_deriv = signal_range_robust * 0.01  # e.g. 1% of signal range
        threshold = max(threshold, min_meaningful_jump_deriv)  # Prevent overly sensitive threshold

        print(f"Advanced-align (derivative): Sensitivity: {segmentation_sensitivity}, Threshold: {threshold:.5f}")

        potential_changepoints = np.where(abs_deriv > threshold)[0]
        if len(potential_changepoints) > 0:
            current_group = [potential_changepoints[0]]
            for i in range(1, len(potential_changepoints)):
                if potential_changepoints[i] - potential_changepoints[i - 1] <= window_size // 2:  # Group if close
                    current_group.append(potential_changepoints[i])
                else:
                    changepoints.append(int(np.median(current_group)))
                    current_group = [potential_changepoints[i]]
            if current_group:  # Add last group
                changepoints.append(int(np.median(current_group)))

    all_segment_points = [0] + sorted(list(set(changepoints))) + [len(signal)]
    segments_info = []
    for i in range(len(all_segment_points) - 1):
        start, end = all_segment_points[i], all_segment_points[i + 1]
        if start >= end: continue  # Skip empty or invalid segments

        seg_data = signal[start:end]  # Use original signal for baseline calculation
        if len(seg_data) > 0:
            baseline = np.percentile(seg_data, 10)  # Robust baseline
        else:
            baseline = aligned_signal[start - 1] if start > 0 else 0.0  # Fallback for empty segment
        segments_info.append({'start': start, 'end': end, 'baseline': baseline})

    print(f"Advanced-align: Signal divided into {len(segments_info)} segments at points {changepoints}")

    if len(segments_info) > 1:
        reference_baseline = segments_info[0]['baseline']  # Align to the first segment's baseline
        for seg_info in segments_info[1:]:  # Iterate from the second segment
            start, end, current_baseline = seg_info['start'], seg_info['end'], seg_info['baseline']
            offset = current_baseline - reference_baseline

            # Determine a minimum offset threshold to avoid correcting tiny fluctuations
            signal_range_robust = np.percentile(signal, 95) - np.percentile(signal, 5) if len(signal) > 10 else 1.0
            if signal_range_robust < 1e-9: signal_range_robust = 1.0
            min_offset_threshold = signal_range_robust * 0.005  # e.g., 0.5% of signal range

            if abs(offset) > min_offset_threshold:  # Only correct if offset is significant
                print(f"Advanced-align: Aligning segment ({start}-{end}) by {-offset:.4f}")

                seg_len = end - start
                if smooth_transitions:
                    # Max transition is e.g. 20% of segment length or the global window_size
                    trans_size = min(seg_len // 5, window_size)
                    if trans_size > 0 and trans_size < seg_len:  # Ensure valid transition size
                        weights = np.linspace(0, 1, trans_size)  # Linear transition
                        # Apply transition at the start of the segment being shifted
                        aligned_signal[start: start + trans_size] -= offset * weights
                        # Apply full correction to the rest of the segment
                        aligned_signal[start + trans_size: end] -= offset
                    elif seg_len > 0:  # If segment too small for transition, apply full correction
                        aligned_signal[start:end] -= offset
                elif seg_len > 0:  # No smooth transition, apply full correction
                    aligned_signal[start:end] -= offset
            else:
                print(f"Advanced-align: Skipping segment ({start}-{end}), offset {offset:.4f} too small.")

    return aligned_signal, changepoints

def robust_baseline_alignment(signal: np.ndarray, time: np.ndarray,
                            manual_points=None,
                            detection_sensitivity: float = 1.0,
                            transition_smoothness: float = 0.5,
                            min_stable_segment_duration_sec: float = 2.0,
                            stability_threshold_factor: float = 0.3,
                            detrend_shifted_segments: bool = True,
                            flatten_short_transients_sec: float = 0.0) -> Tuple[np.ndarray, List[int]]:
    """
    Enhanced baseline alignment (v5) with optional flattening of very short transients.

    Args:
        signal: Signal to process
        time: Corresponding time points
        manual_points: Manually specified points for alignment
        detection_sensitivity: How sensitive the algorithm is to baseline changes (higher = more sensitive)
        transition_smoothness: Controls the smoothness of transitions (higher = smoother)
        min_stable_segment_duration_sec: Minimum duration for a segment to be considered stable
        stability_threshold_factor: Factor to determine stability threshold
        detrend_shifted_segments: Whether to detrend segments after shifting
        flatten_short_transients_sec: Max duration for a transient to be flattened (0 means off)

    Returns:
        aligned_signal: Signal with baselines aligned
        detected_shifts: Indices where baseline shifts were detected
    """
    aligned_signal = signal.copy()
    dt = np.mean(np.diff(time)) if len(time) > 1 else 1.0
    min_stable_segment_len_pts = int(min_stable_segment_duration_sec / dt) if dt > 0 else 10
    max_transient_len_pts = int(flatten_short_transients_sec / dt) if dt > 0 and flatten_short_transients_sec > 0 else 0

    print(
        f"Robust Align v5: Sensitivity={detection_sensitivity:.2f}, Smoothness={transition_smoothness:.2f}, "
        f"Detrend={detrend_shifted_segments}"
    )
    print(
        f"Robust Align v5: Min Stable Seg Dur={min_stable_segment_duration_sec:.2f}s, "
        f"Stability Factor={stability_threshold_factor:.2f}"
    )
    if max_transient_len_pts > 0:
        print(
            f"Robust Align v5: Flattening short transients up to {flatten_short_transients_sec:.2f}s "
            f"({max_transient_len_pts} pts)"
        )

    # Step 1: Estimate Global Baseline
    global_baseline_estimate = np.percentile(signal, 10)
    signal_overall_range = np.percentile(signal, 95) - np.percentile(signal, 5)
    if signal_overall_range < 1e-9: signal_overall_range = 1.0
    print(f"Robust Align v5: Global Baseline Estimate = {global_baseline_estimate:.4f}")

    # Step 2: Detect Initial Candidate Jump Points
    candidate_jump_points = []
    print(f"Robust Align v6_cand: Input detection_sensitivity from GUI: {detection_sensitivity:.2f}")

    if manual_points:
        candidate_jump_points = sorted(
            list(set(idx for idx, stype in manual_points if isinstance(idx, (int, np.integer))))) if isinstance(
            manual_points[0], tuple) else sorted(list(set(manual_points)))
        print(f"Robust Align v6_cand: Using {len(candidate_jump_points)} provided manual candidate points.")
    else:
        print("Robust Align v6_cand: No manual points, detecting candidates automatically.")
        first_deriv = np.diff(signal, prepend=signal[0])
        abs_deriv = np.abs(first_deriv)

        if len(abs_deriv) == 0:  # Should not happen with prepend
            print("Robust Align v6_cand: abs_deriv is empty. No candidates.")
            return aligned_signal, []

        deriv_median = np.median(abs_deriv)
        deriv_mad = np.median(np.abs(abs_deriv - deriv_median))

        # Map sensitivity [0.1, 3.0] to multiplier [~15, ~2.5]
        if detection_sensitivity < 1.0:
            # For sensitivity 0.1 to 1.0, map multiplier from ~20 down to ~5
            mad_multiplier_cand = 20.0 - 15.0 * (detection_sensitivity - 0.1) / 0.9
        elif detection_sensitivity < 2.0:
            # For sensitivity 1.0 to 2.0, map multiplier from ~5 down to ~3
            mad_multiplier_cand = 5.0 - 2.0 * (detection_sensitivity - 1.0) / 1.0
        else:  # sensitivity 2.0 to 3.0
            # For sensitivity 2.0 to 3.0, map multiplier from ~3 down to ~1.5
            mad_multiplier_cand = 3.0 - 1.5 * (detection_sensitivity - 2.0) / 1.0

        mad_multiplier_cand = max(1.5, mad_multiplier_cand)  # Ensure a minimum multiplier

        threshold_cand = deriv_median + mad_multiplier_cand * deriv_mad * 1.4826

        # Additional check: if MAD is very small (flat derivative with spikes),
        # threshold_cand might be too low. Ensure it's also a few times the median.
        if deriv_mad < 1e-9 * deriv_median or deriv_mad < 1e-7:  # If MAD is essentially zero or extremely small
            # For flat derivatives with spikes, the threshold should be significantly above the (low) median
            # Use a high percentile as a fallback, but make it less sensitive if GUI sens is low
            percentile_fallback = 99.9 - detection_sensitivity * 1.5  # e.g. 0.1 -> 99.75; 3.0 -> 95.4
            percentile_fallback = max(95.0, min(99.9, percentile_fallback))
            threshold_cand = np.percentile(abs_deriv, percentile_fallback)
            print(
                f"Robust Align v6_cand: MAD very small, using {percentile_fallback:.1f}th percentile "
                f"fallback for threshold_cand."
            )
        else:  # If MAD is useful, ensure threshold is at least (e.g.) 2x median derivative
            min_thresh_vs_median = deriv_median * (1.5 + (
                    1.0 / max(0.1, detection_sensitivity)))  # e.g., sens 0.1 -> median*11.5; sens 3.0 -> median*1.8
            threshold_cand = max(threshold_cand, min_thresh_vs_median)

        print(f"Robust Align v6_cand: GUI_sens={detection_sensitivity:.2f} -> MAD_mult={mad_multiplier_cand:.2f}")
        print(
            f"Robust Align v6_cand: deriv_median={deriv_median:.4f}, deriv_mad={deriv_mad:.4f}, "
            f"calculated_threshold_cand={threshold_cand:.4f}"
        )

        raw_candidates = np.where(abs_deriv > threshold_cand)[0]

        # Clustering remains important
        cluster_window_cand = max(5, int(min_stable_segment_len_pts * 0.10))  # Smaller cluster window for candidates
        if cluster_window_cand % 2 == 0: cluster_window_cand += 1
        # Make cluster window dependent on dt, e.g., 0.1 seconds worth of points
        cluster_window_cand = max(5, int(0.1 / dt if dt > 0 else 5))

        if len(raw_candidates) > 0:
            _current_cluster = [raw_candidates[0]]
            for i in range(1, len(raw_candidates)):
                if raw_candidates[i] - raw_candidates[i - 1] <= cluster_window_cand:
                    _current_cluster.append(raw_candidates[i])
                else:
                    if _current_cluster:
                        # Take the point with max derivative value in the cluster
                        candidate_jump_points.append(_current_cluster[np.argmax(abs_deriv[_current_cluster])])
                    _current_cluster = [raw_candidates[i]]
            if _current_cluster:
                candidate_jump_points.append(_current_cluster[np.argmax(abs_deriv[_current_cluster])])
        candidate_jump_points = sorted(list(set(candidate_jump_points)))

    print(f"Robust Align v6_cand: Final Candidate Jumps ({len(candidate_jump_points)}): {candidate_jump_points}")

    # Step 2.5: Optional Short Transient Flattening (before segment analysis)
    if max_transient_len_pts > 0 and len(candidate_jump_points) >= 2:
        print("\nRobust Align v5: Attempting to flatten short transients...")
        new_candidate_jumps = []
        i = 0
        while i < len(candidate_jump_points):
            current_jump_onset = candidate_jump_points[i]
            if i + 1 < len(candidate_jump_points):
                next_jump_offset = candidate_jump_points[i + 1]
                transient_len = next_jump_offset - current_jump_onset

                if 0 < transient_len <= max_transient_len_pts:
                    print(
                        f"  Found short transient: onset {current_jump_onset}, offset {next_jump_offset}, "
                        f"len {transient_len} pts."
                    )
                    # Flatten this transient by interpolating between points just outside it
                    interp_start_idx = current_jump_onset - 1
                    interp_end_idx = next_jump_offset + 1

                    if interp_start_idx >= 0 and interp_end_idx < len(aligned_signal):
                        val_before = aligned_signal[interp_start_idx]
                        val_after = aligned_signal[interp_end_idx]

                        points_to_replace = np.arange(current_jump_onset, next_jump_offset)
                        if len(points_to_replace) > 0:
                            interpolated_values = np.linspace(val_before, val_after, num=len(points_to_replace) + 2)[1:-1]
                            if len(interpolated_values) == len(points_to_replace):
                                aligned_signal[points_to_replace] = interpolated_values
                                print(f"    Flattened transient from {current_jump_onset} to {next_jump_offset - 1}.")
                                # This transient is handled. Skip the next_jump_offset as it was part of this.
                                i += 2
                                continue
                            else:
                                print(
                                    f"    Interpolation length mismatch for transient "
                                    f"({current_jump_onset}-{next_jump_offset - 1}). Skipping flattening."
                                )
                        else:
                            print(
                                f"    Transient ({current_jump_onset}-{next_jump_offset - 1}) too short to interpolate. "
                                f"Skipping flattening."
                            )
                    else:
                        print(
                            f"    Not enough context to flatten transient ({current_jump_onset}-{next_jump_offset - 1}). "
                            f"Skipping."
                        )

            # If not a short transient or failed to flatten, keep the current jump and move to the next
            new_candidate_jumps.append(current_jump_onset)
            i += 1
        candidate_jump_points = sorted(list(set(new_candidate_jumps)))  # Update candidate jumps
        print(
            f"Robust Align v5: Candidate Jumps after transient flattening ({len(candidate_jump_points)}): "
            f"{candidate_jump_points}"
        )

    # Step 3: Segment Analysis & Identification of Shifted Stable Segments
    segment_boundaries = [0] + candidate_jump_points + [len(signal)]
    segments_to_align_info = []  # Stores segments that pass criteria for main alignment

    print("\nRobust Align v5: Analyzing Segments for sustained shifts...")
    for i in range(len(segment_boundaries) - 1):
        seg_start = segment_boundaries[i]
        seg_end = segment_boundaries[i + 1]
        if seg_start >= seg_end: continue

        seg_len_pts = seg_end - seg_start

        # If flatten_short_transients was active and this segment is very short,
        # it might have already been handled or is too short to be a "stable baseline".
        if max_transient_len_pts > 0 and seg_len_pts <= max_transient_len_pts:
            print(
                f"  Seg {i} ({seg_start}-{seg_end}): Short (len {seg_len_pts}), likely handled by transient flattening "
                f"or too short for stable baseline. Skipping main alignment."
            )
            continue

        # Segment analysis logic
        padding = min(seg_len_pts // 10, 20)
        analysis_start = seg_start + padding
        analysis_end = seg_end - padding
        if analysis_start >= analysis_end: analysis_start, analysis_end = seg_start, seg_end
        segment_data_for_analysis = aligned_signal[analysis_start:analysis_end]  # Use current aligned_signal
        if len(segment_data_for_analysis) < 5: continue
        local_seg_baseline = np.percentile(segment_data_for_analysis, 10)
        seg_internal_range = np.percentile(segment_data_for_analysis, 90) - np.percentile(segment_data_for_analysis, 10)
        stability_abs_threshold = signal_overall_range * stability_threshold_factor
        is_stable_segment = seg_internal_range < stability_abs_threshold
        offset_from_global = local_seg_baseline - global_baseline_estimate
        min_significant_offset = signal_overall_range * (0.02 / detection_sensitivity)

        print(
            f"  Seg {i} ({seg_start}-{seg_end}): Dur={seg_len_pts * dt:.2f}s, Base={local_seg_baseline:.3f}, "
            f"IntRange={seg_internal_range:.3f} (StabThr={stability_abs_threshold:.3f}), Stable={is_stable_segment}, "
            f"Offset={offset_from_global:.3f} (SigOffThr={min_significant_offset:.3f})"
        )

        if (seg_len_pts >= min_stable_segment_len_pts and
                is_stable_segment and
                abs(offset_from_global) > min_significant_offset):
            segments_to_align_info.append({
                'start_idx': seg_start, 'end_idx': seg_end,
                'offset_to_global': offset_from_global,
                # Store original signal for this segment for detrending, relative to its start
                'seg_data_for_detrend': signal[seg_start:seg_end].copy()
            })
            print(f"    -> Marked Segment {i} for sustained shift alignment.")

    # Step 4: Apply Corrections for Sustained Shifts
    if not segments_to_align_info:
        print("Robust Align v5: No segments met criteria for sustained shift alignment.")
        # If short transient flattening happened, aligned_signal might still be modified.
        return aligned_signal, candidate_jump_points

    print("\nRobust Align v5: Applying Corrections for sustained shifts...")
    segments_to_align_info.sort(key=lambda s: s['start_idx'])

    for i, seg_info in enumerate(segments_to_align_info):
        s_start = seg_info['start_idx']
        s_end = seg_info['end_idx']
        s_offset = seg_info['offset_to_global']
        original_segment_data_for_detrend = seg_info['seg_data_for_detrend']

        print(f"  Correcting sustained shift for segment ({s_start}-{s_end}) by offset: {-s_offset:.4f}")
        seg_actual_len = s_end - s_start
        if seg_actual_len <= 0: continue

        # Values for this segment after level correction, and optional detrending
        segment_level_corrected = original_segment_data_for_detrend - s_offset
        final_corrected_values_for_segment = segment_level_corrected.copy()

        if detrend_shifted_segments and seg_actual_len > 10:  # Min length for sensible detrending
            try:
                seg_time_local = np.arange(seg_actual_len)
                poly_degree_detrend = 1
                coeffs = np.polyfit(seg_time_local, segment_level_corrected, poly_degree_detrend)
                internal_trend = np.polyval(coeffs, seg_time_local)

                # Subtract trend, then re-center around the target global baseline
                detrended_temp = segment_level_corrected - internal_trend
                final_corrected_values_for_segment = detrended_temp - np.mean(detrended_temp) + global_baseline_estimate
                print(
                    f"    Detrended segment. Original mean after offset: {np.mean(segment_level_corrected):.3f}, "
                    f"Final mean: {np.mean(final_corrected_values_for_segment):.3f}"
                )
            except np.linalg.LinAlgError:
                print(f"    Detrending failed for segment ({s_start}-{s_end}). Using level correction only.")
                # final_corrected_values_for_segment remains segment_level_corrected

        # Transition Smoothing
        max_trans_pts = min(seg_actual_len // 3, int(0.5 / dt if dt > 0 else 50))
        max_trans_pts = max(5, max_trans_pts)
        transition_len_pts = int(max_trans_pts * transition_smoothness)

        # Apply to aligned_signal[s_start:s_end]
        if transition_len_pts > 0 and s_start + transition_len_pts < s_end:
            # Smoothly blend from current aligned_signal into the new corrected values
            for k_trans in range(transition_len_pts):
                weight = (k_trans + 1) / transition_len_pts  # Linear weight
                idx_current = s_start + k_trans
                # Value from `aligned_signal` *before* this specific segment's correction is applied
                val_from_previous_state = aligned_signal[idx_current]
                val_target_for_this_point = final_corrected_values_for_segment[k_trans]
                aligned_signal[idx_current] = ((1 - weight) * val_from_previous_state + 
                                             weight * val_target_for_this_point)

            # Apply the rest of the corrected values directly
            if s_start + transition_len_pts < s_end:
                aligned_signal[s_start + transition_len_pts: s_end] = final_corrected_values_for_segment[
                                                                      transition_len_pts:]
        elif seg_actual_len > 0:  # No transition or segment too short
            aligned_signal[s_start: s_end] = final_corrected_values_for_segment

    print("Robust Align v5: Sustained shift alignment process finished.")
    return aligned_signal, candidate_jump_points

def detect_artifacts(signal_data: np.ndarray, threshold: float = 3.0, 
                    window_size: int = None) -> np.ndarray:
    """
    Detect artifacts in the signal data.

    Args:
        signal_data (np.ndarray): Input signal data
        threshold (float): Threshold for artifact detection in standard deviations
        window_size (int): Window size for local statistics calculation

    Returns:
        np.ndarray: Boolean mask where True indicates an artifact
    """
    if window_size is None:
        window_size = min(len(signal_data) // 10, 100)
        window_size = max(window_size, 10)  # Minimum window size
    
    # Calculate rolling mean and standard deviation
    rolling_mean = np.zeros_like(signal_data)
    rolling_std = np.zeros_like(signal_data)
    
    half_window = window_size // 2
    padded_signal = np.pad(signal_data, (half_window, half_window), mode='reflect')
    
    for i in range(len(signal_data)):
        window = padded_signal[i:i+window_size]
        rolling_mean[i] = np.mean(window)
        rolling_std[i] = np.std(window)
    
    # Detect outliers
    z_scores = (signal_data - rolling_mean) / (rolling_std + 1e-10)  # Add small constant to avoid division by zero
    artifacts = np.abs(z_scores) > threshold
    
    # Expand artifact regions slightly to capture edges
    expanded_artifacts = np.zeros_like(artifacts, dtype=bool)
    for i in range(len(artifacts)):
        if artifacts[i]:
            start = max(0, i - 2)
            end = min(len(artifacts), i + 3)
            expanded_artifacts[start:end] = True
    
    return expanded_artifacts

def remove_artifacts_fast(signal_data: np.ndarray, artifact_mask: np.ndarray) -> np.ndarray:
    """
    Remove artifacts from signal data using linear interpolation.
    
    Args:
        signal_data: Input signal data
        artifact_mask: Boolean mask where True indicates artifacts to remove
        
    Returns:
        Corrected signal with artifacts removed
    """
    corrected_signal = signal_data.copy()
    
    # Find contiguous artifact regions
    artifact_regions = []
    in_artifact = False
    start_idx = 0
    
    for i in range(len(artifact_mask)):
        if artifact_mask[i] and not in_artifact:
            # Start of new artifact region
            in_artifact = True
            start_idx = i
        elif not artifact_mask[i] and in_artifact:
            # End of artifact region
            in_artifact = False
            artifact_regions.append((start_idx, i))
    
    # Handle case where artifact extends to the end
    if in_artifact:
        artifact_regions.append((start_idx, len(artifact_mask)))
    
    # Interpolate each region
    for start, end in artifact_regions:
        # Get points just outside the artifact for interpolation
        left_idx = max(0, start - 1)
        right_idx = min(len(signal_data) - 1, end)
        
        # If artifact is at the edge, use nearest good value
        if left_idx == 0 and artifact_mask[left_idx]:
            # Find first non-artifact point
            for i in range(len(artifact_mask)):
                if not artifact_mask[i]:
                    left_idx = i
                    break
        
        if right_idx == len(signal_data) - 1 and artifact_mask[right_idx]:
            # Find last non-artifact point
            for i in range(len(artifact_mask) - 1, -1, -1):
                if not artifact_mask[i]:
                    right_idx = i
                    break
        
        # Only interpolate if we have valid endpoints
        if not artifact_mask[left_idx] and not artifact_mask[right_idx]:
            # Linear interpolation
            x = np.array([left_idx, right_idx])
            y = np.array([signal_data[left_idx], signal_data[right_idx]])
            
            # Points to interpolate
            interp_indices = np.arange(start, end)
            interp_values = np.interp(interp_indices, x, y)
            
            # Apply interpolated values
            corrected_signal[start:end] = interp_values
    
    return corrected_signal

def adaptive_motion_correction(main_signal: np.ndarray, control_signal: np.ndarray, 
                              time: np.ndarray, artifact_threshold: float = 2.0) -> np.ndarray:
    """
    Apply adaptive motion correction using a control signal.
    
    Args:
        main_signal: The primary signal to correct
        control_signal: The control signal for motion reference
        time: Time points
        artifact_threshold: Threshold for motion artifact detection
        
    Returns:
        Motion-corrected signal
    """
    corrected_signal = main_signal.copy()
    
    # Step 1: Detect large motion artifacts in control channel
    control_derivative = np.abs(np.diff(control_signal, prepend=control_signal[0]))
    control_derivative_mad = np.median(np.abs(control_derivative - np.median(control_derivative)))
    artifact_threshold_value = np.median(control_derivative) + artifact_threshold * control_derivative_mad * 1.4826
    
    motion_artifacts = control_derivative > artifact_threshold_value
    
    # Expand artifact mask slightly
    artifact_mask = np.zeros_like(motion_artifacts, dtype=bool)
    for i in range(len(motion_artifacts)):
        if motion_artifacts[i]:
            start = max(0, i - 2)
            end = min(len(motion_artifacts), i + 3)
            artifact_mask[start:end] = True
    
    # Step 2: For non-artifact regions, compute local regression coefficients
    segment_size = max(int(len(main_signal) * 0.05), 20)  # 5% of signal or at least 20 points
    num_segments = len(main_signal) // segment_size
    
    regression_slopes = np.zeros(num_segments)
    segment_centers = np.zeros(num_segments, dtype=int)
    
    for i in range(num_segments):
        start = i * segment_size
        end = min((i + 1) * segment_size, len(main_signal))
        
        # Skip segments with too many artifacts
        if np.mean(artifact_mask[start:end]) > 0.3:  # More than 30% artifacts
            continue
        
        # Compute local regression for clean segments
        try:
            slope, _, _, _, _ = linregress(control_signal[start:end], main_signal[start:end])
            regression_slopes[i] = slope
            segment_centers[i] = (start + end) // 2
        except:
            regression_slopes[i] = 0
    
    # Step 3: Apply adaptive correction
    # Interpolate regression coefficients across the entire signal
    valid_segments = (regression_slopes != 0)
    if np.sum(valid_segments) >= 2:
        # Interpolate coefficients
        valid_centers = segment_centers[valid_segments]
        valid_slopes = regression_slopes[valid_segments]
        
        # Extend to edges to avoid extrapolation issues
        if valid_centers[0] > 0:
            valid_centers = np.insert(valid_centers, 0, 0)
            valid_slopes = np.insert(valid_slopes, 0, valid_slopes[0])
        
        if valid_centers[-1] < len(main_signal) - 1:
            valid_centers = np.append(valid_centers, len(main_signal) - 1)
            valid_slopes = np.append(valid_slopes, valid_slopes[-1])
        
        # Interpolate slopes for each point
        interp_func = interp1d(valid_centers, valid_slopes, kind='linear', bounds_error=False, fill_value=(valid_slopes[0], valid_slopes[-1]))
        all_slopes = interp_func(np.arange(len(main_signal)))
        
        # Apply correction with variable coefficients
        for i in range(len(main_signal)):
            # If point is in artifact region, apply more aggressive correction
            if artifact_mask[i]:
                # For artifacts, use a more uniform correction factor
                # Average of recent valid slopes
                nearby_slopes = all_slopes[max(0, i-segment_size):min(len(all_slopes), i+segment_size+1)]
                correction_factor = np.median(nearby_slopes) if len(nearby_slopes) > 0 else 0
                corrected_signal[i] = main_signal[i] - correction_factor * control_signal[i]
            else:
                # For normal regions, use the interpolated slope
                corrected_signal[i] = main_signal[i] - all_slopes[i] * control_signal[i]
    elif np.sum(valid_segments) == 1:
        # Only one valid segment, use a constant correction
        valid_slope = regression_slopes[valid_segments][0]
        corrected_signal = main_signal - valid_slope * control_signal
    else:
        # No valid segments for regression, fall back to simple artifact removal
        if np.any(artifact_mask):
            corrected_signal = remove_artifacts_fast(main_signal, artifact_mask)
    
    return corrected_signal

def downsample_data(time: np.ndarray, signal: np.ndarray, factor: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample the data by the given factor.

    Args:
        time: Time array
        signal: Signal array
        factor: Downsample factor (1 = no downsampling)

    Returns:
        Tuple of (downsampled_time, downsampled_signal)
    """
    if factor <= 1:
        return time, signal

    # Use stride to downsample
    downsampled_time = time[::factor]
    downsampled_signal = signal[::factor]

    return downsampled_time, downsampled_signal

def center_signal(signal: np.ndarray) -> np.ndarray:
    """
    Center a signal by subtracting its mean.

    Args:
        signal: Signal to center

    Returns:
        Signal centered around zero
    """
    if signal is None or len(signal) == 0:
        return signal

    return signal - np.mean(signal)

def smooth_blanking(signal: np.ndarray, time: np.ndarray, blank_start_idx: int, 
                   blank_end_idx: int, transition_width: float = 0.1) -> np.ndarray:
    """
    Apply gradient-based interpolation for blanking with smooth transitions.

    Args:
        signal: Original signal array
        time: Time points array
        blank_start_idx: Start index of region to blank
        blank_end_idx: End index of region to blank
        transition_width: Width of transition as a fraction of blanked region (0-1)

    Returns:
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
            blanked_signal[pre_trans_start:blank_start_idx] = (weights * signal[pre_trans_start:blank_start_idx] + 
                                                             (1 - weights) * target_line)

        # Create post-transition zone (ramping up)
        post_trans_end = min(len(signal), blank_end_idx + trans_size)
        if blank_end_idx < post_trans_end:
            # Create weights from 0→1 for transition
            weights = np.linspace(0, 1, post_trans_end - blank_end_idx)
            # Get target line from interpolated value to data
            target_line = np.linspace(post_blank_val, signal[post_trans_end - 1], post_trans_end - blank_end_idx)
            # Blend target line with original using weights
            blanked_signal[blank_end_idx:post_trans_end] = (weights * signal[blank_end_idx:post_trans_end] + 
                                                          (1 - weights) * target_line)

        # Linear interpolation for blanked region
        x_interp = np.array([blank_start_idx - 1, blank_end_idx + 1])
        y_interp = np.array([pre_blank_val, post_blank_val])
        blanked_region = np.interp(
            np.arange(blank_start_idx, blank_end_idx + 1),
            x_interp, y_interp
        )
        blanked_signal[blank_start_idx:blank_end_idx + 1] = blanked_region

    return blanked_signal

def process_data(time: np.ndarray, analog_1: np.ndarray, analog_2: np.ndarray = None, 
                digital_1: np.ndarray = None, digital_2: np.ndarray = None,
                low_cutoff: float = 0.001, high_cutoff: float = 1.0, 
                downsample_factor: int = 50, artifact_threshold: float = 3.0,
                drift_correction: bool = True, drift_degree: int = 2,
                external_control: np.ndarray = None, 
                edge_protection: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                      np.ndarray, np.ndarray, np.ndarray, 
                                                      np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process photometry data by applying filters and various signal processing techniques.

    Args:
        time: Time array
        analog_1: Primary signal data
        analog_2: Control signal data (optional)
        digital_1: Digital signal 1 (optional)
        digital_2: Digital signal 2 (optional)
        low_cutoff: Low cutoff frequency for high-pass filter
        high_cutoff: High cutoff frequency for low-pass filter
        downsample_factor: Factor to downsample data
        artifact_threshold: Threshold for artifact detection
        drift_correction: Whether to apply drift correction
        drift_degree: Polynomial degree for drift correction
        external_control: External control signal (optional)
        edge_protection: Whether to apply edge protection during filtering

    Returns:
        Tuple containing processed time and signals, including:
        - processed_time: Processed time array
        - processed_signal: Main processed signal
        - processed_signal_no_control: Processed signal without control channel correction
        - processed_analog_2: Processed control signal
        - downsampled_raw_analog_1: Downsampled raw main signal
        - downsampled_raw_analog_2: Downsampled raw control signal
        - processed_digital_1: Digital signal 1
        - processed_digital_2: Digital signal 2
        - artifact_mask: Boolean mask indicating artifacts
        - drift_curve: Fitted drift curve
    """
    # Check for null or empty inputs
    if time is None or len(time) == 0:
        print("Error: Time array is None or empty")
        return None, None, None, None, None, None, None, None, None, None
    
    # Compute sampling rate
    dt = np.diff(time).mean()
    fs = 1 / dt
    
    # Make copies for processing
    processed_signal = analog_1.copy()
    processed_analog_2 = analog_2.copy() if analog_2 is not None else None
    
    # Make a copy for processing without isosbestic control
    processed_signal_no_control = analog_1.copy()
    
    # Calculate baseline from the RAW signal BEFORE filtering
    stable_start = int(len(analog_1) * 0.2)  # 20% in
    stable_end = int(len(analog_1) * 0.4)    # 40% in
    raw_baseline = np.median(analog_1[stable_start:stable_end])
    
    print(f"Using stable baseline region from {stable_start}-{stable_end}, baseline value: {raw_baseline}")
    
    # Use external control if provided
    if external_control is not None:
        # Check if external control has a different size than the current signal
        if len(external_control) != len(processed_signal):
            print(
                f"External control size ({len(external_control)}) doesn't match signal size "
                f"({len(processed_signal)}). Resizing..."
            )
            
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
                processed_signal_no_control = sosfiltfilt(sos, processed_signal_no_control)
                if processed_analog_2 is not None:
                    processed_analog_2 = sosfiltfilt(sos, processed_analog_2)
        
        # Apply low-pass filter (high_cutoff)
        if high_cutoff > 0 and high_cutoff < fs / 2:
            # Use SOS for low-pass filter too
            sos = butter(2, high_cutoff / (fs / 2), 'low', output='sos')
            processed_signal = sosfiltfilt(sos, processed_signal)
            processed_signal_no_control = sosfiltfilt(sos, processed_signal_no_control)
            if processed_analog_2 is not None:
                processed_analog_2 = sosfiltfilt(sos, processed_analog_2)
        
        # Detect artifacts in the control channel (if available)
        artifact_mask = None
        if control_for_artifacts is not None:
            artifact_mask = detect_artifacts(control_for_artifacts, threshold=artifact_threshold)
            # Double-check mask size matches current signal
            if np.size(artifact_mask) != len(processed_signal):
                print(
                    f"Warning: artifact mask size mismatch! Expected {len(processed_signal)}, "
                    f"got {np.size(artifact_mask)}"
                )
                # Create a safe mask with the correct size
                artifact_mask = np.zeros(len(processed_signal), dtype=bool)
            
            # Store the unprocessed signal for comparison
            processed_signal_no_control = processed_signal.copy()
            
            # Apply motion correction
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
        
        # Special handling for start of recording
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
                processed_signal_no_control[i] = ((1 - weight[i]) * nc_linear_trend[i] + 
                                                weight[i] * processed_signal_no_control[i])
        
        # Find the most stable region for baseline calculation
        window_size = min(int(len(processed_signal) * 0.2), int(20 * fs))  # 20% of signal or 20 seconds
        min_variance = float('inf')
        best_start_idx = 0
        
        # Skip the first 10% that may contain noise or blanked regions
        search_start = max(start_idx, int(len(processed_signal) * 0.1))
        
        # Search through the signal for the most stable window
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
                    processed_signal = correct_drift(processed_signal, method='fitted_curve', window_size=drift_curve)
                
                # Apply same drift correction to no-control signal
                if drift_curve is not None:
                    processed_signal_no_control = correct_drift(processed_signal_no_control, 
                                                              method='fitted_curve', 
                                                              window_size=drift_curve)
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
                print(
                    f"Adjusted baseline from {smart_baseline:.4f} to {adjusted_baseline:.4f} to ensure positive dF/F"
                )
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
        if 'processed_signal_no_control' not in locals():
            processed_signal_no_control = analog_1.copy()
    
    # Ensure all arrays have consistent lengths before returning
    print(f"Final processed array lengths: time={len(processed_time)}, signal={len(processed_signal)}")
    min_len = min(len(processed_time), len(processed_signal))
    processed_time = processed_time[:min_len]
    processed_signal = processed_signal[:min_len]
    if processed_signal_no_control is not None:
        if len(processed_signal_no_control) > min_len:
            processed_signal_no_control = processed_signal_no_control[:min_len]
    if processed_analog_2 is not None:
        if len(processed_analog_2) > min_len:
            processed_analog_2 = processed_analog_2[:min_len]
    if artifact_mask is not None and len(artifact_mask) > min_len:
        artifact_mask = artifact_mask[:min_len]
    if drift_curve is not None and len(drift_curve) > min_len:
        drift_curve = drift_curve[:min_len]
    
    # Return the processed data, downsampled raw data, and artifact mask
    return (processed_time, processed_signal, processed_signal_no_control, processed_analog_2,
            downsampled_raw_analog_1, downsampled_raw_analog_2, processed_digital_1, processed_digital_2,
            artifact_mask, drift_curve)

def calculate_signal_to_noise(signal_data: np.ndarray, noise_estimation_method: str = 'mad',
                           signal_window: int = None, noise_window: int = None) -> float:
    """
    Calculate signal-to-noise ratio (SNR).

    Args:
        signal_data (np.ndarray): Input signal data
        noise_estimation_method (str): Noise estimation method, options: 'mad', 'std', 'diff', 'window'
        signal_window (int): Window size for signal estimation
        noise_window (int): Window size for noise estimation

    Returns:
        float: Signal-to-noise ratio
    """
    # Check input
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    
    # Estimate signal amplitude
    if signal_window is None:
        # Use full signal range
        signal_amplitude = np.max(signal_data) - np.min(signal_data)
    else:
        # Use sliding window to find maximum amplitude
        signal_amplitude = 0
        for i in range(0, len(signal_data) - signal_window + 1):
            window = signal_data[i:i+signal_window]
            amplitude = np.max(window) - np.min(window)
            signal_amplitude = max(signal_amplitude, amplitude)
    
    # Estimate noise level
    if noise_estimation_method.lower() == 'mad':
        # Median absolute deviation - more robust than standard deviation
        median = np.median(signal_data)
        mad = np.median(np.abs(signal_data - median))
        # For normal distribution, MAD needs to be multiplied by constant 1.4826
        noise_level = 1.4826 * mad
    
    elif noise_estimation_method.lower() == 'std':
        # Standard deviation
        noise_level = np.std(signal_data)
    
    elif noise_estimation_method.lower() == 'diff':
        # Using differences between adjacent points
        diffs = np.diff(signal_data)
        noise_level = np.std(diffs) / np.sqrt(2)
    
    elif noise_estimation_method.lower() == 'window':
        # Estimate noise in a specific window
        if noise_window is None:
            # Default to first 10% of data
            noise_window = max(int(len(signal_data) * 0.1), 1)
        
        noise_segment = signal_data[:noise_window]
        noise_level = np.std(noise_segment)
    
    else:
        raise ValueError(f"Unsupported noise estimation method: {noise_estimation_method}")
    
    # Avoid division by zero
    if noise_level == 0:
        return float('inf')
    
    # Calculate SNR
    snr = signal_amplitude / noise_level
    
    return snr

def extract_frequency_components(signal_data: np.ndarray, fs: float = 100.0,
                                method: str = 'fft', normalize: bool = True) -> Dict[str, np.ndarray]:
    """
    Extract frequency components from the signal.

    Args:
        signal_data (np.ndarray): Input signal data
        fs (float): Sampling rate (Hz)
        method (str): Spectral analysis method, options: 'fft', 'periodogram', 'welch'
        normalize (bool): Whether to normalize spectral amplitude

    Returns:
        Dict[str, np.ndarray]: Dictionary containing frequencies and magnitudes
    """
    # Check input
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    
    # Apply selected spectral analysis method
    if method.lower() == 'fft':
        # Fast Fourier Transform
        n = len(signal_data)
        fft_result = np.fft.rfft(signal_data)
        freq = np.fft.rfftfreq(n, d=1/fs)
        
        # Calculate magnitude spectrum
        magnitude = np.abs(fft_result)
        
        # Normalize
        if normalize and np.max(magnitude) > 0:
            magnitude = magnitude / np.max(magnitude)
        
        return {'frequencies': freq, 'magnitudes': magnitude}
    
    elif method.lower() == 'periodogram':
        # Periodogram
        freq, psd = signal.periodogram(signal_data, fs)
        
        # Calculate magnitude spectrum (square root of PSD)
        magnitude = np.sqrt(psd)
        
        # Normalize
        if normalize and np.max(magnitude) > 0:
            magnitude = magnitude / np.max(magnitude)
        
        return {'frequencies': freq, 'magnitudes': magnitude}
    
    elif method.lower() == 'welch':
        # Welch method
        # Use Hanning window and 50% overlap
        nperseg = min(256, len(signal_data))
        freq, psd = signal.welch(signal_data, fs, nperseg=nperseg)
        
        # Calculate magnitude spectrum
        magnitude = np.sqrt(psd)
        
        # Normalize
        if normalize and np.max(magnitude) > 0:
            magnitude = magnitude / np.max(magnitude)
        
        return {'frequencies': freq, 'magnitudes': magnitude}
    
    else:
        raise ValueError(f"Unsupported spectral analysis method: {method}")

def resample_signal(signal_data: np.ndarray, time_data: np.ndarray = None,
                   new_fs: float = None, target_length: int = None,
                   method: str = 'linear') -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample a signal.

    Args:
        signal_data (np.ndarray): Input signal data
        time_data (np.ndarray, optional): Time axis data
        new_fs (float, optional): New sampling rate
        target_length (int, optional): Target signal length
        method (str): Resampling method, options: 'linear', 'cubic', 'nearest', 'quadratic', 'previous', 'next'

    Returns:
        Tuple[np.ndarray, np.ndarray]: Resampled (signal, time) tuple

    Note:
        Either new_fs or target_length must be specified
    """
    # Check input
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    
    if time_data is None:
        time_data = np.arange(len(signal_data))
    
    if not isinstance(time_data, np.ndarray):
        time_data = np.array(time_data)
    
    if new_fs is None and target_length is None:
        raise ValueError("Either new_fs or target_length must be specified")
    
    # Calculate original sampling rate
    if len(time_data) > 1:
        original_fs = 1.0 / np.mean(np.diff(time_data))
    else:
        original_fs = 1.0
    
    # Determine new time axis
    if new_fs is not None:
        # Calculate new time axis using new sampling rate
        total_time = time_data[-1] - time_data[0]
        num_samples = int(total_time * new_fs) + 1
        new_time = np.linspace(time_data[0], time_data[-1], num_samples)
    else:
        # Create new time axis with target length
        new_time = np.linspace(time_data[0], time_data[-1], target_length)
    
    # Use scipy's interp1d for interpolation
    interp_func = interp1d(time_data, signal_data, kind=method, 
                          bounds_error=False, fill_value='extrapolate')
    new_signal = interp_func(new_time)
    
    return new_signal, new_time

def process_batch(signals: Dict[str, np.ndarray], 
                 processing_pipeline: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """
    Apply a processing pipeline to multiple signals.

    Args:
        signals (Dict[str, np.ndarray]): Input signals dictionary
        processing_pipeline (List[Dict[str, Any]]): List of processing steps, each step is a dictionary
                                                with 'function' and 'params' keys

    Returns:
        Dict[str, np.ndarray]: Processed signals dictionary
    """
    # Create output signals dictionary
    processed_signals = {}
    
    # Process each signal
    for signal_name, signal_data in signals.items():
        processed_data = signal_data.copy()
        
        # Apply processing pipeline
        for step in processing_pipeline:
            function_name = step['function']
            params = step.get('params', {})
            
            # Select processing function
            if function_name == 'filter':
                processed_data = apply_filter(processed_data, **params)
            elif function_name == 'smooth':
                processed_data = smooth_signal(processed_data, **params)
            elif function_name == 'normalize':
                processed_data = normalize_signal(processed_data, **params)
            elif function_name == 'baseline':
                processed_data = correct_baseline(processed_data, **params)
            elif function_name == 'drift':
                processed_data = correct_drift(processed_data, **params)
            elif function_name == 'align_baselines':
                if 'time' in params:
                    processed_data, _ = auto_align_baselines(processed_data, params.pop('time'), **params)
                else:
                    print(f"Warning: 'time' parameter missing for align_baselines on {signal_name}")
            elif function_name == 'advanced_align':
                if 'time' in params:
                    processed_data, _ = advanced_baseline_alignment(processed_data, params.pop('time'), **params)
                else:
                    print(f"Warning: 'time' parameter missing for advanced_align on {signal_name}")
            elif function_name == 'custom':
                # Custom processing function
                custom_func = params.pop('function')
                processed_data = custom_func(processed_data, **params)
            else:
                raise ValueError(f"Unknown processing function: {function_name}")
        
        # Store processed signal
        processed_signals[signal_name] = processed_data
    
    return processed_signals

def create_processing_pipeline(steps: List[Dict[str, Any]]) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a reusable signal processing pipeline.

    Args:
        steps (List[Dict[str, Any]]): List of processing steps, each step is a dictionary
                                      with 'function' and 'params'

    Returns:
        Callable[[np.ndarray], np.ndarray]: Function that can be applied to a signal
    """
    def pipeline(signal_data: np.ndarray, time_data: np.ndarray = None) -> np.ndarray:
        """Apply processing pipeline to signal"""
        processed_data = signal_data.copy()
        
        for step in steps:
            function_name = step['function']
            params = step.get('params', {}).copy()  # Make a copy to avoid modifying original
            
            # Add time data if required and available
            if function_name in ['align_baselines', 'advanced_align', 'robust_align'] and time_data is not None:
                params['time'] = time_data
            
            # Select processing function
            if function_name == 'filter':
                processed_data = apply_filter(processed_data, **params)
            elif function_name == 'smooth':
                processed_data = smooth_signal(processed_data, **params)
            elif function_name == 'normalize':
                processed_data = normalize_signal(processed_data, **params)
            elif function_name == 'baseline':
                processed_data = correct_baseline(processed_data, **params)
            elif function_name == 'drift':
                processed_data = correct_drift(processed_data, **params)
            elif function_name == 'align_baselines':
                if 'time' in params:
                    processed_data, _ = auto_align_baselines(processed_data, params.pop('time'), **params)
                else:
                    print("Warning: 'time' parameter missing for align_baselines, skipping step")
            elif function_name == 'advanced_align':
                if 'time' in params:
                    processed_data, _ = advanced_baseline_alignment(processed_data, params.pop('time'), **params)
                else:
                    print("Warning: 'time' parameter missing for advanced_align, skipping step")
            elif function_name == 'robust_align':
                if 'time' in params:
                    processed_data, _ = robust_baseline_alignment(processed_data, params.pop('time'), **params)
                else:
                    print("Warning: 'time' parameter missing for robust_align, skipping step")
            elif function_name == 'custom':
                # Custom processing function
                custom_func = params.pop('function')
                processed_data = custom_func(processed_data, **params)
            else:
                raise ValueError(f"Unknown processing function: {function_name}")
        
        return processed_data
    
    return pipeline
