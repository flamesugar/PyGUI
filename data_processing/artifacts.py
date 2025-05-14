"""
Artifacts detection and handling module.
Provides functions to detect, identify, remove and correct signal artifacts,
as well as assess data quality in photometry recordings.
"""

import numpy as np
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import warnings

def detect_artifacts(signal_data: np.ndarray, threshold: float = 3.0, 
                   window_size: int = None, method: str = 'zscore') -> np.ndarray:
    """
    Detect artifacts in signal data using various detection methods.

    Args:
        signal_data (np.ndarray): Input signal data
        threshold (float): Detection threshold (meaning varies by method)
        window_size (int): Window size for local statistics calculation (None for auto)
        method (str): Detection method: 'zscore', 'mad', 'derivative', 'wavelet', or 'combined'

    Returns:
        np.ndarray: Boolean mask where True indicates artifact presence
    """
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    
    # Auto-determine window size if not provided
    if window_size is None:
        window_size = min(len(signal_data) // 10, 100)
        window_size = max(window_size, 10)  # Ensure minimum window size
    
    # Ensure window size is valid
    window_size = min(window_size, len(signal_data) // 2)
    window_size = max(window_size, 3)  # Minimum 3 points for meaningful statistics
    
    # Calculate artifacts based on selected method
    if method.lower() == 'zscore':
        return _detect_artifacts_zscore(signal_data, threshold, window_size)
    
    elif method.lower() == 'mad':
        return _detect_artifacts_mad(signal_data, threshold, window_size)
    
    elif method.lower() == 'derivative':
        return _detect_artifacts_derivative(signal_data, threshold, window_size)
    
    elif method.lower() == 'wavelet':
        return _detect_artifacts_wavelet(signal_data, threshold, window_size)
    
    elif method.lower() == 'combined':
        return _detect_artifacts_combined(signal_data, threshold, window_size)
    
    else:
        warnings.warn(f"Unknown artifact detection method: {method}. Using 'zscore' method.")
        return _detect_artifacts_zscore(signal_data, threshold, window_size)

def _detect_artifacts_zscore(signal_data: np.ndarray, threshold: float, window_size: int) -> np.ndarray:
    """
    Detect artifacts using z-score (local mean and standard deviation).
    
    Args:
        signal_data: Input signal
        threshold: Z-score threshold
        window_size: Window size for local statistics
    
    Returns:
        Boolean mask where True indicates artifact presence
    """
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
    z_scores = (signal_data - rolling_mean) / (rolling_std + 1e-10)  # Add small value to prevent division by zero
    artifacts = np.abs(z_scores) > threshold
    
    # Expand artifact regions slightly to capture edges
    expanded_artifacts = np.zeros_like(artifacts, dtype=bool)
    for i in range(len(artifacts)):
        if artifacts[i]:
            start = max(0, i - 2)
            end = min(len(artifacts), i + 3)
            expanded_artifacts[start:end] = True
    
    return expanded_artifacts

def _detect_artifacts_mad(signal_data: np.ndarray, threshold: float, window_size: int) -> np.ndarray:
    """
    Detect artifacts using median absolute deviation (more robust than z-score).
    
    Args:
        signal_data: Input signal
        threshold: MAD threshold multiplier
        window_size: Window size for local statistics
    
    Returns:
        Boolean mask where True indicates artifact presence
    """
    # Calculate rolling median and MAD
    rolling_median = np.zeros_like(signal_data)
    rolling_mad = np.zeros_like(signal_data)
    
    half_window = window_size // 2
    padded_signal = np.pad(signal_data, (half_window, half_window), mode='reflect')
    
    for i in range(len(signal_data)):
        window = padded_signal[i:i+window_size]
        rolling_median[i] = np.median(window)
        rolling_mad[i] = np.median(np.abs(window - rolling_median[i]))
    
    # MAD to standard deviation conversion factor for normal distribution
    consistency_constant = 1.4826
    
    # Detect outliers
    deviations = np.abs(signal_data - rolling_median) / (rolling_mad * consistency_constant + 1e-10)
    artifacts = deviations > threshold
    
    # Expand artifact regions slightly
    expanded_artifacts = np.zeros_like(artifacts, dtype=bool)
    for i in range(len(artifacts)):
        if artifacts[i]:
            start = max(0, i - 2)
            end = min(len(artifacts), i + 3)
            expanded_artifacts[start:end] = True
    
    return expanded_artifacts

def _detect_artifacts_derivative(signal_data: np.ndarray, threshold: float, window_size: int) -> np.ndarray:
    """
    Detect artifacts based on signal derivative (helps identify rapid changes).
    
    Args:
        signal_data: Input signal
        threshold: Derivative threshold multiplier
        window_size: Context window size
    
    Returns:
        Boolean mask where True indicates artifact presence
    """
    # Calculate first derivative
    derivative = np.diff(signal_data, prepend=signal_data[0])
    
    # Calculate local MAD of the derivative
    rolling_median = np.zeros_like(derivative)
    rolling_mad = np.zeros_like(derivative)
    
    half_window = window_size // 2
    padded_derivative = np.pad(derivative, (half_window, half_window), mode='reflect')
    
    for i in range(len(derivative)):
        window = padded_derivative[i:i+window_size]
        rolling_median[i] = np.median(window)
        rolling_mad[i] = np.median(np.abs(window - rolling_median[i]))
    
    # Detect outliers in derivative
    consistency_constant = 1.4826
    abs_derivative = np.abs(derivative)
    artifact_threshold = rolling_median + threshold * rolling_mad * consistency_constant
    artifacts = abs_derivative > artifact_threshold
    
    # Group closely spaced artifacts
    expanded_artifacts = np.zeros_like(artifacts, dtype=bool)
    for i in range(len(artifacts)):
        if artifacts[i]:
            start = max(0, i - 3)
            end = min(len(artifacts), i + 4)
            expanded_artifacts[start:end] = True
    
    return expanded_artifacts

def _detect_artifacts_wavelet(signal_data: np.ndarray, threshold: float, window_size: int) -> np.ndarray:
    """
    Detect artifacts using wavelet decomposition (good for identifying transients).
    
    Args:
        signal_data: Input signal
        threshold: Wavelet coefficient threshold
        window_size: Not directly used but kept for API consistency
        
    Returns:
        Boolean mask where True indicates artifact presence
    """
    try:
        import pywt
    except ImportError:
        warnings.warn("PyWavelets not installed. Falling back to z-score method.")
        return _detect_artifacts_zscore(signal_data, threshold, window_size)
    
    # Perform wavelet decomposition
    wavelet = 'db4'  # Daubechies wavelet
    max_level = pywt.dwt_max_level(len(signal_data), pywt.Wavelet(wavelet).dec_len)
    level = min(max_level, 4)  # Use reasonable decomposition level
    
    # Get wavelet coefficients
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    
    # Focus on detail coefficients which capture fast changes
    detail_coeffs = coeffs[1]  # First level detail coefficients
    
    # Calculate threshold for detail coefficients
    coeff_median = np.median(detail_coeffs)
    coeff_mad = np.median(np.abs(detail_coeffs - coeff_median))
    wavelet_threshold = coeff_median + threshold * coeff_mad * 1.4826
    
    # Reconstruct wavelet details to original signal length
    detail_signal = pywt.waverec([np.zeros_like(coeffs[0]), detail_coeffs] + 
                                [np.zeros_like(c) for c in coeffs[2:]], wavelet)
    
    # Trim to match original length if needed
    detail_signal = detail_signal[:len(signal_data)]
    
    # Detect artifacts where detail coefficient is high
    artifacts = np.abs(detail_signal) > wavelet_threshold
    
    # Expand artifact regions
    expanded_artifacts = np.zeros_like(artifacts, dtype=bool)
    for i in range(len(artifacts)):
        if artifacts[i]:
            start = max(0, i - 3)
            end = min(len(artifacts), i + 4)
            expanded_artifacts[start:end] = True
    
    return expanded_artifacts

def _detect_artifacts_combined(signal_data: np.ndarray, threshold: float, window_size: int) -> np.ndarray:
    """
    Combine multiple artifact detection methods for better accuracy.
    
    Args:
        signal_data: Input signal
        threshold: Base threshold for all methods
        window_size: Window size for statistics
        
    Returns:
        Boolean mask where True indicates artifact presence
    """
    # Apply multiple methods with adjusted thresholds
    zscore_artifacts = _detect_artifacts_zscore(signal_data, threshold * 1.2, window_size)
    derivative_artifacts = _detect_artifacts_derivative(signal_data, threshold * 1.0, window_size)
    mad_artifacts = _detect_artifacts_mad(signal_data, threshold * 1.1, window_size)
    
    # Combine results (require at least 2 methods to agree)
    combined_mask = np.zeros_like(zscore_artifacts, dtype=int)
    combined_mask += zscore_artifacts
    combined_mask += derivative_artifacts
    combined_mask += mad_artifacts
    
    # Points where at least 2 methods agree
    final_artifacts = combined_mask >= 2
    
    return final_artifacts

def remove_artifacts(signal_data: np.ndarray, artifact_mask: np.ndarray, 
                    method: str = 'interpolate', 
                    window_size: int = 5,
                    smooth_edges: bool = True) -> np.ndarray:
    """
    Remove artifacts from signal data using different correction methods.
    
    Args:
        signal_data (np.ndarray): Input signal data
        artifact_mask (np.ndarray): Boolean mask where True indicates artifact
        method (str): Removal method: 'interpolate', 'median', 'mean', 'savgol', or 'adaptive'
        window_size (int): Context window size for artifact correction
        smooth_edges (bool): Whether to smooth transitions at artifact boundaries
    
    Returns:
        np.ndarray: Signal with artifacts removed
    """
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    
    if artifact_mask is None or not np.any(artifact_mask):
        return signal_data.copy()  # No artifacts to remove
    
    # Identify contiguous artifact regions
    artifact_regions = _identify_artifact_regions(artifact_mask)
    
    # Apply selected correction method
    if method.lower() == 'interpolate':
        return _remove_artifacts_interpolate(signal_data, artifact_regions, smooth_edges)
    
    elif method.lower() == 'median':
        return _remove_artifacts_median(signal_data, artifact_regions, window_size, smooth_edges)
    
    elif method.lower() == 'mean':
        return _remove_artifacts_mean(signal_data, artifact_regions, window_size, smooth_edges)
    
    elif method.lower() == 'savgol':
        return _remove_artifacts_savgol(signal_data, artifact_regions, window_size, smooth_edges)
    
    elif method.lower() == 'adaptive':
        return _remove_artifacts_adaptive(signal_data, artifact_regions, window_size, smooth_edges)
    
    else:
        warnings.warn(f"Unknown artifact removal method: {method}. Using 'interpolate' method.")
        return _remove_artifacts_interpolate(signal_data, artifact_regions, smooth_edges)

def _identify_artifact_regions(artifact_mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Identify contiguous regions of artifacts.
    
    Args:
        artifact_mask: Boolean mask where True indicates artifact presence
    
    Returns:
        List of (start_idx, end_idx) tuples for each artifact region
    """
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
    
    return artifact_regions

def _remove_artifacts_interpolate(signal_data: np.ndarray, 
                                artifact_regions: List[Tuple[int, int]],
                                smooth_edges: bool) -> np.ndarray:
    """
    Remove artifacts using linear interpolation.
    
    Args:
        signal_data: Input signal
        artifact_regions: List of (start, end) tuples for artifact regions
        smooth_edges: Whether to smooth transitions
    
    Returns:
        Corrected signal
    """
    corrected_signal = signal_data.copy()
    
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
            
            # Apply smooth transition if requested
            if smooth_edges and end - start > 4:
                # Apply smooth transition at start
                transition_length = min(3, (end - start) // 3)
                if transition_length > 0:
                    weights = np.linspace(0, 1, transition_length)
                    for i in range(transition_length):
                        blend_idx = start + i
                        if blend_idx < end:
                            corrected_signal[blend_idx] = ((1 - weights[i]) * signal_data[blend_idx] + 
                                                         weights[i] * interp_values[i])
                    
                    # Update the interpolation range to exclude transition
                    interp_values = interp_values[transition_length:]
                    start = start + transition_length
                
                # Apply smooth transition at end
                if transition_length > 0 and start < end - transition_length:
                    weights = np.linspace(1, 0, transition_length)
                    for i in range(transition_length):
                        blend_idx = end - transition_length + i
                        interp_idx = len(interp_values) - transition_length + i
                        if blend_idx < len(corrected_signal) and interp_idx < len(interp_values):
                            corrected_signal[blend_idx] = (weights[i] * interp_values[interp_idx] + 
                                                         (1 - weights[i]) * signal_data[blend_idx])
                    
                    # Update the interpolation range to exclude transition
                    interp_values = interp_values[:-transition_length]
                    end = end - transition_length
            
            # Apply interpolated values to the non-transition part
            if start < end and len(interp_values) > 0:
                corrected_signal[start:end] = interp_values
    
    return corrected_signal

def _remove_artifacts_median(signal_data: np.ndarray, 
                          artifact_regions: List[Tuple[int, int]],
                          window_size: int,
                          smooth_edges: bool) -> np.ndarray:
    """
    Remove artifacts using median filtering.
    
    Args:
        signal_data: Input signal
        artifact_regions: List of (start, end) tuples for artifact regions
        window_size: Size of window for median calculation
        smooth_edges: Whether to smooth transitions
    
    Returns:
        Corrected signal
    """
    corrected_signal = signal_data.copy()
    
    # Pad signal to handle edge artifacts
    padded_signal = np.pad(signal_data, (window_size, window_size), mode='reflect')
    
    for start, end in artifact_regions:
        # Apply median filter to artifact region
        for i in range(start, end):
            # Extract window around the artifact point
            window_start = i - start + window_size
            local_window = padded_signal[window_start:window_start + window_size]
            
            # Replace with median
            corrected_signal[i] = np.median(local_window)
        
        # Apply smooth transition if requested
        if smooth_edges and end - start > 4:
            # Apply smooth transition at start
            transition_length = min(3, (end - start) // 3)
            if transition_length > 0:
                weights = np.linspace(0, 1, transition_length)
                for i in range(transition_length):
                    blend_idx = start + i
                    if blend_idx < end:
                        corrected_signal[blend_idx] = ((1 - weights[i]) * signal_data[blend_idx] + 
                                                     weights[i] * corrected_signal[blend_idx])
            
            # Apply smooth transition at end
            if transition_length > 0:
                weights = np.linspace(1, 0, transition_length)
                for i in range(transition_length):
                    blend_idx = end - transition_length + i
                    if blend_idx < len(corrected_signal):
                        corrected_signal[blend_idx] = (weights[i] * corrected_signal[blend_idx] + 
                                                     (1 - weights[i]) * signal_data[blend_idx])
    
    return corrected_signal

def _remove_artifacts_mean(signal_data: np.ndarray, 
                         artifact_regions: List[Tuple[int, int]],
                         window_size: int,
                         smooth_edges: bool) -> np.ndarray:
    """
    Remove artifacts using local mean.
    
    Args:
        signal_data: Input signal
        artifact_regions: List of (start, end) tuples for artifact regions
        window_size: Size of window for mean calculation
        smooth_edges: Whether to smooth transitions
    
    Returns:
        Corrected signal
    """
    # Implementation similar to median but using mean
    corrected_signal = signal_data.copy()
    
    # Pad signal to handle edge artifacts
    padded_signal = np.pad(signal_data, (window_size, window_size), mode='reflect')
    
    for start, end in artifact_regions:
        # Apply mean filter to artifact region
        for i in range(start, end):
            # Extract window around the artifact point
            window_start = i - start + window_size
            local_window = padded_signal[window_start:window_start + window_size]
            
            # Replace with mean
            corrected_signal[i] = np.mean(local_window)
        
        # Apply smooth transition if requested (same as in median method)
        if smooth_edges and end - start > 4:
            transition_length = min(3, (end - start) // 3)
            
            # Start transition
            if transition_length > 0:
                weights = np.linspace(0, 1, transition_length)
                for i in range(transition_length):
                    blend_idx = start + i
                    if blend_idx < end:
                        corrected_signal[blend_idx] = ((1 - weights[i]) * signal_data[blend_idx] + 
                                                     weights[i] * corrected_signal[blend_idx])
            
            # End transition
            if transition_length > 0:
                weights = np.linspace(1, 0, transition_length)
                for i in range(transition_length):
                    blend_idx = end - transition_length + i
                    if blend_idx < len(corrected_signal):
                        corrected_signal[blend_idx] = (weights[i] * corrected_signal[blend_idx] + 
                                                     (1 - weights[i]) * signal_data[blend_idx])
    
    return corrected_signal

def _remove_artifacts_savgol(signal_data: np.ndarray, 
                           artifact_regions: List[Tuple[int, int]],
                           window_size: int,
                           smooth_edges: bool) -> np.ndarray:
    """
    Remove artifacts using Savitzky-Golay filter.
    
    Args:
        signal_data: Input signal
        artifact_regions: List of (start, end) tuples for artifact regions
        window_size: Size of window for filter
        smooth_edges: Whether to smooth transitions
    
    Returns:
        Corrected signal
    """
    corrected_signal = signal_data.copy()
    
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Minimum window size for Savitzky-Golay filter
    window_size = max(window_size, 5)
    
    # Apply Savitzky-Golay filter to entire signal
    try:
        # For Savgol, polynomial order must be less than window size
        poly_order = min(3, window_size - 1)
        smoothed_signal = signal.savgol_filter(signal_data, window_size, poly_order)
    except:
        # Fall back to simple smoothing if Savgol fails
        smoothed_signal = gaussian_filter1d(signal_data, window_size / 6)
    
    # Replace only artifact regions with smoothed signal
    for start, end in artifact_regions:
        # Apply the smoothed signal in the artifact region
        corrected_signal[start:end] = smoothed_signal[start:end]
        
        # Apply smooth transition if requested
        if smooth_edges and end - start > 4:
            transition_length = min(3, (end - start) // 3)
            
            # Start transition
            if transition_length > 0:
                weights = np.linspace(0, 1, transition_length)
                for i in range(transition_length):
                    blend_idx = start + i
                    if blend_idx < end:
                        corrected_signal[blend_idx] = ((1 - weights[i]) * signal_data[blend_idx] + 
                                                     weights[i] * smoothed_signal[blend_idx])
            
            # End transition
            if transition_length > 0:
                weights = np.linspace(1, 0, transition_length)
                for i in range(transition_length):
                    blend_idx = end - transition_length + i
                    if blend_idx < len(corrected_signal):
                        corrected_signal[blend_idx] = (weights[i] * smoothed_signal[blend_idx] + 
                                                     (1 - weights[i]) * signal_data[blend_idx])
    
    return corrected_signal

def _remove_artifacts_adaptive(signal_data: np.ndarray, 
                             artifact_regions: List[Tuple[int, int]],
                             window_size: int,
                             smooth_edges: bool) -> np.ndarray:
    """
    Remove artifacts using an adaptive approach that chooses the best method
    based on artifact characteristics.
    
    Args:
        signal_data: Input signal
        artifact_regions: List of (start, end) tuples for artifact regions
        window_size: Size of window for filters
        smooth_edges: Whether to smooth transitions
    
    Returns:
        Corrected signal
    """
    corrected_signal = signal_data.copy()
    
    # Apply different correction methods based on artifact characteristics
    for start, end in artifact_regions:
        artifact_length = end - start
        
        # Choose method based on artifact length
        if artifact_length <= 3:
            # Very short artifacts - use interpolation
            corrected_region = _remove_artifacts_interpolate(
                signal_data, [(start, end)], smooth_edges
            )[start:end]
        
        elif artifact_length <= 10:
            # Short artifacts - use median filter
            corrected_region = _remove_artifacts_median(
                signal_data, [(start, end)], max(window_size, 5), smooth_edges
            )[start:end]
        
        else:
            # Longer artifacts - use Savitzky-Golay filter
            corrected_region = _remove_artifacts_savgol(
                signal_data, [(start, end)], max(window_size, 7), smooth_edges
            )[start:end]
        
        # Apply the corrected region
        corrected_signal[start:end] = corrected_region
    
    return corrected_signal

def classify_artifacts(signal_data: np.ndarray, artifact_mask: np.ndarray, 
                      time_data: np.ndarray = None) -> Dict[str, List[int]]:
    """
    Classify detected artifacts into different categories based on their characteristics.
    
    Args:
        signal_data (np.ndarray): Input signal data
        artifact_mask (np.ndarray): Boolean mask where True indicates artifact
        time_data (np.ndarray, optional): Time data for calculating duration
    
    Returns:
        Dict[str, List[int]]: Dictionary mapping artifact types to lists of indices
    """
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    
    if artifact_mask is None or not np.any(artifact_mask):
        return {'total': [], 'spike': [], 'dropout': [], 'step': [], 'noise': []}
    
    # Initialize classification dictionary
    artifact_classes = {
        'total': [],
        'spike': [],    # Sharp, brief deviations
        'dropout': [],  # Signal loss/dips
        'step': [],     # Sudden baseline shifts
        'noise': []     # High-frequency noise bursts
    }
    
    # Get artifact indices
    artifact_indices = np.where(artifact_mask)[0]
    artifact_classes['total'] = artifact_indices.tolist()
    
    # Identify contiguous artifact regions
    artifact_regions = _identify_artifact_regions(artifact_mask)
    
    # Calculate first derivative
    derivative = np.diff(signal_data, prepend=signal_data[0])
    
    # Analyze each artifact region
    for start, end in artifact_regions:
        region_length = end - start
        
        # Skip very short regions (can't classify reliably)
        if region_length < 2:
            continue
        
        # Extract region data
        region_data = signal_data[start:end]
        region_derivative = derivative[start:end]
        
        # Calculate region characteristics
        region_min = np.min(region_data)
        region_max = np.max(region_data)
        region_range = region_max - region_min
        signal_range = np.percentile(signal_data, 95) - np.percentile(signal_data, 5)
        
        # Calculate context before and after (if available)
        pre_idx = max(0, start - region_length)
        post_idx = min(len(signal_data), end + region_length)
        
        pre_context = signal_data[pre_idx:start]
        post_context = signal_data[end:post_idx]
        
        pre_mean = np.mean(pre_context) if len(pre_context) > 0 else None
        post_mean = np.mean(post_context) if len(post_context) > 0 else None
        
        # CLASSIFICATION LOGIC
        
        # 1. Spikes: brief excursions with large amplitude change
        is_spike = (region_length <= 10 and 
                   region_range > 0.2 * signal_range and 
                   np.max(np.abs(region_derivative)) > 3 * np.std(derivative))
        
        # 2. Dropouts: signal falls significantly below baseline
        below_baseline = False
        if pre_mean is not None and post_mean is not None:
            baseline = (pre_mean + post_mean) / 2
            below_baseline = np.mean(region_data) < baseline - 0.15 * signal_range
        
        is_dropout = (below_baseline and 
                     region_length <= 30 and 
                     region_range < 0.5 * signal_range)
        
        # 3. Steps: sustained baseline shifts
        is_step = False
        if pre_mean is not None and post_mean is not None:
            step_size = abs(post_mean - pre_mean)
            is_step = (step_size > 0.15 * signal_range and 
                      region_length <= 15)
        
        # 4. Noise: high frequency fluctuations
        # Use frequency domain analysis if region is long enough
        is_noise = False
        if region_length >= 8:
            # Calculate high frequency energy
            if region_length >= 16:  # Minimum for FFT
                region_fft = np.fft.rfft(region_data - np.mean(region_data))
                high_freq_energy = np.sum(np.abs(region_fft[len(region_fft)//3:]))
                low_freq_energy = np.sum(np.abs(region_fft[:len(region_fft)//3]))
                is_noise = high_freq_energy > 2 * low_freq_energy
            else:
                # For shorter regions, use simple derivative variance
                is_noise = np.var(region_derivative) > 3 * np.var(derivative)
        
        # Assign the region to appropriate class(es)
        region_indices = np.arange(start, end).tolist()
        
        if is_spike:
            artifact_classes['spike'].extend(region_indices)
        elif is_dropout:
            artifact_classes['dropout'].extend(region_indices)
        elif is_step:
            artifact_classes['step'].extend(region_indices)
        elif is_noise:
            artifact_classes['noise'].extend(region_indices)
    
    return artifact_classes

def evaluate_signal_quality(signal_data: np.ndarray, artifact_mask: np.ndarray = None, 
                          sampling_rate: float = 1.0) -> Dict[str, float]:
    """
    Evaluate overall signal quality based on various metrics.
    
    Args:
        signal_data (np.ndarray): Input signal data
        artifact_mask (np.ndarray, optional): Boolean mask where True indicates artifact
        sampling_rate (float): Sampling rate in Hz
        
    Returns:
        Dict[str, float]: Dictionary of quality metrics
    """
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    
    # Initialize metrics dictionary
    quality_metrics = {}
    
    # 1. Basic statistics
    quality_metrics['mean'] = float(np.mean(signal_data))
    quality_metrics['std'] = float(np.std(signal_data))
    quality_metrics['min'] = float(np.min(signal_data))
    quality_metrics['max'] = float(np.max(signal_data))
    quality_metrics['range'] = float(quality_metrics['max'] - quality_metrics['min'])
    
    # 2. Robust statistics (less affected by outliers)
    quality_metrics['median'] = float(np.median(signal_data))
    p25 = np.percentile(signal_data, 25)
    p75 = np.percentile(signal_data, 75)
    quality_metrics['iqr'] = float(p75 - p25)  # Interquartile range
    
    # 3. Signal-to-noise ratio (SNR)
    # Estimate noise using median absolute deviation (robust)
    median = np.median(signal_data)
    mad = np.median(np.abs(signal_data - median))
    noise_level = 1.4826 * mad  # Convert MAD to std. dev. equivalent
    
    # Signal range (peak-to-peak)
    p05 = np.percentile(signal_data, 5)
    p95 = np.percentile(signal_data, 95)
    signal_range = p95 - p05
    
    # Calculate SNR
    quality_metrics['snr'] = float(signal_range / (noise_level + 1e-10))
    
    # 4. Artifact metrics (if artifact mask provided)
    if artifact_mask is not None:
        n_artifacts = np.sum(artifact_mask)
        artifact_percent = 100 * n_artifacts / len(signal_data)
        quality_metrics['artifact_count'] = int(n_artifacts)
        quality_metrics['artifact_percent'] = float(artifact_percent)
        
        # Calculate "clean" SNR (excluding artifacts)
        if n_artifacts < len(signal_data):
            clean_signal = signal_data[~artifact_mask]
            clean_median = np.median(clean_signal)
            clean_mad = np.median(np.abs(clean_signal - clean_median))
            clean_noise = 1.4826 * clean_mad
            
            clean_p05 = np.percentile(clean_signal, 5)
            clean_p95 = np.percentile(clean_signal, 95)
            clean_range = clean_p95 - clean_p05
            
            quality_metrics['clean_snr'] = float(clean_range / (clean_noise + 1e-10))
    
    # 5. Frequency domain metrics
    try:
        # Calculate power spectral density
        f, psd = signal.welch(signal_data, fs=sampling_rate, 
                             nperseg=min(256, len(signal_data)),
                             scaling='spectrum')
        
        # Total signal power
        total_power = np.sum(psd)
        
        # Power in different frequency bands (assuming sampling_rate is in Hz)
        if len(f) > 3:  # Need at least a few frequency bins
            f_idx = {}
            # Find indices for frequency bands
            # Define indices based on available frequency resolution
            f_max = f[-1]
            f_idx['low'] = np.where(f <= min(1.0, f_max/3))[0]
            f_idx['mid'] = np.where((f > min(1.0, f_max/3)) & (f <= min(5.0, 2*f_max/3)))[0]
            f_idx['high'] = np.where(f > min(5.0, 2*f_max/3))[0]
            
            # Calculate power in each band
            for band in ['low', 'mid', 'high']:
                if len(f_idx[band]) > 0:
                    band_power = np.sum(psd[f_idx[band]])
                    quality_metrics[f'{band}_power'] = float(band_power)
                    quality_metrics[f'{band}_percent'] = float(100 * band_power / (total_power + 1e-10))
            
            # Frequency flatness measure
            if len(psd) > 1 and np.all(psd > 0):
                geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
                arithmetic_mean = np.mean(psd)
                quality_metrics['spectral_flatness'] = float(geometric_mean / (arithmetic_mean + 1e-10))
    except:
        # Skip frequency metrics if calculation fails
        pass
    
    # 6. Stationarity assessment (trend and variance stability)
    # Split signal into chunks and compare means and variances
    if len(signal_data) > 20:
        n_chunks = min(5, len(signal_data) // 10)
        if n_chunks >= 2:
            chunk_size = len(signal_data) // n_chunks
            chunk_means = []
            chunk_vars = []
            
            for i in range(n_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(signal_data))
                chunk = signal_data[start:end]
                
                if artifact_mask is not None:
                    # Exclude artifacts from chunk statistics
                    chunk_mask = artifact_mask[start:end]
                    if np.any(~chunk_mask):
                        chunk = chunk[~chunk_mask]
                
                if len(chunk) > 0:
                    chunk_means.append(np.mean(chunk))
                    chunk_vars.append(np.var(chunk))
            
            if len(chunk_means) >= 2:
                # Coefficient of variation of means and variances
                mean_stability = np.std(chunk_means) / (np.mean(chunk_means) + 1e-10)
                var_stability = np.std(chunk_vars) / (np.mean(chunk_vars) + 1e-10)
                
                quality_metrics['mean_stability'] = float(1.0 / (1.0 + mean_stability))
                quality_metrics['variance_stability'] = float(1.0 / (1.0 + var_stability))
                
                # Overall stationarity score (1=perfectly stationary, 0=non-stationary)
                quality_metrics['stationarity'] = float(np.mean([
                    quality_metrics['mean_stability'],
                    quality_metrics['variance_stability']
                ]))
    
    # 7. Overall quality score (0-100)
    # Combine multiple metrics into a single quality score
    score_components = []
    
    # SNR contribution (0-40 points)
    snr = quality_metrics.get('clean_snr', quality_metrics['snr'])
    snr_score = min(40, max(0, 40 * (1 - np.exp(-snr/3))))
    score_components.append(snr_score)
    
    # Artifact penalty (0-30 points)
    artifact_score = 30
    if 'artifact_percent' in quality_metrics:
        artifact_percent = quality_metrics['artifact_percent']
        artifact_score = max(0, 30 * (1 - artifact_percent/100))
    score_components.append(artifact_score)
    
    # Stationarity contribution (0-30 points)
    stationarity_score = 15  # Default middle value
    if 'stationarity' in quality_metrics:
        stationarity_score = 30 * quality_metrics['stationarity']
    score_components.append(stationarity_score)
    
    # Calculate final quality score
    quality_metrics['quality_score'] = float(sum(score_components))
    
    return quality_metrics

def remove_artifacts_fast(signal_data: np.ndarray, artifact_mask: np.ndarray) -> np.ndarray:
    """
    Fast removal of artifacts using linear interpolation (simplified version).
    
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

def evaluate_control_channels(main_signal: np.ndarray, control_signals: List[np.ndarray]) -> Dict[str, float]:
    """
    Evaluate control channels to determine the best one for artifact correction.
    
    Args:
        main_signal (np.ndarray): Primary signal of interest
        control_signals (List[np.ndarray]): List of potential control signals
        
    Returns:
        Dict[str, float]: Quality scores for each control channel
    """
    if not isinstance(main_signal, np.ndarray):
        main_signal = np.array(main_signal)
    
    # Initialize results dictionary
    control_scores = {}
    
    # Check each control signal
    for i, control in enumerate(control_signals):
        if control is None or len(control) != len(main_signal):
            control_scores[f'control_{i}'] = 0.0
            continue
        
        # Calculate metrics
        metrics = {}
        
        # 1. Signal variance (higher is better for control)
        metrics['variance'] = np.var(control)
        
        # 2. Signal range (percentile-based to avoid outlier influence)
        p05 = np.percentile(control, 5)
        p95 = np.percentile(control, 95)
        metrics['range'] = p95 - p05
        
        # 3. Autocorrelation (temporal stability)
        if len(control) > 2:
            autocorr = np.corrcoef(control[:-1], control[1:])[0, 1]
            metrics['autocorrelation'] = abs(autocorr)
        else:
            metrics['autocorrelation'] = 0.0
        
        # 4. Correlation with main signal (potential artifact connection)
        corr = np.corrcoef(main_signal, control)[0, 1]
        metrics['correlation'] = abs(corr)
        
        # 5. High-frequency content (motion artifacts have high-frequency components)
        try:
            # Use Welch's method to estimate power spectrum
            f, psd = signal.welch(control, nperseg=min(256, len(control)))
            if len(f) > 3:
                # Ratio of high to low frequency power
                high_freq_idx = len(f) // 2
                high_freq_power = np.sum(psd[high_freq_idx:])
                low_freq_power = np.sum(psd[:high_freq_idx])
                metrics['high_freq_ratio'] = high_freq_power / (low_freq_power + 1e-10)
            else:
                metrics['high_freq_ratio'] = 0.0
        except:
            metrics['high_freq_ratio'] = 0.0
        
        # Calculate overall score (weighted sum of metrics)
        # Higher weight to variance, range and autocorrelation
        score = (0.4 * (metrics['variance'] / (np.var(main_signal) + 1e-10)) + 
                0.3 * (metrics['range'] / (np.percentile(main_signal, 95) - np.percentile(main_signal, 5) + 1e-10)) + 
                0.3 * metrics['autocorrelation'])
        
        # Normalize score to 0-100 range
        normalized_score = 100 * score / (1 + score)
        
        control_scores[f'control_{i}'] = float(normalized_score)
    
    return control_scores
