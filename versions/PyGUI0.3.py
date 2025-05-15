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


def read_ppd_file(file_path):
    """
    Reads a .ppd file and returns the header and data.
    """
    with open(file_path, 'rb') as f:
        header_size = int.from_bytes(f.read(2), 'little')
        header_bytes = f.read(header_size)
        header_str = header_bytes.decode('utf-8')
        header = json.loads(header_str)
        data_bytes = f.read()

    return header, data_bytes


def parse_ppd_data(header, data_bytes):
    """
    Parses the data bytes from a .ppd file and returns the analog and digital signals.
    """
    sampling_rate = header['sampling_rate']
    volts_per_division = header['volts_per_division']

    data = np.frombuffer(data_bytes, dtype=np.dtype('<u2'))
    analog = data >> 1
    digital = data & 1

    analog_1 = analog[0::2]
    analog_2 = analog[1::2]
    digital_1 = digital[0::2]
    digital_2 = digital[1::2]

    analog_1 = analog_1 * volts_per_division[0]
    analog_2 = analog_2 * volts_per_division[1]

    time = np.arange(0, len(analog_1) / sampling_rate, 1 / sampling_rate)

    return time, analog_1, analog_2, digital_1, digital_2


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
                 drift_correction=10):
    """
    Processes the data by applying filters, correcting baseline drift, and calculating dF/F.

    Parameters
    -----------
    low_cutoff : float
        Low cutoff frequency for bandpass filter (Hz)
    high_cutoff : float
        High cutoff frequency for bandpass filter (Hz)
    drift_correction : float
        Length of the running-average filter to remove drift, in seconds
    """
    # Compute sampling rate
    dt = np.diff(time).mean()
    fs = 1 / dt

    # Store original digital signals - do not process them
    original_digital_1 = digital_1.copy() if digital_1 is not None else None
    original_digital_2 = digital_2.copy() if digital_2 is not None else None

    # Make a copy of the signal for processing
    processed_signal = analog_1.copy()
    original_signal = analog_1.copy()  # Keep truly original for reference

    # Normalize cutoff frequencies
    nyquist = fs / 2

    # First apply 405nm control correction if available
    if analog_2 is not None and len(analog_2) == len(processed_signal):
        processed_signal = correct_405(processed_signal, analog_2)

    # STEP 1: REFLECTION PADDING TO REDUCE EDGE EFFECTS
    # Create extended signal with reflections at the edges
    signal_length = len(processed_signal)
    reflection_length = min(int(signal_length * 0.1), int(5 * fs))
    left_reflection = processed_signal[1:reflection_length + 1][::-1]
    right_reflection = processed_signal[-reflection_length - 1:-1][::-1]
    extended_signal = np.concatenate((left_reflection, processed_signal, right_reflection))

    # STEP 2: BASIC SIGNAL CLEANING
    # Remove DC offset using median (more robust than mean)
    extended_signal = extended_signal - np.median(extended_signal)

    # STEP 3: REMOVE NOISE SPIKES
    # Use a more robust method based on median absolute deviation
    signal_median = np.median(extended_signal)
    signal_mad = np.median(np.abs(extended_signal - signal_median))
    threshold = 8 * signal_mad  # Conservative threshold (robust version of ~5 sigma)
    mask = np.abs(extended_signal - signal_median) > threshold
    if np.any(mask):
        # Replace outliers with local median values
        for i in np.where(mask)[0]:
            window = 10
            start = max(0, i - window)
            end = min(len(extended_signal), i + window + 1)
            neighbors = extended_signal[start:end].copy()
            # Remove outliers from the neighborhood
            valid_neighbors = neighbors[np.abs(neighbors - signal_median) <= threshold]
            if len(valid_neighbors) > 0:
                extended_signal[i] = np.median(valid_neighbors)
            else:
                extended_signal[i] = signal_median

    # STEP 4: APPLY FREQUENCY FILTERS
    # First low-pass filter if requested
    if high_cutoff < nyquist * 0.9:  # Stay well below Nyquist
        norm_high_cutoff = high_cutoff / nyquist
        try:
            # Use a gentle 2nd order Butterworth
            b_low, a_low = butter(2, norm_high_cutoff, 'low')
            extended_signal = filtfilt(b_low, a_low, extended_signal)
        except Exception as e:
            print(f"Low-pass filter error: {e}")

    # Apply high-pass if requested, but with very low order
    if low_cutoff > 0 and low_cutoff < nyquist * 0.4:  # Reasonable range
        norm_low_cutoff = low_cutoff / nyquist
        try:
            # Use just 1st order to minimize distortion
            b_high, a_high = butter(1, norm_low_cutoff, 'high')
            extended_signal = filtfilt(b_high, a_high, extended_signal)
        except Exception as e:
            print(f"High-pass filter error: {e}")

    # STEP 5: EXTRACT THE PROCESSED SIGNAL FROM EXTENDED SIGNAL
    processed_signal = extended_signal[reflection_length:reflection_length + signal_length]

    # STEP 6: DRIFT CORRECTION - NEW METHOD
    # Using Asymmetric Least Squares smoothing for drift correction
    # This method is better for preserving signal peaks while removing baseline drift
    if drift_correction > 0 and len(processed_signal) > 20:
        # Scale the strength based on the drift_correction parameter, with smooth transition
        lambda_value = max(1, min(1e7, 10 ** (drift_correction / 2)))  # Exponential scaling
        p_value = 0.01  # Asymmetry parameter - strongly penalizes values above the baseline

        try:
            # Always use polynomial fit as baseline first to ensure consistency
            x = np.linspace(0, 1, len(processed_signal))

            # Scale polynomial degree smoothly based on drift correction strength
            # This prevents sudden changes in baseline estimation
            if drift_correction < 4.5:
                poly_degree = 3  # Low drift correction = higher polynomial (more flexible)
            elif drift_correction < 5.0:
                # Gradual transition between 4.5 and 5.0
                # Map 4.5->3 and 5.0->1 with a smooth transition
                t = (drift_correction - 4.5) / 0.5  # 0 to 1 as drift_correction goes from 4.5 to 5.0
                poly_degree = max(1, int(3 - 2 * t))  # Smoothly transition from 3 to 1
            else:
                poly_degree = 1  # High drift correction = lower polynomial (more rigid)

            # Apply polynomial baseline removal
            coeffs = np.polyfit(x, processed_signal, poly_degree)
            polynomial_baseline = np.polyval(coeffs, x)
            processed_signal = processed_signal - polynomial_baseline

            # Then apply ALS with strength proportional to drift_correction
            # Only use ALS for stronger correction needs, with smooth transition
            if drift_correction > 3.0:
                # Scale ALS lambda smoothly to prevent discontinuities
                als_factor = min(1.0, (drift_correction - 3.0) / 2.0)  # Gradual increase from 0 to 1
                scaled_lambda = lambda_value * als_factor

                if scaled_lambda > 10:
                    baseline = als_baseline(processed_signal, lam=scaled_lambda, p=p_value, niter=10)
                    processed_signal = processed_signal - baseline * als_factor
        except Exception as e:
            print(f"Drift correction error: {e}")
            # Already applied polynomial correction above, so no need for additional fallback
    # STEP 7: CALCULATE dF/F
    # Use the 10th percentile as a robust baseline estimate
    F0 = np.percentile(original_signal, 10)  # Using original signal for stability

    # Safety check for division by near-zero
    if abs(F0) < 1e-6:
        F0 = np.mean(np.abs(original_signal))
        if abs(F0) < 1e-6:
            F0 = 1.0  # Safe fallback

    # Calculate dF/F as percentage
    df_f = processed_signal / F0 * 100

    # STEP 8: FINAL CLEANUP - REMOVE EXTREME VALUES
    df_f_median = np.median(df_f)
    df_f_mad = np.median(np.abs(df_f - df_f_median))
    # Use 4 * MAD (approximately 5-6 sigma in normal distributions)
    clip_threshold = 4 * df_f_mad
    df_f = np.clip(df_f, df_f_median - clip_threshold, df_f_median + clip_threshold)

    return time, df_f, original_digital_1, original_digital_2


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


def plot_data(time, signal, raw_analog_1, digital_1=None, digital_2=None, filter_params=None, processor_func=None):
    """
    Plots the signal with zooming and panning.

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
        Function to process data with new filter parameters
    """
    # Create a figure with subplots using GridSpec for more control
    fig = plt.figure(figsize=(10, 6))
    gs = plt.GridSpec(3, 1, height_ratios=[3, 1, 2])

    # Main photometry signal plot
    ax1 = plt.subplot(gs[0, :])
    line, = ax1.plot(time / 60, signal, color='green', linewidth=1)

    # Calculate autoscale based on mean signal standard deviation (5 times)
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    ax1.set_ylim(signal_mean - 5 * signal_std, signal_mean + 5 * signal_std)

    # Add labels and title
    ax1.set_ylabel('dF/F (%)')
    ax1.set_title('Fiber Photometry Data')

    # Hide x-axis labels for top plot (will share with bottom plot)
    ax1.set_xticklabels([])

    # Digital signal plot (if available)
    ax2 = plt.subplot(gs[1, :], sharex=ax1)
    if digital_1 is not None:
        # Make TTL signal more visible with thicker line
        digital_line_1, = ax2.plot(time / 60, digital_1, color='red', linewidth=2)
        # Ensure y-axis is correctly scaled for digital signals
        ax2.set_ylim(-0.1, 1.1)
    ax2.set_ylabel('Stimulus')

    # Add TTL2 if available
    if digital_2 is not None:
        digital_line_2, = ax2.plot(time / 60, digital_2, color='blue', linewidth=2, alpha=0.7)

    # Raw signal plot
    ax3 = plt.subplot(gs[2, :], sharex=ax1)
    ax3.plot(time / 60, raw_analog_1, color='gray', linewidth=1)
    ax3.set_xlabel('Time (min)')
    ax3.set_ylabel('Raw Signal')

    # Initial autoscale for raw signal
    raw_min = np.percentile(raw_analog_1, 1)  # 1st percentile
    raw_max = np.percentile(raw_analog_1, 99)  # 99th percentile
    raw_range = raw_max - raw_min
    if raw_range < 1e-6:  # If range is too small
        raw_range = 1.0

    raw_padding = raw_range * 0.2  # 20% padding
    ax3.set_ylim(raw_min - raw_padding, raw_max + raw_padding)

    # Adjust spacing between subplots for better layout
    plt.subplots_adjust(hspace=0.5)

    # Connect zoom/pan events across subplots - fixed method name
    ax2.sharex = ax1
    ax3.sharex = ax1

    # Initialize matplotlib interactive mode for real-time updates
    plt.ion()

    # Add sliders for filter parameters if provided
    if filter_params is not None and processor_func is not None:
        # Create axes for sliders
        plt.subplots_adjust(bottom=0.25)  # Make room for sliders

        # Low cutoff frequency slider (bandpass lower bound)
        low_slider_ax = plt.axes([0.25, 0.05, 0.65, 0.03])
        low_slider = Slider(
            ax=low_slider_ax,
            label='Low Cutoff (Hz)',
            valmin=0.0001,
            valmax=0.1,
            valinit=filter_params.get('low_cutoff', 0.01) if filter_params else 0.01,
            valfmt='%1.5f',
        )

        # High cutoff frequency slider (bandpass higher bound)
        high_slider_ax = plt.axes([0.25, 0.10, 0.65, 0.03])
        high_slider = Slider(
            ax=high_slider_ax,
            label='High Cutoff (Hz)',
            valmin=0.2,
            valmax=2.0,
            valinit=filter_params.get('high_cutoff', 1.0) if filter_params else 1.0,
            valfmt='%1.2f',
        )

        # Drift correction strength slider
        drift_slider_ax = plt.axes([0.25, 0.15, 0.65, 0.03])
        drift_slider = Slider(
            ax=drift_slider_ax,
            label='Drift Correction',
            valmin=0.0,
            valmax=10.0,
            valinit=filter_params.get('drift_correction', 10.0) if filter_params else 10.0,
            valfmt='%1.1f',
        )

        # Function to update plot with new filter settings - optimized for real-time updates
        def update_plot(val=None):
            if processor_func is not None:
                # Get current slider values
                low_cutoff = low_slider.val
                high_cutoff = high_slider.val
                drift_correction = drift_slider.val

                # Save current view limits
                current_xlim = ax1.get_xlim()

                # Process data with new filter settings
                new_time, new_signal, new_digital_1, new_digital_2 = processor_func(
                    time, raw_analog_1, None, digital_1, digital_2,
                    low_cutoff=low_cutoff,
                    high_cutoff=high_cutoff,
                    drift_correction=drift_correction
                )

                # Update data
                line.set_ydata(new_signal)

                # Auto-scale y-axis to fit the processed signal
                signal_min = np.percentile(new_signal, 1)  # 1st percentile to ignore outliers
                signal_max = np.percentile(new_signal, 99)  # 99th percentile to ignore outliers

                # Add padding to ensure visibility
                y_range = signal_max - signal_min
                if y_range < 1e-6:  # If range is too small
                    y_range = 1.0  # Set default range

                padding = y_range * 0.2  # Add 20% padding

                # Set new y-limits for main plot
                ax1.set_ylim(signal_min - padding, signal_max + padding)

                # Auto-scale raw signal plot as well
                if raw_analog_1 is not None:
                    raw_min = np.percentile(raw_analog_1, 1)  # 1st percentile
                    raw_max = np.percentile(raw_analog_1, 99)  # 99th percentile
                    raw_range = raw_max - raw_min
                    if raw_range < 1e-6:  # If range is too small
                        raw_range = 1.0

                    raw_padding = raw_range * 0.2  # 20% padding
                    ax3.set_ylim(raw_min - raw_padding, raw_max + raw_padding)

                # Restore x limits
                ax1.set_xlim(current_xlim)

                # Update digital lines if present
                if new_digital_1 is not None and digital_1 is not None:
                    digital_line_1.set_ydata(new_digital_1)
                if new_digital_2 is not None and digital_2 is not None:
                    digital_line_2.set_ydata(new_digital_2)

                # Force immediate redraw for real-time updates
                fig.canvas.draw()
                # Flush events to ensure UI responsiveness
                plt.pause(0.001)  # Short pause to allow the GUI to process events

        # This ensures that we update immediately whenever a slider changes
        low_slider.on_changed(update_plot)
        high_slider.on_changed(update_plot)
        drift_slider.on_changed(update_plot)

        # Initial call to set up the sliders and plots
        update_plot(None)

        # Store the filter parameter sliders in the filter_params dictionary
        filter_params['low_cutoff_slider'] = low_slider
        filter_params['high_cutoff_slider'] = high_slider
        filter_params['drift_correction_slider'] = drift_slider

    # Add zooming and panning capabilities
    plt.gcf().canvas.mpl_connect('scroll_event', lambda event: zoom_factory(event, base_scale=1.2))

    # Zoom function that maintains the position under the cursor
    def zoom_factory(event, base_scale=1.1):
        if event.inaxes != ax1 and event.inaxes != ax3:
            return

        # Get the current x and y limits
        ax = event.inaxes
        y_lim = ax.get_ylim()

        # Calculate scale factor based on scroll direction
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale

        if event.inaxes == ax1:  # Main plot zooming
            # Calculate new limits keeping mouse position fixed (y-axis only)
            y_bottom = event.ydata - (event.ydata - y_lim[0]) * scale_factor
            y_top = event.ydata + (y_lim[1] - event.ydata) * scale_factor

            # Set new y-limits for main plot
            ax1.set_ylim(y_bottom, y_top)

            # Scale y-axis for raw signal plot as well
            # Get current raw signal plot y limits
            y_lim_raw = ax3.get_ylim()
            # Apply similar scaling to keep proportions
            range_factor = (y_top - y_bottom) / (y_lim[1] - y_lim[0])
            raw_range = y_lim_raw[1] - y_lim_raw[0]
            raw_mid = (y_lim_raw[0] + y_lim_raw[1]) / 2
            new_raw_range = raw_range * range_factor
            ax3.set_ylim(raw_mid - new_raw_range / 2, raw_mid + new_raw_range / 2)

        fig.canvas.draw_idle()

    # Ensure figure stays open until user closes it
    plt.ioff()  # Turn off interactive mode to wait for user to close
    plt.show(block=True)

    return fig


def open_file_dialog():
    """
    Opens a file dialog and returns the selected file path.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(filetypes=[("PPD Files", "*.ppd")])
    return file_path


def main(low_cutoff=0.005, high_cutoff=5, drift_correction=10):
    print("Starting main function")
    file_path = open_file_dialog()
    if file_path:
        print(f"Selected file: {file_path}")
        try:
            # Read the file
            header, data = read_ppd_file(file_path)
            print("File read successful")

            # Parse the data
            time, analog_1, analog_2, digital_1, digital_2 = parse_ppd_data(header, data)
            print(f"Data parsed: {len(time)} samples")

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

            print("Initial processing starting")
            # Initial processing with default parameters
            time, analog_1, digital_1, digital_2 = process_data(time, analog_1, analog_2, digital_1, digital_2,
                                                                low_cutoff, high_cutoff, drift_correction)
            print("Initial processing complete")

            # Define a function to reprocess data with new parameters
            def reprocess_with_params(time_arg, analog_1_arg, analog_2_arg, digital_1_arg, digital_2_arg,
                                      low_cutoff=0.01, high_cutoff=1.0, drift_correction=10.0):
                print(f"Reprocessing with params: low={low_cutoff}, high={high_cutoff}, drift={drift_correction}")
                # Reprocess with new parameters but keep the original time and raw data
                return process_data(time_arg, analog_1_arg, analog_2_arg, digital_1_arg, digital_2_arg,
                                    low_cutoff=low_cutoff,
                                    high_cutoff=high_cutoff,
                                    drift_correction=drift_correction)

            # Create filter parameters dictionary
            filter_params = {
                'low_cutoff': low_cutoff,
                'high_cutoff': high_cutoff,
                'drift_correction': drift_correction
            }

            print("Starting plot_data function")
            # Plot with processor function for in-place updating
            plot_data(time, analog_1, raw_analog_1, digital_1, digital_2,
                      filter_params=filter_params, processor_func=reprocess_with_params)
            print("plot_data function completed")
        except Exception as e:
            print(f"Error processing file: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No file selected.")


if __name__ == "__main__":
    main()
