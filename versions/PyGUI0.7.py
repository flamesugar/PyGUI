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

        # Calculate the original duration based on sampling rate
        original_duration = len(analog_1) / sampling_rate  # Duration in seconds
        print(f"Original duration: {original_duration:.2f} seconds")

        # If the data is very large, downsample properly to prevent memory issues
        max_points = 500000  # Reasonable maximum points to process
        if len(analog_1) > max_points:
            print(f"Data is very large ({len(analog_1)} points), downsampling to prevent memory issues")

            # Calculate downsample factor
            downsample_factor = len(analog_1) // max_points + 1

            # Create proper time array that preserves the original duration
            num_points = len(analog_1) // downsample_factor
            time = np.linspace(0, original_duration, num_points)

            # Downsample signals using stride instead of resample to preserve signal characteristics
            analog_1 = analog_1[::downsample_factor]
            analog_2 = analog_2[::downsample_factor]
            digital_1 = digital_1[::downsample_factor]
            digital_2 = digital_2[::downsample_factor]

            print(f"Downsampled to {len(analog_1)} points. Time range preserved: 0 to {time[-1]:.2f} seconds")
        else:
            # Create time array using the original sampling rate
            time = np.arange(len(analog_1)) / sampling_rate  # Time in seconds

        # Apply volts per division scaling if available
        if len(volts_per_division) >= 2:
            analog_1 = analog_1 * volts_per_division[0]
            analog_2 = analog_2 * volts_per_division[1]

        print(f"Parsed data: {len(time)} samples, time range: 0 to {time[-1]:.2f} seconds")
        return time, analog_1, analog_2, digital_1, digital_2
    except Exception as e:
        print(f"Error parsing PPD data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def butter_filter(data, cutoff, fs, order=5):
    """Applies a Butterworth filter to the data."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


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
                 downsample_factor=1):
    """Processes the data by applying filters and downsampling."""
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

    # Apply downsampling if needed
    if downsample_factor > 1:
        processed_time, processed_signal = downsample_data(time, processed_signal, int(downsample_factor))

        # Also downsample digital signals if they exist
        if digital_1 is not None:
            _, processed_digital_1 = downsample_data(time, digital_1, int(downsample_factor))
        else:
            processed_digital_1 = None

        if digital_2 is not None:
            _, processed_digital_2 = downsample_data(time, digital_2, int(downsample_factor))
        else:
            processed_digital_2 = None
    else:
        processed_time = time
        processed_digital_1 = digital_1
        processed_digital_2 = digital_2

    # Return the processed data
    return processed_time, processed_signal, processed_digital_1, processed_digital_2


class PhotometryViewer:
    def __init__(self, root, file_path):
        self.root = root
        self.file_path = file_path

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

        # Initial parameters
        self.low_cutoff = 0.01
        self.high_cutoff = 1.0
        self.downsample_factor = 1

        # Process data with initial parameters
        self.processed_time, self.processed_signal, self.processed_digital_1, self.processed_digital_2 = process_data(
            self.time, self.analog_1, self.analog_2, self.digital_1, self.digital_2,
            self.low_cutoff, self.high_cutoff, self.downsample_factor
        )

        # Create the GUI
        self.create_gui()

    def create_gui(self):
        # Create a frame for the matplotlib figure
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Create matplotlib figure
        self.fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 2, 1])

        # Panel 1: Processed signal plot
        self.ax1 = self.fig.add_subplot(gs[0])
        self.line, = self.ax1.plot(self.processed_time / 60, self.processed_signal, 'g-', lw=1)
        self.ax1.set_ylabel('\u0394F/F')
        self.ax1.set_title('Processed Photometry Signal')
        self.ax1.grid(True, alpha=0.3)

        # Auto-scale y-axis to 5*STD
        self.update_y_limits()

        # Panel 2: Raw signal plot
        self.ax2 = self.fig.add_subplot(gs[1], sharex=self.ax1)
        self.raw_line, = self.ax2.plot(self.time / 60, self.raw_analog_1, 'b-', lw=1)
        self.ax2.set_ylabel('Raw Signal')
        self.ax2.set_title('Raw Analog 1 Signal')
        self.ax2.grid(True, alpha=0.3)

        # Panel 3: Combined Digital signals
        self.ax3 = self.fig.add_subplot(gs[2], sharex=self.ax1)
        self.digital_lines = []

        if self.processed_digital_1 is not None:
            line, = self.ax3.plot(self.processed_time / 60, self.processed_digital_1, 'b-', lw=1, label='Digital 1')
            self.digital_lines.append(line)

        if self.processed_digital_2 is not None:
            line, = self.ax3.plot(self.processed_time / 60, self.processed_digital_2, 'r-', lw=1, label='Digital 2')
            self.digital_lines.append(line)

        self.ax3.set_ylabel('Digital Signals')
        self.ax3.set_ylim(-0.1, 1.1)
        self.ax3.set_xlabel('Time (minutes)')
        if self.digital_lines:
            self.ax3.legend(loc='upper right')

        # Create canvas and toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame)
        self.toolbar.update()

        # Create slider frame with more space
        self.slider_frame = tk.Frame(self.root)
        self.slider_frame.pack(fill=tk.X, padx=20, pady=10)

        # Create sliders with larger size
        self.create_sliders()

        # Add zoom buttons (since Ctrl+mouse wheel is problematic)
        self.add_zoom_buttons()

        # Adjust layout
        self.fig.tight_layout()

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

        # Downsample slider - LARGER SIZE (replacing drift correction)
        tk.Label(self.slider_frame, text="Downsample factor:", font=('Arial', 12)).grid(
            row=2, column=0, sticky="w", padx=10, pady=10)
        self.downsample_slider = tk.Scale(self.slider_frame, from_=1, to=100,
                                          resolution=1, orient=tk.HORIZONTAL,
                                          length=800, width=25, font=('Arial', 12),
                                          command=self.update_filter)
        self.downsample_slider.set(self.downsample_factor)
        self.downsample_slider.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

    def add_zoom_buttons(self):
        # Create a frame for zoom buttons
        zoom_frame = tk.Frame(self.root)
        zoom_frame.pack(fill=tk.X, padx=20, pady=5)

        # X-axis zoom buttons
        tk.Label(zoom_frame, text="X-axis zoom:", font=('Arial', 12)).grid(
            row=0, column=0, sticky="w", padx=10, pady=5)

        tk.Button(zoom_frame, text="Zoom In (X)", font=('Arial', 12),
                  command=lambda: self.zoom_x(in_out="in")).grid(
            row=0, column=1, padx=5, pady=5)

        tk.Button(zoom_frame, text="Zoom Out (X)", font=('Arial', 12),
                  command=lambda: self.zoom_x(in_out="out")).grid(
            row=0, column=2, padx=5, pady=5)

        # Y-axis zoom buttons (for processed signal)
        tk.Label(zoom_frame, text="Y-axis zoom (processed):", font=('Arial', 12)).grid(
            row=1, column=0, sticky="w", padx=10, pady=5)

        tk.Button(zoom_frame, text="Zoom In (Y)", font=('Arial', 12),
                  command=lambda: self.zoom_y(self.ax1, "in")).grid(
            row=1, column=1, padx=5, pady=5)

        tk.Button(zoom_frame, text="Zoom Out (Y)", font=('Arial', 12),
                  command=lambda: self.zoom_y(self.ax1, "out")).grid(
            row=1, column=2, padx=5, pady=5)

        # Y-axis zoom buttons (for raw signal)
        tk.Label(zoom_frame, text="Y-axis zoom (raw):", font=('Arial', 12)).grid(
            row=2, column=0, sticky="w", padx=10, pady=5)

        tk.Button(zoom_frame, text="Zoom In (Y)", font=('Arial', 12),
                  command=lambda: self.zoom_y(self.ax2, "in")).grid(
            row=2, column=1, padx=5, pady=5)

        tk.Button(zoom_frame, text="Zoom Out (Y)", font=('Arial', 12),
                  command=lambda: self.zoom_y(self.ax2, "out")).grid(
            row=2, column=2, padx=5, pady=5)

        # Reset view button
        tk.Button(zoom_frame, text="Reset View", font=('Arial', 12, 'bold'),
                  command=self.reset_view, bg='lightblue').grid(
            row=0, column=3, rowspan=3, padx=20, pady=5, sticky="nsew")

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

        # Redraw
        self.canvas.draw_idle()

    def reset_view(self):
        """Reset the view to show all data"""
        # Reset x-axis to show all data
        self.ax1.set_xlim(self.processed_time[0] / 60, self.processed_time[-1] / 60)

        # Reset y-axis for processed signal (to 5*STD)
        self.update_y_limits()

        # Reset y-axis for raw signal
        margin = 0.1 * (self.raw_analog_1.max() - self.raw_analog_1.min())
        self.ax2.set_ylim(self.raw_analog_1.min() - margin, self.raw_analog_1.max() + margin)

        # Reset y-axis for digital signals
        self.ax3.set_ylim(-0.1, 1.1)

        # Redraw
        self.canvas.draw_idle()

    def update_filter(self, _=None):
        """Update filter parameters and reprocess data"""
        self.low_cutoff = self.low_slider.get()
        self.high_cutoff = self.high_slider.get()
        self.downsample_factor = int(self.downsample_slider.get())

        print(f"Processing with: low={self.low_cutoff}, high={self.high_cutoff}, downsample={self.downsample_factor}")

        # Reprocess with new parameters
        self.processed_time, self.processed_signal, self.processed_digital_1, self.processed_digital_2 = process_data(
            self.time, self.raw_analog_1, self.analog_2, self.digital_1, self.digital_2,
            low_cutoff=self.low_cutoff, high_cutoff=self.high_cutoff, downsample_factor=self.downsample_factor
        )

        # Update plots - need to update x and y data since downsampling changes time axis
        self.line.set_xdata(self.processed_time / 60)
        self.line.set_ydata(self.processed_signal)

        # Update digital signals if they exist
        if len(self.digital_lines) > 0 and self.processed_digital_1 is not None:
            self.digital_lines[0].set_xdata(self.processed_time / 60)
            self.digital_lines[0].set_ydata(self.processed_digital_1)
        if len(self.digital_lines) > 1 and self.processed_digital_2 is not None:
            self.digital_lines[1].set_xdata(self.processed_time / 60)
            self.digital_lines[1].set_ydata(self.processed_digital_2)

        # Update y-axis limits for processed signal
        self.update_y_limits()

        # Update x-axis limits to show full data range
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_xlim(self.processed_time[0] / 60, self.processed_time[-1] / 60)

        # Redraw
        self.canvas.draw_idle()

    def update_y_limits(self):
        """Update y-axis limits for processed signal based on 5*STD"""
        mean_val = np.mean(self.processed_signal)
        std_val = np.std(self.processed_signal)
        self.ax1.set_ylim(mean_val - 5 * std_val, mean_val + 5 * std_val)


def main():
    """Main function to start the application."""
    try:
        # Create the root window
        root = tk.Tk()
        root.title("Photometry Signal Viewer")
        root.geometry("1400x1000")  # Larger window size

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