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
        self.drift_correction = 5.0

        # Process data with initial parameters
        self.processed_time, self.processed_signal, self.processed_digital_1, self.processed_digital_2 = process_data(
            self.time, self.analog_1, self.analog_2, self.digital_1, self.digital_2,
            self.low_cutoff, self.high_cutoff, self.drift_correction
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

        # Create slider frame
        self.slider_frame = tk.Frame(self.root)
        self.slider_frame.pack(fill=tk.X, padx=10, pady=5)

        # Create sliders
        self.create_sliders()

        # Connect mouse wheel events for zoom
        self.connect_zoom_events()

        # Adjust layout
        self.fig.tight_layout()

    def create_sliders(self):
        # Low cutoff slider
        tk.Label(self.slider_frame, text="Low cutoff (Hz):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.low_slider = tk.Scale(self.slider_frame, from_=0.001, to=0.5,
                                   resolution=0.001, orient=tk.HORIZONTAL, length=300,
                                   command=self.update_filter)
        self.low_slider.set(self.low_cutoff)
        self.low_slider.grid(row=0, column=1, padx=5, pady=5)

        # High cutoff slider
        tk.Label(self.slider_frame, text="High cutoff (Hz):").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.high_slider = tk.Scale(self.slider_frame, from_=0.1, to=5.0,
                                    resolution=0.1, orient=tk.HORIZONTAL, length=300,
                                    command=self.update_filter)
        self.high_slider.set(self.high_cutoff)
        self.high_slider.grid(row=1, column=1, padx=5, pady=5)

        # Drift correction slider
        tk.Label(self.slider_frame, text="Drift correction:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.drift_slider = tk.Scale(self.slider_frame, from_=0.0, to=20.0,
                                     resolution=0.1, orient=tk.HORIZONTAL, length=300,
                                     command=self.update_filter)
        self.drift_slider.set(self.drift_correction)
        self.drift_slider.grid(row=2, column=1, padx=5, pady=5)

    def update_filter(self, _=None):
        """Update filter parameters and reprocess data"""
        self.low_cutoff = self.low_slider.get()
        self.high_cutoff = self.high_slider.get()
        self.drift_correction = self.drift_slider.get()

        print(f"Processing with: low={self.low_cutoff}, high={self.high_cutoff}, drift={self.drift_correction}")

        # Reprocess with new parameters
        self.processed_time, self.processed_signal, self.processed_digital_1, self.processed_digital_2 = process_data(
            self.time, self.raw_analog_1, self.analog_2, self.digital_1, self.digital_2,
            low_cutoff=self.low_cutoff, high_cutoff=self.high_cutoff, drift_correction=self.drift_correction
        )

        # Update plots
        self.line.set_ydata(self.processed_signal)

        # Update digital signals if they exist
        if len(self.digital_lines) > 0 and self.processed_digital_1 is not None:
            self.digital_lines[0].set_ydata(self.processed_digital_1)
        if len(self.digital_lines) > 1 and self.processed_digital_2 is not None:
            self.digital_lines[1].set_ydata(self.processed_digital_2)

        # Update y-axis limits for processed signal
        self.update_y_limits()

        # Redraw
        self.canvas.draw_idle()

    def update_y_limits(self):
        """Update y-axis limits for processed signal based on 5*STD"""
        mean_val = np.mean(self.processed_signal)
        std_val = np.std(self.processed_signal)
        self.ax1.set_ylim(mean_val - 5 * std_val, mean_val + 5 * std_val)

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
        - Normal scroll: zoom y-axis
        - Ctrl+scroll: zoom x-axis
        """
        # Check if the event occurred within an axis
        if event.inaxes is None:
            return

        # Get the current axis limits
        if self.ctrl_pressed:
            # Ctrl+scroll: zoom x-axis
            base_scale = 1.1  # Zoom factor

            # Get current x limits
            xlim = event.inaxes.get_xlim()
            xdata = event.xdata

            # Calculate new limits
            if event.button == 'up':  # Zoom in
                new_xlim = [xdata - (xdata - xlim[0]) / base_scale,
                            xdata + (xlim[1] - xdata) / base_scale]
            else:  # Zoom out
                new_xlim = [xdata - (xdata - xlim[0]) * base_scale,
                            xdata + (xlim[1] - xdata) * base_scale]

            # Apply new limits to all x-linked axes
            self.ax1.set_xlim(new_xlim)
            # Other axes are linked via sharex
        else:
            # Normal scroll: zoom y-axis (only for the axis being scrolled)
            base_scale = 1.1  # Zoom factor

            # Get current y limits of the axis being scrolled
            ylim = event.inaxes.get_ylim()
            ydata = event.ydata

            # Calculate new limits
            if event.button == 'up':  # Zoom in
                new_ylim = [ydata - (ydata - ylim[0]) / base_scale,
                            ydata + (ylim[1] - ydata) / base_scale]
            else:  # Zoom out
                new_ylim = [ydata - (ydata - ylim[0]) * base_scale,
                            ydata + (ylim[1] - ydata) * base_scale]

            # Apply new limits only to the axis being scrolled
            event.inaxes.set_ylim(new_ylim)

        # Redraw the figure
        self.canvas.draw_idle()


def main():
    """Main function to start the application."""
    try:
        # Create the root window
        root = tk.Tk()
        root.title("Photometry Signal Viewer")
        root.geometry("1200x800")

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