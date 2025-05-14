import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from data_processing import process_data

def create_blanking_panel(self, parent):
    """Create a panel with manual blanking controls"""
    # Create main frame for manual blanking
    self.manual_blanking_frame = ttk.LabelFrame(parent, text="Manual Signal Blanking")
    self.manual_blanking_frame.pack(fill=tk.X, padx=10, pady=5)
    
    # Description label
    description = ttk.Label(
        self.manual_blanking_frame, 
        text="Use this tool to smooth out artifacts by selecting regions on the signal.",
        wraplength=400
    )
    description.pack(padx=10, pady=5)
    
    # Button to enable/disable blanking mode
    self.manual_blank_btn = tk.Button(
        self.manual_blanking_frame,
        text="Enable Selection Mode",
        font=('Arial', self.font_sizes['button'] if hasattr(self, 'font_sizes') else 10),
        bg='lightblue',
        command=self.toggle_blanking_mode
    )
    self.manual_blank_btn.pack(fill=tk.X, padx=10, pady=5)
    
    # Frame for apply and clear buttons (initially disabled)
    button_frame = tk.Frame(self.manual_blanking_frame)
    button_frame.pack(fill=tk.X, padx=10, pady=5)
    
    # Apply blanking button
    self.apply_blanking_btn = tk.Button(
        button_frame,
        text="Apply Blanking",
        font=('Arial', self.font_sizes['button'] if hasattr(self, 'font_sizes') else 10),
        state=tk.DISABLED,
        command=self.apply_blanking
    )
    self.apply_blanking_btn.pack(side=tk.LEFT, padx=5)
    
    # Clear selection button
    self.clear_selection_btn = tk.Button(
        button_frame,
        text="Clear Selection",
        font=('Arial', self.font_sizes['button'] if hasattr(self, 'font_sizes') else 10),
        state=tk.DISABLED,
        command=self.clear_blanking_selection
    )
    self.clear_selection_btn.pack(side=tk.LEFT, padx=5)
    
    # Add cancel all blanking button
    cancel_frame = tk.Frame(self.manual_blanking_frame)
    cancel_frame.pack(fill=tk.X, padx=10, pady=5)
    
    self.cancel_blanking_btn = tk.Button(
        cancel_frame,
        text="Remove All Blanking",
        font=('Arial', self.font_sizes['button'] if hasattr(self, 'font_sizes') else 10),
        bg='lightcoral',
        command=self.cancel_blanking
    )
    self.cancel_blanking_btn.pack(side=tk.LEFT, padx=5)
    
    # Add description for cancel button
    ttk.Label(
        cancel_frame,
        text="Removes all blanking and restores original signal",
        font=('Arial', self.font_sizes['base'] - 1 if hasattr(self, 'font_sizes') else 9),
        foreground='gray'
    ).pack(side=tk.LEFT, padx=5)
    
    return self.manual_blanking_frame

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
        edge_protection=self.edge_protection_var.get() if hasattr(self, 'edge_protection_var') else True
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
            edge_protection=self.edge_protection_var.get() if hasattr(self, 'edge_protection_var') else True
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
