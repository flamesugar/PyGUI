import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from data_processing import detect_artifacts

def create_artifact_panel(self, parent):
    """Create a panel with artifact detection controls"""
    # Create main frame
    artifact_frame = ttk.LabelFrame(parent, text="Artifact Detection")
    artifact_frame.pack(fill=tk.X, padx=10, pady=5)
    
    # Description label
    description = ttk.Label(
        artifact_frame, 
        text="Detect and highlight artifacts in the signal based on threshold settings.",
        wraplength=400
    )
    description.pack(fill=tk.X, padx=10, pady=5)
    
    # Create buttons row
    buttons_frame = ttk.Frame(artifact_frame)
    buttons_frame.pack(fill=tk.X, padx=10, pady=5)
    
    # Button to open artifact settings
    settings_btn = ttk.Button(
        buttons_frame, 
        text="Artifact Settings...",
        command=self.open_artifact_window
    )
    settings_btn.pack(side=tk.LEFT, padx=5)
    
    # Button to highlight artifacts
    highlight_btn = ttk.Button(
        buttons_frame, 
        text="Highlight Artifacts",
        command=self.highlight_artifacts
    )
    highlight_btn.pack(side=tk.LEFT, padx=5)
    
    # Checkbox for showing no-control signal
    control_frame = ttk.Frame(artifact_frame)
    control_frame.pack(fill=tk.X, padx=10, pady=5)
    
    self.show_no_control_var = tk.BooleanVar(value=True)
    show_no_control_check = ttk.Checkbutton(
        control_frame,
        text="Show signal without isosbestic control",
        variable=self.show_no_control_var,
        command=self.toggle_no_control_line
    )
    show_no_control_check.pack(side=tk.LEFT, padx=5)
    
    # Current threshold display
    threshold_frame = ttk.Frame(artifact_frame)
    threshold_frame.pack(fill=tk.X, padx=10, pady=5)
    
    ttk.Label(
        threshold_frame, 
        text="Current artifact threshold:"
    ).pack(side=tk.LEFT, padx=5)
    
    self.threshold_label = ttk.Label(
        threshold_frame,
        text=f"{self.artifact_threshold:.1f}"
    )
    self.threshold_label.pack(side=tk.LEFT, padx=5)
    
    return artifact_frame

def open_artifact_window(self):
    """Open artifact detection settings in a separate window"""
    artifact_window = tk.Toplevel(self.root)
    artifact_window.title("Artifact Detection")
    artifact_window.geometry("700x500")
    artifact_window.transient(self.root)

    # Create main frame
    artifact_frame = ttk.Frame(artifact_window, padding=20)
    artifact_frame.pack(fill=tk.BOTH, expand=True)

    font_size = self.font_sizes['base']
    slider_font_size = self.font_sizes['slider']

    # Artifact threshold slider
    tk.Label(
        artifact_frame,
        text="Artifact threshold:",
        font=('Arial', font_size)).grid(row=0, column=0, sticky="w", padx=10, pady=10)

    self.artifact_slider = tk.Scale(
        artifact_frame,
        from_=1.0, to=10.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        length=400,
        width=25,
        font=('Arial', slider_font_size),
        command=self.update_filter)
    self.artifact_slider.set(self.artifact_threshold)
    self.artifact_slider.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

    # Add description for artifact threshold
    artifact_desc = tk.Label(
        artifact_frame,
        text="Lower values detect more artifacts. Higher values are less sensitive.",
        font=('Arial', font_size - 1),
        fg='gray')
    artifact_desc.grid(row=1, column=0, columnspan=2, sticky="w", padx=10)

    # Button to show artifacts
    highlight_btn = tk.Button(
        artifact_frame,
        text="Highlight Artifacts",
        font=('Arial', font_size),
        bg="lightblue",
        command=self.highlight_artifacts)
    highlight_btn.grid(row=2, column=0, columnspan=2, pady=10)

    # Shared isosbestic control option
    shared_control_frame = tk.Frame(artifact_frame)
    shared_control_frame.grid(row=3, column=0, columnspan=2, sticky="w", padx=10, pady=10)

    self.shared_isosbestic_var = tk.BooleanVar(value=True)
    self.shared_isosbestic_check = tk.Checkbutton(
        shared_control_frame,
        text="Use best isosbestic channel for both files",
        variable=self.shared_isosbestic_var,
        font=('Arial', font_size),
        command=self.update_filter)
    self.shared_isosbestic_check.pack(side=tk.LEFT, padx=5)

    # Add indicator for which control is being used
    self.control_indicator = tk.Label(
        shared_control_frame,
        text="",
        font=('Arial', font_size - 1),
        fg="blue")
    self.control_indicator.pack(side=tk.LEFT, padx=10)

    # No-control signal display options
    tk.Label(
        artifact_frame,
        text="No-Control Signal Display Options:",
        font=('Arial', font_size, 'bold')).grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=5)

    self.show_no_control_var = tk.BooleanVar(value=True)
    self.show_no_control_check = tk.Checkbutton(
        artifact_frame,
        text="Show signal without isosbestic control",
        variable=self.show_no_control_var,
        font=('Arial', font_size),
        command=self.toggle_no_control_line)
    self.show_no_control_check.grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=5)

    # Offset slider for no-control signal
    offset_frame = tk.Frame(artifact_frame)
    offset_frame.grid(row=6, column=0, columnspan=2, sticky="w", padx=10, pady=5)

    tk.Label(
        offset_frame,
        text="No-Control Vertical Offset:",
        font=('Arial', font_size)).pack(side=tk.LEFT, padx=5)

    self.nocontrol_offset_var = tk.DoubleVar(value=0.0)
    self.nocontrol_offset_slider = tk.Scale(
        offset_frame,
        from_=-50.0, to=50.0,
        resolution=1.0,
        orient=tk.HORIZONTAL,
        length=300,
        width=15,
        font=('Arial', slider_font_size - 1),
        command=self.adjust_nocontrol_offset)
    self.nocontrol_offset_slider.pack(side=tk.LEFT, padx=5)

    # Button frame
    button_frame = tk.Frame(artifact_frame)
    button_frame.grid(row=7, column=0, columnspan=2, pady=20)

    # Apply button
    apply_button = tk.Button(
        button_frame,
        text="Apply Changes",
        font=('Arial', font_size),
        bg="lightgreen",
        command=self.update_filter)
    apply_button.pack(side=tk.LEFT, padx=10)

    # Close button
    close_button = tk.Button(
        button_frame,
        text="Close",
        font=('Arial', font_size),
        command=artifact_window.destroy)
    close_button.pack(side=tk.LEFT, padx=10)

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

def update_artifact_markers(self):
    """Update the artifact markers on both plots with null checks"""
    if self.artifact_mask is None or not np.any(self.artifact_mask):
        # No artifacts detected, clear the markers
        self.artifact_markers_processed.set_data([], [])
        if hasattr(self, 'artifact_markers_raw') and self.artifact_markers_raw is not None:
            self.artifact_markers_raw.set_data([], [])
        return

    # Get the time points where artifacts occur (in minutes)
    artifact_times = self.processed_time[self.artifact_mask] / 60

    # Update markers on the processed signal
    artifact_values_processed = self.processed_signal[self.artifact_mask]
    self.artifact_markers_processed.set_data(artifact_times, artifact_values_processed)

    # Update markers on the raw signal (405nm channel) - WITH NULL CHECK
    if hasattr(self, 'artifact_markers_raw') and self.artifact_markers_raw is not None and self.downsampled_raw_analog_2 is not None:
        artifact_values_raw = self.downsampled_raw_analog_2[self.artifact_mask]
        self.artifact_markers_raw.set_data(artifact_times, artifact_values_raw)
    # If artifact_markers_raw is None but we need it, initialize it
    elif self.downsampled_raw_analog_2 is not None:
        artifact_values_raw = self.downsampled_raw_analog_2[self.artifact_mask]
        self.artifact_markers_raw, = self.ax2.plot(
            artifact_times, artifact_values_raw,
            'mo', ms=4, alpha=0.7,
            label='Artifacts (Raw)'
        )

def update_secondary_artifact_markers(self):
    """Update the artifact markers for the secondary file"""
    if not hasattr(self, 'secondary_artifact_mask') or self.secondary_artifact_mask is None or not np.any(self.secondary_artifact_mask):
        # No artifacts detected in secondary, clear the markers
        if hasattr(self, 'secondary_artifact_markers_processed') and self.secondary_artifact_markers_processed is not None:
            self.secondary_artifact_markers_processed.set_data([], [])
        if hasattr(self, 'secondary_artifact_markers_raw') and self.secondary_artifact_markers_raw is not None:
            self.secondary_artifact_markers_raw.set_data([], [])
        return

    # Get the time points where artifacts occur (in minutes)
    secondary_artifact_times = self.secondary_processed_time[self.secondary_artifact_mask] / 60

    # Update markers on the processed signal
    if hasattr(self, 'secondary_artifact_markers_processed') and self.secondary_artifact_markers_processed is not None:
        secondary_artifact_values_processed = self.secondary_processed_signal[self.secondary_artifact_mask]
        self.secondary_artifact_markers_processed.set_data(secondary_artifact_times, secondary_artifact_values_processed)
    
    # Update markers on the raw signal if it exists
    if hasattr(self, 'secondary_artifact_markers_raw') and self.secondary_artifact_markers_raw is not None and \
       hasattr(self, 'secondary_downsampled_raw_analog_2') and self.secondary_downsampled_raw_analog_2 is not None:
        secondary_artifact_values_raw = self.secondary_downsampled_raw_analog_2[self.secondary_artifact_mask]
        self.secondary_artifact_markers_raw.set_data(secondary_artifact_times, secondary_artifact_values_raw)

def determine_better_control_channel(self):
    """
    Determines which file has the more realistic isosbestic control channel (analog_2)
    using the raw (not downsampled) control signals.

    Returns:
    --------
    tuple: (better_control_signal, primary_has_better_control)
        better_control_signal: The raw control signal array (np.array) deemed better.
        primary_has_better_control: Boolean indicating if primary file's control is better.
    """
    if hasattr(self, 'shared_isosbestic_var') and not self.shared_isosbestic_var.get():
        if hasattr(self, 'control_indicator'):
            self.control_indicator.config(text="Each file using its own control")
        return None, True  # No shared control, primary is "better" for itself

    primary_control = self.raw_analog_2 if hasattr(self, 'raw_analog_2') and self.raw_analog_2 is not None else None
    secondary_control = self.secondary_raw_analog_2 if hasattr(self,
                                                              'secondary_raw_analog_2') and self.secondary_raw_analog_2 is not None else None

    primary_label = self.get_region_label_from_filename(self.file_path)
    secondary_label = self.get_region_label_from_filename(
        self.secondary_file_path) if self.secondary_file_path else "Secondary"

    if secondary_control is None and primary_control is None:
        if hasattr(self, 'control_indicator'):
            self.control_indicator.config(text="No control channels available")
        return None, True  # No controls available
    elif secondary_control is None:
        if hasattr(self, 'control_indicator'):
            self.control_indicator.config(text=f"Using {primary_label} control (only one available)")
        return primary_control, True
    elif primary_control is None:
        if hasattr(self, 'control_indicator'):
            self.control_indicator.config(text=f"Using {secondary_label} control (only one available)")
        return secondary_control, False

    # Both controls are available, calculate metrics
    def filtered_range(signal):
        low = np.percentile(signal, 5)
        high = np.percentile(signal, 95)
        return high - low

    def autocorrelation_strength(signal):
        if len(signal) < 3: return 0
        return abs(np.corrcoef(signal[:-1], signal[1:])[0, 1])

    primary_variance = np.var(primary_control)
    secondary_variance = np.var(secondary_control)
    primary_range_val = filtered_range(primary_control)  # Renamed to avoid conflict
    secondary_range_val = filtered_range(secondary_control)  # Renamed to avoid conflict
    primary_autocorr = autocorrelation_strength(primary_control)
    secondary_autocorr = autocorrelation_strength(secondary_control)

    primary_score = (primary_variance * 0.4 + primary_range_val * 0.3 + primary_autocorr * 0.3)
    secondary_score = (secondary_variance * 0.4 + secondary_range_val * 0.3 + secondary_autocorr * 0.3)

    primary_is_better = primary_score >= secondary_score  # Default to primary if scores are equal

    if hasattr(self, 'control_indicator'):
        if primary_is_better:
            self.control_indicator.config(
                text=f"Using {primary_label} control (score: {primary_score:.2f} vs {secondary_score:.2f})")
        else:
            self.control_indicator.config(
                text=f"Using {secondary_label} control (score: {secondary_score:.2f} vs {primary_score:.2f})")

    print(
        f"Control channel assessment: Primary score={primary_score:.3f}, Secondary score={secondary_score:.3f}. Selected: {'Primary' if primary_is_better else 'Secondary'}")

    return (primary_control if primary_is_better else secondary_control), primary_is_better

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
