import tkinter as tk
from tkinter import ttk
import numpy as np
from data_processing import process_data

def create_filter_panel(self, parent):
    """Create a panel with sliders for filter parameters"""
    filter_frame = ttk.Frame(parent)
    
    # Add controls with font sizes from the main class
    font_size = self.font_sizes['base']
    slider_font_size = self.font_sizes['slider']

    # Low cutoff slider
    tk.Label(filter_frame, text="Low cutoff (Hz):", font=('Arial', font_size)).grid(
        row=0, column=0, sticky="w", padx=10, pady=10)
    self.low_slider = tk.Scale(filter_frame, from_=0.0, to=0.01,
                              resolution=0.0001, orient=tk.HORIZONTAL,
                              length=600, width=25, font=('Arial', slider_font_size),
                              command=self.update_filter)
    self.low_slider.set(self.low_cutoff)
    self.low_slider.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

    # High cutoff slider
    tk.Label(filter_frame, text="High cutoff (Hz):", font=('Arial', font_size)).grid(
        row=1, column=0, sticky="w", padx=10, pady=10)
    self.high_slider = tk.Scale(filter_frame, from_=0.1, to=5.0,
                               resolution=0.1, orient=tk.HORIZONTAL,
                               length=600, width=25, font=('Arial', slider_font_size),
                               command=self.update_filter)
    self.high_slider.set(self.high_cutoff)
    self.high_slider.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

    # Downsample slider
    tk.Label(filter_frame, text="Downsample factor:", font=('Arial', font_size)).grid(
        row=2, column=0, sticky="w", padx=10, pady=10)
    self.downsample_slider = tk.Scale(filter_frame, from_=1, to=100,
                                     resolution=1, orient=tk.HORIZONTAL,
                                     length=600, width=25, font=('Arial', slider_font_size),
                                     command=self.update_filter)
    self.downsample_slider.set(self.downsample_factor)
    self.downsample_slider.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
    
    # Artifact threshold slider
    tk.Label(filter_frame, text="Artifact threshold:", font=('Arial', font_size)).grid(
        row=3, column=0, sticky="w", padx=10, pady=10)
    self.artifact_slider = tk.Scale(filter_frame, from_=1.0, to=10.0,
                                   resolution=0.1, orient=tk.HORIZONTAL,
                                   length=600, width=25, font=('Arial', slider_font_size),
                                   command=self.update_filter)
    self.artifact_slider.set(self.artifact_threshold)
    self.artifact_slider.grid(row=3, column=1, padx=10, pady=10, sticky="ew")
    
    # Drift correction checkbox
    drift_frame = ttk.Frame(filter_frame)
    drift_frame.grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=10)
    
    self.drift_var = tk.BooleanVar(value=self.drift_correction)
    drift_check = ttk.Checkbutton(
        drift_frame,
        text="Enable Drift Correction",
        variable=self.drift_var,
        command=self.update_filter
    )
    drift_check.pack(side=tk.LEFT, padx=5)
    
    # Polynomial degree options
    poly_frame = ttk.Frame(drift_frame)
    poly_frame.pack(side=tk.LEFT, padx=20)
    
    ttk.Label(poly_frame, text="Polynomial Degree:").pack(side=tk.LEFT, padx=5)
    self.poly_degree_var = tk.IntVar(value=self.drift_degree)
    
    for i in range(1, 6):  # Degrees 1 through 5
        ttk.Radiobutton(
            poly_frame,
            text=str(i),
            variable=self.poly_degree_var,
            value=i,
            command=self.update_filter
        ).pack(side=tk.LEFT, padx=5)
    
    # Edge protection option if needed
    if hasattr(self, 'edge_protection_var'):
        edge_check = ttk.Checkbutton(
            filter_frame,
            text="Edge Protection",
            variable=self.edge_protection_var,
            command=self.update_filter
        )
        edge_check.grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=10)
    
    # Apply button frame
    button_frame = ttk.Frame(filter_frame)
    button_frame.grid(row=6, column=0, columnspan=2, pady=20)
    
    ttk.Button(
        button_frame,
        text="Apply",
        command=lambda: self.update_filter(force_update=True)
    ).pack(side=tk.LEFT, padx=10)
    
    return filter_frame

def update_filter(self, _=None, preserve_blanking=True, force_update=False):
    """
    Update filter parameters and reprocess data, but only when Apply button is clicked.

    Parameters:
    -----------
    _: event parameter (unused)
    preserve_blanking: whether to preserve manual blanking regions
    force_update: whether to force recalculation (set by Apply button)
    """
    # If not forced by Apply button, just store parameters without updating
    if not force_update:
        return

    try:
        print("Starting filter update...")
        self.root.config(cursor="watch")
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Processing...", fg="blue")
        self.root.update_idletasks()

        # Save current view state
        x_lim = self.ax1.get_xlim() if hasattr(self, 'ax1') and self.ax1 else None
        y1_lim = self.ax1.get_ylim() if hasattr(self, 'ax1') and self.ax1 else None
        y2_lim = self.ax2.get_ylim() if hasattr(self, 'ax2') and self.ax2 else None

        # Backup blanking regions if needed
        blanking_regions_backup = []
        if preserve_blanking and hasattr(self, 'blanking_regions'):
            blanking_regions_backup = self.blanking_regions.copy()

        # Get filter parameters from UI
        self.low_cutoff = float(self.low_slider.get())
        self.high_cutoff = float(self.high_slider.get())
        self.downsample_factor = max(1, int(self.downsample_slider.get()))
        self.artifact_threshold = float(self.artifact_slider.get())
        self.drift_correction = bool(self.drift_var.get())
        self.drift_degree = int(self.poly_degree_var.get())
        edge_protection = bool(self.edge_protection_var.get()) if hasattr(self, 'edge_protection_var') else True

        print(
            f"Parameters: low={self.low_cutoff}, high={self.high_cutoff}, downsample={self.downsample_factor}, artifact_thresh={self.artifact_threshold}, drift={self.drift_correction}, degree={self.drift_degree}, edge_prot={edge_protection}")

        if not hasattr(self, 'time') or self.time is None or not hasattr(self,
                                                                         'raw_analog_1') or self.raw_analog_1 is None:
            print("Error: Missing required primary data for processing.")
            if hasattr(self, 'status_label'): self.status_label.config(text="Error: Missing primary data", fg="red")
            self.root.config(cursor="")
            return

        # Determine the shared control signal to use
        chosen_control_signal_for_sharing, primary_control_is_better = None, True
        if hasattr(self, 'shared_isosbestic_var') and self.shared_isosbestic_var.get():
            chosen_control_signal_for_sharing, primary_control_is_better = self.determine_better_control_channel()

        # Process primary data
        primary_external_control_to_pass = None
        if chosen_control_signal_for_sharing is not None and not primary_control_is_better:
            # If shared control is active and secondary's is better, use it for primary's processing
            primary_external_control_to_pass = chosen_control_signal_for_sharing  # This is secondary's raw_analog_2

        print(
            f"Processing primary data. External control for primary: {'Provided' if primary_external_control_to_pass is not None else 'None'}")
        result = process_data(
            self.time, self.raw_analog_1, self.raw_analog_2,
            self.digital_1, self.digital_2,
            low_cutoff=self.low_cutoff, high_cutoff=self.high_cutoff,
            downsample_factor=self.downsample_factor, artifact_threshold=self.artifact_threshold,
            drift_correction=self.drift_correction, drift_degree=self.drift_degree,
            external_control=primary_external_control_to_pass,
            edge_protection=edge_protection
        )

        if result is None or len(result) < 10:
            print(f"Invalid result from process_data for primary: {result}")
            if hasattr(self, 'status_label'): self.status_label.config(text="Error: Primary processing failed",
                                                                       fg="red")
            self.root.config(cursor="")
            return
        self.processed_time, self.processed_signal, self.processed_signal_no_control, \
            self.processed_analog_2, self.downsampled_raw_analog_1, self.downsampled_raw_analog_2, \
            self.processed_digital_1, self.processed_digital_2, self.artifact_mask, \
            self.drift_curve = result
        print("Primary data processed.")

        # Process secondary file if it exists
        if hasattr(self, 'secondary_file_path') and self.secondary_file_path:
            if not hasattr(self, 'secondary_time') or self.secondary_time is None:
                print("Error: Missing required secondary data for processing (secondary_time).")
            else:
                secondary_external_control_to_pass = None
                if chosen_control_signal_for_sharing is not None and primary_control_is_better:
                    # If shared control is active and primary's is better, use it for secondary's processing
                    secondary_external_control_to_pass = chosen_control_signal_for_sharing  # This is primary's raw_analog_2

                print(
                    f"Processing secondary data. External control for secondary: {'Provided' if secondary_external_control_to_pass is not None else 'None'}")
                secondary_result = process_data(
                    self.secondary_time, self.secondary_raw_analog_1, self.secondary_raw_analog_2,
                    self.secondary_digital_1, self.secondary_digital_2,
                    low_cutoff=self.low_cutoff, high_cutoff=self.high_cutoff,
                    downsample_factor=self.downsample_factor, artifact_threshold=self.artifact_threshold,
                    drift_correction=self.drift_correction, drift_degree=self.drift_degree,
                    external_control=secondary_external_control_to_pass,
                    edge_protection=edge_protection
                )
                if secondary_result is None or len(secondary_result) < 10:
                    print(f"Invalid result from process_data for secondary: {secondary_result}")
                else:
                    self.secondary_processed_time, self.secondary_processed_signal, self.secondary_processed_signal_no_control, \
                        self.secondary_processed_analog_2, self.secondary_downsampled_raw_analog_1, self.secondary_downsampled_raw_analog_2, \
                        self.secondary_processed_digital_1, self.secondary_processed_digital_2, self.secondary_artifact_mask, \
                        self.secondary_drift_curve = secondary_result
                    print("Secondary data processed.")

        if preserve_blanking and blanking_regions_backup:
            print("Reapplying blanking regions...")
            self.blanking_regions = blanking_regions_backup
            self.reapply_all_blanking()  # This will operate on the newly filtered signals

        # Update plot data
        if hasattr(self, 'line') and self.line: self.line.set_data(self.processed_time / 60, self.processed_signal)
        if hasattr(self, 'line_no_control') and self.line_no_control: self.line_no_control.set_data(
            self.processed_time / 60, self.processed_signal_no_control)
        if hasattr(self, 'raw_line') and self.raw_line: self.raw_line.set_data(self.processed_time / 60,
                                                                               self.downsampled_raw_analog_1)
        if hasattr(self,
                   'raw_line2') and self.raw_line2 and self.downsampled_raw_analog_2 is not None: self.raw_line2.set_data(
            self.processed_time / 60, self.downsampled_raw_analog_2)
        if hasattr(self, 'drift_line') and self.drift_line and self.drift_curve is not None:
            baseline_idx = min(int(len(self.drift_curve) * 0.1), 10)
            baseline = np.mean(self.drift_curve[:baseline_idx]) if baseline_idx > 0 and len(
                self.drift_curve) > baseline_idx else np.mean(self.drift_curve) if len(self.drift_curve) > 0 else 0
            df_drift_curve = 100 * (self.drift_curve - baseline) / abs(baseline) if abs(
                baseline) > 1e-9 else self.drift_curve
            self.drift_line.set_data(self.processed_time / 60, df_drift_curve)

        if hasattr(self, 'secondary_line') and self.secondary_line and hasattr(self,
                                                                               'secondary_processed_time') and self.secondary_processed_time is not None:
            self.secondary_line.set_data(self.secondary_processed_time / 60, self.secondary_processed_signal)
            if hasattr(self,
                       'secondary_line_no_control') and self.secondary_line_no_control: self.secondary_line_no_control.set_data(
                self.secondary_processed_time / 60, self.secondary_processed_signal_no_control)
            if hasattr(self,
                       'secondary_drift_line') and self.secondary_drift_line and self.secondary_drift_curve is not None:
                baseline_idx_s = min(int(len(self.secondary_drift_curve) * 0.1), 10)
                baseline_s = np.mean(self.secondary_drift_curve[:baseline_idx_s]) if baseline_idx_s > 0 and len(
                    self.secondary_drift_curve) > baseline_idx_s else np.mean(self.secondary_drift_curve) if len(
                    self.secondary_drift_curve) > 0 else 0
                df_drift_curve_s = 100 * (self.secondary_drift_curve - baseline_s) / abs(baseline_s) if abs(
                    baseline_s) > 1e-9 else self.secondary_drift_curve
                self.secondary_drift_line.set_data(self.secondary_processed_time / 60, df_drift_curve_s)

        self.update_artifact_markers()
        if hasattr(self, 'secondary_processed_signal') and self.secondary_processed_signal is not None:
            if hasattr(self, 'update_secondary_artifact_markers'): self.update_secondary_artifact_markers()

        if x_lim and y1_lim and y2_lim and self.ax1 and self.ax2:  # Check if axes exist
            self.ax1.set_xlim(x_lim)
            self.ax1.set_ylim(y1_lim)
            self.ax2.set_ylim(y2_lim)
        else:  # Fallback to autoscale if limits were not captured or axes don't exist
            if hasattr(self, 'ax1') and self.ax1: self.ax1.relim(); self.ax1.autoscale_view()
            if hasattr(self, 'ax2') and self.ax2: self.ax2.relim(); self.ax2.autoscale_view()

        if hasattr(self, 'canvas'): self.canvas.draw_idle()
        print("Plots updated and canvas redrawn.")

        self.root.config(cursor="")
        if hasattr(self, 'status_label'): self.status_label.config(
            text=f"Filters updated: low={self.low_cutoff:.4f}, high={self.high_cutoff:.2f}", fg="green")
        print("Filter update completed successfully.")

    except Exception as e:
        print(f"Error in update_filter: {e}")
        import traceback
        traceback.print_exc()
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Error updating filters: {e}", fg="red")
        self.root.config(cursor="")

def open_filter_window(self):
    """Open filter settings in a separate window"""
    filter_window = tk.Toplevel(self.root)
    filter_window.title("Signal Filter Settings")
    filter_window.geometry("800x400")
    filter_window.minsize(600, 300)  # Set minimum size
    filter_window.resizable(True, True)  # Make window resizable

    # Create frame for filter controls
    filter_frame = ttk.Frame(filter_window)
    filter_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    # Add controls (copied from original signal_processing_tab)
    font_size = self.font_sizes['base']
    slider_font_size = self.font_sizes['slider']

    # Low cutoff slider
    tk.Label(filter_frame, text="Low cutoff (Hz):", font=('Arial', font_size)).grid(
        row=0, column=0, sticky="w", padx=10, pady=10)
    self.low_slider = tk.Scale(filter_frame, from_=0.0, to=0.01,
                              resolution=0.0001, orient=tk.HORIZONTAL,
                              length=600, width=25, font=('Arial', slider_font_size),
                              command=self.update_filter)
    self.low_slider.set(self.low_cutoff)
    self.low_slider.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

    # High cutoff slider
    tk.Label(filter_frame, text="High cutoff (Hz):", font=('Arial', font_size)).grid(
        row=1, column=0, sticky="w", padx=10, pady=10)
    self.high_slider = tk.Scale(filter_frame, from_=0.1, to=5.0,
                               resolution=0.1, orient=tk.HORIZONTAL,
                               length=600, width=25, font=('Arial', slider_font_size),
                               command=self.update_filter)
    self.high_slider.set(self.high_cutoff)
    self.high_slider.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

    # Downsample slider
    tk.Label(filter_frame, text="Downsample factor:", font=('Arial', font_size)).grid(
        row=2, column=0, sticky="w", padx=10, pady=10)
    self.downsample_slider = tk.Scale(filter_frame, from_=1, to=100,
                                     resolution=1, orient=tk.HORIZONTAL,
                                     length=600, width=25, font=('Arial', slider_font_size),
                                     command=self.update_filter)
    self.downsample_slider.set(self.downsample_factor)
    self.downsample_slider.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

    # Apply and Close buttons
    button_frame = tk.Frame(filter_frame)
    button_frame.grid(row=3, column=0, columnspan=2, pady=20)

    tk.Button(button_frame,
              text="Apply",
              font=('Arial', font_size),
              bg='lightgreen',
              command=lambda: self.update_filter(force_update=True)).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="Close", font=('Arial', font_size),
              command=filter_window.destroy).pack(side=tk.LEFT, padx=10)
