import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt  # 添加缺少的导入
from data_processing import read_ppd_file, parse_ppd_data, process_data

def open_file(self):
    """Open a file dialog to select the primary PPD file"""
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

    file_path = filedialog.askopenfilename(
        title="Select Secondary PPD File",
        filetypes=[("PPD files", "*.ppd"), ("All files", "*.*")]
    )

    if file_path:
        try:
            self.load_secondary_file(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load secondary file: {str(e)}")
            print(f"Error loading secondary file: {e}")
            import traceback
            traceback.print_exc()

def load_file(self, file_path):
    """Load and process a new primary file"""
    self.file_path = file_path
    
    # Update status if label exists
    if hasattr(self, 'status_label'):
        self.status_label.config(text=f"Loading file: {os.path.basename(file_path)}...", fg="blue")
        self.root.update_idletasks()
    
    # Clean up any existing figure
    if hasattr(self, 'fig') and self.fig:
        plt.close(self.fig)
        self.fig = None

    # Read and parse data
    try:
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
        self.processed_time, self.processed_signal, self.processed_signal_no_control, \
            self.processed_analog_2, self.downsampled_raw_analog_1, self.downsampled_raw_analog_2, \
            self.processed_digital_1, self.processed_digital_2, self.artifact_mask, \
            self.drift_curve = result

        # Create the GUI if needed
        self.create_gui()
        
        # Update window title
        self._update_window_title()
        
        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"File loaded: {os.path.basename(file_path)}", fg="green")
    
    except Exception as e:
        print(f"Error loading file: {e}")
        import traceback
        traceback.print_exc()
        messagebox.showerror("Error", f"Failed to load file: {str(e)}")
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Error loading file", fg="red")

def _update_window_title(self):
    """Update the window title to include loaded file names"""
    title = "Photometry Signal Viewer"
    if self.file_path:
        primary_name = os.path.basename(self.file_path)
        title += f" - {primary_name}"
        if hasattr(self, 'secondary_file_path') and self.secondary_file_path:
            secondary_name = os.path.basename(self.secondary_file_path)
            title += f" & {secondary_name}"
    self.root.title(title)

def _add_secondary_data_to_plots(self):
    """Add secondary file data to the plot panels"""
    if not hasattr(self, 'secondary_processed_time') or self.secondary_processed_time is None:
        return
        
    # Get region label for secondary file
    secondary_region = self.get_region_label_from_filename(self.secondary_file_path)
    
    # Update panel 2 title
    if hasattr(self, 'ax2'):
        self.ax2.set_title(f'File 2: {secondary_region}', fontsize=self.font_sizes['title'])
    
    # Create the line for the processed signal
    if hasattr(self, 'ax2'):
        self.secondary_line, = self.ax2.plot(
            self.secondary_processed_time / 60,
            self.secondary_processed_signal,
            'g-', lw=1.5,
            label=f'{secondary_region} Signal'
        )
        
        # Add more secondary lines as needed...
        
        # Update the legend
        if hasattr(self, 'create_checkbox_legend'):
            self.ax2_legend = self.create_checkbox_legend(self.ax2)
    
    # Add digital signals to panel 3 if they exist
    if hasattr(self, 'ax3') and self.secondary_processed_digital_1 is not None:
        line, = self.ax3.plot(
            self.secondary_processed_time / 60,
            self.secondary_processed_digital_1,
            'b--', lw=1,
            label=f'{secondary_region} Digital 1'
        )
        self.secondary_digital_lines.append(line)
        
        # Update the legend for digital panel
        if hasattr(self, 'create_checkbox_legend'):
            self.ax3_legend = self.create_checkbox_legend(self.ax3)
    
    # Redraw canvas
    if hasattr(self, 'canvas'):
        self.canvas.draw_idle()

def _clear_peak_annotations(self):
    """Clear all peak and valley annotations"""
    if hasattr(self, 'peak_annotations'):
        for ann in self.peak_annotations:
            if ann and hasattr(ann, 'remove'):
                ann.remove()
        self.peak_annotations = []
        
    if hasattr(self, 'valley_annotations'):
        for ann in self.valley_annotations:
            if ann and hasattr(ann, 'remove'):
                ann.remove()
        self.valley_annotations = []
        
    if hasattr(self, 'peak_lines'):
        for line in self.peak_lines:
            if line and hasattr(line, 'remove'):
                line.remove()
        self.peak_lines = []
        
    if hasattr(self, 'valley_lines'):
        for line in self.valley_lines:
            if line and hasattr(line, 'remove'):
                line.remove()
        self.valley_lines = []
