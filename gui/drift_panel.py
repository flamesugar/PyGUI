import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from data_processing import fit_drift_curve, correct_drift

def create_drift_panel(self, parent):
    """Create a panel with drift correction controls"""
    # Create frame for drift correction
    drift_frame = ttk.LabelFrame(parent, text="Drift Correction")
    drift_frame.pack(fill=tk.X, expand=False, padx=10, pady=5)
    
    # Enable/disable drift correction
    self.drift_var = tk.BooleanVar(value=self.drift_correction)
    drift_check = ttk.Checkbutton(
        drift_frame, 
        text="Enable Drift Correction", 
        variable=self.drift_var,
        command=lambda: self.update_filter(force_update=True)
    )
    drift_check.pack(anchor=tk.W, padx=10, pady=5)
    
    # Polynomial degree options
    poly_frame = ttk.Frame(drift_frame)
    poly_frame.pack(fill=tk.X, padx=10, pady=5)
    
    ttk.Label(poly_frame, text="Polynomial Degree:").pack(side=tk.LEFT, padx=5)
    self.poly_degree_var = tk.IntVar(value=self.drift_degree)
    
    for i in range(1, 5):  # Degrees 1-4
        ttk.Radiobutton(
            poly_frame,
            text=str(i),
            variable=self.poly_degree_var,
            value=i,
            command=lambda: self.update_filter(force_update=True)
        ).pack(side=tk.LEFT, padx=5)
    
    # Open advanced settings button
    advanced_button = ttk.Button(
        drift_frame,
        text="Advanced Drift Settings...",
        command=self.open_drift_window
    )
    advanced_button.pack(anchor=tk.W, padx=10, pady=10)
    
    return drift_frame

def open_drift_window(self):
    """Open drift correction settings in a separate window"""
    drift_window = tk.Toplevel(self.root)
    drift_window.title("Drift Correction Settings")
    drift_window.geometry("600x300")

    # Create main frame
    drift_frame = ttk.Frame(drift_window)
    drift_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    # Add controls (from original drift_frame)
    font_size = self.font_sizes['base']

    self.drift_var = tk.BooleanVar(value=self.drift_correction)
    self.drift_check = tk.Checkbutton(
        drift_frame, text="Enable Drift Correction",
        variable=self.drift_var, font=('Arial', font_size),
        command=self.update_filter)
    self.drift_check.pack(pady=10)

    # Polynomial degree selection
    poly_frame = tk.Frame(drift_frame)
    poly_frame.pack(pady=10)

    tk.Label(poly_frame, text="Polynomial Degree:", font=('Arial', font_size)).pack(side=tk.LEFT, padx=10)
    self.poly_degree_var = tk.IntVar(value=self.drift_degree)

    for i in range(1, 5):  # Degrees 1-4
        tk.Radiobutton(poly_frame, text=str(i), variable=self.poly_degree_var,
                       value=i, font=('Arial', font_size),
                       command=self.update_filter).pack(side=tk.LEFT, padx=5)

    # Edge protection option
    self.edge_protection_var = tk.BooleanVar(value=True)
    tk.Checkbutton(drift_frame,
                   text="Enable edge protection (reduces distortion at beginning of recording)",
                   variable=self.edge_protection_var,
                   font=('Arial', font_size),
                   command=self.update_filter).pack(pady=20)

    # Apply and Close buttons
    button_frame = tk.Frame(drift_frame)
    button_frame.pack(pady=20)

    tk.Button(button_frame, text="Apply", font=('Arial', font_size),
              command=lambda: self.update_filter(force_update=True)).pack(side=tk.LEFT, padx=10)
    tk.Button(button_frame, text="Close", font=('Arial', font_size),
              command=drift_window.destroy).pack(side=tk.LEFT, padx=10)

def update_drift_correction(self):
    """Apply drift correction to the signal data based on current settings"""
    try:
        # Show processing cursor
        if hasattr(self, 'root'):
            self.root.config(cursor="watch")
        
        # Update status if label exists
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Applying drift correction...", fg="blue")
            self.root.update_idletasks()
        
        # Get current drift settings
        self.drift_correction = self.drift_var.get() if hasattr(self, 'drift_var') else False
        self.drift_degree = self.poly_degree_var.get() if hasattr(self, 'poly_degree_var') else 2
        
        if not self.drift_correction:
            print("Drift correction is disabled")
            if hasattr(self, 'status_label'):
                self.status_label.config(text="Drift correction is disabled", fg="blue")
            if hasattr(self, 'root'):
                self.root.config(cursor="")
            return
        
        # Apply to primary signal if available
        if hasattr(self, 'time') and hasattr(self, 'analog_1') and \
            self.time is not None and self.analog_1 is not None:
            
            print("Fitting drift curve for primary signal...")
            # Fit the drift curve
            self.drift_curve = fit_drift_curve(
                self.time, 
                self.analog_1, 
                self.analog_2 if hasattr(self, 'analog_2') else None,
                degree=self.drift_degree
            )
            
            # Correct the signals
            print("Correcting primary signal drift...")
            if hasattr(self, 'analog_1'):
                self.analog_1 = correct_drift(self.analog_1, self.drift_curve)
                
            # Update any plot elements
            if hasattr(self, 'line') and self.line is not None and hasattr(self, 'processed_signal'):
                # Update will happen through update_filter which will be called separately
                pass
                
            # Update drift line if it exists
            if hasattr(self, 'drift_line') and self.drift_line is not None and hasattr(self, 'drift_curve'):
                # Normalize the drift curve for display
                baseline_idx = min(int(len(self.drift_curve) * 0.1), 10) if len(self.drift_curve) > 0 else 0
                baseline = np.mean(self.drift_curve[:baseline_idx]) if baseline_idx > 0 else 0
                
                # Avoid division by zero
                if abs(baseline) > 1e-9:
                    df_drift_curve = 100 * (self.drift_curve - baseline) / abs(baseline)
                else:
                    df_drift_curve = self.drift_curve
                    
                # Update the drift line
                if hasattr(self, 'processed_time'):
                    self.drift_line.set_data(self.processed_time / 60, df_drift_curve)
            
        # Apply to secondary signal if available
        if hasattr(self, 'secondary_time') and hasattr(self, 'secondary_analog_1') and \
            self.secondary_time is not None and self.secondary_analog_1 is not None:
            
            print("Fitting drift curve for secondary signal...")
            # Fit the secondary drift curve
            self.secondary_drift_curve = fit_drift_curve(
                self.secondary_time, 
                self.secondary_analog_1, 
                self.secondary_analog_2 if hasattr(self, 'secondary_analog_2') else None,
                degree=self.drift_degree
            )
            
            # Correct the secondary signals
            print("Correcting secondary signal drift...")
            if hasattr(self, 'secondary_analog_1'):
                self.secondary_analog_1 = correct_drift(self.secondary_analog_1, self.secondary_drift_curve)
                
            # Update secondary drift line if it exists
            if hasattr(self, 'secondary_drift_line') and self.secondary_drift_line is not None and \
               hasattr(self, 'secondary_drift_curve'):
                
                # Normalize the secondary drift curve
                baseline_idx_s = min(int(len(self.secondary_drift_curve) * 0.1), 10) if len(self.secondary_drift_curve) > 0 else 0
                baseline_s = np.mean(self.secondary_drift_curve[:baseline_idx_s]) if baseline_idx_s > 0 else 0
                
                # Avoid division by zero
                if abs(baseline_s) > 1e-9:
                    df_drift_curve_s = 100 * (self.secondary_drift_curve - baseline_s) / abs(baseline_s)
                else:
                    df_drift_curve_s = self.secondary_drift_curve
                    
                # Update the secondary drift line
                if hasattr(self, 'secondary_processed_time'):
                    self.secondary_drift_line.set_data(self.secondary_processed_time / 60, df_drift_curve_s)
        
        # Update canvas if it exists
        if hasattr(self, 'canvas'):
            self.canvas.draw_idle()
        
        # Restore cursor and update status
        if hasattr(self, 'root'):
            self.root.config(cursor="")
        
        if hasattr(self, 'status_label'):
            self.status_label.config(text="Drift correction applied", fg="green")
            
        print("Drift correction complete")
        
        # Force a full filter update to refresh all data
        if hasattr(self, 'update_filter'):
            self.update_filter(force_update=True)
        
    except Exception as e:
        print(f"Error in update_drift_correction: {e}")
        import traceback
        traceback.print_exc()
        
        # Restore cursor
        if hasattr(self, 'root'):
            self.root.config(cursor="")
            
        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.config(text=f"Error in drift correction: {e}", fg="red")
            
        # Show error message
        messagebox.showerror("Error", f"Failed to apply drift correction:\n{e}")
        
