import numpy as np
import matplotlib.pyplot as plt

def update_panel_y_limits(self):
    """Set appropriate y-axis limits for each panel with signal baseline at 0-10% of the range"""
    try:
        # Panel 1 (Primary) scaling
        if hasattr(self, 'processed_signal') and self.processed_signal is not None:
            # Get a better estimate of the baseline using a lower percentile
            p_min = np.percentile(self.processed_signal, 1)  # Use 1st percentile for lower bound
            p_max = np.percentile(self.processed_signal, 99)  # Use 99th percentile for upper bound

            # Calculate total range
            data_range = p_max - p_min

            # Position the baseline at 10% of the panel height
            # This means the minimum value should be at ~10% from the bottom
            desired_min = p_min - (data_range * 0.1)

            # Give more space at the top for signal peaks
            desired_max = p_max + (data_range * 0.2)

            # Set limits with baseline positioned lower in the panel
            self.ax1.set_ylim(desired_min, desired_max)
            print(f"Set primary panel y-limits to: {self.ax1.get_ylim()}, positioning baseline at ~10%")

        # Panel 2 (Secondary) scaling - ensure 0 is visible
        if hasattr(self, 'secondary_processed_signal') and self.secondary_processed_signal is not None:
            # Use the same approach for the secondary panel
            s_min = np.percentile(self.secondary_processed_signal, 1)
            s_max = np.percentile(self.secondary_processed_signal, 99)

            # Calculate range
            data_range = s_max - s_min

            # Position baseline at 10% of panel height
            desired_min = s_min - (data_range * 0.1)

            # More space at top for peaks
            desired_max = s_max + (data_range * 0.2)

            # Set limits
            self.ax2.set_ylim(desired_min, desired_max)
            print(f"Set secondary panel y-limits to: {self.ax2.get_ylim()}, positioning baseline at ~10%")

    except Exception as e:
        print(f"Error in update_panel_y_limits: {e}")

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
    """Update the artifact markers for the secondary signal"""
    # Check if secondary_artifact_markers_processed exists before trying to set data
    if not hasattr(self, 'secondary_artifact_markers_processed') or self.secondary_artifact_markers_processed is None:
        return

    if (self.secondary_artifact_mask is None or
            not np.any(self.secondary_artifact_mask) or
            len(self.secondary_artifact_mask) != len(self.secondary_processed_time)):
        # No artifacts detected or size mismatch, clear the markers
        self.secondary_artifact_markers_processed.set_data([], [])
        if hasattr(self, 'secondary_artifact_markers_raw') and self.secondary_artifact_markers_raw is not None:
            self.secondary_artifact_markers_raw.set_data([], [])
        return

    # Get the time points where artifacts occur (in minutes)
    artifact_times = self.secondary_processed_time[self.secondary_artifact_mask] / 60

    # Update markers on the processed signal
    artifact_values_processed = self.secondary_processed_signal[self.secondary_artifact_mask]
    self.secondary_artifact_markers_processed.set_data(artifact_times, artifact_values_processed)

    # Update markers on the raw signal (405nm channel)
    if hasattr(self, 'secondary_artifact_markers_raw') and self.secondary_artifact_markers_raw is not None and self.secondary_downsampled_raw_analog_2 is not None:
        artifact_values_raw = self.secondary_downsampled_raw_analog_2[self.secondary_artifact_mask]
        self.secondary_artifact_markers_raw.set_data(artifact_times, artifact_values_raw)

def normalize_raw_signal(self, processed_signal, raw_signal):
    """Normalize raw signal to be on a similar scale as the processed signal"""
    if processed_signal is None or raw_signal is None:
        return raw_signal

    # Get statistics from both signals
    p_mean = np.mean(processed_signal)
    p_std = np.std(processed_signal)
    r_mean = np.mean(raw_signal)
    r_std = np.std(raw_signal)

    # If standard deviation is too small, use a default value
    if r_std < 1e-10:
        r_std = 1.0

    # Scale and shift raw signal to match processed signal scale
    normalized = (raw_signal - r_mean) * (p_std / r_std) + p_mean

    return normalized

def mark_flattened_regions(self):
    """Mark flattened regions in the plot with gray background"""
    # Remove any existing marked regions
    for patch in self.ax1.patches:
        if hasattr(patch, 'is_flattened_marker'):
            patch.remove()

    # Function to detect flat regions
    def find_flat_regions(signal, threshold=1e-6):
        """Find regions where the signal is flat (standard deviation near zero)"""
        flat_regions = []
        in_flat_region = False
        flat_start = 0

        # Use a window to compute rolling standard deviation
        window_size = min(100, len(signal) // 100)
        if window_size < 3:
            return []  # Signal too short

        for i in range(len(signal) - window_size):
            std = np.std(signal[i:i + window_size])
            if std < threshold and not in_flat_region:
                flat_start = i
                in_flat_region = True
            elif std >= threshold and in_flat_region:
                if i - flat_start > window_size:  # Only mark substantial regions
                    flat_regions.append((flat_start, i))
                in_flat_region = False

        # Check if we ended in a flat region
        if in_flat_region and len(signal) - flat_start > window_size:
            flat_regions.append((flat_start, len(signal)))

        return flat_regions

    # Mark flat regions in primary signal
    primary_flats = find_flat_regions(self.processed_signal)
    for start, end in primary_flats:
        if start > 0:  # Only mark flattened beginnings
            patch = plt.Rectangle(
                (self.processed_time[start] / 60, self.ax1.get_ylim()[0]),
                (self.processed_time[end - 1] - self.processed_time[start]) / 60,
                self.ax1.get_ylim()[1] - self.ax1.get_ylim()[0],
                facecolor='lightgray', alpha=0.4, zorder=-100)
            patch.is_flattened_marker = True
            self.ax1.add_patch(patch)

    # Mark flat regions in secondary signal if available
    if hasattr(self, 'secondary_processed_signal') and self.secondary_processed_signal is not None:
        secondary_flats = find_flat_regions(self.secondary_processed_signal)
        for start, end in secondary_flats:
            if start > 0:  # Only mark flattened beginnings
                patch = plt.Rectangle(
                    (self.secondary_processed_time[start] / 60, self.ax1.get_ylim()[0]),
                    (self.secondary_processed_time[end - 1] - self.secondary_processed_time[start]) / 60,
                    self.ax1.get_ylim()[1] - self.ax1.get_ylim()[0],
                    facecolor='mistyrose', alpha=0.4, zorder=-100)
                patch.is_flattened_marker = True
                self.ax1.add_patch(patch)

def create_checkbox_legend(self, ax):
    """Create a custom legend with checkboxes for the given axis"""
    # Get all lines and their labels from the axis
    lines = [line for line in ax.get_lines() if line.get_label()[0] != '_']
    labels = [line.get_label() for line in lines]

    # Create legend with custom handler
    leg = ax.legend(loc='upper right', fontsize=self.font_sizes['base'])

    # Make legend interactive
    leg.set_draggable(True)

    # Store line visibility state
    for line in lines:
        if line not in self.line_visibility:
            self.line_visibility[line] = True

    # Add checkbox functionality
    for i, legline in enumerate(leg.get_lines()):
        legline.set_picker(5)  # Enable picking on the legend line
        legline.line = lines[i]  # Store reference to the actual line

    # Connect the pick event
    self.fig.canvas.mpl_connect('pick_event', self.on_legend_pick)

    return leg

def on_legend_pick(self, event):
    """Handle legend picking to toggle line visibility"""
    # Get all legend lines
    all_legends = []
    if hasattr(self, 'ax1_legend') and self.ax1_legend:
        all_legends.extend(self.ax1_legend.get_lines())
    if hasattr(self, 'ax2_legend') and self.ax2_legend:
        all_legends.extend(self.ax2_legend.get_lines())
    if hasattr(self, 'ax3_legend') and self.ax3_legend:
        all_legends.extend(self.ax3_legend.get_lines())

    # Check if the picked object is a legend line
    if event.artist in all_legends:
        # Get the original line this legend item represents
        line = event.artist.line

        # Toggle visibility
        visible = not line.get_visible()
        line.set_visible(visible)

        # Change the alpha of the legend markers
        event.artist.set_alpha(1.0 if visible else 0.2)

        # Store the new state
        self.line_visibility[line] = visible

        # Redraw the canvas
        self.canvas.draw_idle()
