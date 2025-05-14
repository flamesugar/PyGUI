import matplotlib.pyplot as plt
import numpy as np
import traceback

def connect_zoom_events(self):
    """Connect mouse wheel events for custom zooming"""
    print("Connecting zoom events...")

    # Disconnect existing handlers to avoid duplicates
    if hasattr(self, 'scroll_cid') and self.scroll_cid:
        try:
            self.canvas.mpl_disconnect(self.scroll_cid)
        except:
            pass
    if hasattr(self, 'key_press_cid') and self.key_press_cid:
        try:
            self.canvas.mpl_disconnect(self.key_press_cid)
        except:
            pass
    if hasattr(self, 'key_release_cid') and self.key_release_cid:
        try:
            self.canvas.mpl_disconnect(self.key_release_cid)
        except:
            pass

    # Connect to matplotlib events
    self.scroll_cid = self.canvas.mpl_connect('scroll_event', self.on_scroll)
    self.key_press_cid = self.canvas.mpl_connect('key_press_event', self.on_key_press)
    self.key_release_cid = self.canvas.mpl_connect('key_release_event', self.on_key_release)
    self.ctrl_pressed = False

    print(
        f"Connected events: scroll={self.scroll_cid}, key_press={self.key_press_cid}, key_release={self.key_release_cid}")

    # Also bind to the Tkinter widget as a backup but with the improved method
    self.canvas.get_tk_widget().bind("<MouseWheel>", self._on_mousewheel)
    self.canvas.get_tk_widget().bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)

def _on_mousewheel(self, event):
    """Tkinter event handler for mousewheel"""
    # Convert to matplotlib-style event and call on_scroll
    # Direction is platform-dependent
    if event.delta > 0:
        direction = 'up'
    else:
        direction = 'down'

    # Determine which panel the mouse is over based on y-position
    widget_height = self.canvas.get_tk_widget().winfo_height()
    y_position = event.y

    # Create a synthetic matplotlib event
    class SyntheticEvent:
        def __init__(self, button, inaxes, xdata=None, ydata=None):
            self.button = button
            self.inaxes = inaxes
            self.xdata = xdata
            self.ydata = ydata

    # Simple heuristic: if in top third, use ax1, middle third use ax2, bottom third use ax3
    if y_position < widget_height / 3:
        inaxes = self.ax1
    elif y_position < 2 * widget_height / 3:
        inaxes = self.ax2
    else:
        inaxes = self.ax3

    # Create synthetic event with appropriate panel
    synth_event = SyntheticEvent(direction, inaxes)

    # Pass to on_scroll with the appropriate Ctrl state
    self.on_scroll(synth_event)

def _on_ctrl_mousewheel(self, event):
    """Tkinter event handler for Ctrl+mousewheel - ONLY affect X axis"""
    self.ctrl_pressed = True
    self._on_mousewheel(event) # This eventually calls on_scroll
    self.ctrl_pressed = False
    return "break" # Stop further Tkinter event processing for Ctrl+Wheel

def on_key_press(self, event):
    """Handle key press events to track Ctrl key state"""
    if event.key == 'control':
        self.ctrl_pressed = True
    elif event.key == 'home': # Add shortcut to reset view
        print("Home key pressed - Resetting view")
        self.reset_view()

def on_key_release(self, event):
    """Handle key release events to track Ctrl key state"""
    if event.key == 'control':
        self.ctrl_pressed = False

def on_scroll(self, event):
    """
    Enhanced mouse wheel handler:
    - Regular wheel: zoom Y-axis ONLY of the current panel
    - Ctrl+wheel: zoom X-axis ONLY of ALL panels together, centered on mouse position
    """
    try:
        if event.inaxes is None or not hasattr(self, 'canvas') or self.canvas is None:
            return  # Ignore scroll events outside axes or if canvas isn't ready

        # Determine zoom factor
        base_scale = 1.2  # Sensitivity factor
        if event.button == 'up':
            zoom_factor = 1.0 / base_scale  # Zoom in
        elif event.button == 'down':
            zoom_factor = base_scale  # Zoom out
        else:
            return  # Ignore other scroll buttons

        # Get the specific axes where the scroll happened
        ax_current = event.inaxes

        if self.ctrl_pressed:
            # === CTRL+WHEEL: Zoom ONLY X-axis of ALL panels, centered on mouse position ===
            xlim = ax_current.get_xlim()

            # Get mouse x position in data coordinates
            xdata = event.xdata

            # If mouse is outside valid data area, use center of view
            if xdata is None or not np.isfinite(xdata):
                xdata = (xlim[0] + xlim[1]) / 2

            # Calculate distance from mouse to left and right edges
            x_left_dist = xdata - xlim[0]
            x_right_dist = xlim[1] - xdata

            # Calculate new left and right edges with zoom factor,
            # maintaining mouse position as the center of zoom
            new_left = xdata - x_left_dist * zoom_factor
            new_right = xdata + x_right_dist * zoom_factor

            # Apply new x limits to all axes WITHOUT changing y limits
            for ax in [self.ax1, self.ax2, self.ax3]:
                if ax:
                    # Store current y limits
                    current_ylim = ax.get_ylim()
                    # Set new x limits
                    ax.set_xlim(new_left, new_right)
                    # Restore original y limits to prevent y-axis changes
                    ax.set_ylim(current_ylim)

            print(f"Ctrl+Wheel: X-axis ONLY zoom applied to all panels: [{new_left:.2f}, {new_right:.2f}]")
            
            # Redraw and return early to prevent the regular wheel zoom code
            self.canvas.draw_idle()
            return False

        else:
            # === REGULAR WHEEL: Zoom Y-axis of CURRENT panel ONLY ===
            # Skip y-zoom on the digital plot (ax3)
            if ax_current == self.ax3:
                print("Skipping Y-axis zoom on digital panel (ax3)")
                return

            # Get current y limits for the specific panel being scrolled
            ylim = ax_current.get_ylim()

            # Center zoom on mouse y-coordinate, or use view center if outside data range
            ydata = event.ydata if event.ydata is not None and np.isfinite(event.ydata) else (ylim[0] + ylim[1]) / 2

            # Calculate new height centered on ydata
            new_height = (ylim[1] - ylim[0]) * zoom_factor
            new_ylim = [ydata - new_height * (ydata - ylim[0]) / (ylim[1] - ylim[0]),
                        ydata + new_height * (ylim[1] - ydata) / (ylim[1] - ylim[0])]

            # Apply limits ONLY to the current axis (panel-specific zooming)
            ax_current.set_ylim(new_ylim)
            print(f"Regular Wheel: Y-axis zoom applied to current panel")

            # Redraw the canvas to show the changes
            self.canvas.draw_idle()
            return False

    except Exception as e:
        print(f"Error during scroll event processing: {e}")
        traceback.print_exc()
        return False

def zoom_x(self, in_out="in"):
    """Zoom in or out on the x-axis for all panels"""
    try:
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
            if ax:
                ax.set_xlim(new_xlim)
        
        # Redraw
        self.canvas.draw_idle()
        print(f"X-axis zoom: {in_out}, new limits: {new_xlim}")
        
    except Exception as e:
        print(f"Error in zoom_x: {e}")
        traceback.print_exc()

def pan_x(self, direction="right"):
    """Pan the view left or right"""
    # Get current x-limits
    xlim = self.ax1.get_xlim()

    # Calculate the amount to shift (25% of visible range)
    shift_amount = 0.25 * (xlim[1] - xlim[0])

    # Apply shift based on direction
    if direction == "left":
        new_xlim = [xlim[0] - shift_amount, xlim[1] - shift_amount]
    else:  # right
        new_xlim = [xlim[0] + shift_amount, xlim[1] + shift_amount]

    # Apply to all axes
    for ax in [self.ax1, self.ax2, self.ax3]:
        ax.set_xlim(new_xlim)

    # Redraw
    self.canvas.draw_idle()

def zoom_y_all(self, in_out="in"):
    """Zoom in or out on the y-axis for all panels (except digital)"""
    # Zoom the processed signal panel
    self.zoom_y(self.ax1, in_out)

    # Zoom the raw signal panel
    self.zoom_y(self.ax2, in_out)

    # Don't zoom the digital panel - it should stay fixed

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

def reset_view(self):
    """Reset the view to show all data with each panel scaled independently"""
    try:
        # Clear denoising flag
        self.denoising_applied = False

        # Remove any marked regions
        for patch in self.ax1.patches:
            if hasattr(patch, 'is_flattened_marker'):
                patch.remove()

        # Determine overall time range for both primary and secondary
        min_time = self.time[0] / 60
        max_time = self.time[-1] / 60

        if hasattr(self, 'secondary_time') and self.secondary_time is not None:
            min_time = min(min_time, self.secondary_time[0] / 60)
            max_time = max(max_time, self.secondary_time[-1] / 60)

        # Reset x-axis to show all data (shared across panels)
        for ax in [self.ax1, self.ax2, self.ax3]:
            if ax: ax.set_xlim(min_time, max_time)

        # Reset y-axis for each panel independently with proper baseline positioning
        self.update_panel_y_limits()

        # Reset y-axis for digital signals panel
        if hasattr(self, 'ax3') and self.ax3:
            self.ax3.set_ylim(-0.1, 1.1)

        # Redraw
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.draw_idle()

        print("View reset with independent panel scaling and improved baseline positioning")
    except Exception as e:
        print(f"Error in reset_view: {e}")
        import traceback
        traceback.print_exc()
