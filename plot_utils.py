# plot_utils.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from typing import List, Tuple, Dict, Union, Optional, Any

# Define custom color palette for consistent plotting
FP_COLORS = {
    'signal': '#1f77b4',  # Blue
    'control': '#ff7f0e',  # Orange
    'delta_f': '#2ca02c',  # Green
    'events': '#d62728',   # Red
    'baseline': '#9467bd', # Purple
    'filtered': '#8c564b', # Brown
    'raw': '#7f7f7f',      # Gray
    'background': '#f0f0f0'  # Light gray
}

class PlotSettings:
    """Class to store and manage plot settings"""
    def __init__(self):
        self.line_width = 1.5
        self.marker_size = 3
        self.dpi = 100
        self.figure_size = (10, 6)
        self.grid = True
        self.title_fontsize = 14
        self.axis_fontsize = 12
        self.tick_fontsize = 10
        self.legend_fontsize = 10
        self.show_legend = True
        self.colors = FP_COLORS

    def apply_to_figure(self, fig: Figure) -> None:
        """Apply settings to a matplotlib figure"""
        for ax in fig.get_axes():
            ax.grid(self.grid)
            ax.title.set_fontsize(self.title_fontsize)
            ax.xaxis.label.set_fontsize(self.axis_fontsize)
            ax.yaxis.label.set_fontsize(self.axis_fontsize)
            ax.tick_params(axis='both', labelsize=self.tick_fontsize)
            if ax.get_legend() and self.show_legend:
                ax.legend(fontsize=self.legend_fontsize)


def create_figure(figsize: Optional[Tuple[float, float]] = None, 
                  dpi: Optional[int] = None) -> Tuple[Figure, Any]:
    """Create a new matplotlib figure and axis
    
    Args:
        figsize: Optional tuple of figure dimensions (width, height) in inches
        dpi: Optional dots per inch for the figure
        
    Returns:
        Tuple containing (figure, axis)
    """
    settings = PlotSettings()
    fig = plt.figure(figsize=figsize or settings.figure_size, 
                    dpi=dpi or settings.dpi)
    ax = fig.add_subplot(111)
    return fig, ax


def plot_time_series(ax, x_data: np.ndarray, y_data: np.ndarray, 
                    label: str = None, color: str = None, 
                    linewidth: float = None, alpha: float = 1.0,
                    show_peaks: bool = False, peak_indices: np.ndarray = None) -> None:
    """Plot time series data on a given axis
    
    Args:
        ax: Matplotlib axis to plot on
        x_data: Array of x values (time)
        y_data: Array of y values (signal)
        label: Label for legend
        color: Line color
        linewidth: Width of the plotted line
        alpha: Transparency (0.0 to 1.0)
        show_peaks: Whether to highlight detected peaks
        peak_indices: Indices of detected peaks in the data
    """
    settings = PlotSettings()
    
    # Plot the time series
    ax.plot(x_data, y_data, label=label, 
            color=color, 
            linewidth=linewidth or settings.line_width,
            alpha=alpha)
    
    # If requested, mark the peaks
    if show_peaks and peak_indices is not None and len(peak_indices) > 0:
        ax.scatter(x_data[peak_indices], y_data[peak_indices], 
                  color='red', s=settings.marker_size*2, zorder=5,
                  label='Peaks' if label else None)


def plot_signals_comparison(ax, time: np.ndarray, 
                          signals: Dict[str, np.ndarray], 
                          labels: Dict[str, str] = None,
                          colors: Dict[str, str] = None) -> None:
    """Plot multiple signals on the same axis for comparison
    
    Args:
        ax: Matplotlib axis to plot on
        time: Array of time points
        signals: Dictionary mapping signal names to arrays of y values
        labels: Optional dictionary mapping signal names to display labels
        colors: Optional dictionary mapping signal names to colors
    """
    settings = PlotSettings()
    colors = colors or settings.colors
    
    for signal_name, signal_data in signals.items():
        label = labels.get(signal_name, signal_name) if labels else signal_name
        color = colors.get(signal_name, None)
        plot_time_series(ax, time, signal_data, label=label, color=color)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Signal')
    if settings.show_legend:
        ax.legend()


def plot_heatmap(ax, data: np.ndarray, 
                x_labels: Optional[List] = None, 
                y_labels: Optional[List] = None,
                cmap: str = 'viridis', 
                title: str = None) -> None:
    """Create a heatmap visualization of 2D data
    
    Args:
        ax: Matplotlib axis to plot on
        data: 2D array of values to display
        x_labels: Optional list of x-axis labels
        y_labels: Optional list of y-axis labels 
        cmap: Colormap name
        title: Optional title for the plot
    """
    settings = PlotSettings()
    
    # Create the heatmap
    im = ax.imshow(data, aspect='auto', cmap=cmap)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set labels if provided
    if x_labels:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    
    if y_labels:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
    
    if title:
        ax.set_title(title, fontsize=settings.title_fontsize)


def plot_events(ax, event_times: np.ndarray, 
               y_min: float = None, y_max: float = None,
               color: str = 'red', alpha: float = 0.3,
               label: str = 'Events') -> None:
    """Add vertical lines or shaded regions to mark event times
    
    Args:
        ax: Matplotlib axis to plot on
        event_times: Array of event time points
        y_min, y_max: Min and max y values for the lines (default: axis limits)
        color: Color of event markers
        alpha: Transparency of event markers
        label: Label for legend
    """
    if y_min is None or y_max is None:
        y_min, y_max = ax.get_ylim()
    
    # Plot vertical lines for each event
    for event_time in event_times:
        ax.axvline(x=event_time, ymin=y_min, ymax=y_max, 
                  color=color, alpha=alpha, linestyle='--')
    
    # Add a proxy artist for the legend
    from matplotlib.lines import Line2D
    legend_handle = Line2D([0], [0], color=color, linestyle='--', label=label)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(legend_handle)
    labels.append(label)
    ax.legend(handles, labels)


def plot_errorbar(ax, x_data: np.ndarray, y_data: np.ndarray, 
                 yerr: np.ndarray, label: str = None,
                 color: str = None, alpha: float = 0.3) -> None:
    """Plot data with error bars or confidence intervals
    
    Args:
        ax: Matplotlib axis to plot on
        x_data: Array of x values
        y_data: Array of y values
        yerr: Array of error values or [lower_errors, upper_errors]
        label: Label for legend
        color: Line color
        alpha: Transparency of error region
    """
    settings = PlotSettings()
    color = color or settings.colors['signal']
    
    # Plot the main line
    ax.plot(x_data, y_data, label=label, color=color, 
           linewidth=settings.line_width)
    
    # Add error region
    ax.fill_between(x_data, y_data - yerr, y_data + yerr, 
                   color=color, alpha=alpha)


def create_event_aligned_plot(fig, time_window: Tuple[float, float], 
                             signals: Dict[str, np.ndarray],
                             event_times: np.ndarray,
                             sample_rate: float,
                             baseline_window: Optional[Tuple[float, float]] = None) -> None:
    """Create plot of signals aligned to events with averaging
    
    Args:
        fig: Matplotlib figure to plot on
        time_window: Tuple of (before_event, after_event) in seconds
        signals: Dictionary mapping signal names to arrays of signal data
        event_times: Array of event times in seconds
        sample_rate: Sampling rate in Hz
        baseline_window: Optional window for baseline normalization
    """
    settings = PlotSettings()
    
    # Calculate window size in samples
    window_samples = (int(abs(time_window[0]) * sample_rate),
                     int(time_window[1] * sample_rate))
    total_window = sum(window_samples)
    
    # Create time axis for the window
    window_time = np.linspace(time_window[0], time_window[1], total_window)
    
    # Initialize the plot
    n_signals = len(signals)
    axes = fig.subplots(n_signals, 1, sharex=True)
    if n_signals == 1:
        axes = [axes]
    
    # Process each signal
    for ax_idx, (signal_name, signal_data) in enumerate(signals.items()):
        # Convert event times to sample indices
        event_indices = (np.array(event_times) * sample_rate).astype(int)
        
        # Extract signal segments around each event
        segments = []
        for event_idx in event_indices:
            if event_idx - window_samples[0] >= 0 and event_idx + window_samples[1] < len(signal_data):
                segment = signal_data[event_idx - window_samples[0]:event_idx + window_samples[1]]
                
                # Baseline correction if window specified
                if baseline_window is not None:
                    baseline_start = int((baseline_window[0] - time_window[0]) * sample_rate)
                    baseline_end = int((baseline_window[1] - time_window[0]) * sample_rate)
                    baseline = np.mean(segment[baseline_start:baseline_end])
                    segment = segment - baseline
                
                segments.append(segment)
        
        # Calculate mean and std error
        segments_array = np.vstack(segments)
        mean_signal = np.mean(segments_array, axis=0)
        sem_signal = np.std(segments_array, axis=0) / np.sqrt(len(segments))
        
        # Plot mean with error band
        plot_errorbar(axes[ax_idx], window_time, mean_signal, sem_signal, 
                     label=signal_name, 
                     color=settings.colors.get(signal_name, None))
        
        # Add a vertical line at event onset
        axes[ax_idx].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Set labels
        axes[ax_idx].set_ylabel(signal_name)
        if ax_idx == n_signals - 1:
            axes[ax_idx].set_xlabel('Time from event (s)')
        
        axes[ax_idx].set_title(f'{signal_name} aligned to events (n={len(segments)})')
    
    fig.tight_layout()


def save_figure(fig: Figure, filename: str, dpi: int = None) -> None:
    """Save a matplotlib figure to file
    
    Args:
        fig: Figure to save
        filename: Output filename (extension determines format)
        dpi: Resolution in dots per inch
    """
    settings = PlotSettings()
    fig.savefig(filename, dpi=dpi or settings.dpi, bbox_inches='tight')


def prepare_figure_for_gui(fig: Figure) -> FigureCanvasWxAgg:
    """Prepare a matplotlib figure for embedding in wxPython GUI
    
    Args:
        fig: Matplotlib figure to embed
        
    Returns:
        Canvas object for the wx GUI
    """
    canvas = FigureCanvasWxAgg(None, -1, fig)
    return canvas


def apply_default_styling() -> None:
    """Apply default matplotlib styling for consistent plots"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']


def create_checkbox_legend(ax):
    """Create a custom legend with checkboxes for the given axis
    
    Args:
        ax: Matplotlib axis to create legend for
        
    Returns:
        Legend object
    """
    # Get all lines and their labels from the axis
    lines = [line for line in ax.get_lines() if line.get_label()[0] != '_']
    labels = [line.get_label() for line in lines]
    
    # Create legend
    leg = ax.legend(loc='upper right')
    
    # Make legend draggable
    leg.set_draggable(True)
    
    return leg


def update_panel_y_limits(ax, signal_data: np.ndarray):
    """Set appropriate y-axis limits for a panel
    
    Args:
        ax: Matplotlib axis to update
        signal_data: Signal data to use for limit calculation
    """
    if signal_data is None or len(signal_data) == 0:
        return
    
    # Get percentiles for more robust min/max estimates
    p_min = np.percentile(signal_data, 1)  # 1st percentile
    p_max = np.percentile(signal_data, 99)  # 99th percentile
    
    # Calculate range and add some padding
    data_range = p_max - p_min
    
    # Position the baseline at 10% from bottom
    desired_min = p_min - (data_range * 0.1)
    
    # Add extra room on top for peaks
    desired_max = p_max + (data_range * 0.2)
    
    # Set the limits
    ax.set_ylim(desired_min, desired_max)
