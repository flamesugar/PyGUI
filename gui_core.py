import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, Menu, ttk
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib
import os
import sys
import threading
import queue
import traceback

# Import local modules
from data_processing import (
    read_ppd_file, parse_ppd_data, process_data, 
    fit_drift_curve, correct_drift, butter_filter, 
    downsample_data, center_signal, find_peaks_valleys, 
    calculate_peak_metrics, calculate_valley_metrics,
    detect_artifacts, remove_artifacts_fast
)
from plot_utils import create_checkbox_legend, update_panel_y_limits
from config import get_font_size_for_resolution
