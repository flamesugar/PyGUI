import tkinter as tk
from tkinter import Menu

def create_menu(self):
    """Create main menu bar with restructured categories"""
    menubar = Menu(self.root)
    self.root.config(menu=menubar)

    # 1. File menu
    file_menu = Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open Primary File...", command=self.open_file)
    file_menu.add_command(label="Open Secondary File...", command=self.open_secondary_file)
    file_menu.add_command(label="Clear Primary File", command=self.clear_primary_file)
    file_menu.add_command(label="Clear Secondary File", command=self.clear_secondary_file)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=self.on_closing)

    # 2. Signal menu
    signal_menu = Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Signal", menu=signal_menu)
    signal_menu.add_command(label="Filter Settings...", command=self.open_filter_window)
    signal_menu.add_command(label="Drift Correction...", command=self.open_drift_window)
    signal_menu.add_command(label="Manual Blanking", command=self.toggle_blanking_mode)
    signal_menu.add_command(label="Artifact Detection...", command=self.open_artifact_window)

    # 3. Viewer menu
    viewer_menu = Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Viewer", menu=viewer_menu)
    viewer_menu.add_command(label="Zoom In (X)", command=lambda: self.zoom_x(in_out="in"))
    viewer_menu.add_command(label="Zoom Out (X)", command=lambda: self.zoom_x(in_out="out"))
    viewer_menu.add_command(label="Zoom In (Y)", command=lambda: self.zoom_y_all("in"))
    viewer_menu.add_command(label="Zoom Out (Y)", command=lambda: self.zoom_y_all("out"))
    viewer_menu.add_command(label="Pan Left", command=lambda: self.pan_x(direction="left"))
    viewer_menu.add_command(label="Pan Right", command=lambda: self.pan_x(direction="right"))
    viewer_menu.add_command(label="Reset View", command=self.reset_view)
