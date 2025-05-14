import tkinter as tk
import sys
import traceback
from gui import PhotometryViewer

def main():
    """Main function to start the application."""
    try:
        print("Starting application...")
        root = tk.Tk()
        
        # Get screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Set initial window size (80% of screen)
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)

        # Set minimum size to ensure controls are visible
        min_width = min(1200, screen_width - 100)
        min_height = min(800, screen_height - 100)

        root.minsize(min_width, min_height)
        root.geometry(f"{window_width}x{window_height}")

        # Center the window
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2
        root.geometry(f"+{x_position}+{y_position}")
        
        root.title("Photometry Signal Viewer")
        
        # Create viewer application
        app = PhotometryViewer(root)
        
        # Start main loop
        root.mainloop()
        
    except Exception as e:
        print(f"Error in main function: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
