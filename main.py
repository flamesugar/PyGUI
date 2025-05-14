import tkinter as tk
import sys
import os
import traceback

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # 直接从 gui_core 导入 PhotometryViewer
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
            
            # 创建 PhotometryViewer 实例
            app = PhotometryViewer(root)
            
            # 调用 create_gui 方法初始化界面
            app.create_gui()
            
            # Start main loop
            root.mainloop()
            
        except Exception as e:
            print(f"Error in main function: {e}")
            traceback.print_exc()
            sys.exit(1)

    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"导入错误: {e}")
    print("Python 路径:", sys.path)
    print("当前目录:", os.getcwd())
    print("当前目录文件:", os.listdir(current_dir))
    try:
        print("gui 目录文件:", os.listdir(os.path.join(current_dir, 'gui')))
        # 检查 gui_core.py 文件内容
        print("\n检查 gui_core.py 的前20行内容:")
        with open(os.path.join(current_dir, 'gui', 'gui_core.py'), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:20]):
                print(f"{i+1}: {line.strip()}")
    except Exception as err:
        print(f"检查文件时出错: {err}")
    traceback.print_exc()

    # 设置当前目录为搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 检查文件内容
try:
    print("\n检查 photometry_viewer.py 的内容:")
    with open(os.path.join(current_dir, 'gui', 'photometry_viewer.py'), 'r') as f:
        content = f.read()
        print(content[:500] + "..." if len(content) > 500 else content)
    
    print("\n检查 __init__.py 的内容:")
    with open(os.path.join(current_dir, 'gui', '__init__.py'), 'r') as f:
        print(f.read())
    
    # 尝试直接导入
    print("\n尝试各种导入方式:")
    try:
        from RC.photometry_viewer import PhotometryViewer as PV1
        print("成功从 gui.photometry_viewer 导入 PhotometryViewer")
    except ImportError as e:
        print(f"从 photometry_viewer 导入失败: {e}")
        
    try:
        import RC.photometry_viewer as module
        print(f"成功导入 gui.photometry_viewer 模块，内容: {dir(module)}")
    except ImportError as e:
        print(f"导入 gui.photometry_viewer 模块失败: {e}")
    
except Exception as e:
    print(f"检查文件时出错: {e}")
