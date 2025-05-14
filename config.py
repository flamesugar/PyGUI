# config.py
"""
全局配置和默认参数
"""

# 数据处理默认参数
DEFAULT_LOW_CUTOFF = 0.001
DEFAULT_HIGH_CUTOFF = 1.0
DEFAULT_DOWNSAMPLE_FACTOR = 50
DEFAULT_ARTIFACT_THRESHOLD = 3.0
DEFAULT_DRIFT_CORRECTION = True
DEFAULT_DRIFT_DEGREE = 2
DEFAULT_EDGE_PROTECTION = True

# 峰值检测默认参数
DEFAULT_PEAK_PROMINENCE = 10.0
DEFAULT_PEAK_WIDTH = 2.0
DEFAULT_PEAK_DISTANCE = 5.0
DEFAULT_PEAK_THRESHOLD = 5.0

# 谷值检测默认参数
DEFAULT_VALLEY_PROMINENCE = 8.0
DEFAULT_VALLEY_WIDTH = 2.0
DEFAULT_VALLEY_DISTANCE = 5.0

# PSTH分析默认参数
DEFAULT_PSTH_BEFORE = 60.0
DEFAULT_PSTH_AFTER = 60.0
DEFAULT_SHOW_SEM = True
DEFAULT_SHOW_INDIVIDUAL = False

# 去噪默认参数
DEFAULT_AGGRESSIVE_MODE = True
DEFAULT_REMOVE_GAPS = True
DEFAULT_MAX_GAP = 1.0
DEFAULT_USE_CONTROL = True

# 基线对齐默认参数
DEFAULT_ALIGNMENT_MODE = "simple"
DEFAULT_ALIGNMENT_SENSITIVITY = 1.0
DEFAULT_ALIGNMENT_SMOOTHNESS = 0.5

# UI颜色方案
COLORS = {
    'primary_signal': 'green',
    'secondary_signal': 'red',
    'primary_raw': 'blue',
    'secondary_raw': 'cyan',
    'primary_isosbestic': 'magenta',
    'secondary_isosbestic': 'yellow',
    'primary_drift': 'red',
    'secondary_drift': 'orange',
    'primary_peaks': 'red',
    'secondary_peaks': 'orange',
    'primary_valleys': 'blue',
    'secondary_valleys': 'purple',
    'artifact': 'red',
    'background': 'lightgray'
}

# 按钮颜色
BUTTON_COLORS = {
    'normal': 'lightgray',
    'primary': 'lightblue',
    'success': 'lightgreen',
    'warning': 'lightyellow',
    'danger': 'salmon',
    'reset': 'pink'
}

def get_font_size_for_resolution():
    """计算适合当前屏幕分辨率的字体大小"""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # 隐藏临时窗口

        # 获取屏幕宽度
        screen_width = root.winfo_screenwidth()

        # 根据屏幕宽度确定基础字体大小
        if screen_width >= 3840:  # 4K或更高
            base_size = 16
            button_size = 14
            title_size = 18
        elif screen_width >= 2560:  # 1440p
            base_size = 14
            button_size = 12
            title_size = 16
        elif screen_width >= 1920:  # 1080p
            base_size = 12
            button_size = 10
            title_size = 14
        else:  # 更低分辨率
            base_size = 10
            button_size = 9
            title_size = 12

        root.destroy()

        return {
            'base': base_size,
            'button': button_size,
            'title': title_size,
            'slider': int(base_size * 0.9)
        }
    except:
        # 如果出错则返回默认大小
        return {
            'base': 12,
            'button': 10,
            'title': 14,
            'slider': 11
        }
