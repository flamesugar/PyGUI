"""
数据处理包初始化文件。
提供对数据处理模块中各功能的便捷访问。
"""

# 文件I/O模块函数
from .file_io import (
    load_data, save_data, get_supported_formats,
    export_to_csv, import_from_csv,
    read_ppd_file, parse_ppd_data
)

# 信号处理模块函数
from .signal_processing import (
    apply_filter, normalize_signal, correct_baseline,
    smooth_signal, correct_drift, process_data, 
    fit_drift_curve, butter_filter, downsample_data, 
    center_signal, auto_align_baselines, 
    advanced_baseline_alignment, robust_baseline_alignment
)

# 峰值检测模块函数
from .peak_detection import (
    detect_peaks, get_peak_properties,
    set_peak_threshold, classify_peaks,
    find_peaks_valleys, calculate_peak_metrics, 
    calculate_valley_metrics, calculate_intervals
)

# 伪影处理模块函数
from .artifacts import (
    detect_artifacts, remove_artifacts, remove_artifacts_fast,
    evaluate_signal_quality, classify_artifacts,
    evaluate_control_channels
)

# 版本信息
__version__ = '1.0.0'
