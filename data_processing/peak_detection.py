"""
峰值检测模块。
提供各种峰值检测和特征提取算法，用于分析光度测量数据中的信号事件。
"""

import numpy as np
from scipy import signal
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Union, Optional, Any
import warnings

def detect_peaks(signal_data: np.ndarray, method: str = 'height', 
                **kwargs) -> Dict[str, np.ndarray]:
    """
    检测信号中的峰值。

    Args:
        signal_data (np.ndarray): 输入信号数据
        method (str): 峰值检测方法，可选 'height', 'prominence', 'width', 'wavelet', 'threshold', 'adaptive'
        **kwargs: 特定检测方法的其他参数

    Returns:
        Dict[str, np.ndarray]: 包含峰值索引和属性的字典
    """
    # 检查输入
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    
    # 根据选择的方法应用不同的峰值检测算法
    if method.lower() == 'height':
        return _detect_peaks_height(signal_data, **kwargs)
    elif method.lower() == 'prominence':
        return _detect_peaks_prominence(signal_data, **kwargs)
    elif method.lower() == 'width':
        return _detect_peaks_width(signal_data, **kwargs)
    elif method.lower() == 'wavelet':
        return _detect_peaks_wavelet(signal_data, **kwargs)
    elif method.lower() == 'threshold':
        return _detect_peaks_threshold(signal_data, **kwargs)
    elif method.lower() == 'adaptive':
        return _detect_peaks_adaptive(signal_data, **kwargs)
    else:
        raise ValueError(f"不支持的峰值检测方法: {method}")

def find_peaks_valleys(signal_data: np.ndarray, time: np.ndarray, 
                     prominence: float = 1.0, width: float = None, 
                     distance: float = None, threshold: float = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    使用scipy的find_peaks函数查找信号中的峰值和谷值。

    Args:
        signal_data (np.ndarray): 要分析的信号
        time (np.ndarray): 对应的时间点
        prominence (float): 最小峰值显著性（到相邻谷值的垂直距离）
        width (float or None): 最小峰值宽度
        distance (float or None): 峰值之间的最小水平距离（以秒为单位）
        threshold (float or None): 峰值的最小绝对高度

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: 包含峰值和谷值信息的字典元组
    """
    # 查找峰值
    peak_indices, peak_props = find_peaks(
        signal_data,
        prominence=prominence,
        width=width,
        distance=distance,
        height=threshold
    )

    # 通过反转信号查找谷值
    valley_indices, valley_props = find_peaks(
        -signal_data,
        prominence=prominence,
        width=width,
        distance=distance,
        height=None if threshold is None else -threshold
    )

    # 计算额外的峰值指标
    peak_heights = signal_data[peak_indices]
    peak_times = time[peak_indices]

    # 计算谷值指标
    valley_depths = signal_data[valley_indices]
    valley_times = time[valley_indices]

    # 转换谷值的显著性值（它们是在反转的信号上计算的）
    if 'prominences' in valley_props:
        valley_props['prominences'] = valley_props['prominences'].copy()

    return {
        'indices': peak_indices,
        'times': peak_times,
        'heights': peak_heights,
        'properties': peak_props
    }, {
        'indices': valley_indices,
        'times': valley_times,
        'depths': valley_depths,
        'properties': valley_props
    }

def calculate_peak_metrics(peak_data: Dict[str, Any], valley_data: Dict[str, Any], 
                          signal_data: np.ndarray, time: np.ndarray) -> Dict[str, List[float]]:
    """
    计算检测到的峰值的附加指标。

    Args:
        peak_data (Dict[str, Any]): 包含峰值信息的字典
        valley_data (Dict[str, Any]): 包含谷值信息的字典
        signal_data (np.ndarray): 信号数组
        time (np.ndarray): 时间数组

    Returns:
        Dict[str, List[float]]: 包含附加峰值指标的字典
    """
    # 跟踪执行的调试信息
    print(f"Calculating metrics for {len(peak_data['indices'])} peaks...")

    # 安全检查
    if peak_data is None or valley_data is None:
        print("Peak metrics: Missing peak or valley data")
        return None

    peak_indices = peak_data['indices']
    valley_indices = valley_data['indices']

    if len(peak_indices) == 0:
        print("Peak metrics: No peaks found")
        return None
    if len(valley_indices) == 0:
        print("Peak metrics: No valleys found")
        return None

    # 初始化指标字典
    metrics = {
        'area_under_curve': [],
        'full_width_half_max': [],
        'rise_time': [],
        'decay_time': [],
        'preceding_valley': [],
        'following_valley': []
    }

    # 对于每个峰值，找到其前后最近的谷值
    for peak_idx in peak_indices:
        try:
            # 查找前面的谷值
            preceding_valleys = valley_indices[valley_indices < peak_idx]
            preceding_valley_idx = preceding_valleys[-1] if len(preceding_valleys) > 0 else 0

            # 查找后面的谷值
            following_valleys = valley_indices[valley_indices > peak_idx]
            following_valley_idx = following_valleys[0] if len(following_valleys) > 0 else len(signal_data) - 1

            # 存储谷值索引
            metrics['preceding_valley'].append(preceding_valley_idx)
            metrics['following_valley'].append(following_valley_idx)

            # 计算峰值半高宽
            peak_height = signal_data[peak_idx]
            base_level = min(signal_data[preceding_valley_idx], signal_data[following_valley_idx])
            half_height = base_level + (peak_height - base_level) / 2

            # 查找交叉点
            left_segment = signal_data[preceding_valley_idx:peak_idx + 1]
            right_segment = signal_data[peak_idx:following_valley_idx + 1]

            # 默认值（计算失败时使用）
            fwhm = 0.0
            area = 0.0
            rise_time = 0.0
            decay_time = 0.0

            # 查找信号跨越半高的位置
            left_crosses = np.where(left_segment >= half_height)[0]
            right_crosses = np.where(right_segment <= half_height)[0]

            if len(left_crosses) > 0 and len(right_crosses) > 0:
                left_cross_idx = preceding_valley_idx + left_crosses[0]
                right_cross_idx = peak_idx + right_crosses[-1] if len(right_crosses) > 0 else following_valley_idx

                # 计算半高全宽（以秒为单位）
                fwhm = time[right_cross_idx] - time[left_cross_idx]

                # 计算上升和下降时间
                rise_time = time[peak_idx] - time[left_cross_idx]
                decay_time = time[right_cross_idx] - time[peak_idx]

                # 计算曲线下面积（简单的梯形积分）
                peak_segment = signal_data[preceding_valley_idx:following_valley_idx + 1]
                peak_times = time[preceding_valley_idx:following_valley_idx + 1]

                # 使用trapezoid替代已弃用的trapz
                try:
                    area = trapezoid(peak_segment - base_level, peak_times)
                except ImportError:
                    # 如果scipy.integrate.trapezoid不可用，回退到numpy trapz
                    area = np.trapz(peak_segment - base_level, peak_times)

                # 确保面积为正
                area = max(0, area)

            # 添加计算的指标
            metrics['full_width_half_max'].append(fwhm)
            metrics['rise_time'].append(rise_time)
            metrics['decay_time'].append(decay_time)
            metrics['area_under_curve'].append(area)

        except Exception as e:
            print(f"Error calculating metrics for peak at index {peak_idx}: {e}")
            # 计算失败时添加默认值
            metrics['full_width_half_max'].append(0.0)
            metrics['rise_time'].append(0.0)
            metrics['decay_time'].append(0.0)
            metrics['area_under_curve'].append(0.0)
            # 确保即使计算失败也会添加谷值索引
            if 'preceding_valley' not in metrics:
                metrics['preceding_valley'].append(0)
            if 'following_valley' not in metrics:
                metrics['following_valley'].append(0)

    # 调试输出
    print(f"Finished peak metrics calculation. Example width: {metrics['full_width_half_max'][:5]}")
    return metrics

def calculate_intervals(times: np.ndarray) -> Dict[str, Any]:
    """
    计算连续事件（峰值或谷值）之间的间隔。

    Args:
        times (np.ndarray): 事件时间点数组

    Returns:
        Dict[str, Any]: 包含间隔统计信息的字典
    """
    if times is None or len(times) < 2:
        return None

    # 对时间进行排序以确保时间顺序
    sorted_times = np.sort(times)

    # 计算连续事件之间的间隔
    intervals = np.diff(sorted_times)

    # 基本统计信息
    stats = {
        'mean': np.mean(intervals),
        'median': np.median(intervals),
        'std': np.std(intervals),
        'min': np.min(intervals),
        'max': np.max(intervals),
        'count': len(intervals),
        'intervals': intervals
    }

    return stats

def calculate_valley_metrics(peak_data: Dict[str, Any], valley_data: Dict[str, Any], 
                            signal_data: np.ndarray, time: np.ndarray) -> Dict[str, List[float]]:
    """
    计算检测到的谷值的指标（类似于峰值）。

    Args:
        peak_data (Dict[str, Any]): 包含峰值信息的字典
        valley_data (Dict[str, Any]): 包含谷值信息的字典
        signal_data (np.ndarray): 信号数组
        time (np.ndarray): 时间数组

    Returns:
        Dict[str, List[float]]: 包含谷值指标的字典
    """
    # --- 添加检查 ---
    if valley_data is None or not valley_data.get('indices', np.array([])).size:
        print("Valley metrics: No valleys found.")
        return None  # 没有谷值可以计算指标
    if peak_data is None or not peak_data.get('indices', np.array([])).size:
        print("Valley metrics: No peaks found (needed for context). Returning None.")
        return None  # 需要峰值作为上下文
    # --- 检查结束 ---

    valley_indices = valley_data['indices']
    peak_indices = peak_data['indices']

    metrics = {
        'area_above_valley': [],
        'full_width_half_depth': [],  # 相对于峰值的宽度
        'time_to_preceding_peak': [],
        'time_to_following_peak': [],
        'preceding_peak_idx': [],
        'following_peak_idx': []
    }

    print(f"Calculating metrics for {len(valley_indices)} valleys...")  # 调试打印

    for i, valley_idx in enumerate(valley_indices):
        # 用NaN初始化此谷值的指标
        fwhd_val = np.nan
        area_val = np.nan
        time_to_prec = np.nan
        time_to_foll = np.nan
        prec_peak_idx = np.nan  # 如果未找到，则使用NaN而不是0作为索引
        foll_peak_idx = np.nan  # 使用NaN而不是len-1

        # 查找周围的峰值
        preceding_peaks_indices = np.where(peak_indices < valley_idx)[0]
        following_peaks_indices = np.where(peak_indices > valley_idx)[0]

        valid_prec_peak = False
        if preceding_peaks_indices.size > 0:
            prec_peak_idx = peak_indices[preceding_peaks_indices[-1]]
            time_to_prec = time[valley_idx] - time[prec_peak_idx]
            valid_prec_peak = True

        valid_foll_peak = False
        if following_peaks_indices.size > 0:
            foll_peak_idx = peak_indices[following_peaks_indices[0]]
            time_to_foll = time[foll_peak_idx] - time[valley_idx]
            valid_foll_peak = True

        metrics['preceding_peak_idx'].append(prec_peak_idx)
        metrics['following_peak_idx'].append(foll_peak_idx)
        metrics['time_to_preceding_peak'].append(time_to_prec)
        metrics['time_to_following_peak'].append(time_to_foll)

        # 仅当被有效峰值包围时才计算宽度/面积
        if valid_prec_peak and valid_foll_peak:
            try:
                valley_depth = signal_data[valley_idx]
                # 在访问信号之前确保索引有效
                prec_peak_height = signal_data[int(prec_peak_idx)]
                foll_peak_height = signal_data[int(foll_peak_idx)]

                peak_level = np.mean([prec_peak_height, foll_peak_height])
                half_depth_level = valley_depth + (peak_level - valley_depth) / 2

                # 在半深度处查找宽度
                left_segment_indices = np.arange(int(prec_peak_idx), valley_idx + 1)
                right_segment_indices = np.arange(valley_idx, int(foll_peak_idx) + 1)

                if left_segment_indices.size > 1 and right_segment_indices.size > 1:
                    left_segment = signal_data[left_segment_indices]
                    right_segment = signal_data[right_segment_indices]
                    time_left = time[left_segment_indices]
                    time_right = time[right_segment_indices]

                    # 插值找到精确交叉时间
                    interp_left = interp1d(time_left, left_segment - half_depth_level, kind='linear',
                                         bounds_error=False, fill_value=np.nan)
                    interp_right = interp1d(time_right, right_segment - half_depth_level, kind='linear',
                                          bounds_error=False, fill_value=np.nan)

                    # 使用更简单的基于索引的交叉方法
                    left_crosses = np.where(left_segment <= half_depth_level)[0]
                    right_crosses = np.where(right_segment <= half_depth_level)[0]

                    if left_crosses.size > 0 and right_crosses.size > 0:
                        left_cross_idx_rel = left_crosses[-1]  # 下降途中最后一个低于半深度的点
                        right_cross_idx_rel = right_crosses[0]  # 上升途中第一个低于半深度的点

                        left_cross_idx_abs = int(prec_peak_idx) + left_cross_idx_rel
                        right_cross_idx_abs = valley_idx + right_cross_idx_rel

                        fwhd_val = time[right_cross_idx_abs] - time[left_cross_idx_abs]

                # 计算谷值上方到峰值水平的面积
                segment_indices = np.arange(int(prec_peak_idx), int(foll_peak_idx) + 1)
                if segment_indices.size > 1:
                    segment_signal = signal_data[segment_indices]
                    segment_time = time[segment_indices]
                    area_val = trapezoid(np.maximum(0, peak_level - segment_signal), segment_time)

            except IndexError as ie:
                print(f"IndexError calculating metrics for valley index {valley_idx}: {ie}")
            except Exception as e:
                print(f"Error calculating metrics for valley index {valley_idx}: {e}")

        # 添加计算（或NaN）值
        metrics['full_width_half_depth'].append(fwhd_val)
        metrics['area_above_valley'].append(area_val)

    print(f"Finished valley metrics calculation. Example width: {metrics['full_width_half_depth'][:5]}")  # 调试打印
    return metrics

def get_peak_properties(peak_data: Dict[str, np.ndarray], 
                       signal_data: np.ndarray, time_data: np.ndarray = None,
                       include_valleys: bool = True) -> Dict[str, Any]:
    """
    获取峰值的详细属性。

    Args:
        peak_data (Dict[str, np.ndarray]): 由detect_peaks函数返回的峰值数据
        signal_data (np.ndarray): 原始信号数据
        time_data (np.ndarray, optional): 对应的时间数据，如果为None则使用样本索引
        include_valleys (bool): 是否也检测和包含谷值信息

    Returns:
        Dict[str, Any]: 包含峰值属性的扩展字典
    """
    if peak_data is None or 'peak_indices' not in peak_data:
        return None
    
    # 准备结果字典
    result = peak_data.copy()
    
    # 如果没有提供时间数据，则使用样本索引
    if time_data is None:
        time_data = np.arange(len(signal_data))
    
    # 提取峰值索引
    peak_indices = peak_data['peak_indices']
    
    # 如果需要谷值信息
    if include_valleys:
        # 使用峰值的显著性和宽度参数设置寻找谷值
        prominence = 0.2 * np.median(peak_data.get('prominences', [1.0]))
        
        # 查找谷值（反转信号）
        valley_indices, valley_props = find_peaks(-signal_data, prominence=prominence)
        
        # 添加谷值信息
        result['valley_indices'] = valley_indices
        result['valley_heights'] = signal_data[valley_indices]
        result['valley_prominences'] = valley_props.get('prominences', [])
        
        # 找到每个峰值前后的谷值
        preceding_valleys = []
        following_valleys = []
        
        for peak_idx in peak_indices:
            # 查找峰值前面的谷值
            prev_valleys = valley_indices[valley_indices < peak_idx]
            prev_valley = prev_valleys[-1] if len(prev_valleys) > 0 else None
            preceding_valleys.append(prev_valley)
            
            # 查找峰值后面的谷值
            next_valleys = valley_indices[valley_indices > peak_idx]
            next_valley = next_valleys[0] if len(next_valleys) > 0 else None
            following_valleys.append(next_valley)
        
        result['preceding_valleys'] = np.array(preceding_valleys)
        result['following_valleys'] = np.array(following_valleys)
    
    # 计算峰值时间和宽度（以时间单位）
    result['peak_times'] = time_data[peak_indices]
    
    if 'widths' in peak_data:
        # 将样本宽度转换为时间宽度
        sample_rate = 1.0
        if len(time_data) > 1:
            sample_rate = 1.0 / np.mean(np.diff(time_data))
        
        result['width_times'] = peak_data['widths'] / sample_rate
    
    # 添加峰值形状信息
    if len(peak_indices) > 0:
        # 计算每个峰值的斜率（上升沿和下降沿）
        slopes = []
        for i, peak_idx in enumerate(peak_indices):
            if peak_idx > 0 and peak_idx < len(signal_data) - 1:
                # 计算上升沿和下降沿的平均斜率
                prev_point = peak_idx - 1
                next_point = peak_idx + 1
                
                # 如果有谷值信息，使用谷值
                if include_valleys and preceding_valleys[i] is not None:
                    prev_point = preceding_valleys[i]
                if include_valleys and following_valleys[i] is not None:
                    next_point = following_valleys[i]
                
                # 计算斜率
                rising_slope = (signal_data[peak_idx] - signal_data[prev_point]) / (time_data[peak_idx] - time_data[prev_point])
                falling_slope = (signal_data[peak_idx] - signal_data[next_point]) / (time_data[peak_idx] - time_data[next_point])
                
                slopes.append((rising_slope, falling_slope))
            else:
                slopes.append((0, 0))
        
        result['slopes'] = np.array(slopes)
    
    return result

def set_peak_threshold(signal_data: np.ndarray, threshold_method: str = 'std', 
                      factor: float = 3.0, **kwargs) -> float:
    """
    为峰值检测设置自动阈值。

    Args:
        signal_data (np.ndarray): 输入信号数据
        threshold_method (str): 阈值方法，可选 'std', 'mad', 'percentile', 'otsu'
        factor (float): 阈值因子
        **kwargs: 特定方法的其他参数

    Returns:
        float: 计算的阈值值
    """
    # 检查输入
    if not isinstance(signal_data, np.ndarray):
        signal_data = np.array(signal_data)
    
    # 应用不同的阈值方法
    if threshold_method.lower() == 'std':
        # 基于标准差的阈值
        threshold = np.mean(signal_data) + factor * np.std(signal_data)
    
    elif threshold_method.lower() == 'mad':
        # 基于中位数绝对偏差的阈值（更稳健）
        median = np.median(signal_data)
        mad = np.median(np.abs(signal_data - median))
        threshold = median + factor * mad * 1.4826  # 标准化因子
    
    elif threshold_method.lower() == 'percentile':
        # 基于百分位数的阈值
        percentile = kwargs.get('percentile', 95)
        threshold = np.percentile(signal_data, percentile)
    
    elif threshold_method.lower() == 'otsu':
        # Otsu的二值化方法（通常用于图像处理）
        try:
            from skimage.filters import threshold_otsu
            threshold = threshold_otsu(signal_data)
        except ImportError:
            print("Otsu方法需要scikit-image库。回退到标准差方法。")
            threshold = np.mean(signal_data) + factor * np.std(signal_data)
    
    else:
        raise ValueError(f"不支持的阈值方法: {threshold_method}")
    
    return threshold

def classify_peaks(peak_data: Dict[str, np.ndarray], 
                  method: str = 'kmeans', n_classes: int = 2, 
                  features: List[str] = None, **kwargs) -> np.ndarray:
    """
    根据其属性对峰值进行分类。

    Args:
        peak_data (Dict[str, np.ndarray]): 峰值数据字典
        method (str): 分类方法，可选 'kmeans', 'threshold', 'dbscan', 'gmm'
        n_classes (int): 分类数量（对于kmeans和gmm）
        features (List[str]): 用于分类的特征列表
        **kwargs: 特定分类器的其他参数

    Returns:
        np.ndarray: 峰值类别标签
    """
    # 检查有效的峰值数据
    if peak_data is None or 'peak_indices' not in peak_data:
        return np.array([])
    
    # 提取峰值特征
    if features is None:
        # 默认使用峰值高度和显著性
        features = ['peak_heights', 'prominences']
    
    # 准备特征矩阵
    feature_matrix = []
    for feature in features:
        if feature in peak_data and len(peak_data[feature]) > 0:
            # 归一化特征
            feat_vals = peak_data[feature]
            if np.std(feat_vals) > 0:
                feat_vals = (feat_vals - np.mean(feat_vals)) / np.std(feat_vals)
            feature_matrix.append(feat_vals)
    
    # 检查是否有足够的特征
    if not feature_matrix:
        return np.array([])
    
    # 转置特征矩阵为(n_samples, n_features)
    X = np.column_stack(feature_matrix)
    
    # 应用选择的分类方法
    if method.lower() == 'kmeans':
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_classes, random_state=0, **kwargs)
            labels = kmeans.fit_predict(X)
        except ImportError:
            print("KMeans方法需要scikit-learn库。回退到简单阈值方法。")
            labels = _classify_with_threshold(X, n_classes)
    
    elif method.lower() == 'threshold':
        labels = _classify_with_threshold(X, n_classes)
    
    elif method.lower() == 'dbscan':
        try:
            from sklearn.cluster import DBSCAN
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
        except ImportError:
            print("DBSCAN方法需要scikit-learn库。回退到简单阈值方法。")
            labels = _classify_with_threshold(X, n_classes)
    
    elif method.lower() == 'gmm':
        try:
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=n_classes, random_state=0, **kwargs)
            labels = gmm.fit_predict(X)
        except ImportError:
            print("GMM方法需要scikit-learn库。回退到简单阈值方法。")
            labels = _classify_with_threshold(X, n_classes)
    
    else:
        raise ValueError(f"不支持的分类方法: {method}")
    
    return labels

def _classify_with_threshold(X: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """使用简单阈值的基本分类方法"""
    # 使用第一个特征（通常是峰值高度）
    feature = X[:, 0]
    
    # 计算分位数阈值
    thresholds = [np.percentile(feature, 100 * i / n_classes) for i in range(1, n_classes)]
    
    # 分类
    labels = np.zeros(len(feature), dtype=int)
    for i, threshold in enumerate(thresholds, 1):
        labels[feature >= threshold] = i
    
    return labels

def _detect_peaks_height(signal_data: np.ndarray, height: float = None, 
                        distance: int = 1, **kwargs) -> Dict[str, np.ndarray]:
    """基于高度的峰值检测"""
    # 如果没有指定高度阈值，使用信号平均值加上标准差
    if height is None:
        height = np.mean(signal_data) + np.std(signal_data)
    
    # 使用scipy的find_peaks函数
    peaks, properties = find_peaks(signal_data, height=height, distance=distance, **kwargs)
    
    # 计算其他峰值属性
    if len(peaks) > 0:
        prominences = peak_prominences(signal_data, peaks)[0]
        widths, width_heights, left_ips, right_ips = peak_widths(signal_data, peaks, rel_height=0.5)
    else:
        prominences = np.array([])
        widths = np.array([])
        left_ips = np.array([])
        right_ips = np.array([])
    
    # 构建结果字典
    result = {
        'peak_indices': peaks,
        'peak_heights': signal_data[peaks] if len(peaks) > 0 else np.array([]),
        'prominences': prominences,
        'widths': widths,
        'left_bases': left_ips.astype(int) if len(left_ips) > 0 else np.array([], dtype=int),
        'right_bases': right_ips.astype(int) if len(right_ips) > 0 else np.array([], dtype=int)
    }
    
    return result

def _detect_peaks_prominence(signal_data: np.ndarray, prominence: float = None, 
                           distance: int = 1, **kwargs) -> Dict[str, np.ndarray]:
    """基于显著性的峰值检测"""
    # 如果没有指定显著性阈值，使用信号标准差
    if prominence is None:
        prominence = np.std(signal_data)
    
    # 使用scipy的find_peaks函数
    peaks, properties = find_peaks(signal_data, prominence=prominence, distance=distance, **kwargs)
    
    # 获取显著性值
    if len(peaks) > 0:
        prominences = properties['prominences']
        widths, width_heights, left_ips, right_ips = peak_widths(signal_data, peaks, rel_height=0.5)
    else:
        prominences = np.array([])
        widths = np.array([])
        left_ips = np.array([])
        right_ips = np.array([])
    
    # 构建结果字典
    result = {
        'peak_indices': peaks,
        'peak_heights': signal_data[peaks] if len(peaks) > 0 else np.array([]),
        'prominences': prominences,
        'widths': widths,
        'left_bases': left_ips.astype(int) if len(left_ips) > 0 else np.array([], dtype=int),
        'right_bases': right_ips.astype(int) if len(right_ips) > 0 else np.array([], dtype=int)
    }
    
    return result

def _detect_peaks_width(signal_data: np.ndarray, width: float = None, 
                       prominence: float = 0.1, **kwargs) -> Dict[str, np.ndarray]:
    """基于宽度的峰值检测"""
    # 如果没有指定宽度阈值，使用信号长度的1%
    if width is None:
        width = max(int(len(signal_data) * 0.01), 1)
    
    # 使用scipy的find_peaks函数
    peaks, properties = find_peaks(signal_data, width=width, prominence=prominence, **kwargs)
    
    # 计算峰值属性
    if len(peaks) > 0:
        prominences = peak_prominences(signal_data, peaks)[0]
        widths, width_heights, left_ips, right_ips = peak_widths(signal_data, peaks, rel_height=0.5)
    else:
        prominences = np.array([])
        widths = np.array([])
        left_ips = np.array([])
        right_ips = np.array([])
    
    # 构建结果字典
    result = {
        'peak_indices': peaks,
        'peak_heights': signal_data[peaks] if len(peaks) > 0 else np.array([]),
        'prominences': prominences,
        'widths': widths,
        'left_bases': left_ips.astype(int) if len(left_ips) > 0 else np.array([], dtype=int),
        'right_bases': right_ips.astype(int) if len(right_ips) > 0 else np.array([], dtype=int)
    }
    
    return result

def _detect_peaks_wavelet(signal_data: np.ndarray, wavelet: str = 'ricker', 
                         widths: Union[int, List[int]] = 5, min_snr: float = 3.0, 
                         **kwargs) -> Dict[str, np.ndarray]:
    """使用小波变换的峰值检测"""
    try:
        from scipy.signal import find_peaks_cwt
    except ImportError:
        raise ImportError("此方法需要scipy.signal.find_peaks_cwt")
    
    # 处理widths参数
    if isinstance(widths, int):
        widths = np.arange(1, widths + 1)
    
    # 使用连续小波变换查找峰值
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # 忽略库函数中的警告
        peaks = find_peaks_cwt(signal_data, widths, wavelet=wavelet, min_snr=min_snr, **kwargs)
    
    # 将峰值转换为numpy数组
    peaks = np.array(peaks, dtype=int)
    
    # 计算峰值属性
    if len(peaks) > 0:
        prominences = peak_prominences(signal_data, peaks)[0]
        widths, width_heights, left_ips, right_ips = peak_widths(signal_data, peaks, rel_height=0.5)
    else:
        prominences = np.array([])
        widths = np.array([])
        left_ips = np.array([])
        right_ips = np.array([])
    
    # 构建结果字典
    result = {
        'peak_indices': peaks,
        'peak_heights': signal_data[peaks] if len(peaks) > 0 else np.array([]),
        'prominences': prominences,
        'widths': widths,
        'left_bases': left_ips.astype(int) if len(left_ips) > 0 else np.array([], dtype=int),
        'right_bases': right_ips.astype(int) if len(right_ips) > 0 else np.array([], dtype=int)
    }
    
    return result

def _detect_peaks_threshold(signal_data: np.ndarray, threshold: float = None, 
                          min_distance: int = 1, **kwargs) -> Dict[str, np.ndarray]:
    """基于阈值的峰值检测"""
    # 如果没有指定阈值，使用信号平均值加上标准差
    if threshold is None:
        threshold = np.mean(signal_data) + np.std(signal_data)
    
    # 找到所有超过阈值的点
    above_threshold = signal_data > threshold
    # 找到上升沿和下降沿
    rising_edges = np.where(np.diff(above_threshold.astype(int)) > 0)[0] + 1
    falling_edges = np.where(np.diff(above_threshold.astype(int)) < 0)[0] + 1
    
    # 确保有足够的边缘
    if len(rising_edges) == 0 or len(falling_edges) == 0:
        return {
            'peak_indices': np.array([], dtype=int),
            'peak_heights': np.array([]),
            'prominences': np.array([]),
            'widths': np.array([]),
            'left_bases': np.array([], dtype=int),
            'right_bases': np.array([], dtype=int)
        }
    
    # 处理起始边缘情况
    if rising_edges[0] > falling_edges[0]:
        falling_edges = falling_edges[1:]
    if len(rising_edges) > len(falling_edges):
        rising_edges = rising_edges[:-1]
    
    # 确保有效的区域边界对
    n_regions = min(len(rising_edges), len(falling_edges))
    if n_regions == 0:
        return {
            'peak_indices': np.array([], dtype=int),
            'peak_heights': np.array([]),
            'prominences': np.array([]),
            'widths': np.array([]),
            'left_bases': np.array([], dtype=int),
            'right_bases': np.array([], dtype=int)
        }
    
    # 找到每个区域内的峰值
    peaks = []
    for i in range(n_regions):
        start = rising_edges[i]
        end = falling_edges[i]
        
        # 确保有效区域
        if start < end:
            peak_idx = start + np.argmax(signal_data[start:end])
            peaks.append(peak_idx)
    
    # 转换为numpy数组
    peaks = np.array(peaks, dtype=int)
    
    # 应用最小距离过滤
    if min_distance > 1 and len(peaks) > 1:
        # 按照信号值排序
        sorted_idxs = np.argsort(signal_data[peaks])[::-1]
        sorted_peaks = peaks[sorted_idxs]
        
        # 保持最强的峰值并去除太近的峰值
        kept_peaks = [sorted_peaks[0]]
        for peak in sorted_peaks[1:]:
            if np.min(np.abs(np.array(kept_peaks) - peak)) >= min_distance:
                kept_peaks.append(peak)
        
        peaks = np.array(kept_peaks)
        # 按索引重新排序
        peaks.sort()
    
    # 计算峰值属性
    if len(peaks) > 0:
        prominences = peak_prominences(signal_data, peaks)[0]
        widths, width_heights, left_ips, right_ips = peak_widths(signal_data, peaks, rel_height=0.5)
    else:
        prominences = np.array([])
        widths = np.array([])
        left_ips = np.array([])
        right_ips = np.array([])
    
    # 构建结果字典
    result = {
        'peak_indices': peaks,
        'peak_heights': signal_data[peaks] if len(peaks) > 0 else np.array([]),
        'prominences': prominences,
        'widths': widths,
        'left_bases': left_ips.astype(int) if len(left_ips) > 0 else np.array([], dtype=int),
        'right_bases': right_ips.astype(int) if len(right_ips) > 0 else np.array([], dtype=int)
    }
    
    return result

def _detect_peaks_adaptive(signal_data: np.ndarray, window_size: int = None, 
                          factor: float = 3.0, min_distance: int = 1, 
                          **kwargs) -> Dict[str, np.ndarray]:
    """自适应阈值的峰值检测"""
    # 如果没有指定窗口大小，使用信号长度的10%
    if window_size is None:
        window_size = max(int(len(signal_data) * 0.1), 1)
    
    # 确保窗口大小为奇数
    if window_size % 2 == 0:
        window_size += 1
    
    # 使用滑动窗口计算局部均值和标准差
    local_mean = np.zeros_like(signal_data)
    local_std = np.zeros_like(signal_data)
    
    half_window = window_size // 2
    padded_signal = np.pad(signal_data, half_window, mode='reflect')
    
    for i in range(len(signal_data)):
        window = padded_signal[i:i+window_size]
        local_mean[i] = np.mean(window)
        local_std[i] = np.std(window)
    
    # 计算自适应阈值
    adaptive_threshold = local_mean + factor * local_std
    
    # 找到超过阈值的点
    peaks = []
    for i in range(len(signal_data)):
        if signal_data[i] > adaptive_threshold[i]:
            peaks.append(i)
    
    # 转换为numpy数组
    peaks = np.array(peaks)
    
    # 应用最小距离过滤
    if len(peaks) > 0 and min_distance > 1:
        # 按照信号值排序
        sorted_idxs = np.argsort(signal_data[peaks])[::-1]
        sorted_peaks = peaks[sorted_idxs]
        
        # 保持最强的峰值并去除太近的峰值
        kept_peaks = [sorted_peaks[0]]
        for peak in sorted_peaks[1:]:
            if np.min(np.abs(np.array(kept_peaks) - peak)) >= min_distance:
                kept_peaks.append(peak)
        
        peaks = np.array(kept_peaks)
        # 按索引重新排序
        peaks.sort()
    
    # 计算峰值属性
    if len(peaks) > 0:
        prominences = peak_prominences(signal_data, peaks)[0]
        widths, width_heights, left_ips, right_ips = peak_widths(signal_data, peaks, rel_height=0.5)
    else:
        prominences = np.array([])
        widths = np.array([])
        left_ips = np.array([])
        right_ips = np.array([])
    
    # 构建结果字典
    result = {
        'peak_indices': peaks,
        'peak_heights': signal_data[peaks] if len(peaks) > 0 else np.array([]),
        'prominences': prominences,
        'widths': widths,
        'left_bases': left_ips.astype(int) if len(left_ips) > 0 else np.array([], dtype=int),
        'right_bases': right_ips.astype(int) if len(right_ips) > 0 else np.array([], dtype=int)
    }
    
    return result
