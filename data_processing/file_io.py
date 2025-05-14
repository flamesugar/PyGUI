"""
文件输入输出模块，负责处理各种光度测量数据文件的读取和写入。
支持多种文件格式，提供数据转换和导出功能。
"""

import os
import numpy as np
import pandas as pd
import h5py
import json
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Union, Optional, Any

# 支持的文件格式
SUPPORTED_FORMATS = {
    '.csv': '逗号分隔值文件',
    '.xlsx': 'Excel文件',
    '.txt': '制表符分隔文本文件',
    '.h5': 'HDF5文件',
    '.npy': 'NumPy二进制文件',
    '.mat': 'MATLAB数据文件',
    '.json': 'JSON格式文件',
    '.ppd': '光度记录数据文件'  # 添加PPD格式支持
}

def get_supported_formats() -> Dict[str, str]:
    """
    获取支持的文件格式列表。

    Returns:
        Dict[str, str]: 文件扩展名及其描述的字典
    """
    return SUPPORTED_FORMATS

def detect_file_format(filepath: str) -> str:
    """
    根据文件扩展名检测文件格式。

    Args:
        filepath (str): 文件路径

    Returns:
        str: 文件格式，如果不支持则为空字符串
    """
    _, ext = os.path.splitext(filepath)
    if ext.lower() in SUPPORTED_FORMATS:
        return ext.lower()
    return ""

def load_data(filepath: str, **kwargs) -> Dict[str, Any]:
    """
    加载光度测量数据文件。

    Args:
        filepath (str): 要加载的文件路径
        **kwargs: 加载特定文件格式的附加参数

    Returns:
        Dict[str, Any]: 包含信号数据和元数据的字典

    Raises:
        ValueError: 如果文件格式不支持或文件不存在
        IOError: 如果文件读取失败
    """
    if not os.path.exists(filepath):
        raise ValueError(f"文件不存在: {filepath}")
    
    file_format = detect_file_format(filepath)
    if not file_format:
        raise ValueError(f"不支持的文件格式: {filepath}")
    
    try:
        if file_format == '.csv':
            return _load_csv(filepath, **kwargs)
        elif file_format == '.xlsx':
            return _load_excel(filepath, **kwargs)
        elif file_format == '.txt':
            return _load_txt(filepath, **kwargs)
        elif file_format == '.h5':
            return _load_h5(filepath, **kwargs)
        elif file_format == '.npy':
            return _load_numpy(filepath, **kwargs)
        elif file_format == '.mat':
            return _load_matlab(filepath, **kwargs)
        elif file_format == '.json':
            return _load_json(filepath, **kwargs)
        elif file_format == '.ppd':
            return _load_ppd(filepath, **kwargs)
    except Exception as e:
        raise IOError(f"读取文件失败 {filepath}: {str(e)}")

def save_data(data: Dict[str, Any], filepath: str, file_format: str = None, **kwargs) -> bool:
    """
    保存光度测量数据到文件。

    Args:
        data (Dict[str, Any]): 要保存的数据字典
        filepath (str): 输出文件路径
        file_format (str, optional): 文件格式，如果为None则从文件扩展名推断
        **kwargs: 保存特定文件格式的附加参数

    Returns:
        bool: 保存成功返回True，否则False

    Raises:
        ValueError: 如果文件格式不支持
        IOError: 如果文件写入失败
    """
    if file_format is None:
        file_format = detect_file_format(filepath)
        if not file_format:
            raise ValueError(f"无法从文件名确定格式，请明确指定文件格式: {filepath}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(filepath)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        if file_format == '.csv':
            return _save_csv(data, filepath, **kwargs)
        elif file_format == '.xlsx':
            return _save_excel(data, filepath, **kwargs)
        elif file_format == '.txt':
            return _save_txt(data, filepath, **kwargs)
        elif file_format == '.h5':
            return _save_h5(data, filepath, **kwargs)
        elif file_format == '.npy':
            return _save_numpy(data, filepath, **kwargs)
        elif file_format == '.mat':
            return _save_matlab(data, filepath, **kwargs)
        elif file_format == '.json':
            return _save_json(data, filepath, **kwargs)
        elif file_format == '.ppd':
            return _save_ppd(data, filepath, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")
    except Exception as e:
        raise IOError(f"保存文件失败 {filepath}: {str(e)}")

def import_from_csv(filepath: str, time_column: str = 'time', 
                  signal_columns: List[str] = None, **kwargs) -> Dict[str, Any]:
    """
    从CSV文件导入数据的便捷方法。

    Args:
        filepath (str): CSV文件路径
        time_column (str): 包含时间数据的列名
        signal_columns (List[str], optional): 包含信号数据的列名列表
        **kwargs: 传递给pandas.read_csv的附加参数

    Returns:
        Dict[str, Any]: 包含时间和信号数据的字典
    """
    return _load_csv(filepath, time_column=time_column, 
                    signal_columns=signal_columns, **kwargs)

def export_to_csv(data: Dict[str, Any], filepath: str, 
                include_metadata: bool = True, **kwargs) -> bool:
    """
    将数据导出到CSV文件的便捷方法。

    Args:
        data (Dict[str, Any]): 要导出的数据字典
        filepath (str): 输出CSV文件路径
        include_metadata (bool): 是否在CSV开头包含元数据行
        **kwargs: 传递给pandas.to_csv的附加参数

    Returns:
        bool: 导出成功返回True，否则False
    """
    return _save_csv(data, filepath, include_metadata=include_metadata, **kwargs)

def batch_process(file_list: List[str], process_func, output_dir: str = None, 
                output_suffix: str = '_processed', **kwargs) -> List[str]:
    """
    批量处理多个文件。

    Args:
        file_list (List[str]): 要处理的文件路径列表
        process_func: 应用于每个文件的处理函数
        output_dir (str, optional): 输出目录，如果为None则使用原始文件目录
        output_suffix (str): 添加到处理后文件名的后缀
        **kwargs: 传递给处理函数的附加参数

    Returns:
        List[str]: 成功处理的文件列表
    """
    processed_files = []
    
    for filepath in file_list:
        try:
            # 加载数据
            data = load_data(filepath)
            
            # 应用处理函数
            processed_data = process_func(data, **kwargs)
            
            # 确定输出路径
            filename = os.path.basename(filepath)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}{output_suffix}{ext}"
            
            if output_dir:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_path = os.path.join(output_dir, output_filename)
            else:
                output_path = os.path.join(os.path.dirname(filepath), output_filename)
            
            # 保存处理后的数据
            save_data(processed_data, output_path)
            processed_files.append(output_path)
            
        except Exception as e:
            print(f"处理文件 {filepath} 时出错: {str(e)}")
    
    return processed_files

# PPD文件处理函数
def read_ppd_file(file_path):
    """
    读取.ppd文件并返回头部和数据。
    
    Args:
        file_path (str): PPD文件路径
    
    Returns:
        tuple: (header, data_bytes) - 解析后的头部和原始数据字节
    """
    try:
        with open(file_path, 'rb') as f:
            # 前两个字节是小端字节序的头部长度
            header_len_bytes = f.read(2)
            header_len = int.from_bytes(header_len_bytes, byteorder='little')

            # 读取头部
            header_bytes = f.read(header_len)
            header_str = header_bytes.decode('utf-8')

            # 解析头部JSON
            header = json.loads(header_str)

            # 读取文件其余部分作为数据
            data_bytes = f.read()

            print(f"Successfully read file with header: sampling_rate={header.get('sampling_rate', 'unknown')}")
            return header, data_bytes
    except Exception as e:
        print(f"Error reading PPD file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def parse_ppd_data(header, data_bytes):
    """
    解析.ppd文件中的数据字节，返回模拟和数字信号。
    
    Args:
        header (dict): PPD文件头部数据
        data_bytes (bytes): 原始数据字节
    
    Returns:
        tuple: (time, analog_1, analog_2, digital_1, digital_2) - 解析后的时间和信号数据
    """
    try:
        if header is None or data_bytes is None or len(data_bytes) == 0:
            print("Error: No valid data found")
            return None, None, None, None, None

        # 从头部获取采样率 - 确保是数值
        sampling_rate = float(header.get('sampling_rate', 1000))  # 默认为1000 Hz
        print(f"Using sampling rate: {sampling_rate} Hz")

        # 获取volts_per_division（如果可用）
        volts_per_division = header.get('volts_per_division', [1.0, 1.0])  # 默认为1.0

        # 解析数据字节 - 原始格式是小端字节序的无符号16位整数
        data = np.frombuffer(data_bytes, dtype=np.dtype('<u2'))

        # 检查空数据
        if len(data) == 0:
            print("Warning: No data found in file")
            return None, None, None, None, None

        # 存储原始数据长度作为参考
        original_data_length = len(data)
        print(f"Original data length: {original_data_length} samples")

        # 提取模拟和数字信号
        # 最后一位是数字信号，其余是模拟值
        analog = data >> 1  # 右移去除数字位
        digital = data & 1  # 掩码只获取最后一位

        # 分离通道 - 偶数索引是通道1，奇数索引是通道2
        analog_1 = analog[0::2]
        analog_2 = analog[1::2]
        digital_1 = digital[0::2]
        digital_2 = digital[1::2]

        # 关键修复：确保所有数组长度相同
        min_len = min(len(analog_1), len(analog_2), len(digital_1), len(digital_2))
        analog_1 = analog_1[:min_len]
        analog_2 = analog_2[:min_len]
        digital_1 = digital_1[:min_len]
        digital_2 = digital_2[:min_len]

        # 根据采样率计算原始持续时间
        original_duration = min_len / sampling_rate  # 持续时间（秒）
        print(f"Original duration: {original_duration:.2f} seconds")

        # 如果数据非常大，适当下采样以防止内存问题
        max_points = 500000  # 处理的合理最大点数
        if min_len > max_points:
            print(f"Data is very large ({min_len} points), downsampling to prevent memory issues")

            # 计算下采样因子
            downsample_factor = min_len // max_points + 1

            # 修复：创建与下采样信号具有相同点数的时间数组
            num_points = min_len // downsample_factor

            # 使用步长下采样信号
            analog_1 = analog_1[::downsample_factor][:num_points]  # 确保精确长度
            analog_2 = analog_2[::downsample_factor][:num_points]
            digital_1 = digital_1[::downsample_factor][:num_points]
            digital_2 = digital_2[::downsample_factor][:num_points]

            # 创建具有相同长度的时间数组
            time = np.linspace(0, original_duration, num_points)

            print(f"Downsampled to {len(analog_1)} points. Time range preserved: 0 to {time[-1]:.2f} seconds")
        else:
            # 使用原始采样率创建时间数组
            time = np.arange(min_len) / sampling_rate  # 时间（秒）

        # 验证所有数组长度相同
        assert len(time) == len(analog_1) == len(analog_2) == len(digital_1) == len(digital_2), \
            f"Array length mismatch: time={len(time)}, analog_1={len(analog_1)}, analog_2={len(analog_2)}, " \
            f"digital_1={len(digital_1)}, digital_2={len(digital_2)}"

        # 如果可用，应用volts_per_division缩放（转换为mV）
        if len(volts_per_division) >= 2:
            analog_1 = analog_1 * volts_per_division[0]
            analog_2 = analog_2 * volts_per_division[1]

        print(f"Parsed data: {len(time)} samples, time range: 0 to {time[-1]:.2f} seconds")
        return time, analog_1, analog_2, digital_1, digital_2
    except Exception as e:
        print(f"Error parsing PPD file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def _load_ppd(filepath: str, **kwargs) -> Dict[str, Any]:
    """从PPD文件加载数据"""
    try:
        # 读取PPD文件
        header, data_bytes = read_ppd_file(filepath)
        if header is None or data_bytes is None:
            raise IOError(f"无法读取PPD文件: {filepath}")
        
        # 解析PPD数据
        time, analog_1, analog_2, digital_1, digital_2 = parse_ppd_data(header, data_bytes)
        if time is None:
            raise IOError(f"无法解析PPD数据: {filepath}")
        
        # 构建返回结构
        result = {
            'time': time,
            'signals': {
                'analog_1': analog_1,
                'analog_2': analog_2,
                'digital_1': digital_1,
                'digital_2': digital_2
            },
            'metadata': header,
            'sampling_rate': float(header.get('sampling_rate', 1000))
        }
        
        return result
    
    except Exception as e:
        raise IOError(f"从PPD文件加载数据失败: {str(e)}")

def _save_ppd(data: Dict[str, Any], filepath: str, **kwargs) -> bool:
    """保存数据到PPD文件"""
    try:
        # 提取必要的数据
        time = data['time']
        signals = data['signals']
        metadata = data.get('metadata', {})
        
        # 确保metadata中有采样率
        if 'sampling_rate' not in metadata and 'sampling_rate' in data:
            metadata['sampling_rate'] = data['sampling_rate']
        
        # 检查必要的信号是否存在
        if 'analog_1' not in signals or 'analog_2' not in signals or \
           'digital_1' not in signals or 'digital_2' not in signals:
            raise ValueError("PPD格式需要analog_1, analog_2, digital_1, digital_2信号")
        
        # 转换元数据为JSON字符串
        header_str = json.dumps(metadata)
        header_bytes = header_str.encode('utf-8')
        header_len = len(header_bytes)
        
        # 从信号创建数据数组
        analog_1 = signals['analog_1']
        analog_2 = signals['analog_2']
        digital_1 = signals['digital_1']
        digital_2 = signals['digital_2']
        
        # 确保所有数组长度相同
        min_len = min(len(analog_1), len(analog_2), len(digital_1), len(digital_2))
        analog_1 = analog_1[:min_len]
        analog_2 = analog_2[:min_len]
        digital_1 = digital_1[:min_len]
        digital_2 = digital_2[:min_len]
        
        # 应用反向的volts_per_division缩放（如果适用）
        volts_per_division = metadata.get('volts_per_division', [1.0, 1.0])
        if len(volts_per_division) >= 2 and volts_per_division[0] != 0 and volts_per_division[1] != 0:
            analog_1 = analog_1 / volts_per_division[0]
            analog_2 = analog_2 / volts_per_division[1]
        
        # 创建未分离的模拟和数字数组
        analog = np.zeros(min_len*2, dtype=np.uint16)
        digital = np.zeros(min_len*2, dtype=np.uint16)
        
        # 将通道数据放入交错数组
        analog[0::2] = analog_1.astype(np.uint16)
        analog[1::2] = analog_2.astype(np.uint16)
        digital[0::2] = digital_1.astype(np.uint16)
        digital[1::2] = digital_2.astype(np.uint16)
        
        # 组合模拟和数字数据
        data_array = (analog << 1) | digital
        
        # 写入文件
        with open(filepath, 'wb') as f:
            # 写入头部长度（2字节，小端字节序）
            f.write(header_len.to_bytes(2, byteorder='little'))
            
            # 写入头部
            f.write(header_bytes)
            
            # 写入数据
            f.write(data_array.tobytes())
        
        return True
    
    except Exception as e:
        print(f"保存PPD文件失败: {str(e)}")
        return False

# 以下是我之前提供的其他私有辅助函数，保持不变

def _load_csv(filepath: str, time_column: str = 'time', 
             signal_columns: List[str] = None, **kwargs) -> Dict[str, Any]:
    """从CSV文件加载数据"""
    # CSV加载代码保持不变
    # ...原有代码...
    try:
        # 检查CSV文件的前几行是否包含元数据
        with open(filepath, 'r') as f:
            first_line = f.readline().strip()
        
        metadata = {}
        skip_rows = 0
        
        # 如果以#开头，则为元数据行
        while first_line.startswith('#'):
            # 解析元数据行
            if ':' in first_line:
                key, value = first_line[1:].split(':', 1)
                metadata[key.strip()] = value.strip()
            skip_rows += 1
            with open(filepath, 'r') as f:
                for _ in range(skip_rows):
                    f.readline()
                first_line = f.readline().strip()
        
        # 读取CSV数据
        df = pd.read_csv(filepath, skiprows=skip_rows, **kwargs)
        
        # 确定信号列
        if signal_columns is None:
            # 默认除了时间列外的所有列都是信号列
            signal_columns = [col for col in df.columns if col != time_column]
        
        # 创建返回数据结构
        result = {
            'time': df[time_column].values if time_column in df.columns else np.arange(len(df)),
            'signals': {col: df[col].values for col in signal_columns if col in df.columns},
            'metadata': metadata,
            'sampling_rate': _estimate_sampling_rate(df[time_column].values) if time_column in df.columns else 1.0
        }
        
        return result
    
    except Exception as e:
        raise IOError(f"从CSV加载数据失败: {str(e)}")

# 其余辅助函数保持不变
def _estimate_sampling_rate(time_data: np.ndarray) -> float:
    """估计采样率（赫兹）"""
    if time_data is None or len(time_data) < 2:
        return 1.0
    
    # 计算时间差的中位数
    time_diffs = np.diff(time_data)
    median_diff = np.median(time_diffs)
    
    if median_diff <= 0:
        return 1.0
    
    # 采样率 = 1 / 时间差
    return 1.0 / median_diff

# 这里应添加其他在我的原始代码中存在的_load_*和_save_*函数
