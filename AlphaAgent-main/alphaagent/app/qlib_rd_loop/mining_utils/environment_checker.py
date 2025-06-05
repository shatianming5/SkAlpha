"""
环境检查模块
检查因子挖掘运行环境
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
from alphaagent.app.cli_utils.unicode_handler import safe_print


def check_environment() -> Dict[str, Any]:
    """
    检查运行环境
    
    Returns:
        环境信息字典
    """
    env_info = {}
    
    # 检查Python版本
    env_info['python_version'] = sys.version
    safe_print(f"Python版本: {sys.version.split()[0]}", "ENV")
    
    # 检查工作目录
    env_info['working_dir'] = os.getcwd()
    safe_print(f"工作目录: {os.getcwd()}", "ENV")
    
    # 检查关键目录
    key_dirs = ['log', 'alphaagent', 'git_ignore_folder']
    env_info['directories'] = {}
    
    for dir_name in key_dirs:
        dir_path = Path(dir_name)
        exists = dir_path.exists()
        env_info['directories'][dir_name] = {
            'exists': exists,
            'path': str(dir_path.absolute()) if exists else None
        }
        status = "存在" if exists else "不存在"
        safe_print(f"目录 {dir_name}: {status}", "ENV")
    
    # 检查环境变量
    important_vars = ['PATH', 'PYTHONPATH']
    env_info['env_vars'] = {}
    
    for var in important_vars:
        value = os.environ.get(var)
        env_info['env_vars'][var] = value
        if value:
            safe_print(f"环境变量 {var}: 已设置", "ENV")
        else:
            safe_print(f"环境变量 {var}: 未设置", "ENV")
    
    # 检查内存使用
    try:
        import psutil
        memory = psutil.virtual_memory()
        env_info['memory'] = {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent
        }
        safe_print(f"内存使用: {memory.percent:.1f}% ({memory.available//1024//1024}MB可用)", "ENV")
    except ImportError:
        env_info['memory'] = None
        safe_print("内存信息: 无法获取 (psutil未安装)", "ENV")
    
    return env_info


def check_dependencies() -> Dict[str, bool]:
    """
    检查关键依赖包
    
    Returns:
        依赖包状态字典
    """
    dependencies = {
        'pandas': False,
        'numpy': False,
        'matplotlib': False,
        'qlib': False,
        'fire': False
    }
    
    for package in dependencies.keys():
        try:
            __import__(package)
            dependencies[package] = True
            safe_print(f"依赖包 {package}: 已安装", "DEP")
        except ImportError:
            safe_print(f"依赖包 {package}: 未安装", "DEP")
    
    return dependencies


def validate_data_files():
    """
    验证数据文件是否存在
    
    Returns:
        bool: 数据文件是否完整
    """
    required_files = [
        'daily_pv_all.h5',
        'daily_pv_debug.h5'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        safe_print(f"缺少数据文件: {missing_files}", "WARN")
        return False
    else:
        safe_print("所有必需的数据文件都存在", "OK")
        return True 