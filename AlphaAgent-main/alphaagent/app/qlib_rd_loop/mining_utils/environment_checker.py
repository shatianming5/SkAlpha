"""
环境检查器
"""

import os
from alphaagent.app.cli_utils.unicode_handler import safe_print


def check_environment():
    """
    检查运行环境
    
    Returns:
        dict: 环境信息
    """
    env_info = {
        'cwd': os.getcwd(),
        'use_local': os.getenv('USE_LOCAL', '未设置'),
        'python_path': os.getenv('PYTHONPATH', '未设置'),
    }
    
    safe_print("环境检查:", "CHECK")
    for key, value in env_info.items():
        safe_print(f"   - {key}: {value}")
    
    return env_info


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


def check_dependencies():
    """
    检查依赖包
    
    Returns:
        dict: 依赖检查结果
    """
    dependencies = {
        'pandas': False,
        'numpy': False,
        'qlib': False,
        'fire': False,
        'dotenv': False
    }
    
    for package in dependencies:
        try:
            __import__(package)
            dependencies[package] = True
        except ImportError:
            dependencies[package] = False
    
    safe_print("依赖检查:", "CHECK")
    for package, available in dependencies.items():
        status = "OK" if available else "MISSING"
        safe_print(f"   - {package}: {status}")
    
    return dependencies 