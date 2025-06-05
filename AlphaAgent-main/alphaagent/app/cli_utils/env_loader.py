"""
环境变量加载器
"""

import os
from dotenv import load_dotenv


def load_environment():
    """
    安全加载.env文件，添加错误处理
    
    Returns:
        bool: 是否成功加载
    """
    try:
        if os.path.exists(".env"):
            load_dotenv(".env")
            print("[OK] .env文件加载成功")
            return True
        else:
            print("[WARN] .env文件不存在，使用默认配置")
            return False
    except Exception as e:
        print(f"[WARN] .env文件加载失败: {e}")
        print("使用默认配置继续运行")
        return False


def get_use_local():
    """
    获取USE_LOCAL环境变量
    
    Returns:
        bool: 是否使用本地环境
    """
    use_local = os.getenv("USE_LOCAL", "True").lower()
    return True if use_local in ["true", "1"] else False


def get_timeout():
    """
    获取超时时间设置
    
    Returns:
        int: 超时时间（秒）
    """
    from alphaagent.oai.llm_conf import LLM_SETTINGS
    return LLM_SETTINGS.factor_mining_timeout 