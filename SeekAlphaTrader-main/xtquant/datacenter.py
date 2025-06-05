# coding: utf-8
"""
模拟的datacenter模块
用于替代缺失的编译扩展模块
"""

import time
import threading
from typing import Dict, List, Any, Optional


class IPythonApiClient:
    """模拟的RPC客户端"""
    
    def __init__(self):
        self.initialized = False
        
    def init(self):
        """初始化客户端"""
        self.initialized = True
        return 0
        
    def load_config(self, config_file: str, section: str):
        """加载配置文件"""
        return 0


# 全局状态变量
_config_dir = ""
_data_home_dir = ""
_token = ""
_init_done = False
_server_list_status = {"done": 1, "servers": []}
_auth_markets = {"done": 1, "markets": ["SH", "SZ", "BJ"]}


def rpc_init(config_dir: str) -> int:
    """初始化RPC"""
    global _config_dir
    _config_dir = config_dir
    return 0


def get_local_server_port() -> int:
    """获取本地服务器端口"""
    return 58610


def register_create_nparray(func):
    """注册numpy数组创建函数"""
    pass


def set_config_dir(config_dir: str):
    """设置配置目录"""
    global _config_dir
    _config_dir = config_dir


def set_data_home_dir(data_home_dir: str):
    """设置数据主目录"""
    global _data_home_dir
    _data_home_dir = data_home_dir


def set_token(token: str):
    """设置认证token"""
    global _token
    _token = token


def log_init():
    """初始化日志"""
    pass


def start_init_quote():
    """开始初始化行情"""
    global _init_done
    # 模拟异步初始化
    def _init():
        time.sleep(1)
        global _init_done
        _init_done = True
    
    thread = threading.Thread(target=_init)
    thread.daemon = True
    thread.start()


def get_status() -> Dict[str, Any]:
    """获取状态"""
    return {
        "init_done": _init_done,
        "status": "ok"
    }


def fetch_auth_markets() -> Dict[str, Any]:
    """获取授权市场"""
    return _auth_markets


def fetch_server_list_status() -> bytes:
    """获取服务器列表状态"""
    import json
    return json.dumps(_server_list_status).encode('utf-8')


def fetch_init_result(market_list: List[str]) -> Dict[str, Any]:
    """获取初始化结果"""
    return {
        "done": 1,
        "result": "success",
        "markets": market_list
    }


def shutdown():
    """关闭服务"""
    global _init_done
    _init_done = False


def listen(ip: str, port_start: int, port_end: Optional[int] = None) -> Dict[str, Any]:
    """启动监听服务"""
    port = port_start if port_end is None else port_start
    return {
        "result": 0,
        "port": port,
        "ip": ip
    }


def set_kline_mirror_enabled(markets: List[str]):
    """设置K线镜像功能"""
    pass


def set_allow_optmize_address(allow_list: List[str]):
    """设置允许优化的地址列表"""
    pass


def set_wholequote_market_list(market_list: List[str]):
    """设置全推行情市场列表"""
    pass


def set_future_realtime_mode(enable: bool):
    """设置期货实时模式"""
    pass


def set_watch_market_list(markets: List[str]):
    """设置监控市场列表"""
    pass


def set_index_mirror_enabled(markets: List[str]):
    """设置指标镜像功能"""
    pass 