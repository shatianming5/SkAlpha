"""
超时处理模块
处理因子挖掘过程中的超时控制
"""

import os
import threading
import time
import signal
import functools
from functools import wraps
from alphaagent.log import logger
from alphaagent.app.cli_utils import get_timeout
from alphaagent.app.cli_utils.unicode_handler import safe_print


class TimeoutError(Exception):
    """超时异常"""
    pass


def timeout_handler(signum, frame):
    """超时信号处理器"""
    raise TimeoutError("操作超时")


def force_timeout():
    """
    强制超时装饰器
    为函数添加超时控制
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timeout_seconds = get_timeout()
            
            if timeout_seconds > 0:
                # 设置超时信号
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout_seconds)
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except TimeoutError:
                    safe_print(f"操作超时 ({timeout_seconds}秒)", "TIMEOUT")
                    raise
                finally:
                    # 恢复原始信号处理器
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class TimeoutManager:
    """
    超时管理器
    """
    
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.timer = None
        self.start_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        self.timer = threading.Timer(self.timeout_seconds, self._timeout_handler)
        self.timer.start()
        logger.info(f"开始计时，超时时间: {self.timeout_seconds}秒")
    
    def stop(self):
        """停止计时"""
        if self.timer:
            self.timer.cancel()
            elapsed = time.time() - self.start_time if self.start_time else 0
            logger.info(f"计时停止，已用时: {elapsed:.2f}秒")
    
    def _timeout_handler(self):
        """超时处理"""
        logger.error(f"操作超时，已超过{self.timeout_seconds}秒")
        os._exit(1)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop() 