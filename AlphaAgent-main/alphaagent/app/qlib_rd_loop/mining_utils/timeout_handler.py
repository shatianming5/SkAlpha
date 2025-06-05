"""
超时处理器
"""

import os
import threading
import time
from functools import wraps
from alphaagent.log import logger


def force_timeout():
    """
    强制超时装饰器
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            from alphaagent.oai.llm_conf import LLM_SETTINGS
            seconds = LLM_SETTINGS.factor_mining_timeout
            
            def handle_timeout():
                logger.error(f"强制终止程序执行，已超过{seconds}秒")
                os._exit(1)  # 使用os._exit强制退出

            # 使用threading.Timer替代signal.alarm，支持Windows
            timer = threading.Timer(seconds, handle_timeout)
            timer.start()

            try:
                result = func(*args, **kwargs)
            finally:
                # 取消定时器
                timer.cancel()
            return result
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