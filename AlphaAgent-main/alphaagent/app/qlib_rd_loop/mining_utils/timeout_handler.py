"""
超时处理模块
处理因子挖掘过程中的超时控制
支持Windows和Unix/Linux系统
"""

import os
import sys
import threading
import time
import functools
from alphaagent.log import logger
from alphaagent.app.cli_utils import get_timeout
from alphaagent.app.cli_utils.unicode_handler import safe_print


class TimeoutError(Exception):
    """超时异常"""
    pass


def force_timeout():
    """
    强制超时装饰器
    为函数添加超时控制
    支持跨平台（Windows/Unix/Linux）
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timeout_seconds = get_timeout()
            
            if timeout_seconds <= 0:
                # 没有设置超时，直接执行
                return func(*args, **kwargs)
            
            # 使用threading.Timer实现跨平台超时控制
            result = [None]
            exception = [None]
            finished = [False]
            
            def target():
                """目标函数执行线程"""
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
                finally:
                    finished[0] = True
            
            def timeout_handler():
                """超时处理函数"""
                if not finished[0]:
                    safe_print(f"操作超时 ({timeout_seconds}秒)", "TIMEOUT")
                    logger.error(f"强制终止程序执行，已超过{timeout_seconds}秒")
                    # 跨平台强制退出
                    os._exit(1)
            
            # 启动执行线程
            exec_thread = threading.Thread(target=target)
            exec_thread.daemon = True
            exec_thread.start()
            
            # 启动超时定时器
            timer = threading.Timer(timeout_seconds, timeout_handler)
            timer.start()
            
            try:
                # 等待执行完成
                exec_thread.join(timeout_seconds + 1)
                
                if finished[0]:
                    # 正常完成
                    timer.cancel()
                    if exception[0]:
                        raise exception[0]
                    return result[0]
                else:
                    # 超时了
                    timer.cancel()
                    raise TimeoutError(f"操作超时 ({timeout_seconds}秒)")
                    
            except KeyboardInterrupt:
                timer.cancel()
                safe_print("用户中断操作", "INTERRUPT")
                raise
            except Exception as e:
                timer.cancel()
                raise e
        
        return wrapper
    return decorator


class TimeoutManager:
    """
    超时管理器
    跨平台实现
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
        safe_print(f"强制终止程序执行，已超过{self.timeout_seconds}秒", "TIMEOUT")
        os._exit(1)
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop() 