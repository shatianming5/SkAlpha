"""
进度跟踪器
"""

import time
from alphaagent.app.cli_utils.unicode_handler import safe_print


class ProgressTracker:
    """
    进度跟踪器
    """
    
    def __init__(self, total_steps=5):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.step_times = []
    
    def start(self):
        """开始跟踪"""
        self.start_time = time.time()
        safe_print(f"开始因子挖掘流程，总共{self.total_steps}个步骤", "START")
    
    def next_step(self, step_name):
        """进入下一步"""
        if self.current_step > 0:
            step_duration = time.time() - self.step_start_time
            self.step_times.append(step_duration)
            safe_print(f"步骤{self.current_step}完成，耗时: {step_duration:.2f}秒", "OK")
        
        self.current_step += 1
        self.step_start_time = time.time()
        
        progress = (self.current_step / self.total_steps) * 100
        safe_print(f"步骤{self.current_step}/{self.total_steps}: {step_name} (进度: {progress:.1f}%)", "STEP")
    
    def complete(self):
        """完成跟踪"""
        if self.current_step > 0:
            step_duration = time.time() - self.step_start_time
            self.step_times.append(step_duration)
        
        total_duration = time.time() - self.start_time
        avg_step_time = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        
        safe_print(f"因子挖掘流程完成！", "SUCCESS")
        safe_print(f"总耗时: {total_duration:.2f}秒", "INFO")
        safe_print(f"平均每步耗时: {avg_step_time:.2f}秒", "INFO")
        
        # 显示各步骤耗时
        for i, step_time in enumerate(self.step_times, 1):
            safe_print(f"步骤{i}耗时: {step_time:.2f}秒", "INFO")
    
    def get_progress(self):
        """获取当前进度"""
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress_percent': (self.current_step / self.total_steps) * 100,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0
        } 