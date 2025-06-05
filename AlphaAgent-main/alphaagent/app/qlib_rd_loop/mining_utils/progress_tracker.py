"""
进度跟踪模块
跟踪因子挖掘过程的进度
"""

import time
from typing import Optional
from alphaagent.app.cli_utils.unicode_handler import safe_print


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total_steps: int = 5):
        """
        初始化进度跟踪器
        
        Args:
            total_steps: 总步骤数
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = None
        self.step_times = []
        self.step_names = []
    
    def start(self):
        """开始跟踪"""
        self.start_time = time.time()
        self.current_step = 0
        self.step_times = []
        self.step_names = []
        safe_print(f"开始执行 {self.total_steps} 个步骤", "PROGRESS")
    
    def next_step(self, step_name: str):
        """
        进入下一步
        
        Args:
            step_name: 步骤名称
        """
        current_time = time.time()
        
        if self.current_step > 0:
            # 记录上一步的耗时
            step_duration = current_time - self.step_times[-1]
            safe_print(f"步骤 {self.current_step} 完成，耗时: {step_duration:.2f}秒", "PROGRESS")
        
        self.current_step += 1
        self.step_times.append(current_time)
        self.step_names.append(step_name)
        
        progress_percent = (self.current_step / self.total_steps) * 100
        safe_print(f"[{self.current_step}/{self.total_steps}] ({progress_percent:.1f}%) {step_name}", "PROGRESS")
    
    def complete(self):
        """完成跟踪"""
        if self.start_time is None:
            return
        
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        safe_print("=" * 50, "PROGRESS")
        safe_print("执行完成！", "PROGRESS")
        safe_print(f"总耗时: {total_duration:.2f}秒", "PROGRESS")
        
        if len(self.step_times) > 1:
            safe_print("各步骤耗时:", "PROGRESS")
            for i in range(1, len(self.step_times)):
                step_duration = self.step_times[i] - self.step_times[i-1]
                step_name = self.step_names[i] if i < len(self.step_names) else f"步骤{i}"
                safe_print(f"  {step_name}: {step_duration:.2f}秒", "PROGRESS")
        
        safe_print("=" * 50, "PROGRESS")
    
    def get_elapsed_time(self) -> float:
        """
        获取已用时间
        
        Returns:
            已用时间（秒）
        """
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    def get_estimated_remaining_time(self) -> Optional[float]:
        """
        估算剩余时间
        
        Returns:
            估算剩余时间（秒），如果无法估算则返回None
        """
        if self.current_step == 0 or self.start_time is None:
            return None
        
        elapsed = self.get_elapsed_time()
        avg_time_per_step = elapsed / self.current_step
        remaining_steps = self.total_steps - self.current_step
        
        return avg_time_per_step * remaining_steps 