"""
因子挖掘主流程
使用模块化设计，降低复杂度
"""

from typing import Any
import fire
import threading
import os
import time

from alphaagent.app.qlib_rd_loop.conf import ALPHA_AGENT_FACTOR_PROP_SETTING
from alphaagent.components.workflow.alphaagent_loop import AlphaAgentLoop
from alphaagent.core.exception import FactorEmptyError
from alphaagent.log import logger

# 导入拆解后的模块化工具
from alphaagent.app.qlib_rd_loop.mining_utils import (
    force_timeout, 
    check_environment, 
    ProgressTracker,
    SummaryDisplayer
)
from alphaagent.app.cli_utils import get_use_local, get_timeout
from alphaagent.app.cli_utils.unicode_handler import safe_print


@force_timeout()
def main(path=None, step_n=None, potential_direction=None, stop_event=None):
    """
    Autonomous alpha factor mining. 

    Args:
        path: 会话路径
        step_n: 循环轮数（每轮包含5个步骤：假设生成、因子构造、因子计算、因子回测、反馈生成）
        potential_direction: 初始方向/市场假设
        stop_event: 停止事件

    You can continue running session by

    .. code-block:: python

        dotenv run -- python rdagent/app/qlib_rd_loop/factor_alphaagent.py $LOG_PATH/__session__/1/0_propose  --step_n 1  --potential_direction "[Initial Direction (Optional)]"  # `step_n` is a optional paramter

    """
    safe_print("开始因子挖掘...", "START")
    safe_print(f"参数: path={path}, step_n={step_n}, potential_direction={potential_direction}", "PARAMS")
    
    # 初始化时间统计
    start_time = time.time()
    step_times = []
    total_steps = step_n * 5 if step_n else 5  # 在try块外定义
    
    try:
        # 环境检查
        env_info = check_environment()
        safe_print(f"超时时间: {get_timeout()}秒")
        
        # 创建进度跟踪器
        tracker = ProgressTracker(total_steps=5)
        tracker.start()  # 不传递参数
        
        # 创建AlphaAgent循环
        safe_print("正在初始化AlphaAgent循环...", "INIT")
        model_loop = AlphaAgentLoop(
            ALPHA_AGENT_FACTOR_PROP_SETTING, 
            potential_direction=potential_direction,
            stop_event=stop_event,
            use_local=get_use_local()
        )
        
        # 计算总步骤数
        safe_print(f"将执行 {step_n} 轮循环，总共 {total_steps} 个步骤", "PARAMS")
        
        tracker.next_step("开始因子挖掘流程")
        
        # 记录开始时间
        loop_start_time = time.time()
        
        # 运行主循环
        safe_print("开始执行因子挖掘主循环...", "LOOP")
        model_loop.run(step_n=total_steps, stop_event=stop_event)
        
        # 记录循环结束时间
        loop_end_time = time.time()
        step_times.append(loop_end_time - loop_start_time)
        
        # 显示因子挖掘表现总结
        safe_print("=" * 60, "SUMMARY")
        safe_print("因子挖掘流程完成！正在生成表现总结...", "SUMMARY")
        
        # 使用拆解后的摘要显示器
        summary_displayer = SummaryDisplayer()
        summary_displayer.display_factor_mining_summary(model_loop)
        
        safe_print("=" * 60, "SUMMARY")
        
        tracker.complete()
        safe_print("因子挖掘流程完成！", "SUCCESS")
        
    except KeyboardInterrupt:
        safe_print("用户中断了程序执行", "INTERRUPT")
        logger.info("用户中断了程序执行")
    except Exception as e:
        safe_print(f"执行过程中发生错误: {e}", "ERROR")
        logger.error(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        raise e
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        
        safe_print(f"总耗时: {total_time:.2f}秒", "INFO")
        if total_steps > 0:
            safe_print(f"平均每步耗时: {total_time/total_steps:.2f}秒", "INFO")
        
        # 显示各步骤耗时
        for i, step_time in enumerate(step_times, 1):
            safe_print(f"步骤{i}耗时: {step_time:.2f}秒", "INFO")
        
        safe_print("程序执行结束", "END")
        logger.info("程序执行完成或被终止")


# 原来的display_factor_mining_summary函数已移动到SummaryDisplayer类中

if __name__ == "__main__":
    fire.Fire(main)
