"""
Factor workflow with session control
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
from alphaagent.log.time import measure_time
from alphaagent.oai.llm_conf import LLM_SETTINGS

# 导入新的模块化工具
from alphaagent.app.qlib_rd_loop.mining_utils import force_timeout, check_environment, ProgressTracker
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
        display_factor_mining_summary(model_loop)
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


def display_factor_mining_summary(model_loop):
    """
    显示因子挖掘表现总结
    
    Args:
        model_loop: AlphaAgentLoop实例，包含执行结果
    """
    try:
        from alphaagent.app.cli_utils.unicode_handler import safe_print
        import os
        from pathlib import Path
        
        safe_print("[SUMMARY] 因子挖掘表现总结", "SUMMARY")
        safe_print("-" * 50, "SUMMARY")
        
        # 获取最新的执行结果
        if hasattr(model_loop, 'loop_prev_out') and model_loop.loop_prev_out:
            
            # 1. 显示生成的假设
            if 'factor_propose' in model_loop.loop_prev_out:
                hypothesis = model_loop.loop_prev_out['factor_propose']
                if hasattr(hypothesis, 'hypothesis'):
                    safe_print(f"[SUMMARY] 市场假设: {hypothesis.hypothesis[:100]}...", "SUMMARY")
                else:
                    safe_print("[SUMMARY] 市场假设: 已生成", "SUMMARY")
            
            # 2. 显示构造的因子
            if 'factor_construct' in model_loop.loop_prev_out:
                factors = model_loop.loop_prev_out['factor_construct']
                if hasattr(factors, 'sub_tasks'):
                    safe_print(f"[SUMMARY] 构造因子数量: {len(factors.sub_tasks)}", "SUMMARY")
                    for i, task in enumerate(factors.sub_tasks[:3], 1):  # 只显示前3个
                        if hasattr(task, 'factor_name'):
                            safe_print(f"         {i}. {task.factor_name}", "SUMMARY")
                else:
                    safe_print("[SUMMARY] 构造因子数量: 已完成", "SUMMARY")
            
            # 3. 显示回测结果
            backtest_found = False
            if 'factor_backtest' in model_loop.loop_prev_out:
                backtest_result = model_loop.loop_prev_out['factor_backtest']
                
                # 尝试多种方式获取结果
                result_data = None
                if hasattr(backtest_result, 'result') and backtest_result.result is not None:
                    result_data = backtest_result.result
                elif hasattr(backtest_result, 'backtest_result'):
                    result_data = backtest_result.backtest_result
                elif isinstance(backtest_result, dict):
                    result_data = backtest_result
                
                if result_data is not None and len(result_data) > 0:
                    backtest_found = True
                    safe_print("[SUMMARY] 回测表现:", "SUMMARY")
                    
                    # 显示关键指标
                    key_metrics = {
                        'IC': 'IC (信息系数)',
                        'ICIR': 'ICIR (信息比率)', 
                        'Rank IC': 'Rank IC (排序IC)',
                        'Rank ICIR': 'Rank ICIR (排序信息比率)',
                        'Annual Return': '年化收益率',
                        'Sharpe Ratio': '夏普比率',
                        'Max Drawdown': '最大回撤',
                        'averaged_annualized_roi': '平均年化收益率',
                        'info_ratio': '信息比率',
                        'max_drawdown': '最大回撤',
                        'monthly_win_rate': '月胜率',
                        'overall_trading_win_rate': '总体胜率',
                        'turnover': '换手率',
                        'algorithm_volatility': '算法波动率'
                    }
                    
                    displayed_metrics = 0
                    for metric_key, metric_name in key_metrics.items():
                        if metric_key in result_data:
                            value = result_data[metric_key]
                            if isinstance(value, (int, float)) and not (isinstance(value, float) and (value != value)):  # 排除NaN
                                if metric_key in ['Annual Return', 'averaged_annualized_roi']:
                                    safe_print(f"         {metric_name}: {value:.4f} ({value*100:.2f}%)", "SUMMARY")
                                elif metric_key in ['Max Drawdown', 'max_drawdown']:
                                    safe_print(f"         {metric_name}: {value:.4f} ({value*100:.2f}%)", "SUMMARY")
                                elif metric_key in ['monthly_win_rate', 'overall_trading_win_rate']:
                                    safe_print(f"         {metric_name}: {value:.4f} ({value*100:.2f}%)", "SUMMARY")
                                elif metric_key in ['turnover']:
                                    safe_print(f"         {metric_name}: {value:.4f} ({value*100:.2f}%)", "SUMMARY")
                                else:
                                    safe_print(f"         {metric_name}: {value:.6f}", "SUMMARY")
                                displayed_metrics += 1
                    
                    if displayed_metrics == 0:
                        safe_print("         [无有效指标数据]", "SUMMARY")
                    
                    # 评估表现
                    safe_print("[SUMMARY] 表现评估:", "SUMMARY")
                    ic_value = result_data.get('IC', result_data.get('test_ic', 0))
                    annual_return = result_data.get('Annual Return', result_data.get('averaged_annualized_roi', 0))
                    sharpe_ratio = result_data.get('Sharpe Ratio', result_data.get('sharpe_ratio', 0))
                    
                    if ic_value > 0.05:
                        safe_print("         IC表现: 优秀 (>0.05)", "SUMMARY")
                    elif ic_value > 0.02:
                        safe_print("         IC表现: 良好 (>0.02)", "SUMMARY")
                    elif ic_value > 0:
                        safe_print("         IC表现: 一般 (>0)", "SUMMARY")
                    else:
                        safe_print("         IC表现: 较差 (≤0)", "SUMMARY")
                    
                    if annual_return > 0.1:
                        safe_print("         收益表现: 优秀 (>10%)", "SUMMARY")
                    elif annual_return > 0.05:
                        safe_print("         收益表现: 良好 (>5%)", "SUMMARY")
                    elif annual_return > 0:
                        safe_print("         收益表现: 一般 (>0%)", "SUMMARY")
                    else:
                        safe_print("         收益表现: 较差 (≤0%)", "SUMMARY")
                    
                    if sharpe_ratio > 1.0:
                        safe_print("         风险调整收益: 优秀 (>1.0)", "SUMMARY")
                    elif sharpe_ratio > 0.5:
                        safe_print("         风险调整收益: 良好 (>0.5)", "SUMMARY")
                    elif sharpe_ratio > 0:
                        safe_print("         风险调整收益: 一般 (>0)", "SUMMARY")
                    else:
                        safe_print("         风险调整收益: 较差 (≤0)", "SUMMARY")
            
            if not backtest_found:
                safe_print("[SUMMARY] 回测结果: 无有效数据", "SUMMARY")
            
            # 4. 显示SeekAlpha输出文件位置
            safe_print("[SUMMARY] SeekAlpha输出文件:", "SUMMARY")
            
            # 检查SeekAlpha输出目录
            seekalpha_base = Path("D:/SeekAlphaTrader-main/SeekAlphaTrader-main")
            outputs_dir = seekalpha_base / "outputs"
            figures_dir = outputs_dir / "figures"
            
            if figures_dir.exists():
                # 查找最新的图像文件
                png_files = list(figures_dir.glob("*.png"))
                if png_files:
                    # 按修改时间排序，获取最新的
                    latest_png = max(png_files, key=lambda x: x.stat().st_mtime)
                    safe_print(f"         收益图表: {latest_png}", "SUMMARY")
                else:
                    safe_print("         收益图表: 未找到PNG文件", "SUMMARY")
            else:
                safe_print("         收益图表: 输出目录不存在", "SUMMARY")
            
            # 检查CSV文件
            csv_zip_path = outputs_dir / "因子明细.zip"
            if csv_zip_path.exists():
                safe_print(f"         因子明细: {csv_zip_path}", "SUMMARY")
            else:
                safe_print("         因子明细: 未找到ZIP文件", "SUMMARY")
            
            # 检查当前工作目录的输出
            current_outputs = Path("./outputs")
            if current_outputs.exists():
                current_figures = current_outputs / "figures"
                if current_figures.exists():
                    png_files = list(current_figures.glob("*.png"))
                    if png_files:
                        latest_png = max(png_files, key=lambda x: x.stat().st_mtime)
                        safe_print(f"         本地图表: {latest_png.absolute()}", "SUMMARY")
            
            # 5. 显示反馈建议
            if 'feedback' in model_loop.loop_prev_out:
                feedback = model_loop.loop_prev_out['feedback']
                if hasattr(feedback, 'new_hypothesis'):
                    safe_print("[SUMMARY] 下一步建议:", "SUMMARY")
                    safe_print(f"         {feedback.new_hypothesis[:150]}...", "SUMMARY")
                elif hasattr(feedback, 'feedback'):
                    safe_print("[SUMMARY] 反馈建议:", "SUMMARY")
                    safe_print(f"         {str(feedback.feedback)[:150]}...", "SUMMARY")
        
        else:
            safe_print("[SUMMARY] 无法获取执行结果数据", "SUMMARY")
            safe_print("[SUMMARY] 请检查model_loop.loop_prev_out是否存在", "SUMMARY")
            
    except Exception as e:
        safe_print(f"生成表现总结时发生错误: {e}", "ERROR")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fire.Fire(main)
