"""
摘要显示模块
处理因子挖掘结果的摘要显示
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from alphaagent.app.cli_utils.unicode_handler import safe_print


class SummaryDisplayer:
    """摘要显示器"""
    
    def __init__(self):
        """初始化摘要显示器"""
        pass
    
    def display_factor_mining_summary(self, model_loop):
        """
        显示因子挖掘表现总结
        
        Args:
            model_loop: AlphaAgentLoop实例，包含执行结果
        """
        try:
            safe_print("[SUMMARY] 因子挖掘表现总结", "SUMMARY")
            safe_print("-" * 50, "SUMMARY")
            
            # 获取最新的执行结果
            if hasattr(model_loop, 'loop_prev_out') and model_loop.loop_prev_out:
                self._display_hypothesis(model_loop.loop_prev_out)
                self._display_factors(model_loop.loop_prev_out)
                self._display_backtest_results(model_loop.loop_prev_out)
                self._display_output_files()
                self._display_feedback(model_loop.loop_prev_out)
            else:
                safe_print("[SUMMARY] 无法获取执行结果数据", "SUMMARY")
                safe_print("[SUMMARY] 请检查model_loop.loop_prev_out是否存在", "SUMMARY")
                
        except Exception as e:
            safe_print(f"生成表现总结时发生错误: {e}", "ERROR")
            import traceback
            traceback.print_exc()
    
    def _display_hypothesis(self, loop_prev_out: Dict[str, Any]):
        """显示生成的假设"""
        if 'factor_propose' in loop_prev_out:
            hypothesis = loop_prev_out['factor_propose']
            if hasattr(hypothesis, 'hypothesis'):
                safe_print(f"[SUMMARY] 市场假设: {hypothesis.hypothesis[:100]}...", "SUMMARY")
            else:
                safe_print("[SUMMARY] 市场假设: 已生成", "SUMMARY")
    
    def _display_factors(self, loop_prev_out: Dict[str, Any]):
        """显示构造的因子"""
        if 'factor_construct' in loop_prev_out:
            factors = loop_prev_out['factor_construct']
            if hasattr(factors, 'sub_tasks'):
                safe_print(f"[SUMMARY] 构造因子数量: {len(factors.sub_tasks)}", "SUMMARY")
                for i, task in enumerate(factors.sub_tasks[:3], 1):  # 只显示前3个
                    if hasattr(task, 'factor_name'):
                        safe_print(f"         {i}. {task.factor_name}", "SUMMARY")
            else:
                safe_print("[SUMMARY] 构造因子数量: 已完成", "SUMMARY")
    
    def _display_backtest_results(self, loop_prev_out: Dict[str, Any]):
        """显示回测结果"""
        backtest_found = False
        
        if 'factor_backtest' in loop_prev_out:
            backtest_result = loop_prev_out['factor_backtest']
            result_data = self._extract_result_data(backtest_result)
            
            if result_data is not None and len(result_data) > 0:
                backtest_found = True
                safe_print("[SUMMARY] 回测表现:", "SUMMARY")
                self._display_metrics(result_data)
                self._display_performance_evaluation(result_data)
        
        if not backtest_found:
            safe_print("[SUMMARY] 回测结果: 无有效数据", "SUMMARY")
    
    def _extract_result_data(self, backtest_result):
        """提取回测结果数据"""
        result_data = None
        
        if hasattr(backtest_result, 'result') and backtest_result.result is not None:
            result_data = backtest_result.result
        elif hasattr(backtest_result, 'backtest_result'):
            result_data = backtest_result.backtest_result
        elif isinstance(backtest_result, dict):
            result_data = backtest_result
        
        return result_data
    
    def _display_metrics(self, result_data: Dict[str, Any]):
        """显示关键指标"""
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
                    formatted_value = self._format_metric_value(metric_key, value)
                    safe_print(f"         {metric_name}: {formatted_value}", "SUMMARY")
                    displayed_metrics += 1
        
        if displayed_metrics == 0:
            safe_print("         [无有效指标数据]", "SUMMARY")
    
    def _format_metric_value(self, metric_key: str, value: float) -> str:
        """格式化指标值"""
        percentage_metrics = [
            'Annual Return', 'averaged_annualized_roi',
            'Max Drawdown', 'max_drawdown',
            'monthly_win_rate', 'overall_trading_win_rate',
            'turnover'
        ]
        
        if metric_key in percentage_metrics:
            return f"{value:.4f} ({value*100:.2f}%)"
        else:
            return f"{value:.6f}"
    
    def _display_performance_evaluation(self, result_data: Dict[str, Any]):
        """显示表现评估"""
        safe_print("[SUMMARY] 表现评估:", "SUMMARY")
        
        ic_value = result_data.get('IC', result_data.get('test_ic', 0))
        annual_return = result_data.get('Annual Return', result_data.get('averaged_annualized_roi', 0))
        sharpe_ratio = result_data.get('Sharpe Ratio', result_data.get('sharpe_ratio', 0))
        
        # IC表现评估
        if ic_value > 0.05:
            safe_print("         IC表现: 优秀 (>0.05)", "SUMMARY")
        elif ic_value > 0.02:
            safe_print("         IC表现: 良好 (>0.02)", "SUMMARY")
        elif ic_value > 0:
            safe_print("         IC表现: 一般 (>0)", "SUMMARY")
        else:
            safe_print("         IC表现: 较差 (≤0)", "SUMMARY")
        
        # 收益表现评估
        if annual_return > 0.1:
            safe_print("         收益表现: 优秀 (>10%)", "SUMMARY")
        elif annual_return > 0.05:
            safe_print("         收益表现: 良好 (>5%)", "SUMMARY")
        elif annual_return > 0:
            safe_print("         收益表现: 一般 (>0%)", "SUMMARY")
        else:
            safe_print("         收益表现: 较差 (≤0%)", "SUMMARY")
        
        # 风险调整收益评估
        if sharpe_ratio > 1.0:
            safe_print("         风险调整收益: 优秀 (>1.0)", "SUMMARY")
        elif sharpe_ratio > 0.5:
            safe_print("         风险调整收益: 良好 (>0.5)", "SUMMARY")
        elif sharpe_ratio > 0:
            safe_print("         风险调整收益: 一般 (>0)", "SUMMARY")
        else:
            safe_print("         风险调整收益: 较差 (≤0)", "SUMMARY")
    
    def _display_output_files(self):
        """显示输出文件位置"""
        safe_print("[SUMMARY] 输出文件:", "SUMMARY")
        
        # 检查git_ignore_folder输出
        git_ignore_folder = Path("git_ignore_folder")
        if git_ignore_folder.exists():
            self._check_output_directory(git_ignore_folder)
        else:
            safe_print("         git_ignore_folder: 目录不存在", "SUMMARY")
    
    def _check_output_directory(self, base_dir: Path):
        """检查输出目录"""
        # 检查图表文件
        figures_dir = base_dir / "figures"
        if figures_dir.exists():
            png_files = list(figures_dir.glob("*.png"))
            if png_files:
                latest_png = max(png_files, key=lambda x: x.stat().st_mtime)
                safe_print(f"         收益图表: {latest_png}", "SUMMARY")
            else:
                safe_print("         收益图表: 未找到PNG文件", "SUMMARY")
        else:
            safe_print("         收益图表: figures目录不存在", "SUMMARY")
        
        # 检查因子明细
        factors_dir = base_dir / "因子明细"
        if factors_dir.exists():
            csv_files = list(factors_dir.glob("*.csv"))
            zip_files = list(factors_dir.glob("*.zip"))
            
            if zip_files:
                latest_zip = max(zip_files, key=lambda x: x.stat().st_mtime)
                safe_print(f"         因子明细: {latest_zip}", "SUMMARY")
            elif csv_files:
                safe_print(f"         因子明细: {len(csv_files)}个CSV文件", "SUMMARY")
            else:
                safe_print("         因子明细: 无文件", "SUMMARY")
        else:
            safe_print("         因子明细: 目录不存在", "SUMMARY")
        
        # 检查数据文件
        data_dir = base_dir / "data"
        if data_dir.exists():
            data_files = list(data_dir.glob("*.zip"))
            if data_files:
                latest_data = max(data_files, key=lambda x: x.stat().st_mtime)
                safe_print(f"         数据包: {latest_data}", "SUMMARY")
            else:
                safe_print("         数据包: 无文件", "SUMMARY")
    
    def _display_feedback(self, loop_prev_out: Dict[str, Any]):
        """显示反馈建议"""
        if 'feedback' in loop_prev_out:
            feedback = loop_prev_out['feedback']
            if hasattr(feedback, 'new_hypothesis'):
                safe_print("[SUMMARY] 下一步建议:", "SUMMARY")
                safe_print(f"         {feedback.new_hypothesis[:150]}...", "SUMMARY")
            elif hasattr(feedback, 'feedback'):
                safe_print("[SUMMARY] 反馈建议:", "SUMMARY")
                safe_print(f"         {str(feedback.feedback)[:150]}...", "SUMMARY") 