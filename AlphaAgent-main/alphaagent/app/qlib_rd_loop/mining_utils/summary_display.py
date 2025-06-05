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
            safe_print("因子挖掘表现总结", "SUMMARY")
            safe_print("-" * 50, "SUMMARY")
            
            # 调试信息：检查model_loop的属性
            safe_print(f"model_loop类型: {type(model_loop)}", "SUMMARY")
            safe_print(f"是否有loop_prev_out属性: {hasattr(model_loop, 'loop_prev_out')}", "SUMMARY")
            safe_print(f"是否有final_results属性: {hasattr(model_loop, 'final_results')}", "SUMMARY")
            
            # 获取最新的执行结果 - 优先使用final_results
            results_data = None
            if hasattr(model_loop, 'get_final_results'):
                results_data = model_loop.get_final_results()
                safe_print("使用get_final_results()方法获取结果", "SUMMARY")
            elif hasattr(model_loop, 'final_results') and model_loop.final_results:
                results_data = model_loop.final_results
                safe_print("使用final_results属性获取结果", "SUMMARY")
            elif hasattr(model_loop, 'loop_prev_out') and model_loop.loop_prev_out:
                results_data = model_loop.loop_prev_out
                safe_print("使用loop_prev_out属性获取结果", "SUMMARY")
            
            if results_data:
                safe_print(f"结果数据类型: {type(results_data)}", "SUMMARY")
                safe_print(f"结果数据键: {list(results_data.keys()) if isinstance(results_data, dict) else 'Not a dict'}", "SUMMARY")
                
                self._display_hypothesis(results_data)
                self._display_factors(results_data)
                self._display_backtest_results(results_data)
                self._display_output_files()
                self._display_feedback(results_data)
            else:
                safe_print("无法获取执行结果数据", "SUMMARY")
                safe_print("请检查model_loop的结果存储", "SUMMARY")
                
                # 尝试获取其他可能的结果属性
                if hasattr(model_loop, '__dict__'):
                    safe_print(f"model_loop可用属性: {list(model_loop.__dict__.keys())}", "SUMMARY")
                
        except Exception as e:
            safe_print(f"生成表现总结时发生错误: {e}", "ERROR")
            import traceback
            traceback.print_exc()
    
    def _display_hypothesis(self, loop_prev_out: Dict[str, Any]):
        """显示生成的假设"""
        if 'factor_propose' in loop_prev_out:
            hypothesis = loop_prev_out['factor_propose']
            if hasattr(hypothesis, 'hypothesis'):
                safe_print(f"市场假设: {hypothesis.hypothesis[:100]}...", "SUMMARY")
            else:
                safe_print("市场假设: 已生成", "SUMMARY")
        else:
            safe_print("市场假设: 未找到", "SUMMARY")
    
    def _display_factors(self, loop_prev_out: Dict[str, Any]):
        """显示构造的因子"""
        if 'factor_construct' in loop_prev_out:
            factors = loop_prev_out['factor_construct']
            if hasattr(factors, 'sub_tasks'):
                safe_print(f"构造因子数量: {len(factors.sub_tasks)}", "SUMMARY")
                for i, task in enumerate(factors.sub_tasks[:3], 1):  # 只显示前3个
                    if hasattr(task, 'factor_name'):
                        safe_print(f"   {i}. {task.factor_name}", "SUMMARY")
            else:
                safe_print("构造因子数量: 已完成", "SUMMARY")
        else:
            safe_print("构造因子: 未找到", "SUMMARY")
    
    def _display_backtest_results(self, loop_prev_out: Dict[str, Any]):
        """显示回测结果"""
        backtest_found = False
        
        if 'factor_backtest' in loop_prev_out:
            backtest_result = loop_prev_out['factor_backtest']
            result_data = self._extract_result_data(backtest_result)
            
            if result_data is not None and len(result_data) > 0:
                backtest_found = True
                safe_print("回测表现:", "SUMMARY")
                self._display_metrics(result_data)
                self._display_performance_evaluation(result_data)
        
        if not backtest_found:
            safe_print("回测结果: 无有效数据", "SUMMARY")
    
    def _extract_result_data(self, backtest_result):
        """提取回测结果数据"""
        result_data = None
        
        # 尝试多种方式提取结果数据
        if hasattr(backtest_result, 'result') and backtest_result.result is not None:
            result_data = backtest_result.result
            safe_print(f"从backtest_result.result获取数据，类型: {type(result_data)}", "SUMMARY")
        elif hasattr(backtest_result, 'backtest_result'):
            result_data = backtest_result.backtest_result
            safe_print(f"从backtest_result.backtest_result获取数据，类型: {type(result_data)}", "SUMMARY")
        elif isinstance(backtest_result, dict):
            result_data = backtest_result
            safe_print(f"直接使用字典数据，键: {list(backtest_result.keys())}", "SUMMARY")
        elif hasattr(backtest_result, '__dict__'):
            # 尝试从对象属性中查找结果
            attrs = backtest_result.__dict__
            safe_print(f"backtest_result对象属性: {list(attrs.keys())}", "SUMMARY")
            
            # 查找可能包含结果的属性
            for attr_name in ['result', 'results', 'backtest_result', 'performance', 'metrics']:
                if attr_name in attrs and attrs[attr_name] is not None:
                    result_data = attrs[attr_name]
                    safe_print(f"从属性{attr_name}获取数据，类型: {type(result_data)}", "SUMMARY")
                    break
        
        if result_data is None:
            safe_print(f"无法提取结果数据，backtest_result类型: {type(backtest_result)}", "SUMMARY")
        
        return result_data
    
    def _display_metrics(self, result_data: Dict[str, Any]):
        """显示关键指标"""
        safe_print(f"开始显示指标，数据类型: {type(result_data)}", "SUMMARY")
        
        # 如果是pandas Series，转换为字典
        if hasattr(result_data, 'to_dict'):
            result_data = result_data.to_dict()
            safe_print(f"转换Series为字典，键: {list(result_data.keys())}", "SUMMARY")
        elif not isinstance(result_data, dict):
            safe_print(f"无法处理的数据类型: {type(result_data)}", "SUMMARY")
            safe_print(f"数据内容: {str(result_data)[:200]}...", "SUMMARY")
            return
        
        safe_print(f"可用指标键: {list(result_data.keys())}", "SUMMARY")
        
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
                    safe_print(f"   {metric_name}: {formatted_value}", "SUMMARY")
                    displayed_metrics += 1
        
        if displayed_metrics == 0:
            safe_print("   [无有效指标数据]", "SUMMARY")
            # 显示所有可用的指标以便调试
            safe_print("   可用的原始指标:", "SUMMARY")
            for key, value in result_data.items():
                safe_print(f"     {key}: {value}", "SUMMARY")
    
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
        safe_print("表现评估:", "SUMMARY")
        
        ic_value = result_data.get('IC', result_data.get('test_ic', 0))
        annual_return = result_data.get('Annual Return', result_data.get('averaged_annualized_roi', 0))
        sharpe_ratio = result_data.get('Sharpe Ratio', result_data.get('sharpe_ratio', 0))
        
        # IC表现评估
        if ic_value > 0.05:
            safe_print("   IC表现: 优秀 (>0.05)", "SUMMARY")
        elif ic_value > 0.02:
            safe_print("   IC表现: 良好 (>0.02)", "SUMMARY")
        elif ic_value > 0:
            safe_print("   IC表现: 一般 (>0)", "SUMMARY")
        else:
            safe_print("   IC表现: 较差 (≤0)", "SUMMARY")
        
        # 收益表现评估
        if annual_return > 0.1:
            safe_print("   收益表现: 优秀 (>10%)", "SUMMARY")
        elif annual_return > 0.05:
            safe_print("   收益表现: 良好 (>5%)", "SUMMARY")
        elif annual_return > 0:
            safe_print("   收益表现: 一般 (>0%)", "SUMMARY")
        else:
            safe_print("   收益表现: 较差 (≤0%)", "SUMMARY")
        
        # 风险调整收益评估
        if sharpe_ratio > 1.0:
            safe_print("   风险调整收益: 优秀 (>1.0)", "SUMMARY")
        elif sharpe_ratio > 0.5:
            safe_print("   风险调整收益: 良好 (>0.5)", "SUMMARY")
        elif sharpe_ratio > 0:
            safe_print("   风险调整收益: 一般 (>0)", "SUMMARY")
        else:
            safe_print("   风险调整收益: 较差 (≤0)", "SUMMARY")
    
    def _display_output_files(self):
        """显示输出文件位置"""
        safe_print("输出文件:", "SUMMARY")
        
        # 检查git_ignore_folder输出
        git_ignore_folder = Path("git_ignore_folder")
        if git_ignore_folder.exists():
            self._check_output_directory(git_ignore_folder)
        else:
            safe_print("   git_ignore_folder: 目录不存在", "SUMMARY")
    
    def _check_output_directory(self, base_dir: Path):
        """检查输出目录"""
        # 检查图表文件
        figures_dir = base_dir / "figures"
        if figures_dir.exists():
            png_files = list(figures_dir.glob("*.png"))
            if png_files:
                latest_png = max(png_files, key=lambda x: x.stat().st_mtime)
                safe_print(f"   收益图表: {latest_png}", "SUMMARY")
            else:
                safe_print("   收益图表: 未找到PNG文件", "SUMMARY")
        else:
            safe_print("   收益图表: figures目录不存在", "SUMMARY")
        
        # 检查因子明细
        factors_dir = base_dir / "因子明细"
        if factors_dir.exists():
            csv_files = list(factors_dir.glob("*.csv"))
            zip_files = list(factors_dir.glob("*.zip"))
            
            if zip_files:
                latest_zip = max(zip_files, key=lambda x: x.stat().st_mtime)
                safe_print(f"   因子明细: {latest_zip}", "SUMMARY")
            elif csv_files:
                safe_print(f"   因子明细: {len(csv_files)}个CSV文件", "SUMMARY")
            else:
                safe_print("   因子明细: 无文件", "SUMMARY")
        else:
            safe_print("   因子明细: 目录不存在", "SUMMARY")
        
        # 检查数据文件
        data_dir = base_dir / "data"
        if data_dir.exists():
            data_files = list(data_dir.glob("*.zip"))
            if data_files:
                latest_data = max(data_files, key=lambda x: x.stat().st_mtime)
                safe_print(f"   数据包: {latest_data}", "SUMMARY")
            else:
                safe_print("   数据包: 无文件", "SUMMARY")
    
    def _display_feedback(self, loop_prev_out: Dict[str, Any]):
        """显示反馈建议"""
        if 'feedback' in loop_prev_out:
            feedback = loop_prev_out['feedback']
            if hasattr(feedback, 'new_hypothesis'):
                safe_print("下一步建议:", "SUMMARY")
                safe_print(f"   {feedback.new_hypothesis[:150]}...", "SUMMARY")
            elif hasattr(feedback, 'feedback'):
                safe_print("反馈建议:", "SUMMARY")
                safe_print(f"   {feedback.feedback[:150]}...", "SUMMARY")
        else:
            safe_print("反馈建议: 未找到", "SUMMARY") 