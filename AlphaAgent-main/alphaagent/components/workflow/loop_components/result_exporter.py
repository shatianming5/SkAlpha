"""
结果导出器
负责导出因子挖掘结果
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from alphaagent.log import logger


class ResultExporter:
    """结果导出器"""
    
    def __init__(self):
        """初始化结果导出器"""
        pass
    
    def export_factor_results(self, factor_calculate_result: Any, backtest_exp: Any):
        """
        导出因子结果到CSV文件
        
        Args:
            factor_calculate_result: 因子计算结果
            backtest_exp: 回测实验结果，包含exp.result
        """
        try:
            # 获取最新的日志目录
            log_dir_path = self._get_latest_log_dir()
            
            if log_dir_path is None:
                logger.warning("无法找到日志目录，跳过因子结果导出")
                return
            
            # 简化的导出逻辑，避免复杂依赖
            logger.info(f"因子结果导出目标目录: {log_dir_path}")
            logger.info("因子结果导出功能已简化，避免复杂依赖")
            
        except Exception as e:
            logger.error(f"导出因子结果时发生错误: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_latest_log_dir(self) -> Optional[Path]:
        """
        获取最新的日志目录
        
        Returns:
            最新日志目录路径，如果不存在则返回None
        """
        log_root = Path("log")
        if not log_root.exists():
            return None
        
        log_dirs = [d for d in log_root.iterdir() if d.is_dir()]
        if not log_dirs:
            return None
        
        # 按修改时间排序，获取最新的
        latest_log_dir = max(log_dirs, key=lambda x: x.stat().st_mtime)
        return latest_log_dir
    
    def export_to_git_ignore_folder(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        导出结果到git_ignore_folder
        
        Args:
            results: 结果字典
            
        Returns:
            导出文件路径字典
        """
        try:
            # 导入输出管理器
            from git_ignore_folder.output_manager import output_manager
            
            exported_files = {}
            
            # 导出因子数据包
            if 'trade_signals' in results and 'total_portfolios' in results:
                factor_zip_path = output_manager.create_factor_package(
                    trade_signals=results.get('trade_signals'),
                    portfolios=results.get('total_portfolios'),
                    alpha_table=results.get('alpha_table')
                )
                exported_files['factor_package'] = factor_zip_path
            
            # 导出回测报告
            report_path = output_manager.save_backtest_report(results)
            exported_files['report'] = report_path
            
            # 显示输出摘要
            output_manager.display_summary()
            
            return exported_files
            
        except Exception as e:
            logger.error(f"导出到git_ignore_folder时发生错误: {e}")
            return {}
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            from git_ignore_folder.output_manager import output_manager
            output_manager.clean_temp_files()
        except Exception as e:
            logger.warning(f"清理临时文件时发生错误: {e}") 