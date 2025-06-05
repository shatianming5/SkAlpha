import os
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pickle

from alphaagent.log import logger
from alphaagent.components.coder.factor_coder.factor import FactorTask


class FactorExporter:
    """
    因子导出器，用于将挖掘出来的因子信息和回测结果导出到CSV文件
    """
    
    def __init__(self, output_dir: str = "alphaagent"):
        """
        初始化因子导出器
        
        Args:
            output_dir: 输出目录，默认为alphaagent
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.csv_file_path = self.output_dir / "factor_mining_results.csv"
        
    def extract_factor_info_from_workspace(self, workspace_path: Path) -> Dict[str, Any]:
        """
        从工作空间提取因子信息
        
        Args:
            workspace_path: 工作空间路径
            
        Returns:
            包含因子信息的字典
        """
        factor_info = {}
        
        try:
            # 读取因子代码文件
            factor_py_path = workspace_path / "factor.py"
            if factor_py_path.exists():
                factor_code = self._read_file_with_encoding(factor_py_path)
                if factor_code:
                    # 提取因子表达式和名称
                    lines = factor_code.split('\n')
                    for line in lines:
                        if 'expr = ' in line and '"' in line:
                            # 提取因子表达式
                            start_idx = line.find('"') + 1
                            end_idx = line.rfind('"')
                            if start_idx > 0 and end_idx > start_idx:
                                factor_info['factor_expression'] = line[start_idx:end_idx]
                        elif 'name = ' in line and '"' in line:
                            # 提取因子名称
                            start_idx = line.find('"') + 1
                            end_idx = line.rfind('"')
                            if start_idx > 0 and end_idx > start_idx:
                                factor_info['factor_name'] = line[start_idx:end_idx]
            
            # 读取因子值文件信息
            result_h5_path = workspace_path / "result.h5"
            if result_h5_path.exists():
                factor_info['factor_file_path'] = str(result_h5_path)
                factor_info['factor_file_size'] = result_h5_path.stat().st_size
                
                # 尝试读取因子值的基本统计信息
                try:
                    factor_data = pd.read_hdf(result_h5_path)
                    factor_info['factor_data_shape'] = str(factor_data.shape)
                    factor_info['factor_data_mean'] = float(factor_data.mean()) if not factor_data.empty else 0.0
                    factor_info['factor_data_std'] = float(factor_data.std()) if not factor_data.empty else 0.0
                    factor_info['factor_data_min'] = float(factor_data.min()) if not factor_data.empty else 0.0
                    factor_info['factor_data_max'] = float(factor_data.max()) if not factor_data.empty else 0.0
                except Exception as e:
                    logger.warning(f"无法读取因子数据统计信息: {e}")
                    
            factor_info['workspace_id'] = workspace_path.name
            factor_info['creation_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        except Exception as e:
            logger.error(f"提取因子信息失败: {e}")
            
        return factor_info
    
    def extract_backtest_results(self, workspace_path: Path, backtest_result: pd.Series = None) -> Dict[str, Any]:
        """
        从工作空间提取回测结果
        
        Args:
            workspace_path: 工作空间路径
            backtest_result: 直接传入的回测结果（pandas Series）
            
        Returns:
            包含回测结果的字典
        """
        backtest_results = {}
        
        try:
            # 优先使用传入的回测结果
            if backtest_result is not None and isinstance(backtest_result, pd.Series):
                logger.info("使用传入的回测结果数据")
                for metric_name, metric_value in backtest_result.items():
                    # 添加前缀以区分来源
                    key_name = f'live_backtest_{metric_name}' if not metric_name.startswith('backtest_') else metric_name
                    backtest_results[key_name] = float(metric_value) if pd.notna(metric_value) else 0.0
                return backtest_results
            
            # 如果没有传入结果，则从文件中读取
            # 读取qlib回测结果
            qlib_res_path = workspace_path / "qlib_res.csv"
            if qlib_res_path.exists():
                qlib_results = pd.read_csv(qlib_res_path, index_col=0)
                for metric_name, metric_value in qlib_results.iloc[:, 0].items():
                    backtest_results[f'backtest_{metric_name}'] = float(metric_value) if pd.notna(metric_value) else 0.0
            
            # 读取详细回测报告
            ret_pkl_path = workspace_path / "ret.pkl"
            if ret_pkl_path.exists():
                try:
                    with open(ret_pkl_path, 'rb') as f:
                        ret_data = pickle.load(f)
                    if isinstance(ret_data, pd.DataFrame) and not ret_data.empty:
                        # 提取关键性能指标
                        if 'excess_return_without_cost' in ret_data.columns:
                            excess_returns = ret_data['excess_return_without_cost'].dropna()
                            if not excess_returns.empty:
                                backtest_results['annual_return'] = float(excess_returns.mean() * 252)
                                backtest_results['volatility'] = float(excess_returns.std() * (252**0.5))
                                backtest_results['sharpe_ratio'] = float(backtest_results['annual_return'] / backtest_results['volatility']) if backtest_results['volatility'] > 0 else 0.0
                                backtest_results['max_drawdown'] = float(self._calculate_max_drawdown(excess_returns))
                except Exception as e:
                    logger.warning(f"无法读取详细回测结果: {e}")
                    
        except Exception as e:
            logger.error(f"提取回测结果失败: {e}")
            
        return backtest_results
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        计算最大回撤
        
        Args:
            returns: 收益率序列
            
        Returns:
            最大回撤值
        """
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return abs(drawdown.min())
        except:
            return 0.0
    
    def extract_hypothesis_info(self, log_dir: Path, session_id: int) -> Dict[str, Any]:
        """
        从日志中提取假设信息
        
        Args:
            log_dir: 日志目录
            session_id: 会话ID
            
        Returns:
            包含假设信息的字典
        """
        hypothesis_info = {}
        
        try:
            session_dir = log_dir / "__session__" / str(session_id)
            if session_dir.exists():
                # 读取假设提出阶段的信息（pickle文件）
                propose_file = session_dir / "0_factor_propose"
                if propose_file.exists():
                    try:
                        import pickle
                        with open(propose_file, 'rb') as f:
                            loop_obj = pickle.load(f)
                        
                        # 从loop对象中提取假设信息
                        if hasattr(loop_obj, 'loop_prev_out') and isinstance(loop_obj.loop_prev_out, dict):
                            factor_propose = loop_obj.loop_prev_out.get('factor_propose')
                            if factor_propose and hasattr(factor_propose, '__str__'):
                                hypothesis_text = str(factor_propose)
                                # 提取假设的主要部分
                                if 'Hypothesis:' in hypothesis_text:
                                    hypothesis_start = hypothesis_text.find('Hypothesis:') + len('Hypothesis:')
                                    hypothesis_end = hypothesis_text.find('Concise Observation:', hypothesis_start)
                                    if hypothesis_end == -1:
                                        hypothesis_end = len(hypothesis_text)
                                    hypothesis_description = hypothesis_text[hypothesis_start:hypothesis_end].strip()
                                    hypothesis_info['hypothesis_description'] = hypothesis_description[:500]  # 限制长度
                                
                                # 提取完整的假设信息作为备用
                                hypothesis_info['full_hypothesis'] = hypothesis_text[:1000]  # 限制长度
                    except Exception as e:
                        logger.warning(f"无法从pickle文件提取假设信息: {e}")
                
                # 读取因子构造阶段的信息（pickle文件）
                construct_file = session_dir / "1_factor_construct"
                if construct_file.exists():
                    try:
                        import pickle
                        with open(construct_file, 'rb') as f:
                            loop_obj = pickle.load(f)
                        
                        # 从loop对象中提取构造信息
                        if hasattr(loop_obj, 'loop_prev_out') and isinstance(loop_obj.loop_prev_out, dict):
                            factor_construct = loop_obj.loop_prev_out.get('factor_construct')
                            if factor_construct:
                                # 尝试获取更有意义的构造信息
                                construct_info = []
                                
                                # 检查是否有sub_tasks属性
                                if hasattr(factor_construct, 'sub_tasks'):
                                    sub_tasks = factor_construct.sub_tasks
                                    if isinstance(sub_tasks, list) and len(sub_tasks) > 0:
                                        construct_info.append(f"生成了 {len(sub_tasks)} 个因子任务")
                                        # 获取第一个任务的信息作为示例
                                        first_task = sub_tasks[0]
                                        if hasattr(first_task, 'factor_name'):
                                            construct_info.append(f"示例因子: {first_task.factor_name}")
                                        if hasattr(first_task, 'factor_description'):
                                            construct_info.append(f"描述: {first_task.factor_description}")
                                
                                # 检查是否有其他有用的属性
                                if hasattr(factor_construct, 'experiment_setting'):
                                    construct_info.append(f"实验设置: {factor_construct.experiment_setting}")
                                
                                if construct_info:
                                    hypothesis_info['factor_construction_log'] = "; ".join(construct_info)[:500]
                                else:
                                    # 如果没有找到有用信息，使用类名和基本信息
                                    class_name = type(factor_construct).__name__
                                    hypothesis_info['factor_construction_log'] = f"因子构造对象: {class_name}"
                    except Exception as e:
                        logger.warning(f"无法从pickle文件提取构造信息: {e}")
                        
        except Exception as e:
            logger.warning(f"提取假设信息失败: {e}")
            
        return hypothesis_info
    
    def _read_file_with_encoding(self, file_path: Path) -> str:
        """
        尝试用多种编码读取文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容字符串，如果读取失败返回空字符串
        """
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'latin1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                logger.warning(f"读取文件 {file_path} 时发生错误: {e}")
                break
        
        # 如果所有编码都失败，尝试二进制读取并忽略错误
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                # 尝试解码为字符串，忽略错误
                return content.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.warning(f"无法读取文件 {file_path}: {e}")
            return ""
    
    def export_factor_result(self, 
                           factor_info: Dict[str, Any], 
                           backtest_results: Dict[str, Any], 
                           hypothesis_info: Dict[str, Any] = None) -> None:
        """
        导出因子结果到CSV文件
        
        Args:
            factor_info: 因子信息
            backtest_results: 回测结果
            hypothesis_info: 假设信息
        """
        try:
            # 合并所有信息
            combined_info = {}
            combined_info.update(factor_info)
            combined_info.update(backtest_results)
            if hypothesis_info:
                combined_info.update(hypothesis_info)
            
            # 创建DataFrame
            new_row = pd.DataFrame([combined_info])
            
            # 如果CSV文件已存在，则追加；否则创建新文件
            if self.csv_file_path.exists():
                existing_df = pd.read_csv(self.csv_file_path)
                updated_df = pd.concat([existing_df, new_row], ignore_index=True)
            else:
                updated_df = new_row
            
            # 保存到CSV文件
            updated_df.to_csv(self.csv_file_path, index=False, encoding='utf-8-sig')
            logger.info(f"因子结果已导出到: {self.csv_file_path}")
            
        except Exception as e:
            logger.error(f"导出因子结果失败: {e}")
    
    def scan_and_export_all_factors(self, workspace_root: str = "git_ignore_folder/RD-Agent_workspace") -> None:
        """
        扫描所有工作空间并导出因子结果
        
        Args:
            workspace_root: 工作空间根目录
        """
        workspace_root_path = Path(workspace_root)
        if not workspace_root_path.exists():
            logger.warning(f"工作空间根目录不存在: {workspace_root_path}")
            return
        
        exported_count = 0
        for workspace_dir in workspace_root_path.iterdir():
            if workspace_dir.is_dir():
                try:
                    # 提取因子信息
                    factor_info = self.extract_factor_info_from_workspace(workspace_dir)
                    if not factor_info:
                        continue
                    
                    # 提取回测结果
                    backtest_results = self.extract_backtest_results(workspace_dir)
                    
                    # 导出结果
                    self.export_factor_result(factor_info, backtest_results)
                    exported_count += 1
                    
                except Exception as e:
                    logger.error(f"处理工作空间 {workspace_dir} 失败: {e}")
        
        logger.info(f"共导出 {exported_count} 个因子结果")


# 全局导出器实例
_factor_exporter = None

def get_factor_exporter() -> FactorExporter:
    """获取全局因子导出器实例"""
    global _factor_exporter
    if _factor_exporter is None:
        _factor_exporter = FactorExporter()
    return _factor_exporter

def export_factor_mining_result(workspace_path: Path, log_dir: Path = None, session_id: int = None, backtest_result: pd.Series = None) -> None:
    """
    导出因子挖掘结果的便捷函数
    
    Args:
        workspace_path: 工作空间路径
        log_dir: 日志目录
        session_id: 会话ID
        backtest_result: 直接传入的回测结果（pandas Series）
    """
    exporter = get_factor_exporter()
    
    # 提取因子信息
    factor_info = exporter.extract_factor_info_from_workspace(workspace_path)
    if not factor_info:
        logger.warning(f"无法从工作空间 {workspace_path} 提取因子信息")
        return
    
    # 提取回测结果，优先使用传入的结果
    backtest_results = exporter.extract_backtest_results(workspace_path, backtest_result)
    
    # 提取假设信息
    hypothesis_info = {}
    if log_dir and session_id is not None:
        hypothesis_info = exporter.extract_hypothesis_info(log_dir, session_id)
    
    # 导出结果
    exporter.export_factor_result(factor_info, backtest_results, hypothesis_info) 