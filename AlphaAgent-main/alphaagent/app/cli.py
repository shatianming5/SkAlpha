"""
CLI entrance for all alphaagent application.

This will 
- make alphaagent a nice entry and
- autoamtically load dotenv
"""

import sys
import os

# 确保输出立即显示
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

from alphaagent.app.cli_utils import load_environment

# 加载环境变量
load_environment()

import subprocess
from importlib.resources import path as rpath

import fire

# from alphaagent.app.data_mining.model import main as med_model
# from alphaagent.app.general_model.general_model import (
#     extract_models_and_implement as general_model,
# )
# from alphaagent.app.kaggle.loop import main as kaggle_main
# from alphaagent.app.qlib_rd_loop.factor import main as fin_factor
# from alphaagent.app.qlib_rd_loop.factor_from_report import main as fin_factor_report
from alphaagent.app.qlib_rd_loop.factor_mining import main as mine
from alphaagent.app.qlib_rd_loop.factor_backtest import main as backtest


def mine_wrapper(*args, **kwargs):
    """
    mine命令的包装器，确保输出正常显示
    """
    print("[CLI] 调用mine命令...")
    sys.stdout.flush()
    try:
        result = mine(*args, **kwargs)
        print("[CLI] mine命令执行完成")
        sys.stdout.flush()
        return result
    except Exception as e:
        print(f"[CLI] mine命令执行失败: {e}")
        sys.stdout.flush()
        raise
# from alphaagent.app.qlib_rd_loop.model import main as fin_model
# 简化依赖，避免复杂导入
try:
    from alphaagent.app.utils.health_check import health_check
except ImportError:
    def health_check():
        print("健康检查功能不可用（缺少依赖）")

try:
    from alphaagent.app.utils.info import collect_info
except ImportError:
    def collect_info():
        print("信息收集功能不可用（缺少依赖）")

try:
    from alphaagent.utils.factor_export import FactorExporter
except ImportError:
    class FactorExporter:
        def __init__(self, *args, **kwargs):
            pass
        def scan_and_export_all_factors(self, *args, **kwargs):
            print("因子导出功能不可用（缺少依赖）")


def ui(port=19899, log_dir="", debug=False):
    """
    start web app to show the log traces.
    """
    with rpath("alphaagent.log.ui", "app.py") as app_path:
        cmds = ["streamlit", "run", app_path, f"--server.port={port}"]
        if log_dir or debug:
            cmds.append("--")
        if log_dir:
            cmds.append(f"--log_dir={log_dir}")
        if debug:
            cmds.append("--debug")
        subprocess.run(cmds)


def export_factors(workspace_root="git_ignore_folder/RD-Agent_workspace", output_dir="alphaagent"):
    """
    导出所有挖掘出来的因子信息和回测结果到CSV文件
    
    Args:
        workspace_root: 工作空间根目录，默认为git_ignore_folder/RD-Agent_workspace
        output_dir: 输出目录，默认为alphaagent
    """
    print(f"开始导出因子结果...")
    print(f"工作空间根目录: {workspace_root}")
    print(f"输出目录: {output_dir}")
    
    exporter = FactorExporter(output_dir=output_dir)
    exporter.scan_and_export_all_factors(workspace_root=workspace_root)
    
    print(f"因子结果已导出到: {exporter.csv_file_path}")


def app():
    print("[START] AlphaAgent CLI 启动中...")
    sys.stdout.flush()
    
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        command = sys.argv[1]
        print(f"[CLI] 执行命令: {command}")
        sys.stdout.flush()
        
        if command == "mine":
            # 直接调用mine函数，绕过Fire的缓冲
            import argparse
            parser = argparse.ArgumentParser(description='Factor mining')
            parser.add_argument('--potential_direction', type=str, help='Market hypothesis')
            parser.add_argument('--step_n', type=int, help='Number of loops (each loop has 5 steps: propose, construct, calculate, backtest, feedback)')
            parser.add_argument('--path', type=str, help='Session path')
            
            # 解析从第二个参数开始的参数
            args = parser.parse_args(sys.argv[2:])
            
            print(f"[CLI] 调用mine命令，参数: {args}")
            sys.stdout.flush()
            
            try:
                result = mine(
                    path=args.path,
                    step_n=args.step_n,
                    potential_direction=args.potential_direction
                )
                print("[CLI] mine命令执行完成")
                sys.stdout.flush()
                return result
            except Exception as e:
                print(f"[CLI] mine命令执行失败: {e}")
                sys.stdout.flush()
                raise
        else:
            # 对于其他命令，使用Fire
            fire.Fire(
                {
                    "mine": mine_wrapper,
                    "backtest": backtest,
                    "export_factors": export_factors,
                    "ui": ui,
                    "health_check": health_check,
                    "collect_info": collect_info,
                }
            )
    else:
        # 没有参数时显示帮助
        print("可用命令:")
        print("  mine --potential_direction <hypothesis> --step_n <loops>")
        print("    执行因子挖掘，每轮包含5个步骤：")
        print("    1. 假设生成 (factor_propose)")
        print("    2. 因子构造 (factor_construct)")
        print("    3. 因子计算 (factor_calculate)")
        print("    4. 因子回测 (factor_backtest)")
        print("    5. 反馈生成 (feedback)")
        print("  backtest")
        print("  export_factors")
        print("  ui")
        print("  health_check")
        print("  collect_info")
        sys.stdout.flush()
