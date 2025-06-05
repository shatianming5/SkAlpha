"""
Model workflow with session control
It is from `rdagent/app/qlib_rd_loop/model.py` and try to replace `rdagent/app/qlib_rd_loop/RDAgent.py`
"""

import time
import pandas as pd
from typing import Any

from alphaagent.components.workflow.conf import BaseFacSetting
from alphaagent.core.developer import Developer
from alphaagent.core.proposal import (
    Hypothesis2Experiment,
    HypothesisExperiment2Feedback,
    HypothesisGen,  
    Trace,
)
from alphaagent.core.scenario import Scenario
from alphaagent.core.utils import import_class
from alphaagent.log import logger
from alphaagent.log.time import measure_time
from alphaagent.utils.workflow import LoopBase, LoopMeta
from alphaagent.core.exception import FactorEmptyError
from alphaagent.utils.factor_export import export_factor_mining_result
import threading


import datetime
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tqdm.auto import tqdm

from alphaagent.core.exception import CoderError
from alphaagent.log import logger
from functools import wraps

# 定义装饰器：在函数调用前检查stop_event

            
def stop_event_check(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if STOP_EVENT is not None and STOP_EVENT.is_set():
            # 当收到停止信号时，可以直接抛出异常或返回特定值，这里示例抛出异常
            raise Exception("Operation stopped due to stop_event flag.")
        return func(self, *args, **kwargs)
    return wrapper


class AlphaAgentLoop(LoopBase, metaclass=LoopMeta):
    skip_loop_error = (FactorEmptyError,)
    
    @measure_time
    def __init__(self, PROP_SETTING: BaseFacSetting, potential_direction, stop_event: threading.Event, use_local: bool = True):
        print(f"[INIT] 正在初始化AlphaAgentLoop组件...")
        with logger.tag("init"):
            self.use_local = use_local
            self.current_session_id = 0  # 添加会话ID跟踪
            logger.info(f"初始化AlphaAgentLoop，使用{'本地环境' if use_local else 'Docker容器'}回测")
            
            print(f"[SCENARIO] 正在创建场景 (Scenario)...")
            scen: Scenario = import_class(PROP_SETTING.scen)(use_local=use_local)
            logger.log_object(scen, tag="scenario")
            print(f"[OK] 场景创建完成")

            ### 换成基于初始hypo的，生成完整的hypo
            print(f"[HYPOTHESIS] 正在创建假设生成器...")
            self.hypothesis_generator: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen, potential_direction)
            logger.log_object(self.hypothesis_generator, tag="hypothesis generator")
            print(f"[OK] 假设生成器创建完成")

            ### 换成一次生成10个因子
            print(f"[CONSTRUCTOR] 正在创建因子构造器...")
            self.factor_constructor: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)()
            logger.log_object(self.factor_constructor, tag="experiment generation")
            print(f"[OK] 因子构造器创建完成")

            ### 加入代码执行中的 Variables / Functions
            print(f"[CODER] 正在创建编码器...")
            self.coder: Developer = import_class(PROP_SETTING.coder)(scen)
            logger.log_object(self.coder, tag="coder")
            print(f"[OK] 编码器创建完成")
            
            print(f"[RUNNER] 正在创建运行器...")
            self.runner: Developer = import_class(PROP_SETTING.runner)(scen)
            logger.log_object(self.runner, tag="runner")
            print(f"[OK] 运行器创建完成")

            print(f"[SUMMARIZER] 正在创建总结器...")
            self.summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
            logger.log_object(self.summarizer, tag="summarizer")
            print(f"[OK] 总结器创建完成")
            
            self.trace = Trace(scen=scen)
            
            global STOP_EVENT
            STOP_EVENT = stop_event
            super().__init__()
            print(f"[SUCCESS] AlphaAgentLoop所有组件初始化完成！")

    @classmethod
    def load(cls, path, use_local: bool = True):
        """加载现有会话"""
        instance = super().load(path)
        instance.use_local = use_local
        # 从路径中提取会话ID
        try:
            session_path = Path(path)
            if "__session__" in str(session_path):
                session_parts = str(session_path).split("__session__")[-1].split("/")
                instance.current_session_id = int(session_parts[1]) if len(session_parts) > 1 else 0
            else:
                instance.current_session_id = 0
        except:
            instance.current_session_id = 0
        logger.info(f"加载AlphaAgentLoop，使用{'本地环境' if use_local else 'Docker容器'}回测")
        return instance

    @measure_time
    @stop_event_check
    def factor_propose(self, prev_out: dict[str, Any]):
        """
        提出作为构建因子的基础的假设
        """
        print(f"[STEP1] 步骤1: 正在生成因子假设...")
        with logger.tag("r"):  
            idea = self.hypothesis_generator.gen(self.trace)
            logger.log_object(idea, tag="hypothesis generation")
        print(f"[OK] 因子假设生成完成")
        return idea

    @measure_time
    @stop_event_check
    def factor_construct(self, prev_out: dict[str, Any]):
        """
        基于假设构造多个不同的因子
        """
        print(f"[STEP2] 步骤2: 正在构造因子...")
        with logger.tag("r"): 
            factor = self.factor_constructor.convert(prev_out["factor_propose"], self.trace)
            logger.log_object(factor.sub_tasks, tag="experiment generation")
        print(f"[OK] 因子构造完成")
        return factor

    @measure_time
    @stop_event_check
    def factor_calculate(self, prev_out: dict[str, Any]):
        """
        根据因子表达式计算过去的因子表（因子值）
        """
        print(f"[STEP3] 步骤3: 正在计算因子值...")
        with logger.tag("d"):  # develop
            factor = self.coder.develop(prev_out["factor_construct"])
            logger.log_object(factor.sub_workspace_list, tag="coder result")
        print(f"[OK] 因子值计算完成")
        return factor
    

    @measure_time
    @stop_event_check
    def factor_backtest(self, prev_out: dict[str, Any]):
        """
        回测因子
        """
        print(f"[STEP4] 步骤4: 正在执行因子回测...")
        with logger.tag("ef"):  # evaluate and feedback
            logger.info(f"Start factor backtest (Local: {self.use_local})")
            exp = self.runner.develop(prev_out["factor_calculate"], use_local=self.use_local)
            if exp is None:
                logger.error(f"Factor extraction failed.")
                raise FactorEmptyError("Factor extraction failed.")
            logger.log_object(exp, tag="runner result")
            
            # 导出因子挖掘结果到CSV文件，传递回测结果
            try:
                self.__export_factor_results(prev_out["factor_calculate"], exp)
            except Exception as e:
                logger.warning(f"导出因子结果失败: {e}")
        
        print(f"[OK] 因子回测完成")        
        return exp

    def __export_factor_results(self, factor_calculate_result, backtest_exp):
        """
        导出因子结果到CSV文件
        
        Args:
            factor_calculate_result: 因子计算结果
            backtest_exp: 回测实验结果，包含exp.result
        """
        try:
            # 获取最新的日志目录
            log_dir_path = None
            log_root = Path("log")
            if log_root.exists():
                log_dirs = [d for d in log_root.iterdir() if d.is_dir()]
                if log_dirs:
                    log_dir_path = max(log_dirs, key=lambda x: x.stat().st_mtime)
            
            # 遍历所有工作空间并导出结果
            if hasattr(factor_calculate_result, 'sub_workspace_list'):
                for workspace in factor_calculate_result.sub_workspace_list:
                    if hasattr(workspace, 'workspace_path') and workspace.workspace_path:
                        workspace_path = Path(workspace.workspace_path)
                        if workspace_path.exists():
                            logger.info(f"导出因子结果: {workspace_path}")
                            export_factor_mining_result(
                                workspace_path=workspace_path,
                                log_dir=log_dir_path,
                                session_id=self.current_session_id,
                                backtest_result=backtest_exp.result if backtest_exp else None
                            )
            
            # 更新会话ID
            self.current_session_id += 1
            
        except Exception as e:
            logger.error(f"导出因子结果时发生错误: {e}")

    @measure_time
    @stop_event_check
    def feedback(self, prev_out: dict[str, Any]):
        print(f"📝 步骤5: 正在生成反馈...")
        feedback = self.summarizer.generate_feedback(prev_out["factor_backtest"], prev_out["factor_propose"], self.trace)
        with logger.tag("ef"):  # evaluate and feedback
            logger.log_object(feedback, tag="feedback")
        self.trace.hist.append((prev_out["factor_propose"], prev_out["factor_backtest"], feedback))
        print(f"✅ 反馈生成完成")




class BacktestLoop(LoopBase, metaclass=LoopMeta):
    skip_loop_error = (FactorEmptyError,)
    @measure_time
    def __init__(self, PROP_SETTING: BaseFacSetting, factor_path=None):
        with logger.tag("init"):

            self.factor_path = factor_path

            scen: Scenario = import_class(PROP_SETTING.scen)()
            logger.log_object(scen, tag="scenario")

            self.hypothesis_generator: HypothesisGen = import_class(PROP_SETTING.hypothesis_gen)(scen)
            logger.log_object(self.hypothesis_generator, tag="hypothesis generator")

            self.factor_constructor: Hypothesis2Experiment = import_class(PROP_SETTING.hypothesis2experiment)(factor_path=factor_path)
            logger.log_object(self.factor_constructor, tag="experiment generation")

            self.coder: Developer = import_class(PROP_SETTING.coder)(scen, with_feedback=False, with_knowledge=False, knowledge_self_gen=False)
            logger.log_object(self.coder, tag="coder")
            
            self.runner: Developer = import_class(PROP_SETTING.runner)(scen)
            logger.log_object(self.runner, tag="runner")

            self.summarizer: HypothesisExperiment2Feedback = import_class(PROP_SETTING.summarizer)(scen)
            logger.log_object(self.summarizer, tag="summarizer")
            self.trace = Trace(scen=scen)
            super().__init__()

    def factor_propose(self, prev_out: dict[str, Any]):
        """
        Market hypothesis on which factors are built
        """
        with logger.tag("r"):  
            idea = self.hypothesis_generator.gen(self.trace)
            logger.log_object(idea, tag="hypothesis generation")
        return idea
        

    @measure_time
    def factor_construct(self, prev_out: dict[str, Any]):
        """
        Construct a variety of factors that depend on the hypothesis
        """
        with logger.tag("r"): 
            factor = self.factor_constructor.convert(prev_out["factor_propose"], self.trace)
            logger.log_object(factor.sub_tasks, tag="experiment generation")
        return factor

    @measure_time
    def factor_calculate(self, prev_out: dict[str, Any]):
        """
        Debug factors and calculate their values
        """
        with logger.tag("d"):  # develop
            factor = self.coder.develop(prev_out["factor_construct"])
            logger.log_object(factor.sub_workspace_list, tag="coder result")
        return factor
    

    @measure_time
    def factor_backtest(self, prev_out: dict[str, Any]):
        """
        Conduct Backtesting
        """
        with logger.tag("ef"):  # evaluate and feedback
            exp = self.runner.develop(prev_out["factor_calculate"])
            if exp is None:
                logger.error(f"Factor extraction failed.")
                raise FactorEmptyError("Factor extraction failed.")
            logger.log_object(exp, tag="runner result")
        return exp

    @measure_time
    def stop(self, prev_out: dict[str, Any]):
        exit(0)
