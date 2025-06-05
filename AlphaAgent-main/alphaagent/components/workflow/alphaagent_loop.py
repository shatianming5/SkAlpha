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
# from alphaagent.utils.factor_export import export_factor_mining_result  # 移除复杂依赖
import threading

# 导入拆解后的组件
from alphaagent.components.workflow.loop_components import (
    ComponentInitializer,
    StepExecutor,
    ResultExporter
)


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
            
            # 使用组件初始化器
            initializer = ComponentInitializer(PROP_SETTING, potential_direction, use_local)
            
            # 初始化各个组件
            scen = initializer.initialize_scenario()
            self.hypothesis_generator = initializer.initialize_hypothesis_generator(scen)
            self.factor_constructor = initializer.initialize_factor_constructor()
            self.coder = initializer.initialize_coder(scen)
            self.runner = initializer.initialize_runner(scen)
            self.summarizer = initializer.initialize_summarizer(scen)
            self.trace = initializer.initialize_trace(scen)
            
            # 初始化步骤执行器
            self.step_executor = StepExecutor(
                self.hypothesis_generator,
                self.factor_constructor,
                self.coder,
                self.runner,
                self.summarizer,
                self.trace,
                use_local
            )
            
            # 初始化结果导出器
            self.result_exporter = ResultExporter()
            
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

    @stop_event_check
    def factor_propose(self, prev_out: dict[str, Any]):
        """
        提出作为构建因子的基础的假设
        """
        return self.step_executor.execute_factor_propose(prev_out)

    @stop_event_check
    def factor_construct(self, prev_out: dict[str, Any]):
        """
        基于假设构造多个不同的因子
        """
        return self.step_executor.execute_factor_construct(prev_out)

    @stop_event_check
    def factor_calculate(self, prev_out: dict[str, Any]):
        """
        根据因子表达式计算过去的因子表（因子值）
        """
        return self.step_executor.execute_factor_calculate(prev_out)
    
    @stop_event_check
    def factor_backtest(self, prev_out: dict[str, Any]):
        """
        回测因子
        """
        exp = self.step_executor.execute_factor_backtest(prev_out)
        
        # 导出因子挖掘结果
        try:
            self.result_exporter.export_factor_results(prev_out["factor_calculate"], exp)
        except Exception as e:
            logger.warning(f"导出因子结果失败: {e}")
        
        return exp

    @stop_event_check
    def feedback(self, prev_out: dict[str, Any]):
        """
        生成反馈
        """
        return self.step_executor.execute_feedback(prev_out)




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
