"""
步骤执行器
负责执行AlphaAgent循环中的各个步骤
"""

from typing import Any, Dict
from alphaagent.core.developer import Developer
from alphaagent.core.proposal import (
    Hypothesis2Experiment,
    HypothesisExperiment2Feedback,
    HypothesisGen,  
    Trace,
)
from alphaagent.core.exception import FactorEmptyError
from alphaagent.log import logger
from alphaagent.log.time import measure_time


class StepExecutor:
    """步骤执行器"""
    
    def __init__(self, 
                 hypothesis_generator: HypothesisGen,
                 factor_constructor: Hypothesis2Experiment,
                 coder: Developer,
                 runner: Developer,
                 summarizer: HypothesisExperiment2Feedback,
                 trace: Trace,
                 use_local: bool = True):
        """
        初始化步骤执行器
        
        Args:
            hypothesis_generator: 假设生成器
            factor_constructor: 因子构造器
            coder: 编码器
            runner: 运行器
            summarizer: 总结器
            trace: 跟踪器
            use_local: 是否使用本地环境
        """
        self.hypothesis_generator = hypothesis_generator
        self.factor_constructor = factor_constructor
        self.coder = coder
        self.runner = runner
        self.summarizer = summarizer
        self.trace = trace
        self.use_local = use_local
    
    @measure_time
    def execute_factor_propose(self, prev_out: Dict[str, Any]):
        """
        执行因子假设生成步骤
        
        Args:
            prev_out: 前一步输出
            
        Returns:
            假设生成结果
        """
        print(f"[STEP1] 步骤1: 正在生成因子假设...")
        with logger.tag("r"):  
            idea = self.hypothesis_generator.gen(self.trace)
            logger.log_object(idea, tag="hypothesis generation")
        print(f"[OK] 因子假设生成完成")
        return idea
    
    @measure_time
    def execute_factor_construct(self, prev_out: Dict[str, Any]):
        """
        执行因子构造步骤
        
        Args:
            prev_out: 前一步输出
            
        Returns:
            因子构造结果
        """
        print(f"[STEP2] 步骤2: 正在构造因子...")
        with logger.tag("r"): 
            factor = self.factor_constructor.convert(prev_out["factor_propose"], self.trace)
            logger.log_object(factor.sub_tasks, tag="experiment generation")
        print(f"[OK] 因子构造完成")
        return factor
    
    @measure_time
    def execute_factor_calculate(self, prev_out: Dict[str, Any]):
        """
        执行因子计算步骤
        
        Args:
            prev_out: 前一步输出
            
        Returns:
            因子计算结果
        """
        print(f"[STEP3] 步骤3: 正在计算因子值...")
        with logger.tag("d"):  # develop
            factor = self.coder.develop(prev_out["factor_construct"])
            logger.log_object(factor.sub_workspace_list, tag="coder result")
        print(f"[OK] 因子值计算完成")
        return factor
    
    @measure_time
    def execute_factor_backtest(self, prev_out: Dict[str, Any]):
        """
        执行因子回测步骤
        
        Args:
            prev_out: 前一步输出
            
        Returns:
            回测结果
        """
        print(f"[STEP4] 步骤4: 正在执行因子回测...")
        with logger.tag("ef"):  # evaluate and feedback
            logger.info(f"Start factor backtest (Local: {self.use_local})")
            exp = self.runner.develop(prev_out["factor_calculate"], use_local=self.use_local)
            if exp is None:
                logger.error(f"Factor extraction failed.")
                raise FactorEmptyError("Factor extraction failed.")
            logger.log_object(exp, tag="runner result")
        
        print(f"[OK] 因子回测完成")        
        return exp
    
    @measure_time
    def execute_feedback(self, prev_out: Dict[str, Any]):
        """
        执行反馈生成步骤
        
        Args:
            prev_out: 前一步输出
            
        Returns:
            反馈结果
        """
        print(f"[STEP5] 步骤5: 正在生成反馈...")
        with logger.tag("ef"):  # evaluate and feedback
            # 获取最新的假设
            hypothesis = prev_out.get("factor_propose")
            if hypothesis is None and len(self.trace.hist) > 0:
                hypothesis = self.trace.hist[-1][0]  # 获取最后一个假设
            
            feedback = self.summarizer.generate_feedback(
                prev_out["factor_backtest"], 
                hypothesis, 
                self.trace
            )
            logger.log_object(feedback, tag="feedback generation")
        print(f"[OK] 反馈生成完成")
        return feedback 