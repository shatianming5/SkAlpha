"""
组件初始化器
负责初始化AlphaAgent循环中的各个组件
"""

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


class ComponentInitializer:
    """组件初始化器"""
    
    def __init__(self, prop_setting: BaseFacSetting, potential_direction: str, use_local: bool = True):
        """
        初始化组件初始化器
        
        Args:
            prop_setting: 属性设置
            potential_direction: 潜在方向
            use_local: 是否使用本地环境
        """
        self.prop_setting = prop_setting
        self.potential_direction = potential_direction
        self.use_local = use_local
    
    def initialize_scenario(self) -> Scenario:
        """
        初始化场景
        
        Returns:
            场景实例
        """
        print(f"[SCENARIO] 正在创建场景 (Scenario)...")
        scen: Scenario = import_class(self.prop_setting.scen)(use_local=self.use_local)
        logger.log_object(scen, tag="scenario")
        print(f"[OK] 场景创建完成")
        return scen
    
    def initialize_hypothesis_generator(self, scenario: Scenario) -> HypothesisGen:
        """
        初始化假设生成器
        
        Args:
            scenario: 场景实例
            
        Returns:
            假设生成器实例
        """
        print(f"[HYPOTHESIS] 正在创建假设生成器...")
        hypothesis_generator: HypothesisGen = import_class(self.prop_setting.hypothesis_gen)(
            scenario, self.potential_direction
        )
        logger.log_object(hypothesis_generator, tag="hypothesis generator")
        print(f"[OK] 假设生成器创建完成")
        return hypothesis_generator
    
    def initialize_factor_constructor(self) -> Hypothesis2Experiment:
        """
        初始化因子构造器
        
        Returns:
            因子构造器实例
        """
        print(f"[CONSTRUCTOR] 正在创建因子构造器...")
        factor_constructor: Hypothesis2Experiment = import_class(self.prop_setting.hypothesis2experiment)()
        logger.log_object(factor_constructor, tag="experiment generation")
        print(f"[OK] 因子构造器创建完成")
        return factor_constructor
    
    def initialize_coder(self, scenario: Scenario) -> Developer:
        """
        初始化编码器
        
        Args:
            scenario: 场景实例
            
        Returns:
            编码器实例
        """
        print(f"[CODER] 正在创建编码器...")
        coder: Developer = import_class(self.prop_setting.coder)(scenario)
        logger.log_object(coder, tag="coder")
        print(f"[OK] 编码器创建完成")
        return coder
    
    def initialize_runner(self, scenario: Scenario) -> Developer:
        """
        初始化运行器
        
        Args:
            scenario: 场景实例
            
        Returns:
            运行器实例
        """
        print(f"[RUNNER] 正在创建运行器...")
        runner: Developer = import_class(self.prop_setting.runner)(scenario)
        logger.log_object(runner, tag="runner")
        print(f"[OK] 运行器创建完成")
        return runner
    
    def initialize_summarizer(self, scenario: Scenario) -> HypothesisExperiment2Feedback:
        """
        初始化总结器
        
        Args:
            scenario: 场景实例
            
        Returns:
            总结器实例
        """
        print(f"[SUMMARIZER] 正在创建总结器...")
        summarizer: HypothesisExperiment2Feedback = import_class(self.prop_setting.summarizer)(scenario)
        logger.log_object(summarizer, tag="summarizer")
        print(f"[OK] 总结器创建完成")
        return summarizer
    
    def initialize_trace(self, scenario: Scenario) -> Trace:
        """
        初始化跟踪器
        
        Args:
            scenario: 场景实例
            
        Returns:
            跟踪器实例
        """
        return Trace(scen=scenario) 