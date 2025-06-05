from alphaagent.components.workflow.conf import BasePropSetting
from alphaagent.core.conf import ExtendedSettingsConfigDict


class ModelBasePropSetting(BasePropSetting):
    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_MODEL_", protected_namespaces=())

    # 1) override base settings
    scen: str = "alphaagent.scenarios.qlib.experiment.model_experiment.QlibModelScenario"
    """Scenario class for Qlib Model"""

    hypothesis_gen: str = "alphaagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "alphaagent.scenarios.qlib.proposal.model_proposal.QlibModelHypothesis2Experiment"
    """Hypothesis to experiment class"""

    coder: str = "alphaagent.scenarios.qlib.developer.model_coder.QlibModelCoSTEER"
    """Coder class"""

    runner: str = "alphaagent.scenarios.qlib.developer.model_runner.QlibModelRunner"
    """Runner class"""

    summarizer: str = "alphaagent.scenarios.qlib.developer.feedback.QlibModelHypothesisExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 10
    """Number of evolutions"""


class FactorBasePropSetting(BasePropSetting):
    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())

    # 1) override base settings
    scen: str = "alphaagent.scenarios.qlib.experiment.factor_experiment.QlibFactorScenario"
    """Scenario class for Qlib Factor"""

    hypothesis_gen: str = "alphaagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "alphaagent.scenarios.qlib.proposal.factor_proposal.QlibFactorHypothesis2Experiment"
    """Hypothesis to experiment class"""

    coder: str = "alphaagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER"
    """Coder class"""

    runner: str = "alphaagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"
    """Runner class"""

    summarizer: str = "alphaagent.scenarios.qlib.developer.feedback.QlibFactorHypothesisExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 10
    """Number of evolutions"""
    

class AlphaAgentFactorBasePropSetting(BasePropSetting):
    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())

    # 1) override base settings
    scen: str = "alphaagent.scenarios.qlib.experiment.factor_experiment.QlibAlphaAgentScenario"
    """Scenario class for Qlib Factor"""

    hypothesis_gen: str = "alphaagent.scenarios.qlib.proposal.factor_proposal.AlphaAgentHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "alphaagent.scenarios.qlib.proposal.factor_proposal.AlphaAgentHypothesis2FactorExpression"
    """Hypothesis to experiment class"""

    # coder: str = "alphaagent.scenarios.qlib.developer.factor_coder.QlibFactorCoSTEER"
    coder: str = "alphaagent.scenarios.qlib.developer.factor_coder.QlibFactorParser"
    """Coder class"""

    runner: str = "alphaagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"
    """Runner class"""

    summarizer: str = "alphaagent.scenarios.qlib.developer.feedback.AlphaAgentQlibFactorHypothesisExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 5
    """Number of evolutions"""

class FactorBackTestBasePropSetting(BasePropSetting):
    model_config = ExtendedSettingsConfigDict(env_prefix="QLIB_FACTOR_", protected_namespaces=())

    # 1) override base settings
    scen: str = "alphaagent.scenarios.qlib.experiment.factor_experiment.QlibAlphaAgentScenario"
    """Scenario class for Qlib Factor"""

    hypothesis_gen: str = "alphaagent.scenarios.qlib.proposal.factor_proposal.EmptyHypothesisGen"
    """Hypothesis generation class"""

    hypothesis2experiment: str = "alphaagent.scenarios.qlib.proposal.factor_proposal.BacktestHypothesis2FactorExpression"
    """Hypothesis to experiment class"""

    coder: str = "alphaagent.scenarios.qlib.developer.factor_coder.QlibFactorCoder"
    """Coder class"""

    runner: str = "alphaagent.scenarios.qlib.developer.factor_runner.QlibFactorRunner"
    """Runner class"""

    summarizer: str = "alphaagent.scenarios.qlib.developer.feedback.QlibFactorHypothesisExperiment2Feedback"
    """Summarizer class"""

    evolving_n: int = 1
    """Number of evolutions"""


class FactorFromReportPropSetting(FactorBasePropSetting):
    # 1) override the scen attribute
    scen: str = "alphaagent.scenarios.qlib.experiment.factor_from_report_experiment.QlibFactorFromReportScenario"
    """Scenario class for Qlib Factor from Report"""

    # 2) sub task specific:
    report_result_json_file_path: str = "git_ignore_folder/report_list.json"
    """Path to the JSON file listing research reports for factor extraction"""

    max_factors_per_exp: int = 10000
    """Maximum number of factors implemented per experiment"""

    is_report_limit_enabled: bool = False
    """Limits report processing count if True; processes all if False"""


FACTOR_PROP_SETTING = FactorBasePropSetting()
FACTOR_FROM_REPORT_PROP_SETTING = FactorFromReportPropSetting()
MODEL_PROP_SETTING = ModelBasePropSetting()
ALPHA_AGENT_FACTOR_PROP_SETTING = AlphaAgentFactorBasePropSetting()
FACTOR_BACK_TEST_PROP_SETTING = FactorBackTestBasePropSetting()