from alphaagent.components.coder.CoSTEER import CoSTEER
from alphaagent.components.coder.CoSTEER.config import CoSTEER_SETTINGS
from alphaagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
from alphaagent.components.coder.model_coder.evaluators import ModelCoSTEEREvaluator
from alphaagent.components.coder.model_coder.evolving_strategy import (
    ModelMultiProcessEvolvingStrategy,
)
from alphaagent.core.scenario import Scenario


class ModelCoSTEER(CoSTEER):
    def __init__(
        self,
        scen: Scenario,
        *args,
        **kwargs,
    ) -> None:
        eva = CoSTEERMultiEvaluator(ModelCoSTEEREvaluator(scen=scen), scen=scen)
        es = ModelMultiProcessEvolvingStrategy(scen=scen, settings=CoSTEER_SETTINGS)

        super().__init__(*args, settings=CoSTEER_SETTINGS, eva=eva, es=es, evolving_version=2, scen=scen, **kwargs)
