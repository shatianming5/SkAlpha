# from alphaagent.components.coder.CoSTEER import CoSTEER
# from alphaagent.components.coder.CoSTEER.config import CoSTEER_SETTINGS
# from alphaagent.components.coder.CoSTEER.evaluators import CoSTEERMultiEvaluator
# from alphaagent.core.scenario import Scenario


# class ModelCoSTEER(CoSTEER):
#     def __init__(
#         self,
#         scen: Scenario,
#         *args,
#         **kwargs,
#     ) -> None:
#         eva = CoSTEERMultiEvaluator(
#             ModelCoSTEEREvaluator(scen=scen), scen=scen
#         )  # Please specify whether you agree running your eva in parallel or not
#         es = ModelMultiProcessEvolvingStrategy(scen=scen, settings=CoSTEER_SETTINGS)

#         super().__init__(*args, settings=CoSTEER_SETTINGS, eva=eva, es=es, evolving_version=1, scen=scen, **kwargs)
