"""
"""
from lm_eval.tasks.squad.task import SQUAD

class SQUADNoInd(SQUAD):
    TEXT_SUFFIX = ""

