"""
"""
from lm_eval.tasks.squad_no_template.task import SQUADNoTemplate

class SQUADNoTemplateNoInd(SQUADNoTemplate):
    TEXT_SUFFIX = ""

