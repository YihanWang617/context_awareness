"""
"""
from lm_eval.tasks.quac_no_template.task import QUACNoTemplate

class QUACNoTemplateNoInd(QUACNoTemplate):
    TEXT_SUFFIX = ""

