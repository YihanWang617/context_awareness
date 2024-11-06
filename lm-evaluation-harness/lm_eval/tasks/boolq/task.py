"""
"""
import re
from typing import List

import numpy as np

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
import string
ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return ARTICLES_REGEX.sub(" ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

class BoolQCompletion(ConfigurableTask):
    VERSION = 0
    DATASET_PATH = "google/boolq"
    DATASET_NAME = "default"

    def __init__(self):
        super().__init__(config={"metadata": {"version": self.VERSION}})

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False

    def validation_docs(self):
        return self.dataset["validation"]

    def doc_to_text(self, doc):
        return doc["passage"] + "\nAnswer the question according to the above passage. " + doc["question"].capitalize() + "?"# + " Base your answer more on the given information."
    
    def doc_to_hint(self, doc):
        return ""

    def doc_to_target(self, doc):
        return 'yes' if doc['answer'] else 'no'

    def construct_requests(self, doc, ctx, **kwargs):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        return [
            Instance(
                request_type="generate_until",
                doc=doc,
                arguments=(ctx, {"until": [], "max_gen_toks": 50}, ""),
                idx=0,
                **kwargs,
            ),
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, "\n" + "Yes", ""),
                idx=0,
                **kwargs,
            ),
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, "\n" + "No", ""),
                idx=0,
                **kwargs,
            ),
        ]

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # continuation, (logprob_unanswerable, _) = results
        continuation, (yes_likelihood, _), (no_likelihood, _)= results

        labels = ["yes"] if doc['answer'] else ["no"]
        # print(continuation)
        # import pdb; pdb.set_trace()
        # print(doc['question'], continuation, labels, contains_score(continuation[0], labels))
        return {'accuracy': ((yes_likelihood > no_likelihood) == doc['answer'])}

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            # "contains": np.mean,  # Exact match (the normalized answer exactly match the gold answer)
            "accuracy": np.mean,
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            # "contains": True,  # Exact match (the normalized answer exactly match the gold answer
            "accuracy": True,
        }


def contains_score(prediction: str, labels: List[str]):
    return max(
        int(bool(re.search(re.compile(re.escape(label), re.IGNORECASE), normalize_answer(prediction))))
        for label in labels
    )