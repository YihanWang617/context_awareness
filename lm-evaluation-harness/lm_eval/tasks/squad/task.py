"""
"""
import re
from typing import List

import numpy as np

from lm_eval.api.instance import Instance
from lm_eval.api.task import ConfigurableTask
import string
import datasets
from functools import partial
import evaluate
from lm_eval.api.metrics import mean

ARTICLES_REGEX = re.compile(r"\b(a|an|the)\b", re.UNICODE)

def contains_score(prediction, reference):
    return max(
        int(bool(re.search(re.compile(re.escape(normalize_answer(label)), re.IGNORECASE), normalize_answer(prediction))))
        for label in reference
    )
    # return sum([max(
    #     int(bool(re.search(re.compile(re.escape(label), re.IGNORECASE), prediction)))
    #     for label in reference
    # ) for (prediction, reference) in items])/len(items)

def _squad_metric(predictions, references):
    squad_metric = evaluate.load("squad_v2")
    return squad_metric.compute(predictions=predictions, references=references)


def _squad_agg(key, items):
    predictions, references = zip(*items)

    return _squad_metric(predictions=predictions, references=references).get(key, 0)

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


class SQUAD(ConfigurableTask):
    VERSION = 0
    DATASET_PATH = "rajpurkar/squad"
    DATASET_NAME = "plain_text"
    TEXT_SUFFIX = " [IND]"

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
        return doc["context"] + "\nAnswer the question according to the above passage: " + doc["question"] + self.TEXT_SUFFIX #" Base your answer more on the given information."
        # return doc["context"] + "\n" + doc["question"] + self.TEXT_SUFFIX# + " Base your answer on the given context."
    
    def doc_to_hint(self, doc):
        return ""

    def doc_to_target(self, doc):
        return doc["answers"]["text"]

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
                arguments=(ctx, {"until": [". "], "max_gen_toks": 100}, ""),
                idx=0,
                **kwargs,
            )
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
        continuation = results

        predictions = {
            "id": doc["id"],
            "prediction_text": continuation[0],
            "no_answer_probability": 0,
        }

        references = {
            "id": doc["id"],
            "answers": doc["answers"],
        }

        return {
            # "exact": (
            #     predictions,
            #     references,
            # ),  # Exact match (the normalized answer exactly match the gold answer)
            # "f1": (
            #     predictions,
            #     references,
            # ),
            # "contains": (
            #     continuation[0],
            #     doc["answers"]['text']
            # )
            'contains': contains_score(continuation[0], references['answers']['text'])
        }

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            # "exact": partial(
            #     _squad_agg, "exact"
            # ),  # Exact match (the normalized answer exactly match the gold answer)
            # "f1": partial(
            #     _squad_agg, "f1"
            # ),
            "contains": mean
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "exact": True,  # Exact match (the normalized answer exactly match the gold answer)
            "f1": True,}

