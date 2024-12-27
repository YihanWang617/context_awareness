from langchain.evaluation import load_evaluator
from langchain.chat_models import ChatOpenAI
from statistics import mean
import string

class Evaluator:
    def __init__(self, evaluator_name):
        self.evaluator_name = evaluator_name

    def evaluate_response(self, response):
        pass


class GPT4Evaluator(Evaluator):
    def __init__(self, evaluator_name='gpt4'):
        super().__init__(evaluator_name)

        self.evaluation_model = ChatOpenAI(
                model="gpt-4", temperature=0, openai_api_key=self.openai_api_key
            )

    def evaluate_response(self, response):
        accuracy_criteria = {
            "accuracy": """
            Score 1: The answer is completely unrelated to the reference.
            Score 3: The answer has minor relevance but does not align with the reference.
            Score 5: The answer has moderate relevance but contains inaccuracies.
            Score 7: The answer aligns with the reference but has minor omissions.
            Score 10: The answer is completely accurate and aligns perfectly with the reference.
            Only respond with a numberical score
            """
        }

        # Using GPT-4 to evaluate
        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=accuracy_criteria,
            llm=self.evaluation_model,
        )

        eval_result = evaluator.evaluate_strings(
            # The models response
            prediction=response,
            # The actual answer
            reference=self.needle,
            # The question asked
            input=self.retrieval_question,
        )

        return int(eval_result["score"])
    

class SubstringMatchEvaluator(Evaluator):
    def __init__(self, evaluator_name='substring_match', substr_validation_words=[]):
        super().__init__(evaluator_name)

        self.substr_validation_words = substr_validation_words

    def evaluate_response(self, response):
        response_lower = response.lower()
        if all(word in response_lower for word in self.substr_validation_words):
            return 1
        else:
            return 0
        

class SubwordMatchEvaluator(Evaluator):
    def __init__(self, evaluator_name='subword_match', substr_validation_words=[]):
        super().__init__(evaluator_name)
        self.substr_validation_words = substr_validation_words

    def evaluate_response(self, response):
        for p in string.punctuation:
            response = response.lower().replace(p, ' ')

        response = response.split(' ')
        response = sum([r.split('\n') for r in response], [])
        response = [a.strip() for a in response]
        res = mean(any([r.startswith(word) for r in response]) for word in self.substr_validation_words)
        return res
    

def load_evaluator(evaluation_method, substr_validation_words):
    if evaluation_method == 'gpt4':
        return GPT4Evaluator()
    elif evaluation_method == 'substring_match':
        return SubstringMatchEvaluator(substr_validation_words=substr_validation_words)
    elif evaluation_method == 'subword_match':
        return SubwordMatchEvaluator(substr_validation_words=substr_validation_words)
    else:
        raise NotImplementedError()