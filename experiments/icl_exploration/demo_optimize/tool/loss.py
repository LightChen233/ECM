from typing import Any
import textgrad as tg
from textgrad.loss import MultiFieldTokenParsedEvaluation
class MathPredictionLoss():
    def __init__(self, backward_engine) -> None:
        eval_system_prompt = tg.Variable("You are a language model that evaluates the accuracy of a prediction for a mathematical task. Only call a prediction accurate if it is the same as the ground truth answer.", requires_grad=False, role_description="system prompt for the evaluation")

        evaluation_instruction = "Below is a question from a question-answering task, the ground truth answer, and a prediction. Is the final prediction correct, i.e. the same as the ground truth answer? Say only 1 (yes) or 0 (no). Return your response within <ACCURACY> </ACCURACY> tags. e.g.<ACCURACY> 0 </ACCURACY> or <ACCURACY> 1 </ACCURACY>"
        eval_instruction = tg.Variable(evaluation_instruction, requires_grad=False, role_description="evaluation instruction for the task")
        role_descriptions = [
                "Question for the task",
                "Correct answer",
                "Solution and prediction from the language model"
            ]
        self.eval_fn = MultiFieldTokenParsedEvaluation(
            eval_instruction,
            engine=backward_engine,
            role_descriptions=role_descriptions,
            parse_tags=["<ACCURACY>", "</ACCURACY>"],
            system_prompt=eval_system_prompt
        )
        # return self.eval_fn
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.eval_fn(*args, **kwds)
    
    def backward(self):
        return self.eval_fn.backward()