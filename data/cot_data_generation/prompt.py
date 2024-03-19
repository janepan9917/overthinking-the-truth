from dataclasses import dataclass, field
import tiktoken
import argparse

# @dataclass
# class BasicPrompt():
#     """
#     Assume all prompts take this form:

#     <question_prefix> <sep> <question> <sep> <model_answer_prefix>


#     """
#     question_prefix: str
#     model_answer_prefix: str
#     sep: str = "\n"
#     logit_bias: dict = None

#     def build_prompt(self, question: str) -> str:
#         return self.sep.join([
#             self.question_prefix, 
#             question, 
#             self.model_answer_prefix,
#         ])


@dataclass
class Prompt():
    """
    Assume all prompts take this form:

    <question_prefix> <sep> <question> <sep> <solution_prefix> <sep> <solution> <sep> <suffix>


    """
    question_prefix: str
    solution_prefix: str
    suffix: str
    sep: str = "\n\n"
    logit_bias: dict = None

    def build_prompt(self, question: str, solution: str) -> str:
        return self.sep.join([
            self.question_prefix, 
            question, 
            self.solution_prefix, 
            solution, 
            self.suffix,
        ])

@dataclass
class VerifyPrompt():
    """
    Assume all verifying prompts take this form:

    <question_prefix> <sep> <question> <sep> <suffix>


    """
    question_prefix: str
    suffix: str
    sep: str = "\n\n"
    logit_bias: dict = None
    
    def build_prompt(self, question: str) -> str:
        return self.sep.join([
            self.question_prefix, 
            question, 
            self.suffix,
        ])
    
    def check_model_answer(self, response: str, gt_answer: str) -> bool:
        # import pdb; pdb.set_trace()
        try:
            model_answer = response.split(self.answer_prefix)[1].strip()
            return float(eval(model_answer)) == float(eval(gt_answer))
        except:
            return False


# @dataclass
# class RTEPrompt(VerifyPrompt):
#     question_prefix: str = "Premise:"
#     suffix: str = "Your answer should be Yes or No. Only provide the answer to the question. For instance, if the answer is No, your response should be \"Answer: No\" without intermediate steps."
#     # logit_bias: dict = field(
#     #     default_factory=lambda: {"Yes": 100.0, "No": 100.0}
#     # )
    
#     sep: str = "\n"

@dataclass
class RTEPrompt(VerifyPrompt):
    question_prefix: str = "Premise:"
    suffix: str = ""
    logit_bias: dict = field(
        default_factory=lambda: {"Yes": 100.0, "No": 100.0}
    )
    sep: str = "\n"

@dataclass
class RTECoTPrompt(VerifyPrompt):
    question_prefix: str = "Premise:"
    suffix: str = "Your answer should be Yes or No. First, explain your answer, and end your response with \"Answer:\".\n\n Let's think step-by-step."
    sep: str = "\n"

@dataclass
class GSM8KPrompt(VerifyPrompt):
    question_prefix: str = "Question:"
    suffix: str = "Only provide the answer to the question. For instance, if the answer is 5, your response should be \"Answer: 5\" without intermediate steps."
    sep: str = "\n"

@dataclass
class GSM8KCoTPrompt(VerifyPrompt):
    question_prefix: str = "Question:"
    suffix: str = "Your answer should appear at the end of your response after \"Answer:\". For instance, if your answer is 5, end your response with \"Answer: 5\".\n\n Let's think step-by-step."
    sep: str = "\n"

@dataclass
class CommonsenseQAPrompt(VerifyPrompt):
    question_prefix: str = "Question:"
    logit_bias: dict = field(
        default_factory=lambda: {d: 100 for d in "ABCDE"}
    )
    suffix: str = ""
    sep: str = "\n"

@dataclass
class CommonsenseQACoTPrompt(VerifyPrompt):
    question_prefix: str = "Question:"
    suffix: str = "Your answer should be one of A, B, C, D, or E and appear at the end of your response after \"Answer:\". For instance, if your answer is B, end your response with \"Answer: B\". There is only one right answer, so only put one letter.\n\n Let's think step-by-step."
    sep: str = "\n"

@dataclass
class StrategyQAPrompt(VerifyPrompt):
    question_prefix: str = "Question:"
    suffix: str = ""
    logit_bias: dict = field(
        default_factory=lambda: {"Yes": 100.0, "No": 100.0}
    )
    sep: str = "\n"

@dataclass
class StrategyQACoTPrompt(VerifyPrompt):
    question_prefix: str = "Question:"
    suffix: str = "Your answer should be Yes or No, and should appear at the end of your response after \"Answer:\". For instance, if your answer is Yes, end your response with \"Answer: Yes\".\n\n Let's think step-by-step."
    sep: str = "\n"

@dataclass
class HotpotQAPrompt(VerifyPrompt):
    question_prefix: str = "Below is a list of short articles. At the end is a question which can be answered using the information in the articles. Answer the question based on the information in the articles."
    suffix: str = "Only provide the answer to the question. For instance, if the answer is George Washington, your response should be \"Answer: George Washington\" without intermediate steps."
    sep: str = "\n\n"

@dataclass
class HotpotQACoTPrompt(VerifyPrompt):
    question_prefix: str = "Below is a list of short articles. At the end is a question which can be answered using the information in the articles. Answer the question based on the information in the articles."
    suffix: str = "Your answer should appear at the end of your response after \"Answer:\". For instance, if your answer is George Washington, end your response with \"Answer: George Washington\".\n\n Let's think step-by-step."
    sep: str = "\n"


@dataclass
class StandardPrompt(Prompt):
    question_prefix: str = "Here's a math question:"
    solution_prefix: str = "Here's a proposed solution:"
    suffix: str = "Is each step of the solution correct? If not, what and where is the mistake?"

# @dataclass
# class DoesMistakeExistPrompt(Prompt):
#     question_prefix: str = "Here's a math question:"
#     solution_prefix: str = "Here's a proposed solution:"
#     suffix: str = "Is there a mistake in the solution? Answer \"Yes\" or \"No\"."
#     logit_bias: dict = {"Yes": 100.0, "No": 100.0}

# @dataclass
# class MistakeStepPrompt(Prompt):
#     question_prefix: str = "Here's a math question:"
#     solution_prefix: str = "Here's a proposed solution:"
#     suffix: str = "Which step has a mistake? Answer with the step number. For instance, if Step 1 has a mistake, answer with \"1\"."
#     logit_bias: dict = {range(1, 11): 100.0}


@dataclass
class AlgebraVerifyPrompt(VerifyPrompt):
    question_prefix: str = "Here's a math question:"
    solution_prefix: str = "Answer: x = "
    suffix: str = "What is the value of x? End your solution with the final answer.\n\nStart the final answer with \"Answer: x = \". For instance, if x = 5, end the answer with \"Answer: x = 5\"." 


@dataclass
class ProntoQAVerifyPrompt(VerifyPrompt):
    question_prefix: str = "Consider the information and statement below:"
    solution_prefix: str = "Answer: "
    suffix: str = "Given the information, is the statement true or false?\n\nStart the final answer with \"Answer: \". For instance, if the statement is True, end the answer with \"Answer: True\"." 


def get_prompter(args: argparse.Namespace) -> Prompt:
    """ Gets the prompter. """
    if args.dataset == "rte" and args.cot:
        return RTECoTPrompt()
    elif args.dataset == "rte":
        return RTEPrompt()
    elif args.dataset == "gsm8k" and args.cot:
        return GSM8KCoTPrompt()
    elif args.dataset == "gsm8k":
        return GSM8KPrompt()
    elif args.dataset == "commonsense_qa" and args.cot:
        return CommonsenseQACoTPrompt()
    elif args.dataset == "commonsense_qa":
        return CommonsenseQAPrompt()
    elif args.dataset == "hotpot_qa" and args.cot:
        return HotpotQACoTPrompt()
    elif args.dataset == "hotpot_qa":
        return HotpotQAPrompt()
    elif args.dataset == "strategy_qa" and args.cot:
        return StrategyQACoTPrompt()
    elif args.dataset == "strategy_qa":
        return StrategyQAPrompt()
    else:
        raise ValueError(f"Invalid dataset {args.dataset}")