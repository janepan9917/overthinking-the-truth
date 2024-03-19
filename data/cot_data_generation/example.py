from dataclasses import dataclass, field
import pandas as pd
import copy 
import random
import logging
from typing import List

@dataclass
class Example:
    """ A single example from the dataset. """

    qid: int
    task: str
    question: str                       # The question text
    gt_solution: str                    # The ground truth solution
    solution: str                       # The provided solution

    correct_chatgpt_solution: str = None      # ChatGPT's solution (correct)
    mistake_type: str = "no_mistake"    # The type of mistake 
    mistake_step: int = None            # The step at which the mistake was made

