from dataclasses import dataclass, field
from prompt import *
from utils import *
from datasets import load_dataset
from example import *

@dataclass
class Dataset():
    name: str = None
    fn: str = None
    length: int = None
    examples: List[Example] = field(init=False)
    manual_data: bool = False
    verifier = None
    max_num_examples: int = 10


    def verify_examples():
        raise NotImplementedError()

    def generate_examples(self):
        raise NotImplementedError()

    def make_incomplete_solution():
        raise NotImplementedError()

    def make_typo_solution(self, example) -> (int, str):
        # get the full solution and split it into steps
        full_solution = copy.deepcopy(example["gt_solution"])
        split_step_sol = full_solution.split("\n")
        

        # ensure that the typo is different
        typo_solution = full_solution
        while typo_solution == full_solution: 

            # Last line = Answer, ineligible for typo
            ms = random.randint(0, len(split_step_sol)-2) 

            # remove the prefix, since it contains step number
            prefix = split_step_sol[ms].split(":")[0]
            sol_step = split_step_sol[ms][len(prefix)+1:]
            legal_indices = [i for i, c in enumerate(sol_step) if c in "0123456789"]

            # if there were numbers in this step, randomly replace one
            if len(legal_indices) != 0:
                idx = random.choice(legal_indices)
                typo_step = prefix + ":" + sol_step[:idx] + random.choice("123456789") + sol_step[idx+1:]
                split_step_sol[ms] = typo_step
                typo_solution = "\n".join(split_step_sol)
        
        return ms+1, typo_solution # indexing starts at 1

    # def make_human_plausible_solution():
    #     raise NotImplementedError()

@dataclass
class RTEDataset(Dataset):
    name = "rte"

    def generate_examples(self):
        data = load_dataset("SetFit/rte", split="train")
        
        examples = []
        for i, d in enumerate(data):
            q = f"{d['text1']}\nBased on this premise, can we conclude the hypothesis \"{d['text2']}\" is true?"
            solution = "No" if d["label"] == 1 else "Yes"

            examples.append(Example(
                qid=i,
                task=self.name,
                question=q,
                gt_solution=solution,
                solution=None,
            ))

        return examples
        
@dataclass
class GSM8KDataset(Dataset):
    name = "gsm8k"

    def generate_examples(self):
        data = load_dataset("gsm8k", "main", split="train")

        examples = []
        for i, d in enumerate(data):
            q = d["question"]
            solution = d["answer"].split("####")[-1].strip()

            examples.append(Example(
                qid=i,
                task=self.name,
                question=q,
                gt_solution=solution,
                solution=None,
            ))

        return examples
        


@dataclass
class CommonsenseQADataset(Dataset):
    name = "commonsense_qa"

    def generate_examples(self):
        data = load_dataset("commonsense_qa",  split="train")

        examples = []
        for i, d in enumerate(data):
            answers = "\n".join([f"{l}: {c}" for c, l in zip(d["choices"]["text"], d["choices"]["label"])])
            q = f"{d['question']}\n\nOptions:\n{answers}"

            examples.append(Example(
                qid=d["id"],
                task=self.name,
                question=q,
                gt_solution=d["answerKey"],
                solution=None,
            ))

        return examples


class StrategyQADataset(Dataset):
    name = "strategy_qa"

    def generate_examples(self):
        data = load_dataset("wics/strategy-qa",  split="test")

        examples = []
        for i, d in enumerate(data):
            examples.append(Example(
                qid=d["qid"],
                task=self.name,
                question=d["question"],
                gt_solution="Yes" if d["answer"] == True else "No",
                solution=None,
            ))

        return examples
        
class HotpotQADataset(Dataset):
    name = "hotpot_qa"

    def generate_examples(self):
        data = load_dataset("hotpot_qa", "distractor", split="train")
        examples = []
        for i, d in enumerate(data):
            context = self.get_context(d["context"])
            q = f"Context:\n{context}\n\nQuestion:\n{d['question']}"
            examples.append(Example(
                qid=d["id"],
                task=self.name,
                question=q,
                gt_solution=d["answer"],
                solution=None,
            ))

        return examples
    
    def get_context(self, context_info):
        context = []
        for i in range(len(context_info)):
            title = context_info["title"][i]
            sentences = "".join(context_info["sentences"][i])
            context.append(f"{title}\n{sentences}")

        return "\n\n".join(context)

@dataclass
class ManualDataset(Dataset):
    name = "manual_data"
    manual_data = True
    
    def generate_examples(self, fn: str) -> List[Example]:
        """ Loads examples from a CSV file. """
        examples = []
        df = pd.read_csv(fn)
        for idx, row in df.iterrows():
            examples += self._generate_examples(idx, row)

        for i in range(8):
            logging.info(f"EXAMPLE {i}:\n_________________\n")
            logging.info(f"GT Sol: {examples[i].gt_solution}")
            logging.info(f"Sol: {examples[i].solution}")
            logging.info(f"Mistake: {examples[i].mistake_type}")
            logging.info(f"Mistake Step: {examples[i].mistake_step}\n_________________")
            
        return examples

    def _generate_examples(self, idx, row) -> List[Example]:
        """ 
        Generates examples from a single row. 
        Each row produces 4 examples:
        1) Correct
        2) Manual plausible incorrect
        3) Manually incomplete
        4) Typo 
        """
        examples = []

        # correct solution
        q = row['question']
        gt_sol = row['gt_solution']
        examples.append(Example(
            qid=idx,
            task=row['task'],
            question=q,
            gt_solution=gt_sol,
            correct_chatgpt_solution=row["correct_chatgpt_solution"],
            solution=gt_sol,
        ))

        # manual plausible solution
        examples.append(Example(
            qid=idx,
            task=row['task'],
            question=q,
            gt_solution=gt_sol,
            correct_chatgpt_solution=row["correct_chatgpt_solution"],
            solution=row['plausible_incorrect_solution'],
            mistake_type="plausible_incorrect",
            mistake_step=row['plausible_incorrect_mistake_step'],
        ))

        # incomplete solution
        examples.append(Example(
            qid=idx,
            task=row['task'],
            question=q,
            gt_solution=gt_sol,
            correct_chatgpt_solution=row["correct_chatgpt_solution"],
            solution=row['incomplete_solution'],
            mistake_type="incomplete",
            mistake_step=row['incomplete_solution_mistake_step'],
        ))

        # typo solution
        ms, typo_solution = self.make_typo_solution(row)
        examples.append(Example(
            qid=idx,
            task=row['task'],
            question=q,
            gt_solution=gt_sol,
            correct_chatgpt_solution=row["correct_chatgpt_solution"],
            solution=typo_solution,
            mistake_type="typo",
            mistake_step=ms,
        ))

        return examples


@dataclass
class AlgebraDataset(Dataset):
    name: str = "algebra"
    verifier = AlgebraVerifyPrompt()

    def verify_examples(self, model):
        """
        Filter for examples that the model gets correct.
        
        """
        json_fn = "data/algebra/data_{}.jsonl".format(self.length)
        verified_json_fn = "data/algebra/{}_verified_data_{}.jsonl".format(model, self.length)
        logging.info(f"Saving verified examples to {verified_json_fn}")
        num_corr = 0

        for example in open_json(json_fn):
            prompt = self.verifier.build_prompt(example["question"])
            response = get_response(model, prompt)
            gt_answer = example["gt_answer"]

            logging.info(f"Example:\n_________________\n")
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Response: {response}")
            logging.info(f"GT Sol: {example['gt_solution']}")
            logging.info(f"Correct: {self.verifier.check_model_answer(response, gt_answer)}")

            # if the answer is correct, then save it
            if self.verifier.check_model_answer(response, example["gt_answer"]):
                example["correct_chatgpt_solution"] = response
                write_json(verified_json_fn, example)
                num_corr += 1

        logging.info(f"Saved {num_corr} verified examples to {verified_json_fn}")  


    def generate_examples(self, length):
        json_fn = "data/algebra/verified_data_{}.jsonl".format(length)

        examples = [] 
        for example in open_json(json_fn):
            examples += self._generate_examples(example)


    def _generate_examples(self, example):
        # correct solution
        q = example['question']
        gt_sol = example['gt_solution']

        examples = []
        
        examples.append(Example(
            qid=example["qid"],
            task=self.name,
            question=q,
            gt_solution=gt_sol,
            correct_chatgpt_solution=example["correct_chatgpt_solution"],
            solution=gt_sol,
        ))

        # typo solution
        ms, typo_solution = self.make_typo_solution(example)
        examples.append(Example(
            qid=example["qid"],
            task=self.name,
            question=q,
            gt_solution=gt_sol,
            correct_chatgpt_solution=example["correct_chatgpt_solution"],
            solution=typo_solution,
            mistake_type="typo",
            mistake_step=ms,
        ))

        return examples

@dataclass
class ProntoQADataset(Dataset):
    name: str = "prontoqa"
    verifier = ProntoQAVerifyPrompt()
    
    def verify_examples(self, model):
        json_fn = "data/prontoqa/{}hop_1shot_random_nodistractor_testdistractor.json".format(self.length)
        verified_json_fn = "data/prontoqa/{}_verified_{}hop_1shot_random_nodistractor_testdistractor.jsonl".format(model, self.length)
        logging.info(f"Saving verified examples to {verified_json_fn}")
        num_corr = 0
        
        with open(json_fn) as f:
            examples = json.load(f)

        for i, example in enumerate(list(examples.values())[:self.max_num_examples]):
            example = example["in_context_example0"]
            question = self.build_question(example["question"], example["query"])
            gt_solution = "\n".join(example["chain_of_thought"])
            gt_answer = example["answer"]

            prompt = self.verifier.build_prompt(question)
            response = get_response(model, prompt)

            ex = {
                "qid": i,
                "task": self.name,
                "question": question,
                "gt_solution": gt_solution,
                "gt_answer": gt_answer,

            }

            logging.info(f"Example:\n_________________\n")
            logging.info(f"Prompt: {prompt}")
            logging.info(f"Response: {response}")
            logging.info(f"GT Sol: {gt_solution}")
            logging.info(f"Correct: {self.verifier.check_model_answer(response, gt_answer)}")

            
            # if the answer is correct, then save it
            if self.verifier.check_model_answer(response, gt_answer):
                ex["correct_chatgpt_solution"] = response
                write_json(verified_json_fn, ex)
                num_corr += 1

        logging.info(f"Saved {num_corr} verified examples to {verified_json_fn}")  


    def build_question(self, information, statement):
        return "Information:\n{}\n\nStatement:\n{}".format(information, statement)    

    def generate_examples(self, length):
        json_fn = "data/algebra/verified_data_{}.jsonl".format(length)

        examples = [] 
        for example in open_json(json_fn):
            examples += self._generate_examples(example)


    def _generate_examples(self, example):
        # correct solution
        q = example['question']
        gt_sol = example['gt_solution']

        examples.append(Example(
            qid=example["qid"],
            task=self.name,
            question=q,
            gt_solution=gt_sol,
            correct_chatgpt_solution=example["correct_chatgpt_solution"],
            solution=gt_sol,
        ))

        # typo solution
        ms, typo_solution = self.make_typo_solution(example)
        examples.append(Example(
            qid=example["qid"],
            task=self.name,
            question=q,
            gt_solution=gt_sol,
            correct_chatgpt_solution=example["correct_chatgpt_solution"],
            solution=typo_solution,
            mistake_type="typo",
            mistake_step=ms,
        ))

        # incomplete solution
        ms, typo_solution = self.make_typo_solution(example)
        examples.append(Example(
            qid=example["qid"],
            task=self.name,
            question=q,
            gt_solution=gt_sol,
            correct_chatgpt_solution=example["correct_chatgpt_solution"],
            solution=typo_solution,
            mistake_type="typo",
            mistake_step=ms,
        ))

        return examples


DATASETS = {
    "manual": ManualDataset,
    "rte": RTEDataset,
    "gsm8k": GSM8KDataset,
    "commonsense_qa": CommonsenseQADataset,
    "strategy_qa": StrategyQADataset,
    "hotpot_qa": HotpotQADataset,
}
def get_dataset(args):
    return DATASETS[args.dataset](name=args.dataset)