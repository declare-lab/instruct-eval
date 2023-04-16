from fire import Fire

import bbh
import drop
import mmlu
from human_eval.main import main as humaneval


def main(task_name: str, **kwargs):
    task_map = dict(mmlu=mmlu.main, bbh=bbh.main, drop=drop.main, humaneval=humaneval)
    task_fn = task_map.get(task_name)
    if task_fn is None:
        raise ValueError(f"{task_name}. Choose from {list(task_map.keys())}")

    task_fn(**kwargs)


"""
p main.py --task_name bbh --model_name seq_to_seq --model_path google/flan-t5-xl 
p main.py --task_name mmlu --model_name seq_to_seq --model_path google/flan-t5-xl 
p main.py --task_name humaneval --model_name llama --model_path decapoda-research/llama-7b-hf --n_sample 1
"""


if __name__ == "__main__":
    Fire(main)
