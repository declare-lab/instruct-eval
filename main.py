from fire import Fire

from bbh import main as bbh_main
from mmlu import main as mmlu_main


def main(task_name: str, **kwargs):
    task_map = dict(mmlu=mmlu_main, bbh=bbh_main)
    task_fn = task_map.get(task_name)
    if task_fn is None:
        raise ValueError(f"{task_name}. Choose from {list(task_map.keys())}")

    task_fn(**kwargs)


"""
p main.py --task_name bbh --model_name seq_to_seq --model_path google/flan-t5-xl 
p main.py --task_name mmlu --model_name seq_to_seq --model_path google/flan-t5-xl 
"""


if __name__ == "__main__":
    Fire(main)
