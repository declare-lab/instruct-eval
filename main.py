from fire import Fire

import bbh
import drop
import mmlu
from human_eval.main import main as humaneval


def main(task_name: str, **kwargs):
    task_map = dict(mmlu=mmlu.main, bbh=bbh.main, drop=drop.main, humaneval=humaneval)

    if task_name == "all":
        results = {}
        for name, task_fn in task_map.items():
            score = task_fn(**kwargs)
            results[name] = score
    else:
        task_fn = task_map.get(task_name)
        if task_fn is None:
            raise ValueError(f"{task_name}. Choose from {list(task_map.keys())}")
        score = task_fn(**kwargs)
        results = {task_name: score}

    results = {name: round(score * 100, 2) for name, score in results.items()}
    print(results)
    return results


"""
p main.py --task_name bbh --model_name seq_to_seq --model_path google/flan-t5-xl 
p main.py --task_name mmlu --model_name seq_to_seq --model_path google/flan-t5-xl 
p main.py --task_name humaneval --model_name llama --model_path decapoda-research/llama-7b-hf --n_sample 1

p main.py --task_name all --model_name seq_to_seq --model_path google/flan-t5-xl
{'mmlu': 49.25, 'bbh': 40.26, 'drop': 56.32, 'humaneval': 0.0}

p main.py --task_name all --model_name causal --model_path mosaicml/mpt-7b-instruct
{'mmlu': 32.02, 'bbh': 32.08, 'drop': 23.4, 'humaneval': 15.24}

p main.py --task_name all --model_name causal --model_path mosaicml/mpt-7b
{'mmlu': 30.79, 'bbh': 32.14, 'drop': 21.78, 'humaneval': 14.02}

p main.py --task_name all --model_name causal --model_path mosaicml/mpt-7b-chat
{'mmlu': 37.14, 'bbh': 32.02, 'drop': 20.16, 'humaneval': 17.68}

p main.py --task_name all --model_name causal --model_path Salesforce/codegen-6B-mono
{'mmlu': 26.06, 'bbh': 29.24, 'drop': 12.81, 'humaneval': 26.22}

p main.py --task_name all --model_name seq_to_seq --model_path google/flan-ul2 --load_8bit
OOM at BBH (does not OOM if evaluate each task separately)

p main.py --task_name all --model_name causal --model_path facebook/opt-iml-30b --load_8bit
Average accuracy: 0.386 
OOM at BBH (does not OOM if evaluate each task separately)

python main.py all --model_name rwkv --model_path https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/RWKV-4-Raven-7B-v11-Eng99%25-Other1%25-20230427-ctx8192.pth
{'mmlu': 23.6, 'bbh': 26.96, 'drop': 12.03, 'humaneval': 8.54}

python main.py all --model_name rwkv --model_path https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/RWKV-4-Raven-14B-v11x-Eng99%25-Other1%25-20230501-ctx8192.pth
{'mmlu': 25.63, 'bbh': 28.9, 'drop': 6.09, 'humaneval': 11.59}

python main.py all --model_name llama --model_path TheBloke/stable-vicuna-13B-HF --load_8bit
{'mmlu': 49.16, 'bbh': 37.51, 'drop': 34.26, 'humaneval': 15.85}

"""


if __name__ == "__main__":
    Fire(main)
