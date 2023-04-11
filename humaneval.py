from argparse import Namespace
from typing import Dict

from datasets import load_dataset, get_dataset_config_names, Dataset
from fire import Fire
from tqdm import tqdm

from modeling import select_model, EvalModel

from human_eval.data import write_jsonl, read_problems, HUMAN_EVAL
from human_eval.evaluate_functional_correctness import entry_point

def evaluate(model: EvalModel, dataset: Dataset, ntrain: int, **kwargs) -> dict:
    """
    This will generate two files:
    f"humaneval_{model_name}_predictions.jsonl": output from model
    f"humaneval_{model_name}_predictions.jsonl_result.jsonl": evaluation result using f"humaneval_{model_name}_predictions.jsonl"
    """

    n_sample = kwargs['n_sample']
    best_temperature = {1:0.1, 10:0.6, 100:0.8}
    samples = []
    progress_bar = tqdm(total=len(dataset) * n_sample, desc="Generating samples")
    for task_id in dataset:
        for i in range(n_sample):
            prompt = dataset[task_id]["prompt"]
            temperature = best_temperature[n_sample]
            if temperature > 0:
                completion = model.run(prompt, temperature=temperature, do_sample=True)
            else:
                completion = model.run(prompt)
            ## The program tends to overwrite, we only take the first function
            sample = dict(task_id=task_id, completion=completion.split('\n\n')[0])
            if i == 0: 
                print(dataset[task_id])
                print(sample)
            samples.append(sample)
            progress_bar.update(1)
    progress_bar.close()

    model_name = model.model_path.split('/')[-1]
    pred_filename = f"humaneval_{model_name}_predictions.jsonl"
    write_jsonl(pred_filename, samples)
    print("Evaluating...")
    result = entry_point(pred_filename)

    return result


def main(data_dir: str = "", ntrain: int = 3, **kwargs):
    args = Namespace(**locals())
    model = select_model(max_input_length=1360, max_output_length=512, **kwargs)
    print(locals())

    dataset = read_problems()
    result = evaluate(model, dataset, ntrain=ntrain, **kwargs)

    print(result)


"""
p humaneval.py main  --model_name llama --model_path decapoda-research/llama-7b-hf --n_sample 1
{'pass@1': 0.105}

p humaneval.py main  --model_name llama --model_path chavinlo/alpaca-native --n_sample 1
{'pass@1': 0.105}

p humaneval.py main  --model_name llama --model_path eachadea/vicuna-13b --n_sample 1 --load_8bit
{'pass@1': 0.152}

"""


if __name__ == "__main__":
    Fire()
