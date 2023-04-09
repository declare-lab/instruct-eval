from argparse import Namespace
from typing import Dict

from datasets import load_dataset, get_dataset_config_names, Dataset
from fire import Fire
from tqdm import tqdm

from modeling import select_model, EvalModel

from human_eval.data import write_jsonl, read_problems, HUMAN_EVAL
from human_eval.evaluate_functional_correctness import entry_point

def evaluate(model: EvalModel, dataset: Dataset, ntrain: int) -> dict:

    num_samples_per_task = 100
    best_temperature = {1:0.0, 10:0.6, 100:0.8}
    # best_temperature = {1:0.2, 10:0.8, 100:1.0}
    samples = []
    progress_bar = tqdm(total=len(dataset) * num_samples_per_task, desc="Generating samples")
    for task_id in dataset:
        for _ in range(num_samples_per_task):
            prompt = dataset[task_id]["prompt"]
            temperature = best_temperature[num_samples_per_task]
            if temperature > 0:
                completion = model.run(prompt, temperature=temperature)
            else:
                completion = model.run(prompt)
            samples.append(dict(task_id=task_id, completion=completion))
            progress_bar.update(1)
    progress_bar.close()

    pred_filename = "humaneval_predictions.jsonl"
    write_jsonl(pred_filename, samples)
    print("Evaluating...")
    result = entry_point(pred_filename)

    return result


def main(data_dir: str = "", ntrain: int = 3, **kwargs):
    args = Namespace(**locals())
    model = select_model(max_input_length=2048, max_output_length=32, **kwargs)
    print(locals())

    all_results = []
    dataset = read_problems()
    result = evaluate(model, dataset, ntrain=ntrain)

    print(result)


"""

p humaneval.py main --model_name seq_to_seq --model_path google/flan-t5-xl

p humaneval.py main  --model_name llama --model_path decapoda-research/llama-7b-hf

p humaneval.py main  --model_name llama --model_path chavinlo/alpaca-native

p humaneval.py main --model_name chatglm --model_path THUDM/chatglm-6b

"""


if __name__ == "__main__":
    Fire()
