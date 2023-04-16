from argparse import Namespace

from fire import Fire
from tqdm import tqdm

from human_eval.data import write_jsonl, read_problems
from human_eval.evaluation import evaluate_functional_correctness
from modeling import select_model, EvalModel


def entry_point(
    problem_file: str,
    sample_file: str,
    k: str = "1,10,100",
    n_workers: int = 4,
    timeout: float = 3.0,
):
    """
    Evaluates the functional correctness of generated samples, and writes
    results to f"{sample_file}_results.jsonl.gz"
    """
    k = list(map(int, k.split(",")))
    results = evaluate_functional_correctness(
        sample_file, k, n_workers, timeout, problem_file
    )

    return results


def filter_code(completion: str, model: EvalModel) -> str:
    if 'chatglm' in model.model_path:
        ## Remove boilerplate for the function
        return completion.split('"""\n')[-1].replace("`", "")
    else:
        ## The program tends to overwrite, we only take the first function
        return completion.split("\n\n")[0]


def evaluate(model: EvalModel, data_path: str, **kwargs) -> dict:
    dataset = read_problems(data_path)
    n_sample = kwargs["n_sample"]
    best_temperature = {1: 0.1, 10: 0.6, 100: 0.8}
    samples = []
    progress_bar = tqdm(total=len(dataset) * n_sample, desc="Generating samples")
    for task_id in dataset:
        for i in range(n_sample):
            prompt = dataset[task_id]["prompt"]
            prompt = 'Please complete the following Python code without providing any additional tasks such as testing or explanations\n' + prompt
            temperature = best_temperature[n_sample]
            if temperature > 0:
                completion = model.run(prompt, temperature=temperature, do_sample=True)
            else:
                completion = model.run(prompt)
            sample = dict(task_id=task_id, completion=filter_code(completion, model))
            if i == 0: 
                print(prompt)
                print('-'*100)
                print(filter_code(completion, model))
            samples.append(sample)
            progress_bar.update(1)
    progress_bar.close()

    model_name = model.model_path.replace("/", "_")
    pred_filename = f"humaneval_{model_name}_predictions.jsonl"
    write_jsonl(pred_filename, samples)
    print("Evaluating...")
    result = entry_point(problem_file=data_path, sample_file=pred_filename)
    return result


def main(data_path: str = "human_eval/HumanEval.jsonl.gz", **kwargs):
    args = Namespace(**locals())
    model = select_model(max_input_length=1360, max_output_length=512, **kwargs)
    print(locals())

    result = evaluate(model, data_path, **kwargs)
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
