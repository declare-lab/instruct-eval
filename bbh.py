from argparse import Namespace
from typing import Dict

from datasets import load_dataset, get_dataset_config_names, Dataset
from fire import Fire
from tqdm import tqdm

from modeling import select_model, EvalModel


def format_example(dataset: Dict[str, list], idx: int, include_answer=True):
    prompt = dataset["input"][idx]
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(dataset["target"][idx])
    return prompt


def gen_prompt(dataset: Dict[str, list], k=-1):
    prompt = ""
    if k == -1:
        k = len(dataset)
    for i in range(k):
        prompt += format_example(dataset, i)
    return prompt


def evaluate(model: EvalModel, dataset: Dataset, ntrain: int) -> dict:
    data_train = dataset[:ntrain]
    data_test = dataset[ntrain:]
    is_correct = []

    for i in range(len(dataset) - ntrain):
        # get prompt and make sure it fits
        k = int(ntrain)
        prompt_end = format_example(data_test, i, include_answer=False)
        train_prompt = gen_prompt(data_train, k)
        prompt = train_prompt + prompt_end

        while not model.check_valid_length(prompt) and k > 0:
            k -= 1
            train_prompt = gen_prompt(data_train, k)
            prompt = train_prompt + prompt_end

        label = data_test["target"][i]
        pred = model.run(prompt)
        is_correct.append(pred.strip().startswith(label))
        if i == 0:
            print(dict(prompt=prompt, label=label, pred=pred))

    return dict(score=sum(is_correct) / len(is_correct))


def main(data_dir: str = "lukaemon/bbh", ntrain: int = 3, **kwargs):
    args = Namespace(**locals())
    model = select_model(max_input_length=2048, max_output_length=32, **kwargs)
    print(locals())

    all_results = []
    for name in tqdm(get_dataset_config_names(data_dir)):
        dataset = load_dataset(data_dir, name, split="test")
        result = evaluate(model, dataset, ntrain=ntrain)
        all_results.append(result)
        print(dict(name=name, **result))

    print(dict(average=sum(res["score"] for res in all_results) / len(all_results)))


"""
p bbh.py main "lukaemon/bbh" --model_name seq_to_seq --model_path google/flan-t5-xl 
{'average': 0.40261571422898645}

p bbh.py main "lukaemon/bbh" --model_name llama --model_path decapoda-research/llama-7b-hf
{'average': 0.30963361708212966}

p bbh.py main "lukaemon/bbh" --model_name llama --model_path chavinlo/alpaca-native
{'average': 0.3335667396422546}

p bbh.py main "lukaemon/bbh" --model_name chatglm --model_path THUDM/chatglm-6b
{'average': 0.31384628677534854}

python main.py bbh --model_name llama --model_path chavinlo/alpaca-13b --load_8bit
{'average': 0.33351335206026284}

python main.py bbh --model_name causal --model_path togethercomputer/Pythia-Chat-Base-7B
{'average': 0.29975163365323554}

python main.py bbh --model_name llama --model_path decapoda-research/llama-13b-hf --load_8bit
{'average': 0.3719930899679183}

python main.py bbh --model_name llama --model_path TheBloke/koala-7B-HF --load_8bit
{'average': 0.3118093830908477}

python main.py bbh --model_name llama --model_path TheBloke/koala-13B-HF --load_8bit
{'average': 0.3468942926723247}

"""


if __name__ == "__main__":
    Fire()
