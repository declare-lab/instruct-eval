## 🍮 📚 Flan-Eval: Reproducible Held-Out Evaluation for Instruction Tuning

This repository contains code to evaluate instruction-tuned models such as Alpaca and Flan-T5 on held-out
tasks.
We aim to facilitate simple and convenient benchmarking across multiple tasks and models.

### Why?

Instruction-tuned models such as [Flan-T5](https://arxiv.org/abs/2210.11416)
and [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) represent an exciting direction to approximate the
performance of large language models (LLMs) like ChatGPT at lower cost.
However, it is challenging to compare the performance of different models qualitatively.
To evaluate how well the models generalize across a wide range of unseen and challenging tasks, we can use academic
benchmarks such as [MMLU](https://arxiv.org/abs/2009.03300) and [BBH](https://arxiv.org/abs/2210.09261).
Compared to existing libraries such as [evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
and [HELM](https://github.com/stanford-crfm/helm), this repo enables simple and convenient evaluation for multiple
models.
Notably, we support most models from HuggingFace Transformers 🤗 :

- [AutoModelForCausalLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM) (
  eg [GPT-2](https://huggingface.co/gpt2-xl), [GPT-J](https://huggingface.co/EleutherAI/gpt-j-6b)
  , [OPT-IML](https://huggingface.co/facebook/opt-iml-max-1.3b), [BLOOMZ](https://huggingface.co/bigscience/bloomz-7b1))
- [AutoModelForSeq2SeqLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSeq2SeqLM) (
  eg [Flan-T5](https://huggingface.co/google/flan-t5-xl), [Flan-UL2](https://huggingface.co/google/flan-ul2)
  , [TK-Instruct](https://huggingface.co/allenai/tk-instruct-3b-def))
- [LlamaForCausalLM](https://huggingface.co/docs/transformers/main/model_doc/llama#transformers.LlamaForCausalLM) (
  eg [LLaMA](https://huggingface.co/decapoda-research/llama-7b-hf)
  , [Alpaca](https://huggingface.co/chavinlo/alpaca-native), [Vicuna](https://huggingface.co/chavinlo/vicuna))
- [ChatGLM](https://huggingface.co/THUDM/chatglm-6b)

### Results

| Model Name | Model Path                                                                            | Paper                                                    | Size | MMLU | BBH  | DROP | HumanEval |
|------------|---------------------------------------------------------------------------------------|----------------------------------------------------------|------|------|------|------|-----------|
|            | [GPT-4](https://openai.com/product/gpt-4)                                             | [Link](https://arxiv.org/abs/2303.08774)                 | ?    | 86.4 |      | 80.9 | 67.0      |
|            | [ChatGPT](https://openai.com/blog/chatgpt)                                            | [Link](https://arxiv.org/abs/2303.08774)                 | ?    | 70.0 |      | 64.1 | 48.1      |
| seq_to_seq | [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl)                         | [Link](https://arxiv.org/abs/2210.11416)                 | 3B   | 49.2 | 40.2 | 56.3 |           |
| llama      | [eachadea/vicuna-13b](https://huggingface.co/eachadea/vicuna-13b)                     | [Link](https://vicuna.lmsys.org/)                        | 13B  | 49.7 | 37.1 | 32.9 | 15.2      |
| llama      | [TheBloke/koala-13B-HF](https://huggingface.co/TheBloke/koala-13B-HF)                 | [Link](https://bair.berkeley.edu/blog/2023/04/03/koala/) | 13B  | 44.6 | 34.6 |      | 11.0      |
| llama      | [chavinlo/alpaca-native](https://huggingface.co/chavinlo/alpaca-native)               | [Link](https://crfm.stanford.edu/2023/03/13/alpaca.html) | 7B   | 41.6 | 33.3 | 26.3 | 10.3      |
| llama      | [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf) | [Link](https://arxiv.org/abs/2302.13971)                 | 7B   | 35.2 | 30.9 |      | 10.3      |
| chatglm    | [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)                           | [Link](https://arxiv.org/abs/2210.02414)                 | 6B   | 36.1 | 31.3 |      | 3.1       |

### Example Usage

Evaluate on [Massive Multitask Language Understanding](https://huggingface.co/datasets/lukaemon/mmlu) (MMLU) which
includes exam questions from 57 tasks such as mathematics, history, law, and medicine.
We use 5-shot direct prompting and measure the exact-match score.

```
python main.py mmlu --model_name llama --model_path chavinlo/alpaca-native
# 0.4163936761145136

python main.py mmlu --model_name seq_to_seq --model_path google/flan-t5-xl 
# 0.49252243270189433
```

Evaluate on [Big Bench Hard](https://huggingface.co/datasets/lukaemon/bbh) (BBH) which includes 23 challenging tasks for
which PaLM (540B) performs below an average human rater.
We use 3-shot direct prompting and measure the exact-match score.

```
python main.py bbh --model_name llama --model_path TheBloke/koala-13B-HF --load_8bit
# 0.3468942926723247
```

Evaluate on [DROP](https://huggingface.co/datasets/drop) which is a math question answering benchmark.
We use 3-shot direct prompting and measure the exact-match score.

```
python main.py drop --model_name seq_to_seq --model_path google/flan-t5-xl 
# 0.5632458233890215
```

Evaluate on [HumanEval](https://huggingface.co/datasets/openai_humaneval) which includes 164 coding questions.
We use 0-shot direct prompting and measure the pass@1 score.

```
python main.py humaneval  --model_name llama --model_path eachadea/vicuna-13b --n_sample 1 --load_8bit
# {'pass@1': 0.1524390243902439}
```

### Setup

Install dependencies and download data.

```
conda create -n flan-eval python=3.8 -y
conda activate flan-eval
pip install -r requirements.txt
mkdir -p data
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
tar -xf data/mmlu.tar -C data && mv data/data data/mmlu
```
