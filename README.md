## 🍮 📚 Flan-Eval: Reproducible Held-Out Evaluation for Instruction Tuning

This repository contains code to quantitatively evaluate instruction-tuned models such as Alpaca and Flan-T5 on held-out
tasks.
We aim to include a suite of academic benchmarks, including MMLU, BBH and HumanEval.

### Why?

Instruction-tuned models such as [Flan-T5](https://arxiv.org/abs/2210.11416)
and [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) represent an exciting new direction to approximate the
performance of large language models (LLMs) like ChatGPT at lower cost.
However, it is challenging to compare the performance of different models qualitatively.
To evaluate the instruction-tuned models on a wide range of unseen and challenging tasks, we can look to academic
benchmarks such as [MMLU](https://arxiv.org/abs/2009.03300) and [BBH](https://arxiv.org/abs/2210.09261) which are often
used in literature.
However, the [published models](https://github.com/google-research/FLAN) do not open-source their evaluation suites,
while existing suites such as [evalation-harness](https://github.com/EleutherAI/lm-evaluation-harness) do not focus on
instruction-tuned models.
Hence, we aim to provide a simple evaluation suite that covers both decoder and encoder-decoder
models.

### Usage

```
python mmlu.py main data/mmlu --model_name llama --model_path decapoda-research/llama-7b-hf
# 0.35215781227745335

python mmlu.py main data/mmlu --model_name seq_to_seq --model_path google/flan-t5-xl 
# 0.49252243270189433
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