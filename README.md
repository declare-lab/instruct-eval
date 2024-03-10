## :camel: 🍮 📚 InstructEval: Towards Holistic Evaluation of Instruction-Tuned Large Language Models

[Paper](https://arxiv.org/abs/2306.04757) | [Model](https://huggingface.co/declare-lab/flan-alpaca-gpt4-xl) | [Leaderboard](https://declare-lab.github.io/instruct-eval/)

<p align="center">
  <img src="https://raw.githubusercontent.com/declare-lab/instruct-eval/main/docs/logo.png" alt="" width="300" height="300">
</p>

> 🔥 If you are interested in IQ testing LLMs, check out our new work: [AlgoPuzzleVQA](https://github.com/declare-lab/puzzle-reasoning)

> 📣 Introducing Resta: **Safety Re-alignment of Language Models**. [**Paper**](https://arxiv.org/abs/2402.11746) [**Github**](https://github.com/declare-lab/resta)

> 📣 **Red-Eval**, the benchmark for **Safety** Evaluation of LLMs has been added: [Red-Eval](https://github.com/declare-lab/instruct-eval/tree/main/red-eval)

> 📣 Introducing **Red-Eval** to evaluate the safety of the LLMs using several jailbreaking prompts. With **Red-Eval** one could jailbreak/red-team GPT-4 with a 65.1% attack success rate and ChatGPT could be jailbroken 73% of the time as measured on DangerousQA and HarmfulQA benchmarks. More details are here: [Code](https://github.com/declare-lab/red-instruct) and [Paper](https://arxiv.org/abs/2308.09662).

> 📣 We developed Flacuna by fine-tuning Vicuna-13B on the Flan collection. Flacuna is better than Vicuna at problem-solving. Access the model here [https://huggingface.co/declare-lab/flacuna-13b-v1.0](https://huggingface.co/declare-lab/flacuna-13b-v1.0).

> 📣 The [**InstructEval**](https://declare-lab.net/instruct-eval/) benchmark and leaderboard have been released. 

> 📣 The paper reporting Instruction Tuned LLMs on the **InstructEval** benchmark suite has been released on Arxiv. Read it here: [https://arxiv.org/pdf/2306.04757.pdf](https://arxiv.org/pdf/2306.04757.pdf)

> 📣 We are releasing **IMPACT**, a dataset for evaluating the writing capability of LLMs in four aspects: Informative, Professional, Argumentative, and Creative. Download it from Huggingface: [https://huggingface.co/datasets/declare-lab/InstructEvalImpact](https://huggingface.co/datasets/declare-lab/InstructEvalImpact). 

> 📣 **FLAN-T5** is also useful in text-to-audio generation. Find our work
at [https://github.com/declare-lab/tango](https://github.com/declare-lab/tango) if you are interested.

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
Notably, we support most models from HuggingFace Transformers 🤗 (check [here](./docs/models.md) for a list of models we support):

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

For detailed results, please go to our [leaderboard](https://declare-lab.net/instruct-eval/)

| Model Name | Model Path                                                                                                              | Paper                                                                                                         | Size | MMLU | BBH  | DROP | HumanEval |
|------------|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|------|------|------|------|-----------|
|            | [GPT-4](https://openai.com/product/gpt-4)                                                                               | [Link](https://arxiv.org/abs/2303.08774)                                                                      | ?    | 86.4 |      | 80.9 | 67.0      |
|            | [ChatGPT](https://openai.com/blog/chatgpt)                                                                              | [Link](https://arxiv.org/abs/2303.08774)                                                                      | ?    | 70.0 |      | 64.1 | 48.1      |
| seq_to_seq | [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl)                                                         | [Link](https://arxiv.org/abs/2210.11416)                                                                      | 11B  | 54.5 | 43.9 |      |           |
| seq_to_seq | [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl)                                                           | [Link](https://arxiv.org/abs/2210.11416)                                                                      | 3B   | 49.2 | 40.2 | 56.3 |           |
| llama      | [eachadea/vicuna-13b](https://huggingface.co/eachadea/vicuna-13b)                                                       | [Link](https://vicuna.lmsys.org/)                                                                             | 13B  | 49.7 | 37.1 | 32.9 | 15.2      |
| llama      | [decapoda-research/llama-13b-hf](https://huggingface.co/decapoda-research/llama-13b-hf)                                 | [Link](https://arxiv.org/abs/2302.13971)                                                                      | 13B  | 46.2 | 37.1 | 35.3 | 13.4      |
| seq_to_seq | [declare-lab/flan-alpaca-gpt4-xl](https://huggingface.co/declare-lab/flan-alpaca-gpt4-xl)                               | [Link](https://github.com/declare-lab/flan-alpaca)                                                            | 3B   | 45.6 | 34.8 |      |           |
| llama      | [TheBloke/koala-13B-HF](https://huggingface.co/TheBloke/koala-13B-HF)                                                   | [Link](https://bair.berkeley.edu/blog/2023/04/03/koala/)                                                      | 13B  | 44.6 | 34.6 | 28.3 | 11.0      |
| llama      | [chavinlo/alpaca-native](https://huggingface.co/chavinlo/alpaca-native)                                                 | [Link](https://crfm.stanford.edu/2023/03/13/alpaca.html)                                                      | 7B   | 41.6 | 33.3 | 26.3 | 10.3      |
| llama      | [TheBloke/wizardLM-7B-HF](https://huggingface.co/TheBloke/wizardLM-7B-HF)                                               | [Link](https://arxiv.org/abs/2304.12244)                                                                      | 7B   | 36.4 | 32.9 |      | 15.2      |
| chatglm    | [THUDM/chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)                                                             | [Link](https://arxiv.org/abs/2210.02414)                                                                      | 6B   | 36.1 | 31.3 | 44.2 | 3.1       |
| llama      | [decapoda-research/llama-7b-hf](https://huggingface.co/decapoda-research/llama-7b-hf)                                   | [Link](https://arxiv.org/abs/2302.13971)                                                                      | 7B   | 35.2 | 30.9 | 27.6 | 10.3      |
| llama      | [wombat-7b-gpt4-delta](https://huggingface.co/GanjinZero/wombat-7b-gpt4-delta)                                          | [Link](https://arxiv.org/abs/2304.05302)                                                                      | 7B   | 33.0 | 32.4 |      | 7.9       |
| seq_to_seq | [bigscience/mt0-xl](https://huggingface.co/bigscience/mt0-xl)                                                           | [Link](https://arxiv.org/abs/2210.11416)                                                                      | 3B   | 30.4 |      |      |           |
| causal     | [facebook/opt-iml-max-1.3b](https://huggingface.co/facebook/opt-iml-max-1.3b)                                           | [Link](https://arxiv.org/abs/2212.12017)                                                                      | 1B   | 27.5 |      |      | 1.8       |
| causal     | [OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5](https://huggingface.co/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5) | [Link](https://github.com/LAION-AI/Open-Assistant)                                                            | 12B  | 27.0 | 30.0 |      | 9.1       |
| causal     | [stabilityai/stablelm-base-alpha-7b](https://huggingface.co/stabilityai/stablelm-base-alpha-7b)                         | [Link](https://github.com/Stability-AI/StableLM)                                                              | 7B   | 26.2 |      |      | 1.8       |
| causal     | [databricks/dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b)                                               | [Link](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) | 12B  | 25.7 |      |      | 7.9       |
| causal     | [Salesforce/codegen-6B-mono](https://huggingface.co/Salesforce/codegen-6B-mono)                                         | [Link](https://arxiv.org/abs/2203.13474)                                                                      | 6B   |      |      |      | 27.4      |
  | causal     | [togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1](https://huggingface.co/togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1) | [Link](https://github.com/togethercomputer/RedPajama-Data)                                                             | 7B   | 38.1 | 31.3 | 24.7 | 5.5      |

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

Evaluate on [HumanEval](https://huggingface.co/datasets/openai_humaneval) which includes 164 coding questions in python.
We use 0-shot direct prompting and measure the pass@1 score.

```
python main.py humaneval  --model_name llama --model_path eachadea/vicuna-13b --n_sample 1 --load_8bit
# {'pass@1': 0.1524390243902439}
```

### Setup

Install dependencies and download data.

```
conda create -n instruct-eval python=3.8 -y
conda activate instruct-eval
pip install -r requirements.txt
mkdir -p data
wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -O data/mmlu.tar
tar -xf data/mmlu.tar -C data && mv data/data data/mmlu
```

