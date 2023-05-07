import getpass
from pathlib import Path
from typing import Optional

import openai
import rwkv
import tiktoken
from fire import Fire
from peft import PeftModel
from pydantic import BaseModel
from rwkv.model import RWKV
from rwkv.utils import PIPELINE
from torchvision.datasets.utils import download_url
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModel,
)


class EvalModel(BaseModel, arbitrary_types_allowed=True):
    model_path: str
    max_input_length: int = 512
    max_output_length: int = 512

    def run(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError

    def count_text_length(self, text: str) -> int:
        raise NotImplementedError

    def check_valid_length(self, text: str) -> bool:
        return self.count_text_length(text) <= self.max_input_length


class OpenAIModel(EvalModel):
    model_path: Optional[str]
    api_key: Optional[str]
    use_azure: bool = False
    tokenizer: Optional[tiktoken.Encoding]
    api_endpoint: str = "https://research.openai.azure.com/"
    api_version: str = "2023-03-15-preview"

    def load(self):
        if self.tokenizer is None:
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model_path)
            except Exception as e:
                print(e)
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # chatgpt/gpt-4

        if self.use_azure:
            openai.api_type = "azure"
            openai.api_base = self.api_endpoint
            openai.api_version = self.api_version
            if self.model_path is None:
                self.model_path = getpass.getpass("Model or deployment name: ")

        if self.api_key is None:
            self.api_key = getpass.getpass("API Key: ")
        openai.api_key = self.api_key

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        try:
            response = openai.ChatCompletion.create(
                engine=self.model_path,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            return str(e)

    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer.encode(text))


class SeqToSeqModel(EvalModel):
    model_path: str
    model: Optional[PreTrainedModel]
    tokenizer: Optional[PreTrainedTokenizer]
    lora_path: str = ""
    device: str = "cuda"
    load_8bit: bool = False

    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, **args)
            if self.lora_path:
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model.eval()
            if not self.load_8bit:
                self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_output_length,
            **kwargs,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.tokenizer(text).input_ids)


class CausalModel(SeqToSeqModel):
    def load(self):
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, trust_remote_code=True, **args
            )
            self.model.eval()
            if not self.load_8bit:
                self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            pad_token_id=self.tokenizer.eos_token_id,  # Avoid pad token warning
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)


class LlamaModel(SeqToSeqModel):
    use_template: bool = False
    """
    Not officially supported by AutoModelForCausalLM, so we need the specific class
    Optionally, we can use the prompt template from: https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
    However, initial MMLU experiments indicate that the template is not useful for few-shot settings
    """

    def load(self):
        if self.tokenizer is None:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
        if self.model is None:
            args = {}
            if self.load_8bit:
                args.update(device_map="auto", load_in_8bit=True)
            self.model = LlamaForCausalLM.from_pretrained(self.model_path, **args)
            if self.lora_path:
                self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            self.model.eval()
            if not self.load_8bit:
                self.model.to(self.device)

    def run(self, prompt: str, **kwargs) -> str:
        if self.use_template:
            template = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            )
            text = template.format_map(dict(instruction=prompt))
        else:
            text = prompt

        self.load()
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_output_length,
            **kwargs,
        )
        batch_size, length = inputs.input_ids.shape
        return self.tokenizer.decode(outputs[0, length:], skip_special_tokens=True)


class ChatGLMModel(SeqToSeqModel):
    def load(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        if self.model is None:
            self.model = AutoModel.from_pretrained(
                self.model_path, trust_remote_code=True
            ).half()  # FP16 is required for ChatGLM
            self.model.eval()
            self.model.to(self.device)

    def run(self, prompt: str, **kwargs) -> str:
        self.load()
        response, history = self.model.chat(
            self.tokenizer,
            prompt,
            history=[],
            **kwargs,
        )
        return response


class RWKVModel(EvalModel):
    tokenizer_path: str = (
        "https://github.com/BlinkDL/ChatRWKV/raw/main/20B_tokenizer.json"
    )
    download_root: str = "."
    model: Optional[rwkv.utils.PIPELINE]

    def download(self, url: str) -> str:
        path = Path(self.download_root, Path(url).name)
        if not path.exists():
            download_url(url, root=self.download_root)
        return str(path)

    def load(self):
        model_path = self.download(self.model_path)
        tokenizer_path = self.download(self.tokenizer_path)

        if self.model is None:
            model = RWKV(model=model_path, strategy="cuda fp16")
            self.model = rwkv.utils.PIPELINE(model, tokenizer_path)

    def run(self, prompt: str, **kwargs) -> str:
        # Adapted from: https://github.com/BlinkDL/ChatRWKV/blob/main/v2/benchmark_more.py
        self.load()
        out_tokens = []
        out_last = 0
        out_str = ""
        occurrence = {}
        state = None
        token = None

        ctx = f"Bob: {prompt.strip()}\n\nAlice:"  # Special prompt format
        for i in range(self.max_output_length):
            tokens = self.model.encode(ctx) if i == 0 else [token]

            out, state = self.model.model.forward(tokens, state)
            for n in occurrence:
                out[n] -= 0.2 + occurrence[n] * 0.2

            token = self.model.sample_logits(out, temperature=1.0, top_p=0)
            if token == 0:
                break  # exit when 'endoftext'

            out_tokens += [token]
            occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)

            tmp = self.model.decode(out_tokens[out_last:])
            if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):
                # only print when the string is valid utf-8 and not end with \n
                out_str += tmp
                out_last = i + 1

            if "\n\n" in tmp:
                break  # exit when '\n\n'

        breakpoint()
        return out_str

    def count_text_length(self, text: str) -> int:
        self.load()
        return len(self.model.encode(text))


def select_model(model_name: str, **kwargs) -> EvalModel:
    model_map = dict(
        seq_to_seq=SeqToSeqModel,
        causal=CausalModel,
        llama=LlamaModel,
        chatglm=ChatGLMModel,
        openai=OpenAIModel,
        rwkv=RWKVModel,
    )
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"{model_name}. Choose from {list(model_map.keys())}")
    return model_class(**kwargs)


def test_model(
    prompt: str = "Write an email about an alpaca that likes flan.",
    model_name: str = "seq_to_seq",
    model_path: str = "google/flan-t5-base",
    **kwargs,
):
    model = select_model(model_name, model_path=model_path, **kwargs)
    print(locals())
    print(model.run(prompt))


"""
p modeling.py test_model --model_name causal --model_path gpt2
p modeling.py test_model --model_name llama --model_path decapoda-research/llama-7b-hf
p modeling.py test_model --model_name llama --model_path chavinlo/alpaca-native
p modeling.py test_model --model_name chatglm --model_path THUDM/chatglm-6b
p modeling.py test_model --model_name llama --model_path TheBloke/koala-7B-HF
p modeling.py test_model --model_name llama --model_path eachadea/vicuna-13b --load_8bit
p modeling.py test_model --model_name causal --model_path togethercomputer/GPT-NeoXT-Chat-Base-20B --load_8bit
p modeling.py test_model --model_name llama --model_path huggyllama/llama-7b --lora_path tloen/alpaca-lora-7b
p modeling.py test_model --model_name seq_to_seq --model_path google/flan-t5-xl --lora_path declare-lab/flan-alpaca-xl-lora
p modeling.py test_model --model_name openai --model_path VisualQuestionAnswering --use_azure
p modeling.py test_model --model_name rwkv --model_path https://huggingface.co/BlinkDL/rwkv-4-raven/resolve/main/RWKV-4-Raven-7B-v11-Eng99%25-Other1%25-20230427-ctx8192.pth
p modeling.py test_model --model_name causal --model_path mosaicml/mpt-7b-instruct
"""


if __name__ == "__main__":
    Fire()
