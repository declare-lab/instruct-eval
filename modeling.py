from typing import Optional

from fire import Fire
from pydantic import BaseModel
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


class EvalModel(BaseModel, arbitrary_types_allowed=True):
    max_input_length: int = 512
    max_output_length: int = 512

    def run(self, prompt: str) -> str:
        raise NotImplementedError

    def check_valid_length(self, text: str) -> bool:
        raise NotImplementedError


class SeqToSeqModel(EvalModel):
    model_path: str
    model: Optional[PreTrainedModel]
    tokenizer: Optional[PreTrainedTokenizer]
    device: str = "cuda"

    def load(self):
        if self.model is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
            self.model.eval()
            self.model.to(self.device)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def run(self, prompt: str) -> str:
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_length=self.max_output_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def check_valid_length(self, text: str) -> bool:
        self.load()
        inputs = self.tokenizer(text)
        return len(inputs.input_ids) <= self.max_input_length


def select_model(model_name: str, **kwargs) -> EvalModel:
    if model_name == "seq_to_seq":
        return SeqToSeqModel(**kwargs)
    raise ValueError(f"Invalid name: {model_name}")


def test_model(
    prompt: str = "Write an email about an alpaca that likes flan.",
    model_name: str = "seq_to_seq",
    model_path: str = "google/flan-t5-base",
):
    model = select_model(model_name, model_path=model_path)
    print(model.run(prompt))


if __name__ == "__main__":
    Fire()
