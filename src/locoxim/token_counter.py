from functools import cache, partial
from typing import Callable, Literal

import tiktoken
from transformers import AutoTokenizer


class TokenCounter:
    encode: Callable[[str], list[int]]
    decode: Callable[[list[int]], str]

    def __init__(
        self,
        tokenizer_type: Literal["huggingface", "tiktoken"],
        tokenizer_model: str,
    ):
        self.tokenizer_type = tokenizer_type
        self.tokenizer_model = tokenizer_model

        match tokenizer_type:
            case "huggingface":
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
                self.encode = partial(self.tokenizer.encode, add_special_tokens=False)
                self.decode = self.tokenizer.decode
            case "tiktoken":
                self.tokenizer = tiktoken.encoding_for_model(tokenizer_model)
                self.encode = self.tokenizer.encode
                self.decode = self.tokenizer.decode
            case _:
                raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    @cache
    def token_count(self, text: str) -> int:
        tokens = self.encode(text)
        return len(tokens)
