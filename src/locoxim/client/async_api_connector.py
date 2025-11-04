# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

# credit: https://github.com/adobe-research/NoLiMa/blob/main/evaluation/async_api_connector.py

from functools import cache
from typing import Union

import httpx
import tiktoken
from deocr.engine.args import RenderArgs
from deocr.engine.playwright.async_api import transform
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from langchain_aws import ChatBedrockConverse
from openai import AsyncAzureOpenAI, AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletion
from tenacity import (
    retry,
    retry_if_exception_message,
    retry_if_exception_type,
    stop_after_delay,
    wait_random,
    wait_random_exponential,
)
from transformers import AutoTokenizer
from vertexai.preview.tokenization import get_tokenizer_for_model

from .image_helper import ImageTextPayload

DEFAULT_TOKENIZER_MAPPING = {
    "gemini": "google",
    "vllm": "huggingface",
    "openai": "tiktoken",
    "azure-openai": "tiktoken",
    "aws": "huggingface",
}


class APIConnector:
    """
    APIConnector class to unify the API calls for different API providers

    Args:
        api_key (str): API key for the LLM/VLM service
        api_url (str): API URL for the LLM/VLM service
        api_provider (str): API provider type, e.g., openai, vllm
        model (str): Model name or path
        system_prompt (str, optional): Default system prompt for the model. Default: "You
        are a helpful assistant".
        tokenizer_type (str, optional): Type of tokenizer to use. Default: None.
        tokenizer_model (str, optional): Model path or local dir path for the tokenizer. Default: None.
        kwargs: Additional keyword arguments depending on the API provider
    """

    def __init__(
        self,
        api_key: str,
        api_url: str,
        api_provider: str,
        model: str,
        system_prompt: str = "You are a helpful assistant",
        tokenizer_type: str = None,
        tokenizer_model: str = None,
        **kwargs,
    ) -> None:
        assert api_provider in DEFAULT_TOKENIZER_MAPPING, (
            f"Invalid API provider: {api_provider}"
        )
        assert (
            tokenizer_type is None
            or tokenizer_type in DEFAULT_TOKENIZER_MAPPING.values()
        ), f"Invalid tokenizer type: {tokenizer_type}"

        self.api_provider = api_provider
        self.tokenizer_type = tokenizer_type or DEFAULT_TOKENIZER_MAPPING[api_provider]
        self.tokenizer_model = tokenizer_model or model

        if self.tokenizer_type == "huggingface":
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_model, use_fast=True
            )
        elif self.tokenizer_type == "tiktoken":
            self.tokenizer = tiktoken.get_encoding(self.tokenizer_model)
        elif self.tokenizer_type == "google":
            self.tokenizer = get_tokenizer_for_model(self.tokenizer_model)

        self.default_system_prompt = system_prompt.strip()

        if api_provider == "openai":
            self.api = AsyncOpenAI(
                api_key=api_key,
                base_url=api_url,
            )
        elif api_provider == "azure-openai":
            self.api = AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=api_url,
                api_version=kwargs["azure_api_version"],
            )
        elif api_provider == "vllm":
            self.api = AsyncOpenAI(
                api_key=api_key,
                base_url=api_url,
                max_retries=kwargs["max_retries"],
                timeout=kwargs["timeout"],
            )
        elif api_provider == "gemini":
            self.api = genai.Client(
                vertexai=True,
                project=kwargs["project_ID"],
                location=kwargs["location"],
                http_options=types.HttpOptions(timeout=kwargs["http_timeout"] * 1000),
            )

            self.gemini_retry_config = {
                "initial": kwargs["retry_delay"],
                "maximum": kwargs["retry_max_delay"],
                "multiplier": kwargs["retry_multiplier"],
                "timeout": kwargs["retry_timeout"],
            }

        elif api_provider == "aws":
            self.api = ChatBedrockConverse(
                model=model,
                temperature=kwargs["temperature"],
                max_tokens=kwargs["max_tokens"],
                region_name=kwargs["region"],
                top_p=kwargs["top_p"],
            )

        self.model = model
        self.model_config = kwargs

    def encode(self, text: str) -> list:
        """
        Encodes the given text using the tokenizer

        Parameters:
            text (`str`): Text to encode

        Returns:
            `list`: Tokens
        """
        if self.tokenizer_type == "huggingface":
            return self.tokenizer.encode(text, add_special_tokens=False)
        elif self.tokenizer_type == "tiktoken":
            return self.tokenizer.encode(text)
        elif self.tokenizer_type == "google":
            return self.tokenizer._sentencepiece_adapter._tokenizer.encode(text)
        else:
            raise ValueError(f"Invalid tokenizer type: {self.tokenizer_type}")

    def decode(self, tokens: list) -> str:
        """
        Decodes the given tokens using the tokenizer

        Parameters:
            tokens (`list`): Tokens to decode

        Returns:
            `str`: Decoded text
        """
        if self.tokenizer_type == "huggingface":
            return self.tokenizer.decode(tokens)
        elif self.tokenizer_type == "tiktoken":
            return self.tokenizer.decode(tokens)
        elif self.tokenizer_type == "google":
            return self.tokenizer._sentencepiece_adapter._tokenizer.decode(tokens)
        else:
            raise ValueError(f"Invalid tokenizer type: {self.tokenizer_type}")

    @cache
    def token_count(self, text: str) -> int:
        """
        Returns the token count of the given text

        Parameters:
            text (`str`): Text to count tokens
            use_cache (`bool`, optional): Use cache for token count. Defaults to True.

        Returns:
            `int`: Token count of the text
        """
        if self.tokenizer_type == "tiktoken":
            return len(self.tokenizer.encode(text))
        elif self.tokenizer_type == "huggingface":
            return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        elif self.tokenizer_type == "google":
            return self.tokenizer.count_tokens(text).total_tokens
        else:
            raise ValueError(f"Invalid tokenizer type: {self.tokenizer_type}")

    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: Union[str, ImageTextPayload],
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        use_default_system_prompt: bool = True,
        pure_text: bool = True,
        render_args: RenderArgs | None = None,
    ) -> dict:
        """
        Generates a response using the API with the given prompts

        Parameters:
            system_prompt (`str`): System prompt
            user_prompt (`str`): User prompt
            max_tokens (`int`, optional): Maximum tokens to generate. Defaults to 100.
            temperature (`float`, optional): Temperature. Defaults to 0.0 (Greedy Sampling).
            top_p (`float`, optional): Top-p. Defaults to 1.0.

        Returns:
            `dict`: Response from the API that includes the response, prompt tokens count, completion tokens count, total tokens count, and stopping reason
        """
        messages: list[dict] = []
        # formulate system prompt
        system_prompt = system_prompt.strip()

        # form system prompt
        if use_default_system_prompt and len(self.default_system_prompt) > 0:
            system_msg = {"role": "system", "content": self.default_system_prompt}
            messages.append(system_msg)
        elif len(system_prompt) > 0:
            system_msg = {"role": "system", "content": system_prompt}
            messages.append(system_msg)
        else:
            # no system prompt otherwise, allowed
            pass

        # form user prompt
        if isinstance(user_prompt, str) and (pure_text or render_args is None):
            # plain text input
            messages.append({"role": "user", "content": user_prompt})
        elif isinstance(user_prompt, str):
            payload = ImageTextPayload()
            vision_part, text_part = user_prompt.split("\n\nQuestion:", 1)

            payload.add_text(text_part)
            images = await transform(
                item=vision_part,
                cache_dir=".cache",
                render_args=render_args,
            )
            for image in images:
                payload.add_image_adaptive(image)

            messages.append({"role": "user", "content": payload.to_message_content()})
        elif isinstance(user_prompt, ImageTextPayload):
            messages.append(
                {"role": "user", "content": user_prompt.to_message_content()}
            )
        else:
            raise ValueError(f"Invalid: {type(user_prompt)}, {user_prompt}")

        # call API with retries
        if (
            self.api_provider == "openai"
            or self.api_provider == "vllm"
            or self.api_provider == "azure-openai"
        ):

            @retry(
                reraise=True,
                wait=wait_random(1, 20),
                retry=retry_if_exception_type(RateLimitError),
                stop=stop_after_delay(300),
            )
            async def generate_content() -> "ChatCompletion":
                params = {"model": self.model, "messages": messages, "seed": 43}
                if self.model_config.get("openai_thinking_model", False):
                    params["max_completion_tokens"] = max_tokens
                else:
                    params["max_tokens"] = max_tokens
                    params["temperature"] = temperature
                    params["top_p"] = top_p

                completion = await self.api.chat.completions.create(**params)
                return completion

            completion = await generate_content()

            output = {
                "response": completion.choices[0].message.content,
                "prompt_tokens": completion.usage.prompt_tokens,
                "completion_tokens": completion.usage.completion_tokens,
                "total_tokens": completion.usage.total_tokens,
                "finish_reason": completion.choices[0].finish_reason,
                "cached_tokens": completion.usage.prompt_tokens_details.cached_tokens
                if completion.usage.prompt_tokens_details
                else None,
            }
            if self.model_config.get("openai_thinking_model", False):
                output["reasoning_tokens"] = (
                    completion.usage.completion_tokens_details.reasoning_tokens
                )
            return output

        elif self.api_provider == "gemini":

            @retry(
                reraise=True,
                wait=wait_random_exponential(
                    multiplier=self.gemini_retry_config["multiplier"],
                    max=self.gemini_retry_config["maximum"],
                ),
                stop=stop_after_delay(self.gemini_retry_config["timeout"]),
                retry=retry_if_exception_type(
                    (ClientError, httpx.ConnectTimeout, httpx.TimeoutException)
                ),
            )
            async def generate_content():
                completion = await self.api.aio.models.generate_content(
                    model=self.model,
                    contents=messages[-1]["content"],
                    config=types.GenerateContentConfig(
                        system_instruction=self.default_system_prompt,
                        temperature=temperature,
                        top_p=top_p,
                        max_output_tokens=max_tokens,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=self.model_config["thinking_budget"]
                        )
                        if "thinking_budget" in self.model_config
                        else None,
                        seed=43,
                    ),
                )
                return completion

            completion = await generate_content()

            return {
                "response": completion.text if completion.text is not None else "",
                "prompt_tokens": completion.usage_metadata.prompt_token_count,
                "completion_tokens": completion.usage_metadata.candidates_token_count,
                "total_tokens": completion.usage_metadata.total_token_count,
                "finish_reason": completion.candidates[0].finish_reason,
                "cached_tokens": completion.usage_metadata.cached_content_token_count
                if completion.usage_metadata.cached_content_token_count is not None
                else None,
                "reasoning_tokens": completion.usage_metadata.thoughts_token_count
                if completion.usage_metadata.thoughts_token_count is not None
                else None,
            }
        elif self.api_provider == "aws":

            @retry(
                reraise=True,
                wait=wait_random(5, 20),
                retry=retry_if_exception_message(match=r".*ThrottlingException.*"),
            )
            async def generate_content():
                completion = await self.api.ainvoke(messages)
                return completion

            completion = await generate_content()

            return {
                "response": completion.content,
                "prompt_tokens": completion.usage_metadata["input_tokens"],
                "completion_tokens": completion.usage_metadata["output_tokens"],
                "total_tokens": completion.usage_metadata["total_tokens"],
                "finish_reason": completion.response_metadata["stopReason"],
            }
        else:
            raise ValueError(f"Invalid API provider: {self.api_provider}")
