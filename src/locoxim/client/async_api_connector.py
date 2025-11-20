# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

# credit: https://github.com/adobe-research/NoLiMa/blob/main/evaluation/async_api_connector.py

from pprint import pprint
from typing import TYPE_CHECKING, Literal, Optional, Union

from deocr.engine.playwright.async_api import transform
from langchain_aws import ChatBedrockConverse
from openai import AsyncAzureOpenAI, AsyncOpenAI
from tenacity import (
    retry,
    stop_after_delay,
    wait_random,
)

from ..args import args_to_dict
from ..dataio import api_cache_io, api_cache_path, remove_html_tags, strip
from ..token_counter import TokenCounter
from .image_helper import ImageTextPayload

if TYPE_CHECKING:
    from deocr.engine.args import RenderArgs
    from openai.types.chat import ChatCompletion


DEFAULT_TOKENIZER_MAPPING = {
    "vllm": "huggingface",
    "openai": "tiktoken",
    "azure-openai": "tiktoken",
}


class APIConnector:
    """
    APIConnector class to unify the API calls for different API providers

    Args:
        api_key (str): API key for the LLM/VLM service
        api_url (str): API URL for the LLM/VLM service
        api_provider (str): API provider type, e.g., openai, vllm
        model (str): Model name or path
        system_prompt (str, optional): Default system prompt for the model.
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
        system_prompt: str | None,
        tokenizer_type: Literal["huggingface", "tiktoken"] | None = None,
        tokenizer_model: str | None = None,
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

        self.token_counter = TokenCounter(
            tokenizer_type=tokenizer_type or DEFAULT_TOKENIZER_MAPPING[api_provider],
            tokenizer_model=tokenizer_model or model,
        )

        self.default_system_prompt = strip(system_prompt)

        if api_provider == "openai":
            self.api = AsyncOpenAI(
                api_key=api_key,
                base_url=api_url,
                max_retries=kwargs["max_retries"],
                timeout=kwargs["timeout"],
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

    async def generate_response(
        self,
        system_prompt: str | None,
        user_prompt: Union[str, ImageTextPayload],
        max_tokens: int = 100,
        use_default_system_prompt: bool = True,
        pure_text: bool = True,
        generation_kwargs: dict | None = None,
        extra_kwargs: dict | None = None,
        render_args: Optional["RenderArgs"] = None,
        parent_api_cache_dir: str | None = None,
        verbose: bool = False,
    ) -> dict:
        """
        Generates a response using the API with the given prompts

        Parameters:
            system_prompt (`str`): System prompt
            user_prompt (`str`): User prompt
            max_tokens (`int`, optional): Maximum tokens to generate. Default: 100.
            use_default_system_prompt (`bool`, optional): Whether to use the default system prompt. Default: True.
            pure_text (`bool`, optional): Whether to use pure text in API call. If false, render context as images. Default: True.
            generation_kwargs (`dict`, optional): Additional generation keyword arguments for the API call, e.g. temperature, top_p.
            render_args (`RenderArgs`, optional): Rendering arguments for multimodal inputs.
            verbose (`bool`, optional): Whether to print verbose logs. Default: False.

        Returns:
            `dict`: Response from the API that includes the response, prompt tokens count, completion tokens count, total tokens count, and stopping reason
        """
        cache_path = api_cache_path(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
                "use_default_system_prompt": use_default_system_prompt,
                "pure_text": pure_text,
                "generation_kwargs": generation_kwargs,
                "render_args": args_to_dict(render_args),
                "api_provider": self.api_provider,
                "model": self.model,
            },
            parent=parent_api_cache_dir,
        )
        if (
            parent_api_cache_dir is not None
            and (response := api_cache_io(cache_path)) is not None
        ):
            if verbose:
                print("=== API Call Cached Response ===")
                pprint(response)
                print("===============================")
            return response

        messages: list[dict] = []

        # form system prompt
        system_prompt_content = None

        # highest priority given to cmd line arg i.e. self.default_system_prompt
        if use_default_system_prompt:
            system_prompt_content = self.default_system_prompt
        # fallback to function-scoped, i.e. the system_prompt field in data
        elif system_prompt is not None and len(system_prompt.strip()) > 0:
            system_prompt_content = system_prompt

        # add system prompt to messages, but deliberately optional
        if system_prompt_content is not None:
            messages.append({"role": "system", "content": system_prompt_content})

        # form user prompt
        user_prompt_content = None
        if isinstance(user_prompt, str) and (pure_text or render_args is None):
            # plain text input
            user_prompt_content = remove_html_tags(user_prompt.strip())
        elif isinstance(user_prompt, str) and render_args is not None:
            # multimodal input by guessing
            payload = ImageTextPayload()

            # a bit of hard code, but fine for now
            # if you run into errors here,
            # consider add this delim to task_template
            vision_part, text_part = user_prompt.split("\n\nQuestion:", 1)

            images = await transform(
                item=vision_part,
                cache_dir=".cache",
                render_args=render_args,
            )
            for image in images:
                payload.add_image_adaptive(
                    image,
                    save_format=render_args.save_format,
                    save_kwargs=render_args.save_kwargs,
                )

            payload.add_text(text_part)
            user_prompt_content = payload.to_message_content()
        elif isinstance(user_prompt, ImageTextPayload):
            user_prompt_content = user_prompt.to_message_content()
        else:
            raise ValueError(f"Invalid: {type(user_prompt)}, {user_prompt}")

        # add user prompt to messages
        assert user_prompt_content is not None, "user_prompt_content is None"
        messages.append({"role": "user", "content": user_prompt_content})

        # messages now ready for API call
        if verbose:
            print("=== Raw User Prompt ===")
            pprint(user_prompt)
            print("=======================")
            print("=== API Call Messages ===")
            pprint(messages)
            print("=========================")

        # call API with retries
        if (
            self.api_provider == "openai"
            or self.api_provider == "vllm"
            or self.api_provider == "azure-openai"
        ):

            @retry(
                reraise=True,
                wait=wait_random(1, 20),
                stop=stop_after_delay(300),
            )
            async def generate_content() -> "ChatCompletion":
                params = {"model": self.model, "messages": messages, "seed": 43}
                if self.model_config.get("openai_thinking_model", False):
                    # thinking models, e.g. GPT-5 use another set of params
                    params["max_completion_tokens"] = max_tokens
                    params["reasoning_effort"] = "minimal"
                    params["verbosity"] = "low"
                elif self.model.startswith("gemini"):
                    params["max_tokens"] = max_tokens
                    params["extra_body"] = {
                        "extra_body": {
                            "google": {
                                "thinking_config": {
                                    "thinking_budget": self.model_config.get(
                                        "thinking_budget", 0
                                    ),
                                }
                            }
                        }
                    }
                else:
                    params["max_tokens"] = max_tokens
                    params = params | (generation_kwargs or {})
                    params["extra_body"] = extra_kwargs

                try:
                    completion = await self.api.chat.completions.create(**params)
                    return completion
                except Exception as e:
                    print("=== API Call Exception ===")
                    pprint(params)
                    pprint(messages)
                    print("--------------------------")
                    raise e

            completion = await generate_content()

            output = {
                "response": completion.choices[0].message.content,
                "finish_reason": completion.choices[0].finish_reason,
                "api_cache_path": cache_path,
            }

            # if available, report token usage
            if (usage := completion.usage) is not None:
                output |= {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }

                # if available, report stats for reasoning tokens
                if (completion_details := usage.completion_tokens_details) is not None:
                    output["reasoning_tokens"] = completion_details.reasoning_tokens
                if (prompt_details := usage.prompt_tokens_details) is not None:
                    output["cached_tokens"] = prompt_details.cached_tokens

            if verbose:
                print("=== API Call Response ===")
                pprint(output)
                print("=========================")

            return output

        raise NotImplementedError(f"Unsupported API provider: {self.api_provider}")
