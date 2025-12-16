import asyncio
from typing import TYPE_CHECKING, Generator, TypedDict

import numpy as np
from deocr.engine.playwright.async_api import transform as async_transform

from .client.async_api_connector import APIConnector
from .dataio import BookHaystack, args_to_dict, fill_placeholders, has_placeholder

if TYPE_CHECKING:
    from deocr.engine.args import RenderArgs
    from PIL.Image import Image as PILImage

    from .args import DataArgs, ModelArgs
    from .dataio import QuestionItem


class ContextImagePair(TypedDict):
    problem: str
    answers: list[str]
    images: list["PILImage"]
    _context: str
    _source: "QuestionItem"
    _render_args: "RenderArgs"
    _placement_output: dict


def iter_context_and_images(
    model_args: "ModelArgs",
    data_args: "DataArgs",
    render_args: "RenderArgs",
    question_item: "QuestionItem",
    haystack_path: str,
) -> Generator[ContextImagePair, None, None]:
    """
    Yield tuples (context, context_len, images) incrementally.
    """
    api_connector = APIConnector(**args_to_dict(model_args))
    haystack = BookHaystack(haystack_path)

    rng = np.random.RandomState(question_item.seed)
    loop = asyncio.get_event_loop()

    for _needle_depth_i, _needle_depth_percentage in enumerate(
        np.linspace(
            data_args.document_depth_percent_min,
            data_args.document_depth_percent_max,
            data_args.document_depth_num_tests,
        )
    ):
        needle = question_item.needle
        retrieval_question = question_item.retrieval_question

        selected_character = None
        if needle is None:
            # allow needle to be None, n=1, skip 2nd onwards
            if _needle_depth_i != 0:
                continue
        elif has_placeholder(needle, "{CHAR}"):
            assert isinstance(question_item.character_set, list), (
                f"character_set {type(question_item.character_set)} != list"
            )

            # for reproducibility, changing with depth, but always the same sequence for a given seed
            selected_character = str(rng.choice(question_item.character_set))
            needle = fill_placeholders(needle, "{CHAR}", selected_character)

        if has_placeholder(retrieval_question, "{CHAR}"):
            assert selected_character is not None, (
                "selected_character is None but retrieval_question has placeholder"
            )
            retrieval_question = fill_placeholders(
                retrieval_question, "{CHAR}", selected_character
            )

        placement_output = haystack.generate_w_needle_placement(
            needle=needle,
            token_counter=api_connector.token_counter,
            context_length=data_args.context_length,
            depth=_needle_depth_percentage / 100,
            shift=data_args.shift,
            static_depth=data_args.static_depth,
            distractor=question_item.distractor,
            context=question_item.context,
        )

        images = loop.run_until_complete(
            async_transform(
                item=placement_output["text"],
                cache_dir=".cache",
                render_args=question_item.render_args or render_args,
            )
        )

        context = placement_output.pop("text")
        yield {
            "problem": retrieval_question,
            "answers": question_item.gold_answers or [selected_character],
            "images": images,
            "_context": context,
            "_source": question_item,
            "_render_args": render_args,
            "_placement_output": placement_output,
        }
