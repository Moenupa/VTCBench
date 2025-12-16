# Copyright 2022 Adobe
# All Rights Reserved.

# NOTICE: Adobe permits you to use, modify, and distribute this file in
# accordance with the terms of the Adobe license agreement accompanying
# it.

# credit: https://github.com/adobe-research/NoLiMa/blob/main/evaluation/async_evaluate.py

import asyncio
import json
import os
import os.path as osp
import time
from typing import TYPE_CHECKING, Optional, TypedDict

import numpy as np

from .client.async_api_connector import APIConnector
from .client.image_helper import ImageTextPayload
from .dataio import (
    HASH_CACHE_KEY,
    BookHaystack,
    api_cache_io,
    args_to_dict,
    fill_placeholders,
    get_hash,
    get_hash_str,
    has_placeholder,
    remove_html_tags,
)
from .metric import calc_metrics

if TYPE_CHECKING:
    from deocr.engine.args import RenderArgs

    from .args import DataArgs, ModelArgs, RunArgs
    from .dataio import QuestionItem


def scan_dir_for_hash(
    scan_dir: str,
    target_hash: str,
) -> str | None:
    for filename in os.listdir(scan_dir):
        scan_file = f"{scan_dir}/{filename}"
        with open(scan_file, "r") as file:
            other_result = json.load(file)

        if target_hash == other_result[HASH_CACHE_KEY]:
            return scan_file

    return None


def evaluate(
    model_args: "ModelArgs",
    run_args: "RunArgs",
    data_args: "DataArgs",
    render_args: Optional["RenderArgs"],
    question_item: "QuestionItem",
    haystack_path: str,
    verbose: bool = False,
) -> str:
    api_connector = APIConnector(**args_to_dict(model_args))
    haystack = BookHaystack(haystack_path)

    path_friendly_model_name = model_args.model.replace("/", "_")
    eval_name = f"{path_friendly_model_name}_{haystack.get_hash()[:8]}_{question_item.question_id}_{int(time.time())}"

    results_dir = (
        f"{run_args.result_dir}/{path_friendly_model_name}"
        f"/{get_hash(args_to_dict(data_args) | args_to_dict(model_args) | args_to_dict(render_args))}"
    )
    os.makedirs(results_dir, exist_ok=True)

    # fields that affects id
    outputs = {
        "model_args": args_to_dict(model_args),
        "data_args": args_to_dict(data_args),
        "render_args": args_to_dict(render_args),
        "question_item": args_to_dict(question_item),
        "system_prompt": api_connector.default_system_prompt
        if data_args.use_default_system_prompt
        else question_item.system_prompt,
        "haystack_hash": haystack.get_hash(),
    }
    # snapshot -> hash
    output_hash = get_hash(outputs)
    outputs[HASH_CACHE_KEY] = output_hash

    # lazy add fields for runtime, this may change but does not affect id/result
    outputs["run_args"] = args_to_dict(run_args)
    outputs["eval_name"] = eval_name
    results_for_all_depths: list[dict] = []

    results_path = f"{results_dir}/{eval_name}.json"
    if osp.exists(results_path):
        print(f"{results_path} exists, skipped.")
        return results_path

    # scan all existing results
    # WARN: this could cause a lot of IO traffic
    if run_args.prevent_duplicate_tests:
        scan_dir = osp.dirname(results_path)
        if (
            match := scan_dir_for_hash(
                scan_dir=scan_dir,
                target_hash=output_hash,
            )
        ) is not None:
            print(
                f"Duplicate: {match}, {output_hash}, skipped.",
            )
            return match

    async_tasks = []
    rng = np.random.RandomState(question_item.seed)
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

        task_template = data_args.task_template or question_item.task_template
        assert isinstance(task_template, str), (
            f"task_template is not str type{type(task_template)}"
        )
        filled_template = task_template.format(
            haystack=placement_output["text"], question=retrieval_question
        )

        async_tasks.append(
            api_connector.generate_response(
                system_prompt=question_item.system_prompt,
                user_prompt=filled_template,
                max_tokens=model_args.max_tokens,
                use_default_system_prompt=data_args.use_default_system_prompt,
                pure_text=data_args.pure_text,
                render_args=question_item.render_args or render_args,
                generation_kwargs={
                    "temperature": model_args.temperature,
                    "top_p": model_args.top_p,
                },
                extra_kwargs=model_args.extra_kwargs,
                api_cache_dir=run_args.api_cache_dir,
                verbose=verbose and (_needle_depth_i == 0),
            )
        )

        results_for_all_depths.append(
            {
                "placement_metadata": {
                    k: v for k, v in placement_output.items() if k != "text"
                },
                "gold_answers": question_item.gold_answers
                if selected_character is None
                else [selected_character],
            }
        )

        if (log_dir := run_args.log_dir) is not None:
            log_path = f"{log_dir}/{eval_name}_{str(np.round(_needle_depth_percentage, 3))}.txt"
            with open(log_path, "w") as file:
                file.write(placement_output["text"])

    loop = asyncio.get_event_loop()
    responses: list[dict] = loop.run_until_complete(asyncio.gather(*async_tasks))

    for _res_i, _res in enumerate(responses):
        results_for_all_depths[_res_i]["metric"] = calc_metrics(
            _res["response"],
            gold_answers=results_for_all_depths[_res_i]["gold_answers"],
        )
        results_for_all_depths[_res_i] = results_for_all_depths[_res_i] | _res
        api_cache_io(_res["api_cache_path"], save_response=_res)

    outputs["results"] = results_for_all_depths
    outputs["result_path"] = results_path

    with open(results_path, "w") as file:
        json.dump(outputs, file, indent="\t")

    return results_path


class PregeneratedVQATriplet(TypedDict):
    problem: str
    # [{'bytes': b'...'}, ...]
    images: list[dict[str, bytes]]
    answers: list[str]
    instruction: str | None
    _context: str | None


def evaluate_static(
    model_args: "ModelArgs",
    run_args: "RunArgs",
    # skips data_args, render_args, question_item, haystack_path
    # because data is static i.e. pregenerated
    example: PregeneratedVQATriplet,
    pure_text: bool = True,
    use_default_system_prompt: bool = True,
    verbose: bool = False,
) -> str:
    # example follows the same schema as our huggingface dataset
    # https://huggingface.co/datasets/MLLM-CL/VTCBench
    api_connector = APIConnector(**args_to_dict(model_args))

    path_friendly_model_name = model_args.model.replace("/", "_")

    # expects all the fields to be json-serializable
    example_meta = {
        "problem": example["problem"],
        "answers": example["answers"],
        "images_sha256": [
            get_hash_str(json.dumps(img["bytes"].decode("iso-8859-1")))
            for img in example["images"]
        ],
    }
    eval_name = f"{path_friendly_model_name}_static_{get_hash(example_meta)[:8]}_{int(time.time())}"

    # group by model config only (no data/render args for static)
    results_dir = (
        f"{run_args.result_dir}"
        f"/{path_friendly_model_name}"
        f"/{get_hash(args_to_dict(model_args) | example_meta)}"
    )
    os.makedirs(results_dir, exist_ok=True)

    outputs = {
        "model_args": args_to_dict(model_args),
        "run_args": args_to_dict(run_args),
        "example_meta": example_meta,
        "system_prompt": api_connector.default_system_prompt
        if use_default_system_prompt
        else None,
    }
    output_hash = get_hash(outputs)
    outputs[HASH_CACHE_KEY] = output_hash
    outputs["eval_name"] = eval_name

    results_path = f"{results_dir}/{eval_name}.json"
    if osp.exists(results_path):
        print(f"{results_path} exists, skipped.")
        return results_path

    if run_args.prevent_duplicate_tests:
        scan_dir = osp.dirname(results_path)
        if (
            match := scan_dir_for_hash(scan_dir=scan_dir, target_hash=output_hash)
        ) is not None:
            print(f"Duplicate: {match}, {output_hash}, skipped.")
            return match

    # format user_prompt as ImageTextPayload
    payload: ImageTextPayload = ImageTextPayload()
    if pure_text and example["_context"] is not None:
        payload.add_text(remove_html_tags(example["_context"]))
    else:
        for each_image in example["images"]:
            payload.add_image_adaptive(each_image["bytes"], save_format="jpeg")
    if example["instruction"] is not None:
        payload.add_text(example["instruction"] + example["problem"])
    else:
        payload.add_text(example["problem"])
    async_task = api_connector.generate_response(
        # no system prompt in static data, use the default(model-specific) one
        system_prompt=None,
        user_prompt=payload,
        max_tokens=model_args.max_tokens,
        use_default_system_prompt=use_default_system_prompt,
        pure_text=False,
        render_args=None,
        generation_kwargs={
            "temperature": model_args.temperature,
            "top_p": model_args.top_p,
        },
        extra_kwargs=model_args.extra_kwargs,
        api_cache_dir=run_args.api_cache_dir,
        verbose=verbose,
    )

    loop = asyncio.get_event_loop()
    response: dict = loop.run_until_complete(asyncio.gather(async_task))[0]

    result_entry: dict = {
        "gold_answers": example["answers"],
    }
    result_entry["metric"] = calc_metrics(
        response["response"],
        gold_answers=result_entry["gold_answers"],
    )
    result_entry |= response
    api_cache_io(response["api_cache_path"], save_response=response)

    outputs["results"] = [result_entry]
    outputs["result_path"] = results_path

    with open(results_path, "w") as file:
        json.dump(outputs, file, indent="\t")

    return results_path
