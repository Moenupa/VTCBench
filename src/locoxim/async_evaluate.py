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
from typing import Literal

import numpy as np
from deocr.engine.playwright.async_api import RenderArgs

from .args import (
    DataArgs,
    ModelArgs,
    RunArgs,
    args_to_dict,
)
from .client.async_api_connector import APIConnector
from .dataio import (
    HASH_CACHE_KEY,
    QuestionItem,
    api_cache_io,
    dataclass_to_dict,
    fill_placeholders,
    get_hash,
    has_placeholder,
)
from .NoLiMa.book_haystack import BookHaystack


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


def _evaluate_response(
    response: str,
    gold_answers: list[str],
    metric: Literal["EM", "contains", "lastline_EM", "lastline_contains"],
) -> int:
    assert gold_answers is not None and len(gold_answers) > 0, (
        "gold_answers is None or empty"
    )

    match metric:
        case "EM":
            return int(response.strip() in gold_answers)
        case "contains":
            return int(any([gold_answer in response for gold_answer in gold_answers]))
        case "lastline_EM":
            return int(response.strip().split("\n")[-1] in gold_answers)
        case "lastline_contains":
            return int(
                any(
                    [
                        gold_answer in response.strip().split("\n")[-1]
                        for gold_answer in gold_answers
                    ]
                )
            )
        case _:
            raise ValueError(f"Invalid metric: {metric}")


def evaluate(
    model_args: ModelArgs,
    run_args: RunArgs,
    data_args: DataArgs,
    render_args: RenderArgs | None,
    question_item: QuestionItem,
    haystack_path: str,
    verbose: bool = False,
) -> str:
    api_connector = APIConnector(**args_to_dict(model_args))
    haystack = BookHaystack(haystack_path)

    eval_name = (
        f"{model_args.model}_book_{question_item.question_id}_{int(time.time())}"
        if question_item.question_id != ""
        else f"{model_args.model}_book_{int(time.time())}"
    )

    results_dir = f"{run_args.parent_results_dir}/{model_args.model}/{get_hash(args_to_dict(data_args) | args_to_dict(model_args) | args_to_dict(render_args))}"
    os.makedirs(results_dir, exist_ok=True)

    outputs = {
        "model_args": args_to_dict(model_args),
        "data_args": args_to_dict(data_args),
        "run_args": args_to_dict(run_args),
        "render_args": args_to_dict(render_args),
        "question_item": dataclass_to_dict(question_item),
        "system_prompt": api_connector.default_system_prompt
        if data_args.use_default_system_prompt
        else question_item.system_prompt,
        "haystack_hash": haystack.get_hash(),
    }
    # snapshot -> hash
    outputs[HASH_CACHE_KEY] = get_hash(outputs)

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
        target_hash = outputs[HASH_CACHE_KEY]
        if (
            match := scan_dir_for_hash(
                scan_dir=scan_dir,
                target_hash=target_hash,
            )
        ) is not None:
            print(
                f"Duplicate: {match}, {target_hash}, skipped.",
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
                render_args=render_args,
                generation_kwargs={
                    "temperature": model_args.temperature,
                    "top_p": model_args.top_p,
                },
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
        results_for_all_depths[_res_i]["metric"] = _evaluate_response(
            _res["response"],
            gold_answers=results_for_all_depths[_res_i]["gold_answers"],
            metric=run_args.metric,
        )
        results_for_all_depths[_res_i] = results_for_all_depths[_res_i] | _res
        api_cache_io(_res["api_cache_path"], save_response=_res)

    outputs["results"] = results_for_all_depths
    outputs["result_path"] = results_path

    with open(results_path, "w") as file:
        json.dump(outputs, file, indent="\t")

    return results_path
