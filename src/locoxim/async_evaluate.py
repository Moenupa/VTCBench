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

import numpy as np
from deocr.engine.playwright.async_api import RenderArgs

from .args import (
    DataArgs,
    ModelArgs,
    RunArgs,
    args_to_dict,
)
from .client.async_api_connector import APIConnector
from .dataio import HASH_CACHE_KEY, QuestionItem, fill_placeholders, get_hash
from .NoLiMa.book_haystack import BookHaystack


def has_placeholder(text: str, placeholder: str = "{CHAR}") -> bool:
    return placeholder in text


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


# modified from NoLiMa_Tester
# https://github.com/adobe-research/NoLiMa/blob/cb14780b249fecf2851127b2101a062c1b2c6430/evaluation/async_evaluate.py#L24
class NeedleHaystackTester:
    def __init__(
        self,
        model_args: ModelArgs,
        run_args: RunArgs,
        data_args: DataArgs,
        render_args: RenderArgs | None,
        question_item: QuestionItem,
        haystack_path: str,
    ) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.run_args = run_args
        self.render_args = render_args  # optional, only for VLMs

        self.api_connector = APIConnector(**args_to_dict(model_args))
        self.question_item = question_item

        self.haystack = BookHaystack(haystack_path)

        os.makedirs(self.results_dir, exist_ok=True)

        self.eval_name = (
            f"{model_args.model}_book_{self.question_item.question_id}_{int(time.time())}"
            if self.question_item.question_id != ""
            else f"{model_args.model}_book_{int(time.time())}"
        )

    @property
    def results_dir(self) -> str:
        return f"{self.run_args.parent_results_dir}/{get_hash(args_to_dict(self.data_args) | args_to_dict(self.model_args) | args_to_dict(self.render_args))}"

    @property
    def log_placements(self) -> bool:
        return self.run_args.log_placements_dir != ""

    def _evaluate_response(self, response: str, gold_answers: list[str] = None) -> int:
        if gold_answers is None:
            gold_answers = self.question_item.gold_answers
        assert gold_answers is not None and len(gold_answers) > 0, (
            "gold_answers is None or empty"
        )

        match self.run_args.metric:
            case "EM":
                return int(response.strip() in gold_answers)
            case "contains":
                return int(
                    any([gold_answer in response for gold_answer in gold_answers])
                )
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
                raise ValueError(f"Invalid metric: {self.run_args.metric}")

    def evaluate(self) -> None:
        outputs = {
            "model_args": args_to_dict(self.model_args),
            "data_args": args_to_dict(self.data_args),
            "run_args": args_to_dict(self.run_args),
            "render_args": args_to_dict(self.render_args),
            "question_item": self.question_item.__dict__(),
            "system_prompt": self.api_connector.default_system_prompt
            if self.data_args.use_default_system_prompt
            else self.question_item.system_prompt,
            "haystack_hash": self.haystack.get_hash(),
        }
        # snapshot -> hash
        outputs[HASH_CACHE_KEY] = get_hash(outputs)

        outputs["eval_name"] = self.eval_name
        results_for_all_depths = []

        results_path = f"{self.results_dir}/{self.eval_name}.json"
        if osp.exists(results_path):
            print(f"{results_path} exists, skipped.")
            return

        # scan all existing results
        # WARN: this could cause a lot of IO traffic
        if self.run_args.prevent_duplicate_tests:
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
                return

        async_tasks = []
        rng = np.random.RandomState(self.question_item.seed)
        for depth_percentage in np.linspace(
            self.data_args.document_depth_percent_min,
            self.data_args.document_depth_percent_max,
            self.data_args.document_depth_num_tests,
        ):
            needle_depth = depth_percentage / 100
            api_output = {}
            needle = self.question_item.needle
            retrieval_question = self.question_item.retrieval_question
            if "{CHAR}" in needle:
                assert isinstance(self.question_item.character_set, list), (
                    f"character_set {type(self.question_item.character_set)} != list"
                )

                # for reproducibility, changing with depth, but always the same sequence for a given seed
                selected_character = str(rng.choice(self.question_item.character_set))
                needle = fill_placeholders(needle, "{CHAR}", selected_character)
                self.question_item.gold_answers = [selected_character]
            else:
                selected_character = None
            api_output["selected_character"] = selected_character

            if has_placeholder(retrieval_question):
                assert selected_character is not None, (
                    "selected_character is None but retrieval_question has placeholder"
                )
                retrieval_question = fill_placeholders(
                    retrieval_question, "{CHAR}", selected_character
                )

            placement_output = self.haystack.generate_w_needle_placement(
                needle=needle,
                token_count_func=self.api_connector.token_count,
                encoding_func=self.api_connector.encode,
                decoding_func=self.api_connector.decode,
                context_length=self.data_args.context_length,
                depth=needle_depth,
                shift=self.data_args.shift,
                static_depth=self.data_args.static_depth,
                distractor=self.question_item.distractor,
            )

            task_template = (
                self.data_args.task_template or self.question_item.task_template
            )
            assert isinstance(task_template, str), (
                f"task_template is not str type{type(task_template)}"
            )
            filled_template = task_template.format(
                haystack=placement_output["text"], question=retrieval_question
            )

            async_tasks.append(
                self.api_connector.generate_response(
                    system_prompt=self.model_args.system_prompt,
                    user_prompt=filled_template,
                    max_tokens=self.model_args.max_tokens,
                    use_default_system_prompt=self.data_args.use_default_system_prompt,
                    pure_text=self.data_args.pure_text,
                    render_args=self.render_args,
                    generation_kwargs={
                        "temperature": self.model_args.temperature,
                        "top_p": self.model_args.top_p,
                        "top_k": self.model_args.top_k,
                    }
                )
            )

            api_output["placement_metadata"] = {
                k: v for k, v in placement_output.items() if k != "text"
            }

            results_for_all_depths.append(api_output)

            if self.log_placements:
                placement_log_path = os.path.join(
                    self.run_args.log_placements_dir,
                    f"{self.eval_name}_{str(np.round(depth_percentage, 3))}.txt",
                )
                with open(placement_log_path, "w") as file:
                    file.write(placement_output["text"])

        loop = asyncio.get_event_loop()
        responses: list[dict] = loop.run_until_complete(asyncio.gather(*async_tasks))

        for i in range(len(responses)):
            results_for_all_depths[i]["metric"] = (
                self._evaluate_response(
                    responses[i]["response"],
                    gold_answers=[results_for_all_depths[i]["selected_character"]],
                )
                if has_placeholder(self.question_item.needle)
                else self._evaluate_response(responses[i]["response"])
            )
            for k, v in responses[i].items():
                results_for_all_depths[i][k] = v

        outputs["results"] = results_for_all_depths
        outputs["result_path"] = results_path

        with open(results_path, "w") as file:
            json.dump(outputs, file, indent="\t")

        return
