#!/bin/python3
import json
import os.path as osp
import re
import sys
from glob import glob

import pandas as pd
from tqdm.contrib.concurrent import process_map

from locoxim.metric import calc_metrics


def remove_think_tags(text: str) -> str:
    # Remove <think>...</think> tags and their content
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def recalc_metric(response: str, gold_answers: list) -> dict:
    return calc_metrics(
        response=remove_think_tags(response),
        gold_answers=gold_answers,
    )


def wrap_as_dict(obj: dict | int, default_key: str = "EM") -> dict:
    if isinstance(obj, dict):
        return obj
    else:
        return {default_key: obj}


def read_worker(fp: str) -> list[dict]:
    if fp.endswith(".json"):
        json_data: dict = json.load(open(fp))
        results: list[dict] = json_data["results"]
        data_args: dict = json_data["data_args"]
        model_id: str = json_data["model_args"]["model"]
        render_css: str = json_data["render_args"].get("css", "")
        # redo evaluation to ensure consistency
        return [
            recalc_metric(result["response"], result["gold_answers"])
            | data_args
            | {
                "render_css": render_css,
                "collection_id": osp.dirname(fp),
                "json_id": osp.basename(fp),
                "model_id": model_id,
            }
            for result in results
        ]
    assert False


def get_args(defaults: list[str] | None = None) -> list[str]:
    if len(sys.argv) > 1:
        files = []
        for arg in sys.argv[1:]:
            if osp.isdir(arg):
                files.extend(glob(osp.join(arg, "**/*.json"), recursive=True))
            else:
                files.extend(glob(arg, recursive=True))
        return files
    elif defaults is not None:
        return defaults

    return glob("**/*.json", recursive=True)


if __name__ == "__main__":
    files = get_args()

    results: list[list[dict]] = process_map(
        read_worker,
        files,
        max_workers=16,
        chunksize=1,
        desc="Collecting",
    )

    df = pd.DataFrame([item for sublist in results for item in sublist])

    # TODO: add count of samples
    df = (
        df.groupby(["collection_id"])
        .agg(
            {
                "contains_all": "mean",
                "ROUGE-L": "mean",
                "json_id": "count",
                "context_length": "mean",
                "needle_set_path": "first",
                "model_id": "first",
                "render_css": "first",
            }
        )
        .reset_index()
    )
    df[["contains_all", "ROUGE-L"]] = (df[["contains_all", "ROUGE-L"]] * 100.0).round(2)
    df.to_json("all_results.jsonl", index=False, lines=True, orient="records")
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    ):
        print(
            df.set_index(
                ["render_css", "model_id", "needle_set_path", "context_length"]
            ).sort_index()
        )
