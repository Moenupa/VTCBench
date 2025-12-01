#!/bin/python3
import json
import os.path as osp
import sys
from glob import glob

import pandas as pd
from tqdm.contrib.concurrent import process_map

from locoxim.metric import calc_metrics

__doc__ = """
Collect results from multiple json files and summarize the results by metadata,
such as VLM model name, data info, render args, shown as a table.
"""


def read_worker(fp: str) -> list[dict]:
    if fp.endswith(".json"):
        json_data: dict = json.load(open(fp))
        results: list[dict] = json_data["results"]
        data_args: dict = json_data["data_args"]
        model_id: str = json_data["model_args"]["model"]
        render_css: str = json_data["render_args"].get("css", "")
        # redo evaluation to ensure consistency
        return [
            calc_metrics(result["response"], result["gold_answers"])
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
        df.groupby(
            [
                "needle_set_path",
                "collection_id",
            ]
        )
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
    METRIC_COLUMNS = list({"contains", "contains_all", "ROUGE-L"} & set(df.columns))
    df[METRIC_COLUMNS] = (df[METRIC_COLUMNS] * 100.0).round(2)
    df.to_json("all_results.jsonl", index=False, lines=True, orient="records")
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    ):
        print(
            df.set_index(
                [
                    "render_css",
                    "model_id",
                    "needle_set_path",
                    "context_length",
                ]
            ).sort_index()
        )
