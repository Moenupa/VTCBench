#!/bin/python3
import json
import os.path as osp
import sys
from glob import glob

import pandas as pd
from tqdm.contrib.concurrent import process_map

from vtcbench.metric import calc_metrics

__doc__ = """
Collect results from multiple json files and summarize the results by metadata,
such as VLM model name, data info, render args, shown as a table.
"""

GROUP_BY_COLS: list[str] = [
    "needle_set_path",
    "model_id",
    "collection_id",
]
AGG_MAPPING: dict[str, str] = {
    "contains_all": "mean",
    "ROUGE-L": "mean",
    "json_id": "count",
    "context_length": "mean",
    "render_css": "first",
}
METRIC_COLUMNS: list[str] = [
    "contains",
    "contains_all",
    "ROUGE-L",
]


def _safe_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [k for k in columns if k in df.columns]


def read_worker(fp: str) -> list[dict]:
    if fp.endswith(".json"):
        json_data: dict = json.load(open(fp))
        results: list[dict] = json_data["results"]
        model_id: str = json_data["model_args"]["model"]

        # optional args, may be missing for static dataset
        data_args: dict = json_data.get("data_args", {})
        question_id: str = json_data.get("question_item", {}).get("question_id", "")
        render_css: str = json_data.get("render_args", {}).get("css", "")
        # redo evaluation to ensure consistency
        return [
            calc_metrics(result["response"], result["gold_answers"])
            | data_args
            | {
                "question_id": question_id[:8],
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
    # avoid hashing issue
    df[_safe_columns(df, GROUP_BY_COLS)] = df[_safe_columns(df, GROUP_BY_COLS)].astype(
        str
    )

    # aggregate results
    df = (
        df.groupby(_safe_columns(df, GROUP_BY_COLS))
        .agg({k: v for k, v in AGG_MAPPING.items() if k in df.columns})
        .reset_index()
    )
    metric_cols = _safe_columns(df, METRIC_COLUMNS)
    df[metric_cols] = (df[metric_cols] * 100.0).round(2)

    # output as file & stdout
    df.to_json("all_results.jsonl", index=False, lines=True, orient="records")
    print_idx_cols = [
        "render_css",
        "model_id",
        "needle_set_path",
        "context_length",
    ]
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    ):
        print(df.set_index(_safe_columns(df, print_idx_cols)).sort_index())
