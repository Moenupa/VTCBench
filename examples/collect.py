#!/bin/python3
import json
import os.path as osp
import sys
from glob import glob

import pandas as pd
from tqdm.contrib.concurrent import process_map


def read_worker(fp: str) -> dict:
    if fp.endswith(".json"):
        json_data: dict = json.load(open(fp))
        results: list[dict] = json_data["results"]
        data_args: dict = json_data["data_args"]
        return [
            {"collection_id": osp.dirname(fp), "json_id": osp.basename(fp)}
            | data_args
            | result
            for result in results
        ]
    assert False


def get_args(defaults: str = None) -> str:
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
        .agg({"metric": "mean", "json_id": "count", "context_length": "mean"})
        .reset_index()
    )
    df.to_json("all_results.jsonl", index=False, lines=True, orient="records")
    print(df.set_index("collection_id"))
