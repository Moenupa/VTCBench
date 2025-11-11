import json
import os
from copy import deepcopy
from glob import iglob

from deocr.engine.playwright.async_api import RenderArgs
from jsonargparse import ArgumentParser
from numpy.random import RandomState
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from locoxim.args import DataArgs, ModelArgs, RunArgs
from locoxim.async_evaluate import evaluate
from locoxim.dataio import API_CACHE_DIR, NeedleTestConfig, iter_question_items


def _worker(kwargs):
    evaluate(**kwargs)


def run_test(
    model_args: ModelArgs,
    data_args: DataArgs,
    run_args: RunArgs,
    render_args: RenderArgs,
):
    with open(data_args.needle_set_path, "r") as file:
        _raw_dict: list[dict] = json.load(file)
        experiment_config: list[NeedleTestConfig] = [
            NeedleTestConfig(**e) for e in _raw_dict
        ]
    os.makedirs(API_CACHE_DIR, exist_ok=True)

    # an experiment is a json, containing multiple tests, with a test_id and its args
    questions = [
        question
        for test_config in experiment_config
        for question in iter_question_items(
            test_config,
            base_seed=run_args.base_seed,
        )
    ]

    tasks: list[dict] = []
    for haystack_path in iglob(f"{data_args.haystack_dir}/*"):
        for question_item in questions:
            tasks.append(
                {
                    "model_args": deepcopy(model_args),
                    "data_args": deepcopy(data_args),
                    "run_args": deepcopy(run_args),
                    "render_args": deepcopy(render_args),
                    # below independent stuff
                    "question_item": question_item,
                    "haystack_path": haystack_path,
                }
            )
    if run_args.num_tasks is not None and run_args.num_tasks < len(tasks):
        rng = RandomState(run_args.base_seed)
        tasks = rng.choice(tasks, size=run_args.num_tasks).tolist()

    if run_args.num_workers <= 1:
        with tqdm(total=len(tasks)) as pbar:
            for task in tasks:
                result = evaluate(**task, verbose=(pbar.n == 0))
                pbar.set_postfix(result=result)
                pbar.update()
    else:
        process_map(
            _worker,
            tasks,
            max_workers=run_args.num_workers,
            chunksize=1,
        )


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_class_arguments(ModelArgs, "model")
    parser.add_class_arguments(DataArgs, "data")
    parser.add_class_arguments(RunArgs, "run")
    parser.add_class_arguments(RenderArgs, "render")

    args = parser.parse_args()

    model_args: ModelArgs = args.model
    data_args: DataArgs = args.data
    run_args: RunArgs = args.run
    render_args: RenderArgs = args.render

    run_test(
        model_args=model_args,
        data_args=data_args,
        run_args=run_args,
        render_args=render_args,
    )
