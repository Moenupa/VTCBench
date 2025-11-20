import json
import os
from copy import deepcopy
from glob import iglob

from deocr.engine.args import RenderArgs
from jsonargparse import ArgumentParser
from numpy.random import RandomState
from tqdm.contrib.concurrent import process_map

from locoxim.args import DataArgs, ModelArgs, RunArgs
from locoxim.dataio import NeedleTestConfig, iter_question_items
from locoxim.img_counter import iter_context_and_images


def _worker(kwargs):
    return [
        (context_len, len(images))
        for _, context_len, images in iter_context_and_images(**kwargs)
    ]


def run_smoke_test(
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
    if run_args.parent_api_cache_dir is not None:
        os.makedirs(run_args.parent_api_cache_dir, exist_ok=True)

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
                    "render_args": deepcopy(render_args),
                    # below independent stuff
                    "question_item": question_item,
                    "haystack_path": haystack_path,
                }
            )
    # respect max number of tasks, if valid
    if run_args.num_tasks is not None and (0 < run_args.num_tasks < len(tasks)):
        rng = RandomState(run_args.base_seed)
        tasks = rng.choice(tasks, size=run_args.num_tasks).tolist()  # type: ignore

    img_counts: list[list[tuple[int, int]]] = process_map(
        _worker,
        tasks,
        max_workers=run_args.num_workers,
        chunksize=1,
    )

    # flatten
    flat_img_counts = [item for sublist in img_counts for item in sublist]
    if run_args.verbose:
        print(flat_img_counts)

    # compute avg (context length/n_images) ratio
    total_context_length = sum([cl for cl, _ in flat_img_counts])
    total_n_images = sum([nimg for _, nimg in flat_img_counts])
    avg_compratio = total_context_length / total_n_images if total_n_images > 0 else 0.0
    print(f"Average context length / n_images ratio: {avg_compratio:.2f}")
    return flat_img_counts


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

    run_smoke_test(
        model_args=model_args,
        data_args=data_args,
        run_args=run_args,
        render_args=render_args,
    )
