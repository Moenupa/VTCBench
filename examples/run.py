import json
from copy import deepcopy
from glob import iglob

from deocr.engine.playwright.async_api import RenderArgs
from jsonargparse import ArgumentParser

# from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

from locoxim.args import DataArgs, ModelArgs, RunArgs
from locoxim.async_evaluate import NeedleHaystackTester
from locoxim.dataio import NeedleTestConfig


def _worker(kwargs):
    tester = NeedleHaystackTester(**kwargs)
    tester.evaluate()


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

    # an experiment is a json, containing multiple tests, with a test_id and its args
    questions = [
        question
        for test_config in experiment_config
        for question in test_config.iter_question_items(
            default_system_prompt=model_args.system_prompt,
            default_task_template=data_args.task_template,
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
        # increment base seed for every different haystack
        run_args.base_seed = run_args.base_seed + 100

    with tqdm(total=len(tasks)) as pbar:
        for task in tasks:
            tester = NeedleHaystackTester(**task, verbose=(pbar.n == 0))
            result = tester.evaluate()
            pbar.set_postfix(result=result)
            pbar.update()


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
