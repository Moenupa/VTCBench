import json
import os
from copy import deepcopy
from glob import iglob

from deocr.engine.args import RenderArgs
from jsonargparse import ArgumentParser
from numpy.random import RandomState
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from locoxim.args import DataArgs, ModelArgs, RunArgs
from locoxim.async_evaluate import evaluate
from locoxim.dataio import NeedleTestConfig, iter_question_items

__doc__ = """
A script to run VTCBench-M experiments, results are saved to disk following RunArgs settings.
A sample structure look like:

```
.
├── results/
│   ├── gpt-5/
│   │   ├── <longlong_hash>/
│   │   │   ├── gpt-5_book_<haystack>_<test_id>_<question_id>_<questiontype>_<timestamp>.json
│   │   │   ├── gpt-5_<haystack>_<test_id>_<question_id>_<questiontype>_<timestamp>.json
```

each json file contains:
{
    "model_args": {...},
    "data_args": {...},
    "run_args": {...},
    "render_args": {...},
    "question_item": {...},
    "haystack_path": "...",
    "response": "...",
    "system_prompt": "You are a helpful assistant.",
	"haystack_hash": "1f7bdc697141b888565c851e6c34dbd649aaf580280eae92199561c4f0e93dab",
	"hash_sha256": "e020d51012134539143004e6d8b99ca2ca3a6c4af948b1d49596dd983b66b6c0",
	"eval_name": "Qwen2.5-VL-7B-Instruct_book_0401_T15_C02_onehop_1763530329",
	"results": [
		{
			"placement_metadata": {
				"static_depth": null,
				"token_depth": 0,
				"depth": 0.0,
				"context_length_wo_needle": 1000
			},
			"gold_answers": [
				"Gary"
			],
			"metric": {
				"EM": 1,
				"contains": 1,
				"contains_all": 1.0,
				"lastline_EM": 1,
				"lastline_contains": 1,
				"ROUGE-L": 1.0
			},
			"response": "Gary",
			"prompt_tokens": 2119,
			"completion_tokens": 2,
			"total_tokens": 2121,
			"finish_reason": "stop",
			"cached_tokens": null,
			"api_cache_path": null
		},
		...
	],
    "result_path": "..."
}
"""


def _worker(kwargs):
    evaluate(**kwargs)


def run_test(
    model_args: ModelArgs,
    data_args: DataArgs,
    run_args: RunArgs,
    render_args: RenderArgs,
):
    experiment_config: list[NeedleTestConfig] = []
    for needle_path in data_args.needle_set_path:
        with open(needle_path, "r") as file:
            _raw_dict: list[dict] = json.load(file)
            experiment_config += [NeedleTestConfig(**e) for e in _raw_dict]
    if run_args.api_cache_dir is not None:
        os.makedirs(run_args.api_cache_dir, exist_ok=True)

    # an experiment is a json in NoLiMA, it contains multiple tests,
    #   where a test has a test_id, test templates and several question items,
    #       where a question item comes with a preset dict[str, list] inputs and gts,
    #       filling placeholders in the test templates.
    #   note: a test may have multiple test templates (difficulty), e.g. onehop, twohop
    #   final_prompt=template.format(**inputs), permuted to get MxN combinations
    #   template->question_type; inputs->NeedleTestConfig.tests
    #   so a question_id is haystack + test_id + question_type + index_for_inputs
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
    # respect max number of tasks, if valid
    if run_args.num_tasks is not None and (0 < run_args.num_tasks < len(tasks)):
        rng = RandomState(run_args.base_seed)
        tasks = rng.choice(tasks, size=run_args.num_tasks).tolist()  # type: ignore

    if run_args.num_workers <= 1:
        with tqdm(total=len(tasks)) as pbar:
            for task in tasks:
                result = evaluate(**task, verbose=(pbar.n == 0) and run_args.verbose)
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
