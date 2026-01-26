import json
import os
from glob import iglob

from datasets import Dataset
from deocr.engine.args import RenderArgs
from jsonargparse import ArgumentParser
from numpy.random import RandomState
from tqdm.contrib.concurrent import process_map

from vtcbench.args import DataArgs, ModelArgs, RunArgs
from vtcbench.client.image_helper import image_object_to_bytes
from vtcbench.dataio import NeedleTestConfig, args_to_dict, iter_question_items
from vtcbench.dry_evaluate import iter_context_and_images

__doc__ = """
This script is the dry-run version of run.py, here we are interested in the data.
It renders the images and output them as PILImage objects, counts them and computes
the average context length / n_images ratio.

This is still not the vision-text compression ratio, as there is token_per_image factor missing,
which you should calculate manually for the VLM you are using.

You can also modify the counting function with some saving or api input smoke test to further examine the data.
"""


def _worker(kwargs) -> list[dict]:
    samples: list[dict] = []
    for result in iter_context_and_images(**kwargs):
        result = {
            "problem": result["problem"],
            "images": [
                {
                    "bytes": image_object_to_bytes(
                        img,
                        kwargs["render_args"].save_format,
                        kwargs["render_args"].save_kwargs,
                    )
                }
                for img in result["images"]
            ],
            "answers": result["answers"],
            "_context": result["_context"],
            "_render_args": json.dumps(args_to_dict(kwargs["render_args"])),
            "_source": json.dumps(args_to_dict(result["_source"])),
        }
        samples.append(result)
    return samples


def run_smoke_test(
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

    questions = [
        question
        for test_config in experiment_config
        for question in iter_question_items(
            test_config,
            base_seed=run_args.base_seed,
        )
    ]

    tasks: list[dict] = []
    alternating_render_args: list[RenderArgs] = [
        RenderArgs(
            pagesize=render_args.pagesize,
            marginLeft=render_args.marginLeft,
            marginRight=render_args.marginRight,
            marginTop=render_args.marginTop,
            marginBottom=render_args.marginBottom,
            forceOnePage=render_args.forceOnePage,
            autoAdjustHeight=render_args.autoAdjustHeight,
            savePDF=False,
            saveImage=False,
            dpi=render_args.dpi,
            overwrite=False,
            save_format=render_args.save_format,
            save_kwargs=render_args.save_kwargs,
            css=f"*{{font-size:{FONTSIZE:d}px !important;font-family:'{FONTFAMILY}' !important;line-height:{LINEHEIGHT:.1f} !important;}}",
            css_path=render_args.css_path,
        )
        for FONTFAMILY in ["Helvetica", "Times New Roman", "Courier New"]
        # we have established in our ablation
        # that marginal benefit point is around 18px font size
        for FONTSIZE in [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        # lineheight=2 is bad for compression and not commonly used in real world
        for LINEHEIGHT in [1.0, 1.2, 1.5]
    ]
    for haystack_path in iglob(f"{data_args.haystack_dir}/*"):
        for question_item in questions:
            tasks.append(
                {
                    "model_args": model_args,
                    "data_args": data_args,
                    # below independent stuff
                    "question_item": question_item,
                    "haystack_path": haystack_path,
                }
            )
    tasks = [
        t
        | {
            "render_args": render_args,
        }
        for t in tasks
        for render_args in alternating_render_args
    ]

    # respect max number of tasks, if valid
    if run_args.num_tasks is not None and (0 < run_args.num_tasks < len(tasks)):
        rng = RandomState(run_args.base_seed)
        tasks = rng.choice(tasks, size=run_args.num_tasks).tolist()  # type: ignore

    assert run_args.num_workers > 1
    samples_2d = process_map(
        _worker,
        tasks,
        max_workers=run_args.num_workers,
        chunksize=1,
    )
    # flatten to 1d
    Dataset.from_list([e for _l in samples_2d for e in _l]).to_parquet("output.parquet")


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
