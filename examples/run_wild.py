import os

from datasets import load_dataset
from jsonargparse import ArgumentParser
from numpy.random import RandomState
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from vtcbench.args import ModelArgs, RunArgs
from vtcbench.async_evaluate import evaluate_static

__doc__ = """
"""


def _select_instruction_prompt(split: str) -> str:
    match split:
        case "Retrieval":
            return (
                "Answer a question based on the above book snippet. Your answer"
                " should be short and based on either explicitly stated facts"
                " or strong, logical inferences. Return only the final answer"
                " with no additional explanation or reasoning. Question: "
            )
        case "Reasoning":
            return (
                "Answer a question based on the above book snippet. Some special"
                " magic numbers are hidden within the following text. Make sure"
                " to memorize it. I will quiz you about the numbers afterwards."
                " Question: "
            )
        case "Memory":
            return (
                "Based on the above context, write an answer in the form of a"
                " short phrase for the following question. Answer with exact"
                " words from the context whenever possible. Question: "
            )
        case _:
            raise ValueError(f"Unknown split: {split}")


def _worker(kwargs):
    evaluate_static(**kwargs)


def run_wild(
    model_args: ModelArgs,
    run_args: RunArgs,
    data_path: str,
    split: str,
    pure_text: bool,
):
    if run_args.api_cache_dir is not None:
        os.makedirs(run_args.api_cache_dir, exist_ok=True)

    dataset = load_dataset(
        data_path, split=split, columns=["problem", "answers", "images", "_context"]
    )

    tasks: list[dict] = []
    for example in dataset:
        assert isinstance(example, dict)
        tasks.append(
            {
                "model_args": model_args,
                "run_args": run_args,
                # below independent stuff
                "example": example | {"instruction": _select_instruction_prompt(split)},
                "pure_text": pure_text,
            }
        )
    # respect max number of tasks, if valid
    if run_args.num_tasks is not None and (0 < run_args.num_tasks < len(tasks)):
        rng = RandomState(run_args.base_seed)
        tasks = rng.choice(tasks, size=run_args.num_tasks).tolist()  # type: ignore

    if run_args.num_workers <= 1:
        with tqdm(total=len(tasks)) as pbar:
            for task in tasks:
                result = evaluate_static(
                    **task, verbose=(pbar.n == 0) and run_args.verbose
                )
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
    parser.add_class_arguments(RunArgs, "run")
    parser.add_argument(
        "--data.path",
        type=str,
        default="MLLM-CL/VTCBench",
        help="Local/HuggingFace dir path to the dataset",
    )
    parser.add_argument(
        "--data.split",
        type=str,
        default="Retrieval",
        choices=["Retrieval", "Reasoning", "Memory"],
        help="Data split to use",
    )
    parser.add_argument(
        "--data.pure_text",
        type=bool,
        default=False,
        help="Whether to use pure text data",
    )

    args = parser.parse_args()

    model_args: ModelArgs = args.model
    run_args: RunArgs = args.run
    data_path: str = args.data.path
    split: str = args.data.split
    pure_text: bool = args.data.pure_text

    run_wild(
        model_args=model_args,
        run_args=run_args,
        data_path=data_path,
        split=split,
        pure_text=pure_text,
    )
