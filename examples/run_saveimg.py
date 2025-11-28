import asyncio
import json
import os
import os.path as osp
import sys

from deocr.engine.args import RenderArgs
from deocr.engine.playwright.async_api import transform
from tqdm.contrib.concurrent import process_map


def trim_img_path(img_path: str) -> str:
    out = img_path.split("images/", 1)[1]
    return f"images/{out}"


def add_image(input_jsonl: str):
    assert "-S/" in input_jsonl, (
        "Input jsonl path must contain '-S/' to indicate source set."
    )

    output_jsonl = input_jsonl.replace("-S/", "-S-img/")
    os.makedirs(osp.dirname(output_jsonl), exist_ok=True)

    with open(input_jsonl, "r", encoding="utf-8") as f:
        lines = f.readlines()
        objs = [json.loads(line) for line in lines]

    async_tasks = []
    for obj in objs:
        async_tasks.append(
            transform(
                obj["_context"],
                f"{osp.dirname(output_jsonl)}/images",
                RenderArgs(**(obj["_render_args"] | {"saveImage": True})),
            )
        )

    loop = asyncio.get_event_loop()
    image_paths: list[list[str]] = loop.run_until_complete(asyncio.gather(*async_tasks))

    for obj, line_paths in zip(objs, image_paths):
        obj["images"] = [trim_img_path(p) for p in line_paths]
        with open(output_jsonl, "a", encoding="utf-8") as f_out:
            f_out.write(json.dumps(obj) + "\n")


if __name__ == "__main__":
    process_map(
        add_image,
        sys.argv[1:],
        max_workers=os.cpu_count() or 1,
        chunksize=1,
    )
