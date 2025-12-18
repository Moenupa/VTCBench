---
layout: page
title: Evaluation
permalink: /eval/
---

# Evaluation Guide

This project adopts a server-client architecture. 
We require a running OpenAI-compatible LLM/VLM server 
(e.g., vLLM Serving[^1], OpenAI API[^2], etc.) 
to provide LLM/VLM inference services.

## Evaluation Framework (Client)

This repo provides the evaluation framework, i.e. the client side of the project. 
To set up the evaluation framework, you can use `uv` (recommended) or `pip`:

```sh
uv venv
uv sync
uv run playwright install chromium
```

```sh
# or using pip:
pip install -e .
playwright install chromium
```

<details><summary>More on playwright Installation...</summary>

This project depends on [DeOCR](https://pypi.org/project/deocr/), which in turn
depends on [Playwright](https://pypi.org/project/playwright/) to do text-to-image using a browser.

Below is a copy of DeOCR's installation instruction. Please follow instructions 
from [DeOCR](https://pypi.org/project/deocr/) whenever possible.

```sh
pip install deocr[playwright,pymupdf]
# activate your python environment, then install playwright deps
playwright install chromium
```

If you have trouble installing playwright, or have host-switching problems (e.g., slurm), we suggest a hacky fix like this:

```sh
# put libasound.so.2 file (a fake one is also fine) in $HOME/.local/lib
# and then export lib path for playwright to find it:
export LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib
```

</details>

We provide ready-to-use **shell/slurm scripts** for parallel evaluation in 
the [`slurm/`](../slurm) folder that are equivalent to the following.

**VTCBench** evaluation:

```sh
uv run examples/run.py \
  --model config/model/qwen_2.5_vl_7b.json \
  --data config/data/nolima.json \
  --data.context_length 1000 \
  --render config/render/default.yml \
  # --run.num_tasks 1 # for smoke test
```

**VTCBench-Wild** evaluation:

```sh
# no rendering and context length params, because they come from -wild dataset
uv run examples/run_wild.py \
  --model config/model/qwen_2.5_vl_7b.json \
  --data.path MLLM/VTCBench \
  --data.split Retrieval \
  # --run.num_tasks 1 # for smoke test
```

Collect results by running `uv run examples/collect.py results`, or 
`uv run examples/collect.py /path/to/results/`.
This will print a table like below:

```
                     contains_all  ROUGE-L  json_id
render_css model_id                                
           Qwen3-8B         99.38    74.35      800
```

## vLLM Serving

To setup a [vLLM](https://github.com/vllm-project/vllm) serving endpoint, 
please refer to the vLLM Serving Documentation[^1].

A simple example to get you started, using deps from [pyproject.toml](./pyproject.toml):

```sh
# mkdir ../vllm-0.11
# set up a vllm environment seperately, parallel to this repo.
uv venv
uv add vllm==0.11.0 # optionally flash-attn https://github.com/Dao-AILab/flash-attention
# serve your model
vllm serve Qwen/Qwen3-VL-2B-Instruct --port 8001
# to test your endpoint
curl http://localhost:8001/v1/models
```

### Known Dependency Constraints

Following are our dependency recommendations for known models to avoid potential issues.
Upgrade or downgrade with caution.

|                                        Model Name                                         |               Dependency               |
| :---------------------------------------------------------------------------------------: | :------------------------------------: |
|            [Qwen3-VL Series](https://huggingface.co/collections/Qwen/qwen3-vl)            |  `vllm==0.11.0, transformers==4.57.1`  |
| [moonshotai/Kimi-VL-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct) |    `vllm==0.9.2, transformers<4.54`    |
|       [InternVL3.5 Series](https://huggingface.co/collections/OpenGVLab/internvl35)       | `vllm==0.10.1.1, transformers==4.57.1` |


[^1]: https://docs.vllm.ai/en/stable/cli/serve/
[^2]: https://platform.openai.com/docs/api-reference/introduction
