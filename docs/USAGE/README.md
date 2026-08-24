---
layout: page
title: Evaluation
permalink: /eval/
redirect_from: 
  - /usage/
  - /docs/USAGE
---

# Evaluation Guide

We natively support **VTCBench-Wild** evaluation on
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
and [evalscope](https://github.com/modelscope/evalscope).

This page documents how to evaluate **VTCBench** and **VTCBench-Wild** with our 
framework. We decouple LLM generation and evaluation, and therefore require an
OpenAI-compatible LLM/VLM server (e.g., vLLM Serving[^1], OpenAI API[^2], etc.) 
to provide LLM/VLM inference services.

## Evaluation Framework (Client)

The git repo provides the evaluation framework, i.e. the client side. 
To set up the evaluation framework, you can use `uv` (recommended) or `pip`:

```sh
git clone https://github.com/Moenupa/VTCBench.git && cd VTCBench
uv sync
uv run playwright install chromium
```

```sh
# or using pip:
pip install -e .
playwright install chromium
```

<details><summary>Troubleshooting Playwright Installation</summary>

This project depends on [DeOCR](https://pypi.org/project/deocr/), which in turn
depends on [Playwright](https://pypi.org/project/playwright/) for text-to-image rendering in a browser.

```sh
pip install deocr[playwright,pymupdf]
playwright install chromium
```

If you have trouble installing Playwright, or have host-switching problems (e.g., Slurm), we suggest a hacky fix like this:

```sh
# put libasound.so.2 file (a fake one is also fine) in $HOME/.local/lib
# and then export lib path for playwright to find it:
export LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.local/lib
```

</details>

We provide ready-to-use **shell/slurm scripts** for parallel evaluation in 
the [`slurm/`](../slurm) folder that are equivalent to the following.

### VTCBench

```sh
uv run examples/run.py \
  --model config/model/qwen_2.5_vl_7b.json \
  --data config/data/nolima.json \
  --data.context_length 1000 \
  --render config/render/default.yml \
  # --run.num_tasks 1 # for smoke test
```

### VTCBench-Wild

VTCBench-Wild uses pre-rendered static images, so no rendering or context length parameters are needed.

```sh
uv run examples/run_wild.py \
  --model config/model/qwen_2.5_vl_7b.json \
  --data.path MLLM/VTCBench \
  --data.split Retrieval \
  # --run.num_tasks 1 # for smoke test
```

**Key arguments:**

| Argument       | Description                                          |
| -------------- | ---------------------------------------------------- |
| `--model`      | Path to model config JSON                            |
| `--data.path`  | Hugging Face dataset path                            |
| `--data.split` | Dataset split: `Retrieval`, `Reasoning`, or `Memory` |

### Collecting Results

After running evaluations, collect and summarize results:

```sh
uv run examples/collect.py results
# or point to a specific directory:
uv run examples/collect.py /path/to/results/
```

This aggregates results by model and render settings, producing a table like:

```
                     contains_all  ROUGE-L  json_id
render_css model_id                                
           Qwen3-8B         99.38    74.35      800
```

**Environment variables for advanced usage:**

| Variable          | Effect                                                         |
| ----------------- | -------------------------------------------------------------- |
| `WILD=1`          | Collect VTCBench-Wild results (skips `collection_id` grouping) |
| `PLOT_TRADEOFF=1` | Generate font-size vs. performance tradeoff plots              |
| `PLOT_HEATMAP=1`  | Generate depth-vs-context-length heatmaps                      |
| `PLOT_OUTDIR=fig` | Output directory for plots (default: `fig/`)                   |

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
