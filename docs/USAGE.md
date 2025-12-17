# Usage

## VTCBench

## VTCBench-Wild

## Evaluation Framework (This repo)

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

<details><summary>More on playwright...</summary>

This project depends on [DeOCR](https://pypi.org/project/deocr/), which in turn depends on [Playwright](https://pypi.org/project/playwright/) to do text-to-image using a browser.

Below is a copy of DeOCR's installation instruction. Please follow the instruction from [DeOCR](https://pypi.org/project/deocr/) whenever possible.

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
