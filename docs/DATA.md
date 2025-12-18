---
layout: page
title: Data
permalink: /data/
---

# Data

## VTCBench-Wild

[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-md.svg)][hfvtcbench]

Evaluation on a permutation of `dataset × task × formatting` is a huge task.

We hope to simpify it with **VTCBench-Wild**, a static yet wild version of VTCBench, 
sampling data from diverse datasets, tasks and common document formats.  2.2k dataset

```sh
hf download --repo-type dataset MLLM-CL/VTCBench --local-dir data/VTCBench
```

[hfvtcbench]: https://huggingface.co/datasets/MLLM-CL/VTCBench

## Dataset Overview

|   VTCBench    |       Dataset       |    Metric     |      Needle      |   Haystack    | Evaluated by  |            License             |
| :-----------: | :-----------------: | :-----------: | :--------------: | :-----------: | :-----------: | :----------------------------: |
| VTC-Retrieval |  [RULER][gitruler]  | `containsAll` | word/uuid/number |     essay     | Completion/QA |   [Apache-2.0][gitrulerLCS]    |
| VTC-Reasoning | [NoLiMa][gitnolima] | `containsAll` | character/event  |     book      |      QA       | [Adobe Research][gitnolimaLCS] |
|  VTC-Memory   | [LoCoMo][gitlocomo] |   `ROUGE-L`   |       _NA_       | conversations |      QA       |  [CC BY-NC 4.0][gitlocomoLCS]  |

Metrics:

- `containsAny` checks if the prediction is a substring of the (or any one of) ground truth, e.g.:
  - $1.0$ with `pred="magic number is 6822442"`, `gt=["6822442"]`
  - $1.0$ with `pred="magic number is 6822442"`, `gt=["1234567", "6822442"]`
  - $0.0$ with `pred="magic number is 1234567"`, `gt=["6822442"]`
- `containsAll` checks if the prediction contains all of the ground truths, e.g.:
  - $1.0$ with `pred="magic number is 6822442"`, `gt=["6822442"]`
  - $0.5$ with `pred="magic number is 6822442"`, `gt=["1234567", "6822442"]`
  - $0.0$ with `pred="magic number is 1234567"`, `gt=["6822442"]`
- `ROUGE-L` is computed using [`rouge-score`](https://pypi.org/project/rouge-score).
- For details, refer to the implementation: [metrics.py](../src/vtcbench/metric.py).

## VTC-Retrieval (RULER)

Download from our own fork of [RULER Huggingface](https://huggingface.co/datasets/MLLM-CL/RULER).

We converted 4 tasks from RULER paper, namely **\[S,MK,MV,MQ\]-NIAH**.
Each task contains 30 samples, 10 for each needle k-v type: (word-number, uuid-number, word-word).

```sh
hf download --repo-type dataset MLLM-CL/RULER --local-dir data/RULER
```

<details><summary>A sample data point for S-NIAH (word-number)</summary>

- Needle: `One of the special magic numbers for yielding-grain is: 6822442.`
- Question Template: `{haystack_w_needle} What is the special magic number for yielding-grain mentioned in the provided text?`

</details>

## VTC-Reasoning (NoLiMa)

Download via [NoLiMa Huggingface][hfnolima]:

```sh
hf download --repo-type dataset amodaresi/NoLiMa --local-dir data/NoLiMa
```

<details><summary>Optionally, if you have a custom path</summary>

Modify config file path accordingly: [config/data/nolima.json](../config/data/nolima.json).

```json
{
  "needle_set_path": ["data/NoLiMa/needlesets/needle_set.json"],
  "haystack_dir": "data/NoLiMa/haystack/rand_shuffle",
  "...": "..."
}
```

</details>

## VTC-Memory (LoCoMo)

Download from [LoCoMo Github][gitlocomo]

```sh
mkdir -p data/LoCoMo
wget -P data/LoCoMo https://raw.githubusercontent.com/snap-research/locomo/refs/heads/main/data/locomo10.json
python examples/convert.py data/LoCoMo/locomo10.json
```

> [!NOTE]
> Be aware that LoCoMo does not have haystacks,
> and context is directly provided in the needle's `context` field.  
> Random haystack will result in non-relevant context for needles/QAs, simply put:  
> ❌$M$ haystacks \* $N$ needles/QAs;  
> ✔️$N$ needles/QAs with their own context.

<details><summary>Optionally, if you have a custom path</summary>

`examples/convert.py` will output folders parallel to the input file.  
Modify config file path accordingly: [config/data/locomo.json](../config/data/locomo.json).

```json
{
  "needle_set_path": ["data/LoCoMo/needlesets/4_SingleHop.json"],
  "haystack_dir": "data/LoCoMo/haystack",
  "...": "..."
}
```

</details>

[gitruler]: https://github.com/NVIDIA/RULER
[gitrulerLCS]: https://github.com/NVIDIA/RULER/blob/main/LICENSE
[gitnolima]: https://github.com/Adobe-Research/NoLiMa
[gitnolimaLCS]: https://github.com/Adobe-Research/NoLiMa/blob/main/LICENSE
[hfnolima]: https://huggingface.co/datasets/amodaresi/NoLiMa
[gitlocomo]: https://github.com/snap-research/locomo
[gitlocomoLCS]: https://github.com/snap-research/locomo/blob/main/LICENSE.txt

## Rendered Samples

|  VTC-Retrieval   |   VTC-Reasoning   |    VTC-Memory     |
| :--------------: | :---------------: | :---------------: |
| ![][sampleruler] | ![][samplenolima] | ![][samplelocomo] |

[sampleruler]: ../assets/data_samples/ruler_sample.jpeg
[samplenolima]: ../assets/data_samples/nolima_sample.jpeg
[samplelocomo]: ../assets/data_samples/locomo_sample.jpeg
