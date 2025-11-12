# Data Preparation

## Dataset Info

| Dataset | metrics | Needle | Haystack  | License |
|:-:|:-:|:-:|:-:|:-:|
| [RULER][gitruler]    | `contains` | numbers, uuid, QAs| noise, essay | [Apache-2.0][gitrulerLCS] |
| [NoLiMa][gitnolima]  | `EM`, `contains`,<br/> `lastline_EM`, `lastline_contains` | QAs | book | [Adobe Research][gitnolimaLCS] |
| [LoCoMo][gitlocomo]  | `ROUGE` | - | conversation | [CC BY-NC 4.0][gitlocomoLCS] |

Metrics:

- `contains` checks if the prediction is a substring of the (or any one of) ground truth.
- `rouge` is computed using the [`rouge-score`](https://pypi.org/project/rouge-score) package.

## RULER

```sh
# we prepared a simple test set for **Single-NIAH** task defined in RULER paper
hf download --repo-type dataset MLLM-CL/RULER --local-dir data/RULER
```

<details><summary>A sample data point</summary>

- Needle: `One of the special magic numbers for yielding-grain is: 6822442.`
- Question Template: `{haystack_w_needle} What is the special magic number for yielding-grain mentioned in the provided text?`

</details>

## NoLiMa

Download via [NoLiMa Huggingface][hfnolima]:

```sh
hf download --repo-type dataset amodaresi/NoLiMa --local-dir data/NoLiMa
```

<details><summary>Optionally, if you have a custom path</summary>

Modify config file path accordingly: [config/data/nolima.json](../config/data/nolima.json).

```json
{
  "needle_set_path": "data/NoLiMa/needlesets/needle_set.json",
  "haystack_dir": "data/NoLiMa/haystack/rand_shuffle",
  "...": "..."
}
```

</details>

## LoCoMo

Download from [LoCoMo Github][gitlocomo]

```sh
mkdir -p data/LoCoMo
wget -P data/LoCoMo https://github.com/snap-research/locomo/blob/main/data/locomo10.json
python examples/convert.py data/LoCoMo/locomo10.json
```

<details><summary>Optionally, if you have a custom path</summary>

Modify config file path accordingly: [config/data/locomo.json](../config/data/locomo.json).

```json
{
  "needle_set_path": "data/LoCoMo/needlesets/4_SingleHop.json",
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
