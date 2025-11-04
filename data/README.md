# Data Preparation

## Dataset Info

| Dataset | metrics | Needle | Haystack  | License |
|:-:|:-:|:-:|:-:|:-:|
| [RULER][gitruler]    | `contains` | numbers, uuid, QAs| noise, essay | [Apache-2.0][gitrulerLCS] |
| [NoLiMa][gitnolima]  | `EM`, `contains`,<br/> `lastline_EM`, `lastline_contains` | | | [Adobe Research][gitnolimaLCS] |

`metric.contains` checks if the prediction is a substring of the ground truth.

## RULER

```sh
git clone https://github.com/NVIDIA/RULER
cd RULER/scripts/data

# use scripts/data/prepare.py to prepare data
# deps are minimal, but tokenizer lib is required if passed --prepare_for_ns
# which triggers the use of tokenizer to count tokens.
python prepare.py \
    --save_dir /path/to/folder --task <task_name> \
    --max_seq_length 8192 \
    --tokenizer_path /path/to/tokenizer --tokenizer_type hf

# use python prepare.py --help to see all options
```

<details><summary>A sample data point</summary>

```json
{
    "index": 2169,
    "input": "long, long, string",
    "outputs": ["5437923"], 
    "length": 8177, 
    "length_w_model_temp": 8177, 
    "answer_prefix": " The special magic number for jittery-hospital mentioned in the provided text is", 
    "token_position_answer": 569
}
```

</details>

## NoLiMa

Download via [Hugging Face][hfnolima]:

```sh
hf download --repo-type dataset amodaresi/NoLiMa --local-dir /path/to/local/folder
```

TODO.

[gitruler]: https://github.com/NVIDIA/RULER
[gitrulerLCS]: https://github.com/NVIDIA/RULER/blob/main/LICENSE
[gitnolima]: https://github.com/Adobe-Research/NoLiMa
[gitnolimaLCS]: https://github.com/Adobe-Research/NoLiMa/blob/main/LICENSE
[hfnolima]: https://huggingface.co/datasets/amodaresi/NoLiMa
