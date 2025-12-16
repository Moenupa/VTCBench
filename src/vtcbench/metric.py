import re
from typing import Literal

from rouge_score import rouge_scorer


def remove_think_tags(text: str) -> str:
    # Remove <think>...</think> tags and their content
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def calc_metrics(
    response: str,
    gold_answers: list[str],
    metric: list[
        Literal[
            "EM",
            "contains",
            "contains_all",
            "lastline_EM",
            "lastline_contains",
            "ROUGE-L",
        ]
    ]
    | None = None,
) -> dict[str, float | int]:
    assert gold_answers is not None and len(gold_answers) > 0, (
        "gold_answers is None or empty"
    )
    # make sure gold answers are stripped strings, not int/float/etc.,
    # otherwise the 'contains' metric may fail
    gold_answers = [str(ans).strip().lower() for ans in gold_answers]
    response = remove_think_tags(response).strip().lower()
    if metric is None:
        metric = [
            "EM",
            "contains",
            "contains_all",
            "lastline_EM",
            "lastline_contains",
            "ROUGE-L",
        ]

    scores = {}
    for each_metric in metric:
        match each_metric:
            case "EM":
                scores[each_metric] = int(response in gold_answers)
            case "contains":
                scores[each_metric] = int(
                    any([f"{gold_answer}" in response for gold_answer in gold_answers])
                )
            case "contains_all":
                # all gold answers should be contained in the response
                # if so metric==1, can be fractional
                scores[each_metric] = float(
                    sum([f"{gold_answer}" in response for gold_answer in gold_answers])
                    / len(gold_answers)
                )
            case "lastline_EM":
                scores[each_metric] = int(response.split("\n")[-1] in gold_answers)
            case "lastline_contains":
                scores[each_metric] = int(
                    any(
                        [
                            gold_answer in response.split("\n")[-1]
                            for gold_answer in gold_answers
                        ]
                    )
                )
            case "ROUGE-L":
                scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                if len(gold_answers) == 0:
                    scores[each_metric] = 0.0
                else:
                    scores[each_metric] = max(
                        s["rougeL"].fmeasure
                        for s in [scorer.score(response, ref) for ref in gold_answers]
                    )
            case _:
                raise ValueError(f"Invalid metric: {each_metric}")
    return scores
