import json
import os
import os.path as osp
import sys
from typing import TypedDict

from locoxim.dataio import NeedleTestConfig, dataclass_to_dict


class QA(TypedDict):
    question: str
    answer: str
    evidence: list[str]
    category: int


class LoCoMoFormat(TypedDict):
    qa: list[QA]
    conversation: dict[str, str | list[str]]
    """
    "conversation": {
      "speaker_a": "Caroline",
      "speaker_b": "Melanie",
      "session_1_date_time": "1:56 pm on 8 May, 2023",
      "session_1": [
        {
          "speaker": "Caroline",
          "dia_id": "D1:1",
          "text": "Hey Mel! Good to see you! How have you been?"
        },
        {
          "speaker": "Melanie",
          "dia_id": "D1:2",
          "text": "Hey Caroline! Good to see you! I'm swamped with the kids & work. What's up with you? Anything new?"
        },
        {
          "speaker": "Caroline",
          "dia_id": "D1:3",
          "text": "I went to a LGBTQ support group yesterday and it was so powerful."
        },
    """
    sample_id: str
    observation: dict
    session_summary: str
    event_summary: str


LOCOMO_CATEGORIES = {
    4: "SingleHop",
    1: "MultiHop",
    2: "Temporal",
    3: "OpenDomain",
    # 5: "Adversarial",
}


def prepare_configs(
    locomo_data: list[LoCoMoFormat],
    category_id: int,
) -> list[NeedleTestConfig]:
    # extract locomo's needle data (only the QA) to NeedleTestConfig format
    assert category_id in LOCOMO_CATEGORIES, f"Invalid category_id {category_id}"
    category_name = LOCOMO_CATEGORIES[category_id]

    needles: list[NeedleTestConfig] = []
    for sample in locomo_data:
        # filter qa by x["category"] == category
        qas = [x for x in sample["qa"] if x["category"] == category_id]
        # for each qa like below, create a NeedleTestConfig
        # {
        #     "question": "When did Caroline go to the LGBTQ support group?",
        #     "answer": "7 May 2023",
        #     "evidence": [
        #         "D1:3"
        #     ],
        #     "category": 2
        # },
        needle = NeedleTestConfig(
            id=f"{sample['sample_id']}_C{category_id}",
            reasoning_type=category_name,
            system_prompt=None,
            task_template=None,
            needle=None,
            questions={
                category_name: "{1}",
            },
            character_set=None,
            tests={
                f"Q{qi}": {
                    "input_args": [qa["question"]],
                    "gold_answers": [qa["answer"]],
                }
                for qi, qa in enumerate(qas)
            },
            distractors=None,
            context=prepare_context(sample["conversation"]),  # type: ignore
        )
        needles.append(dataclass_to_dict(needle))  # type: ignore
    return needles


def dialogue_to_text(speaker: str, dia_id: str, text: str, blip_caption: str) -> str:
    if blip_caption == "":
        return f"<span class='dialogue' data-speaker='{speaker}' data-dia-id='{dia_id}'>{text}</span>"
    return f"<span class='dialogue' data-speaker='{speaker}' data-dia-id='{dia_id}'>[{blip_caption}] {text}</span>"


def prepare_context(conversation: dict[str, str | list[dict[str, str]]]) -> str:
    out = ""
    # locomo has no more than 40 sessions, so 50 is safe
    for i in range(1, 50):
        session_key = f"session_{i}"
        session_dt_key = f"session_{i}_date_time"
        if session_key not in conversation:
            break

        out += "<div class='session'>"
        out += f"<span class='timestamp'>{conversation[session_dt_key]}</span>"
        for turn in conversation[session_key]:
            assert isinstance(turn, dict)
            out += dialogue_to_text(
                turn["speaker"],
                turn["dia_id"],
                turn["text"],
                turn.get("blip_caption", ""),
            )
        out += "</div>"
    return out


if __name__ == "__main__":
    locomo_path = sys.argv[1]
    locomo_data: list[LoCoMoFormat] = json.load(open(locomo_path, "r"))
    dirname = osp.dirname(locomo_path)

    needleset_dir = f"{dirname}/needlesets/"
    os.makedirs(needleset_dir, exist_ok=True)

    for category_id in LOCOMO_CATEGORIES.keys():
        d = prepare_configs(locomo_data, category_id=category_id)
        out_path = (
            f"{needleset_dir}/{category_id}_{LOCOMO_CATEGORIES[category_id]}.json"
        )
        with open(out_path, "w") as f:
            json.dump(d, f, indent="\t")

    haystack_dir = f"{dirname}/haystack/"
    os.makedirs(haystack_dir, exist_ok=True)
    # prepare an empty haystack file txt
    with open(f"{haystack_dir}/fake.txt", "w") as f:
        f.write("")
