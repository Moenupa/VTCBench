import json
from dataclasses import dataclass, is_dataclass
from functools import cache
from hashlib import sha256
from typing import Any, Generator, Optional

HASH_CACHE_KEY = "hash_sha256"


def dataclass_to_dict(instance) -> dict:
    assert is_dataclass(instance)
    return {
        field.name: getattr(instance, field.name)
        for field in instance.__dataclass_fields__.values()
        if not field.name.startswith("_")
    }


def fill_placeholders(template: str, placeholder: str, value: str) -> str:
    # try to replace this placeholder in the template
    if placeholder in template:
        return template.replace(placeholder, value)
    # otherwise return the original
    return template


def get_hash(
    config_or_data: dict[str, Any],
    debug: bool = False,
) -> str:
    """
    Get a hash string for a given configuration or data dictionary.

    Args:
        config_or_data (dict): Configuration or data dictionary.

    Returns:
        str: Hash string.
    """
    assert isinstance(config_or_data, dict), (
        f"Expected dict, got {type(config_or_data)}"
    )

    if debug:
        for k, v in config_or_data.items():
            try:
                json.dumps(v)
            except TypeError as e:
                raise e

    json_str = json.dumps(config_or_data, sort_keys=True)
    hash_str = get_hash_str(json_str)

    return hash_str


@cache
def get_hash_str(input_str: str) -> str:
    return sha256(input_str.encode("utf-8")).hexdigest()


def _try_parse(text: str | None) -> list[str] | None:
    if text is None:
        return None

    # not a string then assume it is parsed already
    if not isinstance(text, str):
        return text

    # try to parse as list
    try:
        parsed: list[str] = json.loads(text)
        return parsed

    # if not JSON-decodable, return as single-item list
    except json.JSONDecodeError:
        return [text]


def get_distractor_template(
    test_config: "NeedleTestConfig",
    distractor_key: str,
) -> str | None:
    if test_config.distractors is None:
        return None
    return test_config.distractors.get(distractor_key, None)


def iter_question_items(
    test_config: "NeedleTestConfig",
    base_seed: int = 42,
) -> Generator["QuestionItem", None, None]:
    for question_batch_type, question_batch_template in test_config.questions.items():
        for question_batch_id, question_batch_kwargs in test_config.tests.items():
            yield QuestionItem.from_template_with_placeholders(
                test_id=test_config.id,
                system_prompt=None
                if test_config.system_prompt is None
                else test_config.system_prompt.strip(),
                task_template=test_config.task_template,
                question_batch_id=question_batch_id,
                question_batch_type=question_batch_type,
                question_batch_template=question_batch_template,
                question_batch_args=question_batch_kwargs["input_args"],
                needle_template=test_config.needle,
                gold_answers=question_batch_kwargs.get("gold_answers", None),
                base_seed=base_seed,
                character_set=test_config.character_set,
                distractor_template=get_distractor_template(
                    test_config, question_batch_type
                ),
            )


@dataclass
class NeedleTestConfig:
    """
    A test, i.e. a dict in json file containing M*N (question, args) combinations

    Args:
        id (str): Unique ID for the test.
        reasoning_type (str): Type of reasoning ability being tested (e.g., world knowledge).
        system_prompt (str | None): Optional system prompt override.
        task_template (str): Template for the task, filled with needle and question.
        needle (str): Template for the needle.
        questions (dict[str, str]): Mapping from question type/difficulty to question templates, e.g. one-hop, two-hop.
        character_set (list[str] | None): List of characters for needle placement.
        tests (dict[str, dict[str, list]]): Mapping from question IDs to their configurations.
        distractors (dict[str, str] | None): Optional mapping from distractor keys to templates.
    """

    id: str
    reasoning_type: str
    system_prompt: str | None
    task_template: str
    needle: str
    questions: dict[str, str]
    character_set: list[str] | None
    # tests: {"test_id": {"input_args": [...]}}
    tests: dict[str, dict[str, list]]
    distractors: dict[str, str] | None = None


@dataclass
class QuestionItem:
    """
    A single question item, directly used for evaluation

    Args:
        question_id (str): Unique ID for the question.
        system_prompt (Optional[str]): System prompt for the model.
        task_template (str): Task template for the question.
        needle (Optional[str]): Needle string to fill in haystack; None if no needle placement.
        retrieval_question (str): Question string with placeholders filled.
        gold_answers (Optional[list[str]]): Correct answers for evaluation.
        character_set (Optional[list[str]]): Character set for needle placement.
        distractor (Optional[str]): Distractor string with placeholders filled, if any.
        seed (Optional[int]): Seed for random operations.
    """

    question_id: str
    system_prompt: Optional[str]
    task_template: str

    needle: Optional[str]
    retrieval_question: str
    gold_answers: Optional[list[str]]
    character_set: Optional[list[str]]
    distractor: Optional[str]

    seed: int

    @classmethod
    def from_template_with_placeholders(
        cls,
        test_id: str,
        system_prompt: str | None,
        task_template: str | None,
        # args for each question
        question_batch_id: str,
        question_batch_type: str,
        question_batch_template: str,
        question_batch_args: list[str],
        needle_template: str,
        gold_answers: Optional[list[str]],
        base_seed: int,
        character_set: Optional[list[str]] = None,
        distractor_template: Optional[str] = None,
    ) -> "QuestionItem":
        # fill in the placeholders, taking question_args one by one
        for argc, argv in enumerate(question_batch_args):
            _place_holder = "{" + str(argc + 1) + "}"
            question_batch_template = fill_placeholders(
                question_batch_template, _place_holder, argv
            )
            needle_template = fill_placeholders(needle_template, _place_holder, argv)

            if distractor_template is not None:
                distractor_template = fill_placeholders(
                    distractor_template, _place_holder, argv
                )

        new_item = cls(
            question_id=f"{test_id}_{question_batch_id}_{question_batch_type}",
            system_prompt=system_prompt,
            task_template=task_template,
            needle=needle_template,
            retrieval_question=question_batch_template,
            gold_answers=gold_answers,
            character_set=character_set,
            distractor=distractor_template,
            seed=(base_seed + hash(f'{test_id}_{question_batch_id}_{question_batch_type}')) % (2**32),
        )
        return new_item
