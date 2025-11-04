import json
from dataclasses import dataclass, is_dataclass
from functools import cache
from hashlib import sha256
from typing import Any, Generator

HASH_CACHE_KEY = "hash_sha256"


def dataclass_to_dict(instance) -> dict:
    """
    Convert a dataclass instance to a dictionary, including embedded dataclasses.
    """
    assert is_dataclass(instance)
    return {
        field.name: (
            dataclass_to_dict(getattr(instance, field.name))
            if is_dataclass(getattr(instance, field.name))
            else getattr(instance, field.name)
        )
        for field in instance.__dataclass_fields__.values()
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


@dataclass
class NeedleTestConfig:
    """
    A test, i.e. a dict in json file containing M*N (question, args) combinations

    Args:
        id (str): Unique test ID
        reasoning_type (str): Ability tested (e.g., world knowledge)
        system_prompt (str): System prompt override
        task_template (str): Task template, to be filled by haystack (w. needle) and question
        needle (str): Needle template
        questions (dict[str, str]): Mapping of question difficulty to question templates
        character_set (list[str]): List of characters for needle placement
        tests (dict[str, dict[str, list]]): Mapping of question IDs to their configs
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

    def get_distractor_template(self, distractor_key: str) -> str | None:
        if self.distractors is None:
            return None
        return self.distractors.get(distractor_key, None)

    def __iter__(self):
        return self.iter_question_items()

    def iter_question_items(
        self,
        default_system_prompt: str | None = None,
        default_task_template: str | None = None,
        base_seed: int = 42,
    ) -> Generator["QuestionItem", None, None]:
        for question_batch_type, question_batch_template in self.questions.items():
            for question_batch_id, question_batch_kwargs in self.tests.items():
                yield QuestionItem.from_template_with_placeholders(
                    test_id=self.id,
                    system_prompt=default_system_prompt or self.system_prompt,
                    task_template=default_task_template or self.task_template,
                    question_batch_id=question_batch_id,
                    question_batch_type=question_batch_type,
                    question_batch_template=question_batch_template,
                    question_batch_args=question_batch_kwargs["input_args"],
                    needle_template=self.needle,
                    gold_answers=question_batch_kwargs.get("gold_answers", None),
                    base_seed=base_seed,
                    character_set=self.character_set,
                    distractor_template=self.get_distractor_template(
                        question_batch_type
                    ),
                )

    def __hash__(self):
        return hash(self.__dict__())

    def __dict__(self):
        return dataclass_to_dict(self)

    def __json__(self):
        return json.dumps(self.__dict__(), sort_keys=True)

    def __str__(self):
        return self.__json__()


@dataclass
class QuestionItem:
    """
    A single question item, directly used for evaluation

    Args:
        question_id (str): Unique ID for the question
        system_prompt (str): System prompt for the model
        task_template (str): Task template for the question
        needle (str): Needle string with placeholders filled
        retrieval_question (str): Question string with placeholders filled
        gold_answers (str): Correct answers for evaluation
        character_set (str): Character set for needle placement
        distractor (str | None): Distractor string with placeholders filled, if any
        seed (int): Seed for random operations
    """

    question_id: str
    system_prompt: str | None
    task_template: str

    needle: str
    retrieval_question: str
    gold_answers: list[str] | None
    character_set: list[str] | None
    distractor: str | None

    seed: int

    def __hash__(self):
        return hash(self.__dict__())

    def __dict__(self):
        return dataclass_to_dict(self)

    @classmethod
    def from_template_with_placeholders(
        cls,
        test_id: str,
        system_prompt: str | None,
        task_template: str,
        # args for each question
        question_batch_id: str,
        question_batch_type: str,
        question_batch_template: str,
        question_batch_args: list[str],
        needle_template: str,
        gold_answers: list[str] | None,
        base_seed: int,
        character_set: list[str] | None = None,
        distractor_template: str | None = None,
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
            seed=base_seed + int(test_id[:4]),
        )
        return new_item
