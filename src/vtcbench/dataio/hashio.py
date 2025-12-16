import json
import os.path as osp
import pickle as pkl
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from typing import Any

HASH_CACHE_KEY = "hash_sha256"


def args_to_dict(args) -> dict:
    """
    Convert dataclass arguments to a dictionary.
    """
    if args is None:
        return {}
    if is_dataclass(args):
        return {k: v for k, v in asdict(args).items() if not k.startswith("_")}

    # Namespace, or other dict-like
    return {k: v for k, v in dict(args).items() if not k.startswith("_")}


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


def get_hash_str(input_str: str) -> str:
    return sha256(input_str.encode("utf-8")).hexdigest()


def api_cache_path(messages: dict, parent: str | None) -> str | None:
    if parent is None:
        return None

    msg_str = json.dumps(messages, sort_keys=True)
    msg_hash = get_hash_str(msg_str)
    return f"{parent}/{msg_hash}.pkl"


def api_cache_io(
    cache_path: str | None,
    save_response: dict | None = None,
) -> dict | None:
    if cache_path is None:
        return None

    # need to work with async functions
    if osp.exists(cache_path):
        with open(cache_path, "rb") as file:
            response: dict = pkl.load(file)
            return response
    elif save_response is not None:
        # no existing cache, and saving enabled
        with open(cache_path, "wb") as file:
            pkl.dump(save_response, file)

    return None
