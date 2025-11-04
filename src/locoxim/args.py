from dataclasses import dataclass, field

DEFAULT_TASK_TEMPLATE = "You will answer a question based on the following book snippet:\n\n{haystack}\n\nUse the information provided in the book snippet to answer the question. Your answer should be short and based on either explicitly stated facts or strong, logical inferences.\n\nQuestion: {question}\n\n Return only the final answer with no additional explanation or reasoning."


def args_to_dict(args) -> dict:
    """
    Convert dataclass arguments to a dictionary.
    """
    if args is None:
        return {}
    return {k: v for k, v in dict(args).items() if not k.startswith("_")}


@dataclass
class ModelArgs:
    """
    Arguments for LLM/VLM Identifier and its configuration
    """

    # api configs
    model: str = field(
        default="Qwen2.5-VL-7B-Instruct",
        metadata={"help": "HF model path or local dir path for the LLM/VLM"},
    )
    api_key: str = field(
        default="EMPTY",
        metadata={"help": "API key for the LLM/VLM service"},
    )
    api_url: str = field(
        default="http://localhost:1025/v1",
        metadata={"help": "API URL for the LLM/VLM service"},
    )
    api_provider: str = field(
        default="vllm",
        metadata={"help": "API provider type, e.g., openai, vllm, etc."},
    )

    # model configs
    max_tokens: int = field(
        default=192,
        metadata={"help": "Maximum tokens to generate"},
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature for generation"},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p sampling parameter"},
    )
    system_prompt: str = field(
        default="You are a helpful assistant",
        metadata={"help": "Default system prompt for the model"},
    )

    # networking stuff
    timeout: int = field(
        default=200,
        metadata={"help": "Timeout for API requests in seconds"},
    )
    max_retries: int = field(
        default=3,
        metadata={"help": "Maximum number of retries for API requests"},
    )

    # computing tokens to figure out compression ratio
    tokenizer_type: str = field(
        default="huggingface",
        metadata={"help": "Type of tokenizer to use"},
    )
    tokenizer_model: str = field(
        default=None,
        metadata={
            "help": "Model path or local dir path, e.g. Qwen/Qwen2.5-VL-7B-Instruct."
        },
    )


@dataclass
class DataArgs:
    """
    Arguments for data sources, settings, and templates
    """

    needle_set_path: str = field(
        default="data/NoLiMa/needlesets/needle_set.json",
        metadata={"help": "Path to a file containing the needle tests configuration"},
    )
    haystack_dir: str = field(
        default="data/NoLiMa/haystack/rand_shuffle",
        metadata={"help": "Directory containing the haystack files"},
    )
    task_template: str | None = field(
        default=DEFAULT_TASK_TEMPLATE,
        metadata={"help": "Task template name overriding ones specified in needle set"},
    )
    use_default_system_prompt: bool = field(
        default=True,
        metadata={"help": "Use default system prompt"},
    )
    context_length: int = field(
        default=None,
        metadata={"help": "Context length for the needle placement"},
    )
    document_depth_percent_min: float = field(
        default=0,
        metadata={"help": "Minimum document depth percentage"},
    )
    document_depth_percent_max: float = field(
        default=100,
        metadata={"help": "Maximum document depth percentage"},
    )
    document_depth_num_tests: int = field(
        default=35,
        metadata={"help": "Number of points between min and max depth"},
    )
    shift: float = field(
        default=0,
        metadata={
            "help": "Shift for needle placement, applied to the beginning of the haystack"
        },
    )
    static_depth: float = field(
        default=None,
        metadata={"help": "Static depth for needle placement"},
    )
    pure_text: bool = field(
        default=True,
        metadata={
            "help": "If true, use pure text as input to LLM/VLM. Otherwise, embed context in images."
        },
    )


@dataclass
class RunArgs:
    """
    Arguments for identifying, collecting, and reporting runs.
    """

    metric: str = field(
        default="EM",
        metadata={"help": "Evaluation metric"},
    )
    log_placements_dir: str = field(
        default="",
        metadata={
            "help": "Directory to save needle placement logs. If empty, no logs are saved."
        },
    )
    parent_results_dir: str = field(
        default="evaluation_results",
        metadata={"help": "Parent directory to save results"},
    )
    base_seed: int = field(
        default=42,
        metadata={"help": "Base seed for random operations"},
    )
    prevent_duplicate_tests: bool = field(
        default=False,
        metadata={
            "help": "Prevent duplicate tests in the evaluation by scanning other parallel results. Warning: this slows down evaluation significantly."
        },
    )
