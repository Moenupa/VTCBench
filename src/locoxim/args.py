from dataclasses import dataclass, field


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
    openai_thinking_model: bool = field(
        default=False,
        metadata={
            "help": "Enables special kwargs handling for OpenAI models, e.g. gpt-5",
        },
    )

    # model configs
    max_tokens: int = field(
        default=192,
        metadata={"help": "Maximum tokens to generate"},
    )
    # optional arguments for generation
    temperature: float | None = field(
        default=None,
        metadata={"help": "Temperature for generation"},
    )
    top_p: float | None = field(
        default=None,
        metadata={"help": "Top-p sampling parameter"},
    )
    extra_kwargs: dict | None = field(
        default=None,
        metadata={"help": "Additional sampling kwargs for generation"},
    )
    system_prompt: str | None = field(
        default="You are a helpful assistant.",
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
    tokenizer_type: str | None = field(
        default="huggingface",
        metadata={"help": "Type of tokenizer to use"},
    )
    tokenizer_model: str | None = field(
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

    needle_set_path: list[str] = field(
        default_factory=list,
        metadata={"help": "Path to a file containing the needle tests configuration"},
    )
    haystack_dir: str = field(
        default="data/NoLiMa/haystack/rand_shuffle",
        metadata={"help": "Directory containing the haystack files"},
    )
    task_template: str | None = field(
        default=None,
        metadata={"help": "Task template name overriding ones specified in needle set"},
    )
    use_default_system_prompt: bool = field(
        default=True,
        metadata={"help": "Use default system prompt"},
    )
    context_length: int | None = field(
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
    shift: int = field(
        default=0,
        metadata={
            "help": "Shift for needle placement, applied to the beginning of the haystack"
        },
    )
    static_depth: float | None = field(
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

    log_dir: str | None = field(
        default=None,
        metadata={
            "help": "Log directory to save intermediate output. If None, no logs are saved."
        },
    )
    base_seed: int = field(
        default=42,
        metadata={"help": "Base seed for random operations"},
    )
    prevent_duplicate_tests: bool = field(
        default=False,
        metadata={
            "help": "Prevent duplicate tests in the evaluation by scanning other results. Warning: this slows down evaluation significantly."
        },
    )
    num_workers: int = field(
        default=1,
        metadata={"help": "Number of parallel workers for evaluation"},
    )
    parent_results_dir: str = field(
        default="results",
        metadata={"help": "Parent directory to save results"},
    )
    api_cache_dir: str | None = field(
        default=".cache/api_calls",
        metadata={
            "help": "Parent dir to store api call-response cache to avoid redundant calls. If None, disable caching."
        },
    )
    image_dir: str | None = field(
        default=".cache/images",
        metadata={
            "help": "Parent dir to store intermediate images generated during evaluation."
        },
    )
    num_tasks: int | None = field(
        default=50,
        metadata={"help": "Number of tasks to sample for evaluation"},
    )
    verbose: bool = field(
        default=True,
        metadata={"help": "Show verbose output for the first sample only"},
    )
