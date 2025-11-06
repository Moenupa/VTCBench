This project expects that LLM/VLM is seperately deployed, such as [vLLM](https://github.com/vllm-project/vllm), OpenAI API, etc.

A simple example to get you started, using deps from [pyproject.toml](./pyproject.toml):

```sh
# set up a vllm environment seperately, using pyproject.toml
uv venv
uv sync --all-extras
# serve your model
vllm serve Qwen/Qwen3-VL-2B-Instruct --port 8001
# to test your endpoint
curl http://localhost:8001/v1/models
```
