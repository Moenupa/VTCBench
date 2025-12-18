# VTCBench: Can Vision-Language Models Understand Long Contexts with Vision-Text Compression?

<div align="center">
  <a href="https://arxiv.org/abs/2512.15649">
    <img src="https://img.shields.io/badge/2512.15649-B31B1B?logo=arxiv" alt="Arxiv: 2512.15649" />
  </a>
  <a href="https://huggingface.co/datasets/MLLM-CL/VTCBench">
    <img src="https://img.shields.io/badge/Hugging_Face-orange?logo=huggingface" alt="Hugging Face" />
  </a>
  <a href="https://creativecommons.org/licenses/by-nc/4.0/">
    <img src="https://img.shields.io/badge/CC_BY--NC_4.0-ED592F?logo=creativecommons&logoColor=white" alt="License: CC BY-NC 4.0" />
  </a>
  <a href="./CITATION.bib">
    <img src="https://img.shields.io/badge/CITATION-grey" alt="Citation" />
  </a>
</div>

VTCBench is the first comprehensive benchmark specifically designed to evaluate
 the long-context understanding capabilities of Vision-Language Models (VLMs) 
 within the Vision-Text Compression (VTC) paradigm.

<div align="center">
  <img width="47%" src="assets/vtc_pipeline.jpg" />
  <img width="51%" src="assets/vtcbench_tasks.jpg" />
</div>

VTC is an emerging framework that converts long texts into dense 2D visual 
representations (images), achieving token compression ratios of 2-10x 
compared to standard text tokenization. VTCBench rigorously assesses whether 
VLMs can actually understand this compressed information or if they are merely
performing surface-level OCR.

## 🚀 Key Features

- **Three Core Tasks**: Evaluates VLMs across Retrieval, Reasoning, and Memory.
- **VTCBench-Wild**: A variant designed to simulate real-world visual diversity 
  (e.g., varying fonts, backgrounds, and layouts)
- **Two Evaluation Settings**: 
  - Predefined VTC Ratio: Predetermines the compression ratio (e.g., $r_\texttt{VTC}=2.0$)
    to compare model intelligence at a standardized information density
  - Predefined Rendering: Uses a fixed document format (12-pt Helvetica, 96 DPI) 
    to simulate realistic document processing666.
- **Extensive Model Coverage**: Benchmarks 13 leading models including GPT-5, 
  Gemini-2.5 Pro, Gemma, Glyph, Qwen2.5 & Qwen3 & InternVL3.5 series, and more.

## 📊 Benchmark Tasks

### 1. VTC-Retrieval

A visual "Needle-In-A-Haystack" (NIAH) test. It evaluates the model's ability 
to locate specific "needles" (key-value pairs) embedded within a large "haystack"
of distractors. Sub-tasks: Single-needle, Multi-keys, Multi-values, and Multi-queries.

### 2. VTC-Reasoning

Assesses associative reasoning by minimizing literal overlap between the query 
and the context. Success requires the model to infer latent associations rather 
than simple keyword matching.

### 3. VTC-Memory

Evaluates long-term dialogue memory using multi-turn conversations.
Sub-tasks: Single-hop, Multi-hop, Temporal reasoning, and Open-domain knowledge.

### 4. VTCBench-Wild

A more challenging variant of the above tasks, introducing visual noise and 
diversity to simulate real-world document conditions.

## 📈 Main Findings

![vtcbench_results](assets/vtcbench_results.jpg)

- **Perception $\neq$ Comprehension**: While many VLMs excel at OCR and simple 
  retrieval, their performance collapses on reasoning and memory tasks 
  compared to text-only LLMs.
- **Length Fragility**: VLM performance degrades significantly as the context 
  length increases (e.g., from 1k up to 32k tokens).
- **Parameter Sensitivity**: VTC performance is highly sensitive to font size 
  and the spatial positioning of information

## 🛠 Usage & Data

Please refer to the [Usage Guide](docs/USAGE.md) for instructions on how to use VTCBench.
