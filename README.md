# VTCBench: Can Vision-Language Models Understand Long Contexts with Vision-Text Compression?

<div align="center">
  <a href="https://arxiv.org/abs/2512.15649">
    <img src="https://img.shields.io/badge/2512.15649-B31B1B?logo=arxiv" alt="Arxiv: 2512.15649" /></a>
  <a href="https://huggingface.co/datasets/MLLM-CL/VTCBench">
    <img src="https://img.shields.io/badge/Hugging_Face-orange?logo=huggingface" alt="Hugging Face" /></a>
  <a href="https://creativecommons.org/licenses/by-nc/4.0/">
    <img src="https://img.shields.io/badge/CC_BY--NC_4.0-ED592F?logo=creativecommons&logoColor=white" alt="License: CC BY-NC 4.0" /></a>
  <a href="./CITATION.cff">
    <img src="https://img.shields.io/badge/CITATION-grey" alt="Citation" /></a>
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

<table>
<tr>
<th>Task</th>
<th>Task Categories</th>
<th>Task Description</th>
<th>Context Example</th>
<th>Evaluation Example</th>
<th>Visual Text Example</th>
</tr>

<tr>
<td>VTC-Retrieval (NIAH)</td>
<td>Lexical Matching, Multi-Hop Tracing, Aggregation</td>
<td>
  A visual "Needle-In-A-Haystack" (NIAH) test. Requires locating specific 
  "needles" (key-value pairs) embedded within a large "haystack" of distractors.
  <p>Sub-tasks: Single-needle, Multi-keys, Multi-values, and Multi-queries.</p>
</td>
<td>
  (Dynamic 
  <span style='color: orange'>query/key</span>-<span style='color: teal'>value</span> 
  with types: 
  <span style='color: orange'>word</span>-<span style='color: teal'>word</span>,
  <span style='color: orange'>word</span>-<span style='color: teal'>number</span>, 
  <span style='color: orange'>uuid</span>-<span style='color: teal'>number</span>)
  <div style='color: gray'>(essays...)</div>
  One of the special magic numbers for 
  <span style='color: orange'>long-context</span> is: 
  <span style='color: teal'>2026</span>.
  <div style='color: grey'>...One of the special magic numbers for distracting-information is: 2025.</div>
</td>
<td>
  <div><b>QA Variant:</b></div>
  <i>Q:</i> What's the special magic number for <span style='color: orange'>long-context</span>?
  <i>A:</i> <span style='color: teal'>2026</span>.
  <div><b>Completion Variant:</b></div>
  <i>Prompt:</i> one of the special magic number for <span style='color: orange'>long-context</span> is:
  <i>Completion:</i> <span style='color: teal'>2026</span>.
</td>
<td><img src="assets/data_samples/ruler_sample.jpeg"/></td>
</tr>

<tr>
<td>VTC-Reasoning  (NIAH)</td>
<td>Associative Reasoning, Question-Answering</td>
<td>
  Minimized literal overlap between query and key. 
  Requires inferring latent associations rather than simple keyword matching.
</td>
<td>
  (Dynamic 
  <span style='color: orange'>query/key</span>-<span style='color: teal'>value</span> 
  with types: 
  <span style='color: orange'>event/action</span>-<span style='color: teal'>person</span>)
  <div style='color: gray'>(books...)</div>
  There was a <span style='color: orange'>vegan</span> guest, named <span style='color: teal'>Katie</span>.
</td>
<td>
  <div><b>One-Hop Reasoning:</b></div>
  <i>Q:</i> Which character cannot eat <span style='color: orange'>fish-based</span> meals?
  <i>A:</i> <span style='color: teal'>Katie</span>.
  <div><b>Two-Hop Reasoning:</b></div>
  <i>Q:</i> Which character cannot eat <span style='color: orange'>Brandade</span> meals?
  <i>A:</i> <span style='color: teal'>Katie</span>.
</td>
<td><img src="assets/data_samples/nolima_sample.jpeg"/></td>
</tr>

<tr>
<td>VTC-Memory (QA)</td>
<td>Memory, Question-Answering</td>
<td>
  Multi-turn conversations testing long-term memory.
  <p>Sub-tasks: Single-hop, Multi-hop, Temporal reasoning, and Open-domain knowledge.</p>
</td>
<td>
  (No dynamic 
  <span style='color: orange'>query/key</span>-<span style='color: teal'>value</span>,
  fully static.)
  <div style='color: gray'>(conversations...)</div>
  <i style='color: orange'>Caroline</i>: <span style='color: teal'>Researching adoption agencies</span>&mdash;it's
  been a dream to have a family and give a loving home to kids who need it.
</td>
<td>
  <i>Q:</i> What did <span style='color: orange'>Caroline</span> research?
  <i>A:</i> <span style='color: teal'>Adoption agencies</span>.
</td>
<td><img src="assets/data_samples/locomo_sample.jpeg"/></td>
</tr>

</table>

### VTCBench-Wild

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
