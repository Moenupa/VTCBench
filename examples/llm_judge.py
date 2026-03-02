import json
import os.path as osp
import re
import sys
from glob import glob
from typing import Any, TypedDict

from openai import OpenAI
from tqdm.contrib.concurrent import process_map, thread_map

# 初始化OpenAI客户端
# 注意：需要设置OPENAI_API_KEY环境变量
try:
    client = OpenAI()
except Exception:
    print(
        "WARN: Please set a valid OPENAI_API_KEY environment variable. OpenAI client initialization failed."
    )
    client = None


class ExampleMetadata(TypedDict):
    problem: str
    answers: list[str]


class ResultEntry(TypedDict):
    result_path: str  # index
    example_meta: ExampleMetadata  # where question and gt stored
    results: list[dict]  # where pred stored under results[0]["response"]


# 评分提示词
QUESTION_QUALITY_PROMPT_EN_NO_COT = (
    "Please as a grading expert, judge whether the final answers given by the candidates below are consistent with "
    "the standard answers, that is, whether the candidates answered correctly. "
    "Here are some evaluation criteria: "
    "1. Please refer to the given standard answer. You don't need to re-generate the answer to the question because "
    "the standard answer has been given. You only need to judge whether the candidate's answer is consistent with the "
    "standard answer according to the form of the question. THE STANDARD ANSWER IS ALWAYS CORRECT AND THE QUESTION IS "
    "PERFECTLY VALID. NEVER QUESTION THEM. "
    "2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES. "
    "3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression, some "
    "answers may be a textual description, as long as the meaning expressed is the same. Before making a judgment, "
    "please understand the question and the standard answer first, and then judge whether the candidate's answer is "
    "correct.If the standard answer does not specify a unit, but the candidate's answer includes a unit that is "
    "correct for the value given, consider it correct. "
    "4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions, "
    "fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered correct "
    "as long as it matches the standard answer, regardless of whether the reasoning process is correct. For "
    "multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or blanks must "
    "be answered correctly and match the standard answer exactly to be deemed correct. "
    "5. If the prediction is given with \\boxed{{}}, please ignore the \\boxed{{}} and only judge whether the "
    "candidate's answer is consistent with the standard answer. "
    "6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of unnormal repetitive "
    "content, or irrelevant to the question, saying it can't answer the question because some irresistible factors, "
    "like ethical issues, no enough information, etc.), select option C (INVALID).Please judge whether the following "
    "answers are consistent with the standard answer based on the above criteria. Grade the predicted answer of this "
    "new question as one of: "
    "A: CORRECT "
    "B: INCORRECT "
    "C: INVALID "
    'Just return the letters "A", "B", or "C", with no text around it. '
    "Here is your task. Simply reply with either CORRECT, INCORRECT, or INVALID. Don't apologize or correct yourself "
    "if there was a mistake; we are just trying to grade the answer. "
    "\n<Original Question Begin>: "
    "{question} "
    "\n<Original Question End> "
    "\n<Standard Answer Begin>: "
    "{gold_answer} "
    "\n<Standard Answer End> "
    "\n<Candidate's Answer Begin>: "
    "{llm_response} "
    "\n<Candidate's Answer End> "
    "\nJudging the correctness of the candidate's answer:"
)


def extract_final_answer(response_text: str) -> str:
    """
    从带有思考过程的回复中提取最终答案
    如果有<think>标签，则只取<think>之后的内容
    """
    # 查找<think>标签后的最终答案
    thinking_match = re.search(r"<think>(.*?)</think>(.*)", response_text, re.DOTALL)
    if thinking_match:
        # 如果找到<think>标签，只取</think>后面的部分作为最终答案
        return thinking_match.group(2).strip()

    # 如果没有找到标准标签，尝试查找其他可能的格式
    final_answer_match = re.search(r"</think>(.*)", response_text, re.DOTALL)
    if final_answer_match:
        return final_answer_match.group(1).strip()

    # 否则返回整个回答
    return response_text.strip()


def evaluate_answer_with_gpt(
    question: str, gold_answer: str, llm_response: str, max_retries: int = 3
) -> str:
    """
    使用GPT判断答案正确性，添加失败重试逻辑
    """
    if client is None:
        return "C"  # 如果没有可用的OpenAI客户端，则返回无效

    # 提取最终答案
    final_answer = extract_final_answer(llm_response)

    for attempt in range(max_retries):
        try:
            prompt = QUESTION_QUALITY_PROMPT_EN_NO_COT.format(
                question=question, gold_answer=gold_answer, llm_response=final_answer
            )

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )

            result = response.choices[0].message.content.strip()
            # 确保返回的是A、B或C之一
            if result in ["A", "B", "C"]:
                return result
            else:
                print(f"第{attempt + 1}次尝试：GPT返回无效结果: {result}")
                if attempt < max_retries - 1:
                    continue
                else:
                    return "C"  # 如果返回的不是预期结果，则认为无效
        except Exception as e:
            print(f"第{attempt + 1}次尝试：GPT评估出错: {e}")
            if attempt < max_retries - 1:
                continue
            else:
                return "C"  # 出错时返回无效

    return "C"  # 所有重试都失败时返回无效


def process_single_entry(entry_data: ResultEntry) -> dict[str, Any] | None:
    try:
        result_path = entry_data["result_path"]
        question = entry_data["example_meta"]["problem"]
        gold_answers = entry_data["example_meta"]["answers"]
        response = entry_data["results"][0]["response"]

        gold_answer_str = ", ".join(gold_answers) if gold_answers else ""

        result = evaluate_answer_with_gpt(question, gold_answer_str, response)

        return {
            "result_path": result_path,
            "question": question,
            "gold_answer": gold_answer_str,
            "model_response": response,
            "evaluation": result,
        }
    except Exception as e:
        print(f"处理 {entry_data['result_path']} 时出错: {e}")
        return None


def process_entries(entries: list[ResultEntry], max_workers: int = 5) -> dict[str, Any]:
    results = thread_map(
        process_single_entry,
        entries,
        max_workers=max_workers,
        desc="LLM Judging",
    )
    correct_count = sum(1 for r in results if r and r["evaluation"] == "A")
    results.sort(key=lambda x: x["result_path"])
    return {
        "results": results,
        "total": len(results),
        "correct_count": correct_count,
        "accuracy": correct_count / len(results) if results else 0,
    }


def load_json(fp: str) -> ResultEntry:
    return json.load(open(fp))


def folder_to_entries(folder_path: str) -> list[ResultEntry]:
    files = glob(f"{folder_path}/*/*.json", recursive=True)
    # read each json file as a ResultEntry
    entries: list[ResultEntry] = process_map(
        load_json,
        files,
        max_workers=10,
        desc=f"Loading JSON files from {folder_path}/*/*.json",
    )
    return entries


if __name__ == "__main__":
    for folder_path in sys.argv[1:]:
        if not osp.isdir(folder_path):
            print(f"usage: python {sys.argv[0]} <results_folder_for_model> ...")
            continue

        if existing_results := glob(f"{folder_path}/*.json"):
            print(f"Evaluation skipped: {existing_results}. Remove to re-evaluate.")
            continue

        entries = folder_to_entries(folder_path)
        if len(entries) == 0:
            print(f"Skipped {folder_path}: no json files found.")
            continue

        result = process_entries(entries, max_workers=10)

        with open(f"{folder_path}/evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"-> {folder_path}/evaluation_results.json")
        print(f"{result['accuracy']:.2%}={result['correct_count']}/{result['total']}")
