import argparse
import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Tuple

import torch
from dotenv import load_dotenv
from tqdm import tqdm

from src.utils import getData, saveData

# 固定随机种子
seed = 42
if hasattr(torch, "manual_seed"):
    torch.manual_seed(seed)
random.seed(seed)
if hasattr(torch, "cuda") and hasattr(torch.cuda, "manual_seed_all"):
    torch.cuda.manual_seed_all(seed)

load_dotenv()  # 加载 .env 文件

DEFAULT_TEST_FILE = Path("dataset/test.json")
DEFAULT_SAVE_DIR = Path("quiz_results/qwen_7b_full_quiz_gemini_40303")


def configure_logging() -> None:
    """Configure application logging once at startup."""

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)",
        handlers=[console_handler],
    )
def prepare_quiz_dataset(input_file: Path, save_dir: Path) -> Tuple[list, Path]:
    """Prepare quiz data without mutating the source dataset."""

    os.makedirs(save_dir, exist_ok=True)
    quiz_data = getData(str(input_file))
    annotated_data: list = []
    for item in quiz_data:
        annotated_item = {
            **item,
            "quiz_url": str(save_dir / f"{item['id']}.md"),
        }
        annotated_data.append(annotated_item)

    annotated_path = save_dir / f"{input_file.stem}_with_quiz_urls.json"
    saveData(annotated_data, str(annotated_path))
    return annotated_data, annotated_path


def build_generator_model() -> Tuple[object, object]:
    """Construct generator model/tokenizer placeholders based on configuration."""

    from src.config.llms import generator_model, qwen_tokenizer_path
    from transformers import AutoTokenizer

    if generator_model == "qwen":
        tokenizer = AutoTokenizer.from_pretrained(
            qwen_tokenizer_path,
            trust_remote_code=True,
            use_fast=True,
            padding_side="left",
        )
        tokenizer.pad_token = tokenizer.eos_token

        model = None
    else:
        model = None
        tokenizer = None
    return model, tokenizer


def run(quiz_data: Iterable[dict], save_dir: Path, skip_first: int, recursion_limit: int) -> None:
    """Run the quiz generation pipeline against the provided dataset."""

    from src.graph.builder import build_main, build_rag
    from src.graph.nodes.quiz_types import embeddings, reranker

    graph = build_main()
    model, tokenizer = build_generator_model()

    for idx, file_item in enumerate(tqdm(quiz_data)):
        if idx < skip_first:
            continue
        user_input = file_item["query"]

        graph.invoke(
            {
                "messages": [{"role": "user", "content": user_input}],
                "ori_query": user_input,
                "quiz_url": file_item["quiz_url"],
                "rag_graph": build_rag(),
                "search_before_planning": False,
                "generate_tokenizer": tokenizer,
                "generate_model": model,
                "rag": {
                    "embedding_model": embeddings,
                    "reranker_model": reranker,
                },
            },
            config={"recursion_limit": recursion_limit},
        )

from evaluate import evaluate_quiz


def _evaluation_tail() -> str:
    from src.config.llms import eval_llm_type, eval_model

    if eval_llm_type == "hkust":
        return ""
    return f"_{eval_llm_type}_{eval_model}"


def evaluate_dataset(quiz_data: Iterable[dict], save_dir: Path) -> None:
    print("开始evaluate")
    tail = _evaluation_tail()
    evaluate_quiz(quiz_data, f"{save_dir}/eval_result{tail}.jsonl")


def statistic(save_dir: Path) -> None:
    tail = _evaluation_tail()
    cnt = defaultdict(int)
    eval_res = getData(f"{save_dir}/eval_result{tail}.jsonl")
    for item in eval_res:
        for key in item['eval_res']:
            cnt[key] += item['eval_res'][key]
    for key in cnt:
        print(key, cnt[key] / len(eval_res))
    llama3_1 = getData(f"{save_dir}/eval_result_ollama_llama3.1:70b.jsonl")
    qwen3 = getData(f"{save_dir}/eval_result_ollama_qwen3:32b.jsonl")
    r1 = getData(f"{save_dir}/eval_result.jsonl")
    n = llama3_1 + qwen3 + r1
    exist_n = 0
    if len(llama3_1) > 0:
        exist_n += 1
    if len(qwen3) > 0:
        exist_n += 1
    if len(r1) > 0:
        exist_n += 1
    cnt_all = defaultdict(int)
    for item in llama3_1:
        for key in item['eval_res']:
            cnt_all[key] += item['eval_res'][key] / exist_n
    for item in qwen3:
        for key in item['eval_res']:
            cnt_all[key] += item['eval_res'][key] / exist_n
    for item in r1:
        for key in item['eval_res']:
            cnt_all[key] += item['eval_res'][key] / exist_n
    ave = 0
    for key in cnt_all:
        ave += cnt_all[key] / len(eval_res)
        print(key, cnt_all[key] / len(eval_res))
    print("平均分：", ave / len(cnt_all))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuizManus pipeline controller")
    parser.add_argument("--input-file", type=Path, default=DEFAULT_TEST_FILE, help="Path to the source dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SAVE_DIR, help="Directory to store generated quizzes and outputs")
    parser.add_argument("--skip-first", type=int, default=0, help="Number of records to skip before processing")
    parser.add_argument("--recursion-limit", type=int, default=100, help="LangGraph recursion limit")
    parser.add_argument("--cuda-device", type=str, default=None, help="CUDA device id to expose (unset by default)")
    parser.add_argument(
        "--mode",
        choices=["run", "test", "statistic"],
        default="run",
        help="Execution mode: run pipeline, evaluate results, or print statistics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()

    if args.cuda_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    logger = logging.getLogger(__name__)

    if args.mode == "statistic":
        statistic(args.output_dir)
        return

    quiz_data, annotated_path = prepare_quiz_dataset(args.input_file, args.output_dir)
    logger.info("Prepared quiz dataset copy at %s", annotated_path)

    if args.mode == "run":
        run(quiz_data, args.output_dir, args.skip_first, args.recursion_limit)
    elif args.mode == "test":
        evaluate_dataset(quiz_data, args.output_dir)


if __name__ == '__main__':
    main()
