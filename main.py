import argparse
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoTokenizer, BitsAndBytesConfig
from vllm import LLM, SamplingParams
from langgraph.graph import MessagesState

from src.config.llms import eval_llm_type, eval_model, generator_model, qwen_tokenizer_path
from src.graph.builder import build_main, build_rag
from src.graph.nodes.quiz_types import embeddings, reranker
from src.utils import getData, saveData
from tqdm import tqdm

# 固定随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
# 如果是GPU环境
torch.cuda.manual_seed_all(seed)

load_dotenv()  # 加载 .env 文件

DEFAULT_TEST_FILE = Path("dataset/test.json")
DEFAULT_SAVE_DIR = Path("quiz_results/qwen_7b_full_quiz_gemini_40303")


## 配置logging
# import sys
# import logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)",
#     handlers=[
#         logging.StreamHandler(sys.stdout) # 输出到标准输出
#     ]
#     # force=True # Python 3.8+ 如果需要覆盖其他可能的早期配置
# )
import sys
import logging

# 创建 StreamHandler 实例
console_handler = logging.StreamHandler(sys.stdout)

# *** 在这里设置 StreamHandler 的级别为 INFO ***
console_handler.setLevel(logging.INFO)

# 配置 basicConfig，将设置好级别的 console_handler 传入
# basicConfig 的 level 仍然可以保留 DEBUG，这样如果以后你添加了其他 Handler (比如 FileHandler)
# 它们可以根据自己的设置来决定是否处理 DEBUG 消息
logging.basicConfig(
    level=logging.DEBUG, # Logger 的级别仍然是 DEBUG，可以捕获所有消息
    format="%(asctime)s - %(name)s - [%(levelname)s] - %(message)s (%(filename)s:%(lineno)d)",
    handlers=[
        console_handler # 将已设置好级别的 handler 传入
    ]
    # force=True # Python 3.8+ 如果需要覆盖其他可能的早期配置
)
def prepare_quiz_dataset(input_file: Path, save_dir: Path):
    """Prepare quiz data without mutating the source dataset."""

    os.makedirs(save_dir, exist_ok=True)
    quiz_data = getData(str(input_file))
    annotated_data = []
    for item in quiz_data:
        annotated_item = {
            **item,
            "quiz_url": str(save_dir / f"{item['id']}.md"),
        }
        annotated_data.append(annotated_item)

    annotated_path = save_dir / f"{input_file.stem}_with_quiz_urls.json"
    saveData(annotated_data, str(annotated_path))
    return annotated_data, annotated_path


def build_generator_model():
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


def run(quiz_data, save_dir: Path, skip_first: int, recursion_limit: int):
    graph = build_main()
    model, tokenizer = build_generator_model()

    for idx, file_item in enumerate(tqdm(quiz_data)):
        if idx < skip_first:
            continue
        user_input = file_item['query']

        graph.invoke({
            "messages": [{"role": "user", "content": user_input}],
            "ori_query": user_input,
            "quiz_url": file_item['quiz_url'],
            "rag_graph": build_rag(),
            "search_before_planning": False,
            "generate_tokenizer": tokenizer,
            "generate_model": model,
            "rag": {
                "embedding_model": embeddings,
                "reranker_model": reranker
            }
        },
        config={"recursion_limit": recursion_limit})

from evaluate import evaluate_quiz

if eval_llm_type == "hkust":
    tail = ""
else:
    tail = f"_{eval_llm_type}_{eval_model}"


def evaluate_dataset(quiz_data, save_dir: Path):
    print("开始evaluate")
    evaluate_quiz(quiz_data, f"{save_dir}/eval_result{tail}.jsonl")


def statistic(save_dir: Path):
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


def parse_args():
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


def main():
    args = parse_args()
    if args.cuda_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    quiz_data, annotated_path = prepare_quiz_dataset(args.input_file, args.output_dir)
    logger = logging.getLogger(__name__)
    logger.info("Prepared quiz dataset copy at %s", annotated_path)

    if args.mode == "run":
        run(quiz_data, args.output_dir, args.skip_first, args.recursion_limit)
    elif args.mode == "test":
        evaluate_dataset(quiz_data, args.output_dir)
    elif args.mode == "statistic":
        statistic(args.output_dir)


if __name__ == '__main__':
    main()
