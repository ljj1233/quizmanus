import os
from src.graph.builder import build_rag, build_main
from langgraph.graph import MessagesState
from dotenv import load_dotenv
from src.graph.nodes.quiz_types import embeddings, reranker
from src.config.llms import (
    openai_model, openai_api_key, openai_api_base, 
    llm_type, generator_model
)
import os
import numpy as np
from src.utils import getData, get_json_result, saveData
from tqdm import tqdm

# 固定随机种子
seed = 42
np.random.seed(seed)

load_dotenv()  # 加载 .env 文件

def run():
    graph = build_main()
    
    # 使用API配置
    api_config = {
        "model": openai_model,
        "api_key": openai_api_key,
        "api_base": openai_api_base,
        "llm_type": llm_type
    }

    test_file_path = "dataset/test.json"
    save_dir = "quiz_results/qwen_14b_quiz_1072"
    os.makedirs(save_dir, exist_ok=True)
    tmp_test = getData(test_file_path)
    for item in tmp_test:
        item['quiz_url'] = os.path.join(save_dir, f"{item['id']}.md")
    saveData(tmp_test, test_file_path)
    
    for idx, file_item in tqdm(enumerate(getData(test_file_path))):
        user_input = file_item['query']

        graph.invoke({
            "messages": [{"role": "user", "content": user_input}],
            "ori_query": user_input,
            "quiz_url": file_item['quiz_url'],
            "rag_graph": build_rag(),
            "search_before_planning": False,
            "api_config": api_config,
            "rag": {
                "embedding_model": embeddings,
                "reranker_model": reranker
            }
        },
        config={"recursion_limit": 100})

from evaluate import evaluate_quiz

import json
def test():
    print("开始evaluate")
    evaluate_quiz(getData("dataset/test.json"),"quiz_results/qwen_14b_quiz_1072/eval_result.jsonl")


from collections import *
def statistic():
    cnt = defaultdict(int)
    eval_res = getData("quiz_results/qwen_14b_quiz_1072/eval_result.jsonl")
    for item in eval_res:
        for key in item['eval_res']:
            cnt[key]+=item['eval_res'][key]
    for key in cnt:
        print(key,cnt[key]/len(eval_res))
# run()
test()
statistic()