import os
import logging
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
import signal
import sys
import atexit
import time
from pymilvus import connections

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quizmanus.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("quizmanus")

# 固定随机种子
seed = 42
np.random.seed(seed)

load_dotenv()  

# 定义清理函数
def cleanup_resources():
    """清理程序使用的资源，避免异常退出"""
    print("正在清理资源...")
    try:
        # 显式释放BGE模型资源
        if hasattr(embeddings, '_pool') and embeddings._pool is not None:
            embeddings._pool.shutdown()
            embeddings._pool = None
        
        if hasattr(reranker, '_pool') and reranker._pool is not None:
            reranker._pool.shutdown()
            reranker._pool = None
            
        # 关闭数据库连接
        try:
            connections.disconnect("default")
            print("数据库连接已关闭")
        except Exception as e:
            print(f"关闭数据库连接时出错: {e}")
            
        # 确保在退出前等待所有异步任务完成
        time.sleep(1)
        
        print("资源清理完成")
    except Exception as e:
        print(f"清理资源时出错: {e}")

# 注册清理函数，在程序退出时调用
atexit.register(cleanup_resources)

# 注册信号处理，确保在Ctrl+C等情况下也能清理资源
def signal_handler(sig, frame):
    print('接收到终止信号，正在清理...')
    cleanup_resources()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run():
    logger.info("开始构建主图")
    graph = build_main()
    
    # 使用API配置
    api_config = {
        "model": openai_model,
        "api_key": openai_api_key,
        "api_base": openai_api_base,
        "llm_type": llm_type
    }
    logger.info(f"使用模型配置: {api_config['model']}, 类型: {api_config['llm_type']}")

    # 读取测试数据
    test_file_path = "dataset/test.json"
    save_dir = "quiz_results/qwen_14b_quiz_1072"
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"读取测试数据: {test_file_path}")
    
    # 尝试读取测试数据，如果文件不存在则创建示例数据
    try:
        tmp_test = getData(test_file_path)
        logger.info(f"测试数据包含 {len(tmp_test)} 条记录")
    except Exception as e:
        logger.warning(f"读取测试数据失败: {e}，创建示例数据")
        tmp_test = [{
            "id": "example_001",
            "query": "生成18道人教版高中生物（必修一：分子与细胞基础）的题目，包含5道单选题、5道填空题、5道判断题和3道主观题，需涵盖细胞结构、酶活性和细胞呼吸知识点"
        }]
        os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
        saveData(tmp_test, test_file_path)
    
    # 为每条记录设置输出路径
    for item in tmp_test:
        item['quiz_url'] = os.path.join(save_dir, f"{item['id']}.md")
    saveData(tmp_test, test_file_path)
    
    # 处理每条记录
    for idx, file_item in tqdm(enumerate(getData(test_file_path))):
        user_input = file_item['query']
        logger.info(f"处理第 {idx+1}/{len(tmp_test)} 条记录, ID: {file_item['id']}")
        logger.info(f"查询内容: {user_input[:100]}...")
        logger.info(f"输出路径: {file_item['quiz_url']}")
        
        try:
            logger.info("构建RAG图")
            rag_graph = build_rag()
            logger.info("开始调用图处理")
            
            graph.invoke({
                "messages": [{"role": "user", "content": user_input}],
                "ori_query": user_input,
                "quiz_url": file_item['quiz_url'],
                "rag_graph": rag_graph,
                "search_before_planning": False,
                "api_config": api_config,
                "rag": {
                    "embedding_model": embeddings,
                    "reranker_model": reranker
                }
            },
            config={"recursion_limit": 100})
            
            logger.info(f"成功处理记录 ID: {file_item['id']}")
        except Exception as e:
            logger.error(f"处理记录 ID: {file_item['id']} 时出错: {str(e)}", exc_info=True)

from evaluate import evaluate_quiz

import json
def test():
    logger.info("开始评估")
    evaluate_quiz(getData("dataset/test.json"),"quiz_results/qwen_14b_quiz_1072/eval_result.jsonl")
    logger.info("评估完成")


from collections import *
def statistic():
    logger.info("开始统计评估结果")
    cnt = defaultdict(int)
    eval_res = getData("quiz_results/qwen_14b_quiz_1072/eval_result.jsonl")
    for item in eval_res:
        for key in item['eval_res']:
            cnt[key]+=item['eval_res'][key]
    for key in cnt:
        result = cnt[key]/len(eval_res)
        logger.info(f"指标 {key}: {result}")
        print(key, result)
    logger.info("统计完成")

if __name__ == "__main__":
    # 调用run函数处理测试数据集
    run()
    # test()
    # statistic()