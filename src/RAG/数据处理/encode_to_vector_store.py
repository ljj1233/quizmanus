import os
import sys
import json
from tqdm import tqdm
from openai import OpenAI
from src.config.llms import (
    openai_api_key, openai_api_base, 
    embedding_model, api_timeout, 
    api_max_retries, api_batch_size
)
from src.config.rag import DB_URI, COLLECTION_NAME
from src.RAG.vector_store_utils import get_collection

def get_embeddings(texts, client, model=embedding_model):
    """使用OpenAI API获取文本的嵌入向量"""
    embeddings = []
    for text in tqdm(texts, desc="生成嵌入向量"):
        response = client.embeddings.create(
            model=model,
            input=text
        )
        embeddings.append(response.data[0].embedding)
    return embeddings

def main():
    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=api_timeout,
        max_retries=api_max_retries
    )

    # 读取数据
    data_path = "path/to/your/data.json"  # 请替换为实际的数据路径
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 准备文本
    all_res = []
    all_text = []
    for item in data:
        all_res.append({
            "subject": item['subject'].strip(),
            "grade": item['grade'].strip(),
            "title": item['title'].strip(),
            "content": item['课本内容'].strip()
        })
        all_text.append(item['content'].strip())

    # 获取嵌入向量
    docs_embeddings = get_embeddings(all_text, client)

    # 连接到向量数据库
    db_uri = DB_URI
    col_name = COLLECTION_NAME
    col = get_collection(db_uri, col_name)

    # 将数据插入向量数据库
    for i, (doc, embedding) in enumerate(zip(all_res, docs_embeddings)):
        col.insert({
            "id": i,
            "subject": doc["subject"],
            "grade": doc["grade"],
            "title": doc["title"],
            "content": doc["content"],
            "embedding": embedding
        })

    print("数据已成功编码并存储到向量数据库中")

if __name__ == "__main__":
    main() 