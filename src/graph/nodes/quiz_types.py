# from typing_extensions import TypedDict
from langgraph.graph import MessagesState, StateGraph
from typing import List, Dict, TypedDict, Literal, Optional, Annotated, Union
from operator import add
from langgraph.graph import StateGraph
from openai import OpenAI
from src.config.llms import openai_api_key, openai_api_base
from pymilvus.model.embedding import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction

# 初始化OpenAI客户端
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base
)

# 初始化BGE模型
embeddings = BGEM3EmbeddingFunction(
    model_name = "/hpc2hdd/home/fye374/models/BAAI/bge-m3",
    use_fp16=False, 
    device="cuda"
)

# 初始化BGE重排序模型
reranker = BGERerankFunction(
    model_name="/hpc2hdd/home/fye374/models/BAAI/bge-reranker-v2-m3",  
    device="cuda",
    use_fp16=False
)

# 以下是旧的实现，保留为注释以供参考
"""
class EmbeddingFunction:
    def __init__(self, model_name="text-embedding-3-small"):
        self.model_name = model_name
        self.client = client

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """使用OpenAI API进行文本嵌入，批量调用"""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [item.embedding for item in response.data]

class RerankFunction:
    def __init__(self, model_name="text-embedding-3-small"):
        self.model_name = model_name
        self.client = client

    def __call__(self, query: str, documents: List[str], top_k: int = 3) -> List[str]:
        """使用OpenAI API进行文档重排序"""
        # 获取查询的嵌入向量
        query_embedding = self.client.embeddings.create(
            model=self.model_name,
            input=[query]
        ).data[0].embedding

        # 获取文档的嵌入向量（批量）
        doc_embeddings = self.client.embeddings.create(
            model=self.model_name,
            input=documents
        ).data

        # 计算相似度并排序
        similarities = []
        for doc_embedding in doc_embeddings:
            similarity = self._cosine_similarity(query_embedding, doc_embedding.embedding)
            similarities.append(similarity)

        # 获取top_k个最相关的文档
        top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
        return [documents[i] for i in top_indices]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        import numpy as np
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 初始化embedding和reranker
embeddings = EmbeddingFunction()
reranker = RerankFunction()
"""

class RAGState(MessagesState):
    
    hyde_query: str
    selected_subject: str
    retrieved_docs: List[str]
    reranked_docs: List[str]
    embedding_model: BGEM3EmbeddingFunction
    reranker_model: BGERerankFunction
    enable_browser: bool
    outer_knowledge: str


class State(MessagesState):
    """State for the agent system, extends MessagesState with next field."""

    # # Constants
    # TEAM_MEMBERS: list[str]
    # TEAM_MEMBER_CONFIGRATIONS: dict[str, dict]

    # Runtime Variables
    # messages: Annotated[List,add]
    ori_query: str
    rag_graph: StateGraph
    existed_qa: Annotated[List,add]
    next: str
    full_plan: str
    deep_thinking_mode: bool
    search_before_planning: bool
    next_work: str
    rag: RAGState
    quiz_url: str


