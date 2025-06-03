import os
from .ALL_KEYS import common_openai_key, common_openai_base_url

# API配置
openai_model = "Qwen/Qwen2.5-72B-Instruct"  # 修改为ModelScope支持的模型ID
openai_api_key = common_openai_key
openai_api_base = common_openai_base_url

# DeepSeek模型配置
deepseek_model = "deepseek-ai/DeepSeek-V3-0324"

# 模型类型配置
llm_type = "openai"  # openai ollama qwen

# 生成模型配置
generator_model = "deepseek"  # 使用 DeepSeek 模型作为生成器

# Ollama配置
ollama_model = "qwen2.5:72b"

# Embedding配置
embedding_model = "text-embedding-3-small"  # OpenAI的embedding模型
embedding_dimension = 1536  # embedding维度

# Reranker配置
reranker_model = "text-embedding-3-small"  # 使用相同的embedding模型进行重排序
reranker_top_k = 3  # 重排序返回的文档数量

# API调用配置
api_timeout = 300  # API调用超时时间（秒）- 增加到5分钟
api_max_retries = 2  # API调用最大重试次数 - 减少重试次数但增加间隔
api_retry_interval = 5  # 重试间隔时间（秒）- 增加间隔时间
api_batch_size = 10  # 批量处理的大小
