import os
from pathlib import Path

from vllm import SamplingParams

from .env import env_or_all_keys, get_common_openai_settings, get_path_from_env

openai_model = os.getenv("OPENAI_MODEL", "deepseek-v3-250324")
common_openai = get_common_openai_settings()
openai_api_key = common_openai.api_key
openai_api_base = common_openai.base_url
llm_type = os.getenv("LLM_TYPE", "ollama")  # openai ollama qwen gemini

outer_knowledge_llm_type = os.getenv("OUTER_KNOWLEDGE_LLM_TYPE", "gemini")
planner_llm_type = os.getenv("PLANNER_LLM_TYPE", "gemini")
reporter_llm_type = os.getenv("REPORTER_LLM_TYPE", "gemini")
supervisor_llm_type = os.getenv("SUPERVISOR_LLM_TYPE", "gemini")
gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
gemini_api_key = env_or_all_keys("GEMINI_API_KEY", "common_openai_key", common_openai.api_key)
gemini_api_base = env_or_all_keys("GEMINI_API_BASE", "common_openai_base_url", common_openai.base_url)

generator_model = os.getenv("GENERATOR_MODEL", "qwen")  # gemini qwen

_project_root = Path(__file__).resolve().parents[2]
_default_model_path = _project_root / "models" / "qwen2.5-7b-lora-gaokao-60265"
_default_tokenizer_path = _project_root / "models" / "Qwen" / "Qwen2.5-7B-Instruct"
qwen_model_path = get_path_from_env("QWEN_MODEL_PATH", str(_default_model_path))
qwen_tokenizer_path = get_path_from_env("QWEN_TOKENIZER_PATH", str(_default_tokenizer_path))

vllm_sampling_params = SamplingParams(
    temperature=0.1,
    top_p=1,
    max_tokens=1024,
    # num_beams=1,    # 如需 beam search 可加
)

ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:30b")
ollama_num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "25000"))


# eval_model = "qwen3:32b"
eval_model = os.getenv("EVAL_MODEL", "llama3.1:70b")
eval_llm_type = os.getenv("EVAL_LLM_TYPE", "ollama")  # openai ollama hkust

# ollama_num_ctx = 25600
# eval_llm_type = "hkust" #openai ollama hkust
