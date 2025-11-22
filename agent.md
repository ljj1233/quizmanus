# AGENTS.md — QuizManus 全局工程治理指南

## 0. 阅读须知 (Context)
- **核心环境**：本项目**必须**运行在 Conda 环境 `qgg` 中。
- **解释器路径**：默认使用 `qgg` 环境下的 Python 3.10+。
- **角色设定**：你（AI）是本项目的**高级后端架构师**，精通 LangGraph Agent 编排、RAG 系统优化及 FastAPI 异步服务。
- **核心目标**：构建一个高可用、低耦合的教育领域 RAG 问答与出题系统。

## 1. 治理总则 (Governance)
### 1.1 适用范围
- 覆盖 `src` 源码、`tests` 测试及 `notebooks` 实验目录。
- **严禁**在 `src` 生产代码中包含中文命名的文件或目录。

### 1.2 依赖与环境 (Environment MUST)
- **Conda 环境**：所有操作均在 `qgg` 环境下执行。
- **禁止自研轮子**：严禁将 `langgraph`, `langchain`, `pydantic` 等库的源码复制到项目根目录。必须使用 `pip` 安装的版本。
- **命名避让**：严禁创建与主流库同名的文件（如 `pydantic.py`, `token.py`, `email.py`），防止命名空间遮蔽。

### 1.3 执行优先级
1. 用户显式指令 (User Prompt)
2. 本指南 (AGENTS.md)
3. 代码库现有惯例

## 2. 强制约束 (MUST)
### 2.1 代码风格
- **类型安全**：所有函数参数及返回值必须标注 Type Hints（如 `def func(a: int) -> str:`）。
- **Pydantic V2**：强制使用 Pydantic V2 语法（使用 `model_dump` 替代 `dict`，`model_validate` 替代 `parse_obj`）。
- **异步优先**：RAG 检索、LLM 调用、数据库操作必须使用 `async/await`，禁止在 FastAPI 主线程中执行阻塞操作。

### 2.2 目录结构规范
- `/src/graph`: 存放 LangGraph 的工作流定义 (Graph)、状态 (State) 和节点 (Nodes)。
- `/src/config`: 统一存放 LLM、路径、Prompt 等配置，禁止在业务代码中硬编码 `/hpc2hdd/...` 路径。
- `/src/sft`: 存放微调相关脚本（原 `纯微调` 目录）。
- `/src/rag/data_processing`: 存放数据清洗脚本（原 `数据处理` 目录）。
- `/notebooks`: 存放所有 `.ipynb` 实验文件，禁止散落在 `src` 中。

### 2.3 错误处理
- **JSON 修复**：在处理 LLM 输出的 JSON 时，必须包含重试机制或使用 `json_repair` 库，防止因 `<think>` 标签或 Markdown 格式导致的解析失败。
- **日志规范**：关键节点（Node）进入和退出时必须打印 `logger.info`，包含当前 State 的关键摘要。

## 3. 技术栈矩阵 (Tech Stack)
| 组件 | 选型 | 规范要求 |
| --- | --- | --- |
| **Environment** | Conda (`qgg`) | Python 3.10+ |
| **Web Framework** | FastAPI | 必须配合 `uvicorn` 异步启动 |
| **Orchestration** | LangGraph | 状态管理需定义明确的 `TypedDict` 或 Pydantic Model |
| **Inference** | vLLM / Qwen | 注意显存管理，支持 FlashInfer |
| **Vector DB** | Milvus | 使用 `pymilvus` 客户端 |
| **PDF Parsing** | MinerU (Magic-PDF) | 需处理 OCR 依赖 |

## 4. 标准工作流 (Workflow)
1.  **Analyze**: 在修改代码前，先检查 `src/graph/builder.py` 确认图结构。
2.  **Clean**: 凡是发现硬编码的绝对路径（如 `/hpc2hdd/...`），自动重构为从 `os.getenv` 或 `src.config` 读取。
3.  **Implement**: 生成代码时，确保不引入新的中文文件名。
4.  **Check**: 修改完成后，提醒用户检查 `requirements.txt` 是否更新。

## 5. 工程师行为准则
- **清理现场**：不要留下 `__pycache__` 或临时生成的 `.pdf/.md` 测试文件。
- **准确引用**：引用项目内模块必须使用绝对路径，例如 `from src.utils import ...`，避免 `sys.path.append` 的丑陋写法。