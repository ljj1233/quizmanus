# QuizManus Pipeline Handbook

This document describes the end-to-end workflow of the QuizManus generation system, covering the CLI entrypoints, LangGraph topology, and the responsibilities of the key nodes that cooperate to plan, generate, validate, and report quiz items.

## Entry Execution Flow (`main.py`)

### Argument parsing and environment setup
- `parse_args()` defines the supported modes (`run`, `test`, `statistic`), dataset paths, recursion limits, and optional CUDA exposure flags.
- `configure_logging()` installs a single console logger at startup so downstream modules do not reconfigure logging.
- When `--cuda-device` is provided, the value is written to `CUDA_VISIBLE_DEVICES` before any model code is invoked.
- The module seeds Python and PyTorch RNGs, then loads environment variables via `load_dotenv()`.

### Dataset preparation and invocation
- `prepare_quiz_dataset()` copies the source dataset and appends a `quiz_url` for each record, persisting the annotated copy alongside the target output directory.
- `build_generator_model()` lazily constructs the tokenizer/model pair based on `src.config.llms.generator_model`, deferring heavy imports until needed.
- `run()` builds the LangGraph pipelines (`build_main()`, `build_rag()`), loops over the prepared dataset, and invokes the graph with per-item state (messages, original query, quiz URL, RAG config, and recursion limits).
- `evaluate_dataset()` triggers `evaluate_quiz()` on the generated quizzes; `statistic()` aggregates evaluation JSONL outputs across multiple models and prints averaged scores.

## Graph Topology (`src/graph/builder.py`)

### Main workflow graph
- `build_main()` wires the orchestrator graph that coordinates planning, generation, validation, and reporting. Nodes: `coordinator` → `planner` → `supervisor`, plus callable generator nodes (`rag_er`, `rag_and_browser`), a `critic`, and the `reporter`. The compiled graph enforces these names for supervisor routing decisions.

### RAG subgraph
- `build_rag()` constructs the retrieval-and-generation subgraph used by generator nodes: `rewrite` → `retriever` → `router` → `reranker` → `generator`. This subgraph can run synchronously or asynchronously (via `ainvoke`) depending on the calling node.

## Node Responsibilities (`src/graph/nodes/nodes.py`)

### Coordinator
- Reads the original user query and decides whether to hand off to the planner. It sanitizes model output by stripping `<think>` blocks, then routes either to `planner` or ends the conversation early.

### Planner
- Uses the planner prompt to produce a `QuizPlan` with a subject and a list of `PlanStep`s. The plan is validated against configured subjects and must include at least one generator step.
- Executes generator steps concurrently via `run_generator_concurrently()`, which spawns `_generate_single()` tasks to call the RAG subgraph. Generator outputs include question payloads and summarized fingerprints for the supervisor.
- Updates state with the serialized plan, the messages emitted by generator agents, planned and pending generator steps, and subject metadata for downstream nodes.

### Generator nodes (`main_rag`, `main_rag_browser`)
- Invoke the shared RAG subgraph with browser disabled (`rag_er`) or enabled (`rag_and_browser`), respectively.
- Capture the latest question content and fingerprint (`latest_fingerprint`), synthesizing a concise summary via `_summarize_fingerprint()` for supervisor consumption.
- Transition control to the `critic` after each generation.

### Critic
- Validates each generated question before it reaches the supervisor. Checks include: non-empty question content, fingerprint deduplication against `question_fingerprints`, and difficulty validation when provided.
- On pass: appends the fingerprint and returns to the supervisor. On rejection: records retry counts per generator step, emits feedback, and either reroutes back to the appropriate generator (respecting `MAX_CRITIC_RETRIES`) or yields to the supervisor for manual decisions.

### Supervisor
- Applies the supervisor prompt with a trimmed message history (last 10 entries) and enforces structured outputs via `SupervisorDecision` parsing. Responses lacking required fields fall back to `DEFAULT_SUPERVISOR_DECISION`.
- When the decision is `FINISH`, it calls `fill_missing_questions()` to generate any remaining planned items or cover missing knowledge points. The supervisor refuses to finish if required counts or knowledge points are unmet, rerouting to the appropriate generator with updated `next_work` instructions.
- On successful completion, writes the reporter content (or the latest message content) to `quiz_url`.

### Reporter
- Builds a concise payload from `meta_history` fingerprints (or the raw questions) and uses the reporter prompt to synthesize the final quiz report. The reporter’s response is returned to the supervisor for finalization.

## Supporting Utilities

- Fingerprint summaries: `_summarize_fingerprint()` renders topic/focus/type metadata into short strings for message passing and supervision.
- Missing-point handling: `_merge_missing_points()` merges supervisor-detected gaps with required points tracked in state; `_build_next_work()` crafts concrete instructions for generators based on pending steps and missing knowledge.
- Retry tracking: `_step_key()` and `generator_retry_counts` ensure the critic can cap retries per generator step before escalating to the supervisor.

## RAG Components (`src/graph/nodes/rag_nodes.py`, `src/graph/nodes/quiz_types.py`)

- The RAG pipeline stages (`rag_hyde`, `rag_retrieve`, `rag_router`, `rag_reranker`, `rag_generator`) are defined in `rag_nodes.py` and rely on shared `State` definitions from `quiz_types.py`. They handle query rewriting, retrieval, reranking, and final question drafting used by generator nodes.

## Testing Notes

- Core behaviors of the CLI entrypoint and utilities are covered by `tests/test_main.py` and `tests/test_utils.py`, respectively. Tests can be run with `pytest -q` from the repository root.
