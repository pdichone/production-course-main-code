# Production AI Agents with LangChain & LangGraph: Course Code

This is the code for the course **Production AI Agents with LangChain & LangGraph**.
Every lecture has a script here. Run it, break it, change it.

This is the maintained repo for the course. Fixes land here first.

## 1. Setup (5 minutes)

You need **Python 3.12+** and **[uv](https://docs.astral.sh/uv/)**.

```bash
git clone https://github.com/pdichone/production-course-main-code.git
cd production-course-main-code
uv sync                 # creates .venv and installs the exact versions in uv.lock
cp .env.example .env    # then paste your keys into .env
```

Check that it works:

```bash
uv run python main.py   # calls OpenAI and Anthropic once each, prints "Setup complete!"
```

## 2. API keys

Put these in `.env`. Never commit `.env`; it is already in `.gitignore`.

| Key | Needed for | Required? |
|---|---|---|
| `OPENAI_API_KEY` | Most lectures (`gpt-6-luna`, OpenAI embeddings) | Yes |
| `ANTHROPIC_API_KEY` | Claude examples (`main.py`, `working_with_llms.py`, `error_handling.py`) | Recommended |
| `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, `LANGSMITH_TRACING` | Tracing and observability lectures | Optional |

Some embeddings demos can also use a local model through [Ollama](https://ollama.com). No key needed, but Ollama must be running.

## 3. Run a lecture

```bash
uv run python <file>.py
```

Many files have several demo functions. Open the file, look at the `if __name__ == "__main__":` block at the bottom, and uncomment the demo you want.

## 4. What's in here

| Topic | Files |
|---|---|
| Setup and LangChain basics | `main.py`, `core_concepts.py`, `working_with_llms.py`, `chains_v1.py` |
| Prompts and output parsing | `prompt_messages.py`, `prompt_templates_all.py`, `output_parsers_demo.py`, `output_parsers_final.py` |
| Memory | `conversation_memory.py` |
| Section 1 project | `smart_bot_section1.py` |
| Documents, embeddings, RAG | `document_loaders.py`, `text_splitters.py`, `embeddings.py`, `embeddings_deep.py`, `vector_stores.py`, `rag_pipeline.py`, `advanced_rag.py` |
| Section 2 project | `research_assistant.py` |
| LangGraph fundamentals | `langgraph_core.py`, `first_graph.py`, `conditional_edges.py`, `cycles_loops.py`, `checkpointing.py`, `human_in_loop.py`, `tool_calling_agent.py` |
| Multi-agent systems | `multi-agent.py`, `supervisor_agent.py`, `hierarchical_agents.py`, `agent_handoffs.py`, `agent_communication.py`, `parallel_agents.py` |
| Section 4 project | `multi_agent_research_system.py` |
| Production patterns | `langsmith_setup.py`, `monitoring.py`, `error_handling.py`, `cost_optimization.py`, `security_patterns.py`, `testing_patterns.py` |

`docs/langchain_demo.pdf` is the sample document the loader and RAG lectures use.

## 5. Model updates

Model providers retire models. When that happens, the code here gets updated and may differ
slightly from the video. Look for comments that start with `# NOTE (2026-11):` - they explain
what changed and why.

Current Claude model: `claude-sonnet-5-5` (Claude Sonnet 4.5 retires on 2026-11-30).
Claude Sonnet 5.5 rejects a non-default `temperature`, so Claude examples no longer pass one.

Current OpenAI model: `gpt-6-luna` (OpenAI shuts down `gpt-4o` and `gpt-4o-mini` on 2027-01-14).
The premium tier in the routing/fallback demos uses `gpt-6-sol`. Both need `reasoning_effort="none"`
to accept `temperature` or tools, so every OpenAI call passes it.

## 6. Troubleshooting

- **`ModuleNotFoundError`**: run `uv sync` again, and run scripts with `uv run python ...`.
- **`greenlet` / SQLAlchemy error in `conversation_memory.py`**: pull the latest code and run `uv sync`. `greenlet` is now a direct dependency.
- **`401` / authentication error**: check the key in `.env` and that there are no quotes or spaces around it.
- **Anthropic `400` mentioning `temperature`**: remove `temperature=` from the `ChatAnthropic` call.
- **OpenAI `400` "Only the default (1) value is supported" or "Function tools with reasoning_effort are not supported"**: add `reasoning_effort="none"` to the `ChatOpenAI` / `init_chat_model` call.
- **OpenAI `404` model not found for `gpt-4o-mini`**: that model is shut down. Use `gpt-6-luna` plus `reasoning_effort="none"`.
