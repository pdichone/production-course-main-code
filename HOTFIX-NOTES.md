# Hotfix notes: Claude Sonnet 4.5 -> Claude Sonnet 5.5 (2026-11)

Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) retires on 2026-11-30. The Claude examples now
use `claude-sonnet-5-5`. This file lists where the code no longer matches the recorded video.

## Applies to every Claude example

Each `ChatAnthropic` / `init_chat_model(..., model_provider="anthropic")` call now passes
`thinking={"type": "between_tools"}`. Sonnet 5.5 thinks by default, which makes LangChain return
`response.content` as a list of blocks instead of a string. `between_tools` is Sonnet 5.5's
thinking-off setting, so the output looks like the video. Sonnet 5.5 only - remove it if you
switch to another Claude model.

## Lectures whose behavior changed

| File | Line(s) | Video shows | Code does now |
|---|---|---|---|
| `main.py` | 27-29 | `ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)` | `claude-sonnet-5-5`, no `temperature`. Sonnet 5.5 returns a 400 for any non-default temperature. Setup check only, output unchanged. |
| `working_with_llms.py` | 34-40 | `init_chat_model(..., temperature=0.7, streaming=True, max_retries=3)` for Claude, to show the same params work across providers | `claude-sonnet-5-5`, no `temperature` (400 on Sonnet 5.5). The OpenAI calls keep `temperature=0.7`. The "swap providers, keep params" point now has an exception: newer Claude models reject non-default sampling params. |
| `working_with_llms.py` | 64-70 | Comparison dict key and model `claude-sonnet-4-5-20250929`, `temperature=0.7` | Key and model `claude-sonnet-5-5`, no `temperature`. The comparison is no longer at equal temperature. |
| `error_handling.py` | 162-170 | Fallback chain: Claude at `temperature=0` as the last fallback | `claude-sonnet-5-5`, no `temperature`. Claude's fallback answer is no longer pinned to temperature 0. |
| `core_concepts.py` | 129-133 | `ChatAnthropic(model="claude-sonnet-4-5-20250929")` (created, not invoked) | `claude-sonnet-5-5` with `between_tools`. No visible change. |

## Other fixes in this branch

- `pyproject.toml` / `uv.lock`: `greenlet>=3.3.1` added as a direct dependency. SQLAlchemy only
  pulls `greenlet` in on some CPU types (not Apple Silicon `arm64`), so a fresh install on an
  M-series Mac broke `conversation_memory.py` (`SQLChatMessageHistory`).
- `README.md`: written (was empty) - setup, keys, file map, troubleshooting.
- `.env.example`: added, lists `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and the optional LangSmith keys.

## Checked, no change needed

- No assistant prefill, `budget_tokens`, or forced `tool_choice` on Claude.
  (`with_structured_output` forces a tool in LangChain - it is only used with OpenAI models here.)
- No retired `claude-3*` model IDs.

---

# Hotfix notes: gpt-4o / gpt-4o-mini -> gpt-6-luna (2026-11)

OpenAI shuts down `gpt-4o`, `gpt-4o-mini`, `gpt-4.1` and `gpt-4.1-mini` on 2027-01-14. Its listed
replacement is `gpt-6-luna`. Every OpenAI chat model in this repo now uses it. Embedding models
(`text-embedding-3-*`) are not affected and did not change.

## Applies to every OpenAI example

The video shows `ChatOpenAI(model="gpt-4o-mini", temperature=0)` (or `init_chat_model("gpt-4o-mini", ...)`).
The code now shows `ChatOpenAI(model="gpt-6-luna", reasoning_effort="none", temperature=0)`.

`gpt-6-luna` is a reasoning model. Without `reasoning_effort="none"` it returns a 400 for any
`temperature` other than 1, and a 400 for tool calling on Chat Completions. With it, `temperature`,
`bind_tools`, `with_structured_output`, streaming and `max_tokens` all work as in the video
(LangChain sends `max_tokens` as `max_completion_tokens` for you). Tested live on the locked stack.

No lecture uses a `stop=` sequence or a text-based ReAct agent, so no agent type changed.

## Lectures whose behavior changed

| File | Video shows | Code does now |
|---|---|---|
| `working_with_llms.py` (1.4) | Comparison of `gpt-4o-mini` vs `gpt-4o`; exercise `models=["gpt-4o-mini", "gpt-4o"]` | `gpt-6-luna` vs `gpt-6-sol` (OpenAI's current small and large GPT-6 models, both with `reasoning_effort="none"`). Dict keys and printed labels changed. |
| `cost_optimization.py` (5.5) | Router: simple -> `gpt-4o-mini` ($0.00015/1K input), complex -> `gpt-4o` ($0.0025/1K) | Simple -> `gpt-6-luna` ($0.0001/1K input, $0.10/1M), complex -> `gpt-6-sol` ($0.002/1K input, $2.00/1M). Printed model names and cost estimates differ. |
| `error_handling.py` (5.4) | Fallback chain `gpt-4o-mini` -> `gpt-4o` -> Claude | `gpt-6-luna` -> `gpt-6-sol` -> `claude-sonnet-5-5`. Same shape: cheap OpenAI, premium OpenAI, then a different provider. Labels in the output changed. |
| `smart_bot_section1.py` (1.6) | `SmartQABot(model_name="gpt-4o-mini", temperature=0.3)` | Default `gpt-6-luna`; `reasoning_effort="none"` added inside the class so the `temperature` argument still works. |
| `conversation_memory.py` (2.8, trimming demo) | `trim_messages(..., token_counter=llm)` | `token_counter=count_tokens_approximately`. langchain-openai has no tokenizer for `gpt-6-luna` yet and raises `NotImplementedError`. The approximate counter keeps a slightly different number of messages than the video. |
| `main.py` (1.2) | `ChatOpenAI(model_name="gpt-4o-mini", temperature=0)` | `gpt-6-luna` + `reasoning_effort="none"`. The printed `response_metadata` shows the new model name. |

Everywhere else the only visible change is the model name and the extra argument. Answers are
worded differently from the recordings, as you'd expect from a new model.

## Checked

- All 33 changed lesson files ran end to end against the live API (default `__main__` demo),
  plus the `conversation_memory.py` trimming demo. All passed.
- `text-embedding-3-small` is not on OpenAI's deprecation list: unchanged.
