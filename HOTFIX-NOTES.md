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
