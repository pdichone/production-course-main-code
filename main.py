from dotenv import load_dotenv
from importlib.metadata import version

load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():

    # Test openai
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    response = llm.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatOpenAI: {response}")

    # Test anthropic
    # NOTE (2026-11): Claude Sonnet 4.5 retires 2026-11-30, so this uses claude-sonnet-5-5.
    # temperature=0 is dropped (Sonnet 5.5 rejects non-default temperature with a 400).
    # thinking="between_tools" keeps the plain-text reply shown in the video.
    llm_anthropic = ChatAnthropic(
        model="claude-sonnet-5-5", thinking={"type": "between_tools"}
    )
    response_anthropic = llm_anthropic.invoke("Say 'setup complete!' in one word")
    print(f"Response from ChatAnthropic: {response_anthropic}")

    print("Setup complete!")


if __name__ == "__main__":
    main()
