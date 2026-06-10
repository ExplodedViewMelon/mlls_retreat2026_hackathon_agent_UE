# pip install -U "autogen-agentchat" "autogen-ext[openai]"
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient


def get_client():
    base_url = "https://hackerton2026.compute.dtu.dk/v1"
    api_key = "REDACTED"

    client = OpenAIChatCompletionClient(
        model="google/gemma-4-26b-a4b",
        # model="alibaba/qwen-3.6-35b-a3b",
        api_key=api_key,
        base_url=base_url,
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            family="unknown",
            structured_output=True,
        ),
    )
    return client


async def hello_world() -> None:
    client = get_client()
    agent = AssistantAgent("assistant", client, system_message="")
    print(await agent.run(task="Say 'Hello World!'"))


if __name__ == "__main__":
    asyncio.run(hello_world())
