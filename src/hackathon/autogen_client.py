# pip install -U "autogen-agentchat" "autogen-ext[openai]"
import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

from hackathon.settings import settings


def get_client() -> OpenAIChatCompletionClient:
    base_url = "https://hackerton2026.compute.dtu.dk/v1"

    client = OpenAIChatCompletionClient(
        model="google/gemma-4-26b-a4b",
        # model="alibaba/qwen-3.6-35b-a3b",
        base_url=base_url,
        api_key=settings.llm_token,
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
