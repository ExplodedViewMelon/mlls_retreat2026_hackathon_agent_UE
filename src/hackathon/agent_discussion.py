import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

from hackathon.autogen_client import get_client
from hackathon.hotpotqa import get_n_questions

client = get_client()

# get 1 question
hotpotqa_dataset_simple = get_n_questions(n_questions=10, n_titles=2)
dataset_row = hotpotqa_dataset_simple[3]  # 0 is impossible

agents = []
for article in dataset_row.wikipedia_articles:
    import re

    article_title_normalized = re.sub(r"[^A-Za-z0-9_-]", "_", article.title.replace(" ", "_"))
    article_title_normalized = re.sub(r"^[^A-Za-z_]+", "_", article_title_normalized)

    agent = AssistantAgent(
        name=f"expert_{article_title_normalized}",
        model_client=client,
        system_message=(
            "You are an expert on a particular wikipedia subject. "
            "You will find your relevant wikipedia material attached. "
            "You will help a group of agents answer a question. "
            "Each member of the group is an expert on different subjects. "
            "You will have to combine your information to reach an answer. "
            "You each have UNIQUE information and do NOT share the same attached article. "
            "You can trust the other agents. "
            "When your group has clearly reached a shared conclusion, write the word CONSENSUS. "
            f"\nAttached wikipedia article:\n\n{article.title}\n {article.content}"  # noqa
        ),
    )
    agents.append(agent)

termination = TextMentionTermination("CONSENSUS") | MaxMessageTermination(max_messages=12)


group_chat = RoundRobinGroupChat(
    participants=agents,
    termination_condition=termination,
)

TOPIC = dataset_row.question


async def main() -> None:
    print("Question:", dataset_row.question)

    print("=" * 60)
    print("ROUND-ROBIN DISCUSSION")
    print(f"Topic: {TOPIC}")
    print("=" * 60)
    print()

    stream = group_chat.run_stream(task=TOPIC)
    await Console(stream)

    print()
    print("=" * 60)
    print("Discussion ended.")

    print("Question:", dataset_row.question)
    print("Ground truth:", dataset_row.answer)


if __name__ == "__main__":
    asyncio.run(main())
