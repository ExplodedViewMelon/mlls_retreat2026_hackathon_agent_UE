import asyncio
import re

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console

from hackathon.autogen_client import get_client
from hackathon.hotpotqa import get_n_questions_distractor


def normalize_wikipedia_title(name: str) -> str:
    title_normalized = re.sub(r"[^A-Za-z0-9_-]", "_", name.replace(" ", "_"))
    title_normalized = re.sub(r"^[^A-Za-z_]+", "_", title_normalized)
    return title_normalized


client = get_client()

# get 1 question
hotpotqa_dataset_simple = get_n_questions_distractor(n_questions=10, n_titles=2)
dataset_row = hotpotqa_dataset_simple[1]

agents = []
for sentences in dataset_row.different_sentences:
    article_title_normalized = normalize_wikipedia_title(sentences.title)

    agent = AssistantAgent(
        name=f"expert_{article_title_normalized}",
        model_client=client,
        system_message=(
            "You are an expert on a particular wikipedia subject. "
            "You will find the relevant wikipedia material attached. "
            "You will help a group of agents answer a question. "
            "You are the ONLY agent with the attached information. "
            "The other agents have a DIFFERENT article attached. "
            "Therefore each member of the group is an expert on a different subject. "
            "You will have to share information to reach an answer. "
            "You can trust the other agents. "
            "When your group has clearly reached a shared conclusion, write the word CONSENSUS. "
            f"\nAttached wikipedia article:\n\n{sentences.title}\n {sentences.sentences}"  # noqa
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
    print("Articles:")
    for sentences in dataset_row.different_sentences:
        print(f"{sentences.title}")

    print("=" * 60)
    print("ROUND-ROBIN DISCUSSION")
    print(f"Topic: {TOPIC}")
    print("=" * 60)
    print()

    stream = group_chat.run_stream(task=TOPIC)
    last_message = await Console(stream)

    print()
    print("=" * 60)
    print("Discussion ended.")

    last_message_str = last_message.messages[-1].content  # type: ignore

    summary_agent = AssistantAgent("summary_agent", client)
    summarization = await summary_agent.run(
        task=(
            "Summarize the following answer as ultra compact keywords i.e. < 5 words. "
            f"Question: {dataset_row.question}. "
            f"Answer to summarize: {last_message_str} "
        )
    )

    print("Predicted answer summary:", summarization.messages[-1].to_text())
    print("Question:", dataset_row.question)
    print("Ground truth:", dataset_row.answer)


if __name__ == "__main__":
    asyncio.run(main())
