import asyncio
import re
from abc import ABC, abstractmethod
from typing import cast

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from pydantic import BaseModel

from hackathon.autogen_client import get_client
from hackathon.hotpotqa import Question_distractor, get_n_questions_distractor

client = get_client()


class AgentDiscussion(ABC):
    class Discussion(BaseModel):
        messages_str: str

    @staticmethod
    @abstractmethod
    async def perform_discussion(
        question_entry: Question_distractor, do_stream: bool
    ) -> Discussion: ...


async def llm_extract_answer(conversation: str, question: str) -> str:
    extractor_system_prompt = (
        "You are an answer extraction specialist. Given a question and a longer discussion"
        " or expert answer, extract the single most direct and concise answer.\n"
        "Rules:\n"
        "- Extract only the core answer "
        " — a word, name, number, short phrase, or at most 1–2 sentences\n"
        "- Do not include reasoning, explanation, or context unless it is essential to the answer\n"
        "- If the answer is a proper noun (person, place, organization), return just that noun\n"
        "- If the answer is a yes/no, return just 'Yes' or 'No'\n"
        "- Match the style of these examples:\n"
        "Question: What is the capital city of Australia?\n"
        "Answer: Canberra\n"
        "Question: The novel 'Frankenstein' was written by which author?\n"
        "Answer: Mary Shelley\n"
        "Question: Michael Jordan won six NBA championships, all with which team?\n"
        "Answer: The Chicago Bulls\n"
    )
    summary_agent = AssistantAgent("summary_agent", client, system_message=extractor_system_prompt)
    answer_summary_raw = await summary_agent.run(
        task=(f"Question: {question}. Conversation to extract answer from: {conversation} ")
    )
    return answer_summary_raw.messages[-1].to_text()


def normalize_wikipedia_title(name: str) -> str:
    title_normalized = re.sub(r"\W", "_", name, flags=re.ASCII)
    title_normalized = re.sub(r"^(?=\d)", "_", title_normalized)
    title_normalized = re.sub(r"_+", "_", title_normalized).strip("_")
    if not title_normalized:
        return "_"
    if title_normalized[0].isdigit():
        title_normalized = f"_{title_normalized}"
    return title_normalized


class TurnTakingFlat(AgentDiscussion):
    @staticmethod
    async def perform_discussion(
        question_entry: Question_distractor, do_stream: bool
    ) -> AgentDiscussion.Discussion:
        ##### Initialize agents

        TOPIC = question_entry.question

        n_agents = len(question_entry.different_sentences)
        client = get_client()
        agents = []
        for sentences in question_entry.different_sentences:
            article_title_normalized = normalize_wikipedia_title(sentences.title)
            for i, sentence in enumerate(sentences):
                agent = AssistantAgent(
                    name=f"expert_{article_title_normalized}_{i}",
                    model_client=client,
                    system_message=(
                        "You are an expert on a particular wikipedia subject. "
                        "You will find the relevant wikipedia material attached. "
                        "You will help a group of agents answer a question. "
                        "You are the ONLY agent with the attached information. "
                        "The other agents have DIFFERENT information attached. "
                        "Multiple agents may have information about the same subject. "
                        "Therefore each member of the group is an expert on a different subject. "
                        "You will have to share information to reach an answer. "
                        "You can trust the other agents. "
                        "When your group has clearly reached a shared conclusion, "
                        "write the answer following the word CONSENSUS. "
                        "Only write the word CONSENSUS when the task is over. "
                        "All agents share this system prompt. "
                        f"You are in total {n_agents} agents. "
                        "Make sure to hear everyone's opinion before submitting the answer."
                        f"\nAttached wikipedia article:\n\n{sentences.title}\n {sentence}"  # noqa
                    ),
                )
                agents.append(agent)

        termination = TextMentionTermination("CONSENSUS") | MaxMessageTermination(max_messages=12)

        group_chat = RoundRobinGroupChat(
            participants=agents,
            termination_condition=termination,
        )

        ##### Start discussion
        if do_stream:
            group_chat_stream = group_chat.run_stream(task=TOPIC)
            result = await Console(group_chat_stream)
        else:
            result = await group_chat.run(task=TOPIC)

        messages = cast(list[BaseChatMessage], result.messages)

        messages_str = ""
        for message in messages:
            agent = message.source
            content = message.content  # type: ignore

            messages_str += f"{agent}:\n"
            messages_str += f"{content}\n\n"

        return AgentDiscussion.Discussion(messages_str=messages_str)


async def single_run() -> None:
    # get 1 data row
    hotpotqa_dataset_simple = get_n_questions_distractor(n_titles=4)
    question_entry = hotpotqa_dataset_simple[1]

    do_stream = True
    run_result = await TurnTakingFlat.perform_discussion(question_entry, do_stream=do_stream)

    answer = await llm_extract_answer(run_result.messages_str, question_entry.question)

    print("#" * 10)
    print("Titles:")
    for sentences in question_entry.different_sentences:
        print(" -", sentences.title)
    print("Question:", question_entry.question)
    print("Ground truth:", question_entry.answer)
    print("Prediction summary:", answer)

    # # display
    # print("Question:", dataset_row.question)
    # print("Articles:")
    # for sentences in dataset_row.different_sentences:
    #     print(f"{sentences.title}")

    # if do_stream:
    #     print("=" * 60)
    #     print("ROUND-ROBIN DISCUSSION")
    #     print(f"Topic: {dataset_row.question}")
    #     print("=" * 60)
    #     print()

    #     print()
    #     print("=" * 60)
    #     print("Discussion ended.")

    # print("Predicted answer summary:", run_results.answer_summary)
    # print("Question:", dataset_row.question)
    # print("Ground truth:", dataset_row.answer)


if __name__ == "__main__":
    asyncio.run(single_run())
