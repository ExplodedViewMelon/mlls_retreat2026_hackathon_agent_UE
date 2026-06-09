import asyncio
import re
from typing import Sequence

import tqdm
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from pydantic import BaseModel

from hackathon.autogen_client import get_client
from hackathon.hotpotqa import Question_distractor, get_n_questions_distractor


def normalize_wikipedia_title(name: str) -> str:
    title_normalized = re.sub(r"[^A-Za-z0-9_-]", "_", name.replace(" ", "_"))
    title_normalized = re.sub(r"^[^A-Za-z_]+", "_", title_normalized)
    return title_normalized


class SingleRun(BaseModel):
    dataset_row: Question_distractor
    messages: Sequence[BaseAgentEvent | BaseChatMessage]
    answer_summary: str


async def process_question(
    client, dataset_row: Question_distractor, do_stream: bool = True
) -> SingleRun:
    ##### Initialize agents

    TOPIC = dataset_row.question
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
                "When your group has clearly reached a shared conclusion, "
                "write the answer following the word CONSENSUS. "
                f"\nAttached wikipedia article:\n\n{sentences.title}\n {sentences.sentences}"  # noqa
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

    messages = result.messages

    last_message = messages[-1].content  # type: ignore

    summary_agent = AssistantAgent("summary_agent", client)
    answer_summary_raw = await summary_agent.run(
        task=(
            "Summarize the following answer as ultra compact keywords i.e. < 10 words. "
            f"Question: {dataset_row.question}. "
            f"Answer to summarize: {last_message} "
        )
    )
    answer_summary = answer_summary_raw.messages[-1].to_text()

    return SingleRun(dataset_row=dataset_row, messages=messages, answer_summary=answer_summary)


class BenchmarkResult(BaseModel):
    single_runs: list[SingleRun]


async def main() -> None:
    client = get_client()

    # get 1 data row
    hotpotqa_dataset_simple = get_n_questions_distractor(n_titles=2)
    hotpotqa_dataset_simple = hotpotqa_dataset_simple[:]
    do_stream = False

    single_runs: list[SingleRun] = []

    print("Starting benchmark")
    for dataset_row in tqdm.tqdm(
        hotpotqa_dataset_simple, desc="Benchmarking", total=len(hotpotqa_dataset_simple)
    ):
        # process
        try:
            run_result = await process_question(client, dataset_row, do_stream=do_stream)
            single_runs.append(run_result)
        except Exception:
            pass

    benchmark_result = BenchmarkResult(single_runs=single_runs)
    with open("benchmark_result.json", "w", encoding="utf-8") as f:
        f.write(benchmark_result.model_dump_json(indent=2))

    print(benchmark_result)

    print("BENCHMARK SUMMARY:")
    for single_run in single_runs:
        print("ID:", single_run.dataset_row.id)
        print("Pred:", single_run.answer_summary)
        print("Ground truth:", single_run.dataset_row.answer)
        print("----")
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
    asyncio.run(main())
