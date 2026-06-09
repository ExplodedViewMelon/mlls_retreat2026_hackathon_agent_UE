import asyncio
import re
from typing import Any, Coroutine, Sequence, TypeVar

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


T = TypeVar("T")


async def gather_custom(processes: Sequence[Coroutine[Any, Any, T]]) -> list[T]:
    return list(await asyncio.gather(*processes))


async def gather_custom_with_semaphore(
    processes: Sequence[Coroutine[Any, Any, T]], max_concurrency: int
) -> list[T]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _run_with_semaphore(process: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await process

    return list(await asyncio.gather(*(_run_with_semaphore(process) for process in processes)))


class SingleRun(BaseModel):
    dataset_row: Question_distractor
    conversation: str
    messages_raw: Sequence[BaseAgentEvent | BaseChatMessage]
    answer_summary: str


async def process_question(
    client, dataset_row: Question_distractor, do_stream: bool = True
) -> SingleRun:
    ##### Initialize agents

    TOPIC = dataset_row.question

    n_agents = len(dataset_row.different_sentences)

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
                f"You are in total {n_agents} agents. "
                "Make sure to hear everyone's opinion before submitting the answer."
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

    messages_str = ""
    for message in messages:
        agent = message.source
        content = message.content  # type: ignore

        messages_str += f"{agent}:\n"
        messages_str += f"{content}\n\n"

    # last_message = messages[-1].content  # type: ignore

    summary_agent = AssistantAgent("summary_agent", client)
    answer_summary_raw = await summary_agent.run(
        task=(
            "Extract the answer from the following conversation. "
            "Answer should be detailed while being less than 10 words. "
            f"Question: {dataset_row.question}. "
            f"Answer to summarize: {messages_str} "
        )
    )
    answer_summary = answer_summary_raw.messages[-1].to_text()

    return SingleRun(
        dataset_row=dataset_row,
        conversation=messages_str,
        messages_raw=messages,
        answer_summary=answer_summary,
    )


class BenchmarkResult(BaseModel):
    single_runs: list[SingleRun]


async def single_run() -> None:
    client = get_client()

    # get 1 data row
    hotpotqa_dataset_simple = get_n_questions_distractor(n_titles=4)
    dataset_row = hotpotqa_dataset_simple[1]

    do_stream = True
    run_result = await process_question(client, dataset_row, do_stream=do_stream)

    print("#" * 10)
    print("Titles:")
    for sentences in dataset_row.different_sentences:
        print(" -", sentences.title)
    print("Question:", run_result.dataset_row.question)
    print("Ground truth:", run_result.dataset_row.answer)
    print("Prediction summary:", run_result.answer_summary)


async def run_benchmark() -> None:
    client = get_client()

    # get 1 data row
    hotpotqa_dataset_simple = get_n_questions_distractor(n_titles=4)
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
        print("Question:", single_run.dataset_row.question)
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
