import asyncio
import re
from types import CoroutineType
from typing import Any, Coroutine, Sequence, TypeVar

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from pydantic import BaseModel
from tqdm.asyncio import tqdm as atqdm

from hackathon.autogen_client import get_client
from hackathon.hotpot_evalaute_f1 import f1_score
from hackathon.hotpotqa import Question_distractor, get_n_questions_distractor


async def llm_extract_answer(client, conversation: str, question: str) -> str:
    extractor_system_prompt = (
        "You are an answer extraction specialist. Given a question and a longer discussion or expert answer, extract the single most direct and concise answer.\n"
        "Rules:\n"
        "- Extract only the core answer — a word, name, number, short phrase, or at most 1–2 sentences\n"
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


T = TypeVar("T")


async def gather_custom_with_semaphore(
    processes: Sequence[Coroutine[Any, Any, T]], max_concurrency: int
) -> list[T]:
    semaphore = asyncio.Semaphore(max_concurrency)

    n_processes = len(processes)

    async def _run_with_semaphore(process: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await process

    return list(
        await atqdm.gather(
            *(_run_with_semaphore(process) for process in processes),
            desc="Running benchmark",
            total=n_processes,
        )
    )


class SingleRun(BaseModel):
    dataset_row: Question_distractor
    conversation: str
    messages_raw: Sequence[BaseAgentEvent | BaseChatMessage]
    answer_summary: str


async def process_question(dataset_row: Question_distractor, do_stream: bool = True) -> SingleRun:
    ##### Initialize agents

    TOPIC = dataset_row.question

    n_agents = len(dataset_row.different_sentences)
    client = get_client()
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

    answer_summary = await llm_extract_answer(client, messages_str, dataset_row.question)

    return SingleRun(
        dataset_row=dataset_row,
        conversation=messages_str,
        messages_raw=messages,
        answer_summary=answer_summary,
    )


class BenchmarkResult(BaseModel):
    single_runs: list[SingleRun]


async def single_run() -> None:
    # get 1 data row
    hotpotqa_dataset_simple = get_n_questions_distractor(n_titles=4)
    dataset_row = hotpotqa_dataset_simple[1]

    do_stream = True
    run_result = await process_question(dataset_row, do_stream=do_stream)

    print("#" * 10)
    print("Titles:")
    for sentences in dataset_row.different_sentences:
        print(" -", sentences.title)
    print("Question:", run_result.dataset_row.question)
    print("Ground truth:", run_result.dataset_row.answer)
    print("Prediction summary:", run_result.answer_summary)


async def run_benchmark() -> None:
    # get 1 data row
    hotpotqa_dataset_simple = get_n_questions_distractor()
    hotpotqa_dataset_simple = hotpotqa_dataset_simple[:10]
    do_stream = False

    single_runs_futures: list[CoroutineType[Any, Any, SingleRun]] = []

    print("Starting benchmark")

    # # process sequentially
    # for dataset_row in tqdm(
    #     hotpotqa_dataset_simple, desc="Benchmarking", total=len(hotpotqa_dataset_simple)
    # ):

    # try:
    #     run_result = await process_question(client, dataset_row, do_stream=do_stream)
    #     single_runs.append(run_result)
    # except Exception:
    #     pass

    # proces concurrently
    for dataset_row in hotpotqa_dataset_simple:
        single_runs_futures.append(process_question(dataset_row, do_stream=do_stream))

    single_runs = await gather_custom_with_semaphore(single_runs_futures, 10)

    benchmark_result = BenchmarkResult(single_runs=single_runs)
    with open("benchmark_result.json", "w", encoding="utf-8") as f:
        f.write(benchmark_result.model_dump_json(indent=2))

    print(benchmark_result)

    print("BENCHMARK SUMMARY:")
    for single_run in single_runs:
        prediction = single_run.answer_summary
        ground_truth = single_run.dataset_row.answer
        f1, precision, recall = f1_score(prediction, ground_truth)

        print("ID:", single_run.dataset_row.id)
        print("Question:", single_run.dataset_row.question)
        print("Pred:", single_run.answer_summary)
        print("Ground truth:", single_run.dataset_row.answer)
        print("-")
        print(f"F1: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("#" * 10)
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
    asyncio.run(run_benchmark())
    # asyncio.run(single_run())
