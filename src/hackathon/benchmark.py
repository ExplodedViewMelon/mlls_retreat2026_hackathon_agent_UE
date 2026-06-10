import asyncio
from abc import ABC, abstractmethod
from typing import Any, Coroutine, Sequence, TypeVar

from autogen_agentchat.agents import AssistantAgent
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm as atqdm

from hackathon.agent_discussion import AgentDiscussion, TurnTakingFlat, llm_extract_answer
from hackathon.autogen_client import get_client
from hackathon.hotpot_evalaute_f1 import f1_score
from hackathon.hotpotqa import Question_distractor, get_n_questions_distractor

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


class LikelihoodEvaluation(ABC):
    @staticmethod
    @abstractmethod
    async def perform_evaluation(messages_str: str, question: str) -> float: ...


class LikelihoodEvaluationJudge(LikelihoodEvaluation):
    @staticmethod
    async def perform_evaluation(messages_str: str, question: str) -> float:
        class AnswerEvaluation(BaseModel):
            likelihood: float = Field(
                ..., description="Likelihood of the answer being correct. Should be between 0 and 1"
            )  # noqa

        client = get_client()

        evaluate_answer_system_prompt = (
            "Consider the attached reasoning of multiple agents communicating their partial information."
            " Evaluate the likelihood of the answer being correct."
        )
        evaluate_agent = AssistantAgent(
            "evaluate_agent",
            client,
            system_message=evaluate_answer_system_prompt,
            output_content_type=AnswerEvaluation,
        )
        answer_summary_raw = await evaluate_agent.run(
            task=(
                f"Question: {question}. Conversation to evaluate: {messages_str}. "
                "Return a structured completion as a JSON string with a 'likelihood' field"
            )
        )
        return answer_summary_raw.messages[-1].content.likelihood  # type: ignore


class SingleRun(BaseModel):
    dataset_row: Question_distractor
    discussion: AgentDiscussion.Discussion
    answer_summary: str
    likelihood: float


class BenchmarkResult(BaseModel):
    runs: list[SingleRun]


async def process_question_pipeline(
    question_entry: Question_distractor,
    agents_discussion: AgentDiscussion,
    likelihood_evaluation: LikelihoodEvaluation,
    do_stream: bool,
) -> SingleRun:
    discussion = await agents_discussion.perform_discussion(question_entry, do_stream=do_stream)
    answer = await llm_extract_answer(discussion.messages_str, question_entry.question)
    likelihood = await likelihood_evaluation.perform_evaluation(
        discussion.messages_str, question_entry.question
    )

    return SingleRun(
        dataset_row=question_entry,
        discussion=discussion,
        answer_summary=answer,
        likelihood=likelihood,
    )


async def run_benchmark(
    dataset: list[Question_distractor],
    agents_discussion: AgentDiscussion,
    likelihood_evaluation: LikelihoodEvaluation,
    do_stream: bool,
    n_concurrent_processes: int,
) -> None:
    # proces concurrently
    runs_futures = [
        process_question_pipeline(
            question_entry, agents_discussion, likelihood_evaluation, do_stream
        )
        for question_entry in dataset
    ]

    runs = await gather_custom_with_semaphore(runs_futures, n_concurrent_processes)

    benchmark_result = BenchmarkResult(runs=runs)
    with open("benchmark_result.json", "w", encoding="utf-8") as f:
        f.write(benchmark_result.model_dump_json(indent=2))


def print_benchmark_details(benchmark: BenchmarkResult) -> None:
    print(benchmark)

    print("BENCHMARK SUMMARY:")
    for single_run in benchmark.runs:
        prediction = single_run.answer_summary
        ground_truth = single_run.dataset_row.answer
        f1, precision, recall = f1_score(prediction, ground_truth)

        print("ID:", single_run.dataset_row.id)
        print("Question:", single_run.dataset_row.question)
        print("Pred:", single_run.answer_summary)
        print("Ground truth:", single_run.dataset_row.answer)
        print("Likelihood:", single_run.likelihood)
        print("-")
        print(f"F1: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print("#" * 10)


if __name__ == "__main__":
    do_stream = False
    n_concurrent_processes = 10
    use_n_datapoints = 10

    dataset = get_n_questions_distractor()[:use_n_datapoints]
    agent_discussion = TurnTakingFlat()
    likelihood_evaluation = LikelihoodEvaluationJudge()
    benchmark_future = run_benchmark(
        dataset, agent_discussion, likelihood_evaluation, do_stream, n_concurrent_processes
    )

    asyncio.run(benchmark_future)

    # asyncio.run(run_benchmark())
    # asyncio.run(single_run())
