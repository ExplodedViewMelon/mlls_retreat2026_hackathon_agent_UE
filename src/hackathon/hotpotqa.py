import pprint
from typing import Literal

from datasets import load_dataset
from pydantic import BaseModel

from hackathon.wikipedia_client import WikipediaArticle, get_wikipedia_content


class Question_fullwiki(BaseModel):
    id: str
    question: str
    answer: str
    level: str
    type: str
    wikipedia_articles: list[WikipediaArticle]


class Sentences(BaseModel):
    title: str
    sentences: list[str]


class Question_distractor(BaseModel):
    id: str
    question: str
    answer: str
    level: str
    type: str
    different_sentences: list[Sentences]


def get_n_questions_fullwiki(
    n_questions: int = 10,
    n_titles: int = 2,
    level: str = "hard",
    type: Literal["bridge", "comparison"] = "bridge",
) -> list[Question_fullwiki]:
    ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train")
    ds = ds.filter(
        lambda x: x["level"] == level
        and x["type"] == type
        and len(x["context"]["title"]) == n_titles
    ).select(range(n_questions))
    to_return: list[Question_fullwiki] = []
    for q in ds:
        articles = [get_wikipedia_content(title) for title in q["context"]["title"]]

        question_object = Question_fullwiki(
            id=q["id"],
            question=q["question"],
            answer=q["answer"],
            level=q["level"],
            type=q["type"],
            wikipedia_articles=articles,
        )

        to_return.append(question_object)

    return to_return


def get_n_questions_distractor(
    n_titles: int = 2,
    level: str = "hard",
    type: Literal["bridge", "comparison"] = "bridge",
) -> list[Question_distractor]:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    ds = ds.filter(
        lambda x: x["level"] == level
        and x["type"] == type
        and len(x["context"]["title"]) == n_titles
    )
    to_return: list[Question_distractor] = []
    for q in ds:
        different_sentences = [
            Sentences(title=title, sentences=sentences)
            for title, sentences in zip(
                q["context"]["title"], q["context"]["sentences"], strict=False
            )
        ]
        question_object = Question_distractor(
            id=q["id"],
            question=q["question"],
            answer=q["answer"],
            level=q["level"],
            type=q["type"],
            different_sentences=different_sentences,
        )

        to_return.append(question_object)

    return to_return


if __name__ == "__main__":
    # ds = get_n_questions()
    # print(len(ds))
    # print(ds)
    # pprint.pprint(ds[0])

    # make object
    questions = get_n_questions_distractor(n_titles=3)

    for question in questions:
        total = 0
        for sentences in question.different_sentences:
            total += len(sentences.sentences)

        print(total)
