import pprint
from typing import Literal

from datasets import load_dataset
from pydantic import BaseModel

from hackathon.wikipedia_client import WikipediaArticle, get_wikipedia_content


class Question(BaseModel):
    id: str
    question: str
    answer: str
    level: str
    type: str
    wikipedia_articles: list[WikipediaArticle]


def get_n_questions(
    n_questions: int = 10,
    n_titles: int = 2,
    level: str = "hard",
    type: Literal["bridge", "comparison"] = "bridge",
    subname: Literal["fullwiki", "distractor"] = "distractor",
) -> list[Question]:
    ds = load_dataset("hotpotqa/hotpot_qa", subname, split="train")
    ds = ds.filter(
        lambda x: x["level"] == level
        and x["type"] == type
        and len(x["context"]["title"]) == n_titles
    ).select(range(n_questions))
    to_return: list[Question] = []
    for q in ds:
        articles = [get_wikipedia_content(title) for title in q["context"]["title"]]

        question_object = Question(
            id=q["id"],
            question=q["question"],
            answer=q["answer"],
            level=q["level"],
            type=q["type"],
            wikipedia_articles=articles,
        )

        to_return.append(question_object)

    return to_return


if __name__ == "__main__":
    # ds = get_n_questions()
    # print(len(ds))
    # print(ds)
    # pprint.pprint(ds[0])

    # make object
    questions = get_n_questions()
    pprint.pprint(questions[0])
