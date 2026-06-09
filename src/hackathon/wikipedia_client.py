import wikipedia
from pydantic import BaseModel

wikipedia.set_user_agent("MyHackathonApp/1.0 (oliversande@gmail.com)")


class WikipediaArticle(BaseModel):
    title: str
    url: str
    content: str


def get_wikipedia_content(title: str) -> WikipediaArticle:
    page = wikipedia.page(title, auto_suggest=False)

    return WikipediaArticle(title=page.title, content=page.content, url=page.url)


if __name__ == "__main__":
    titles = [
        "Verano de Escándalo (1998)",
        "Triplemanía VII",
        "Protection racket",
        "E. Gordon Gee",
        "Badr Hari",
        "Guerra de Titanes (1998)",
        "Global Fighting Championship",
        "Outrageous Betrayal",
        "Betting controversies in cricket",
        "Prosecution of gender-targeted crimes",
    ]

    for title in titles:
        page = wikipedia.page(title, auto_suggest=False)

        print(page.url)
