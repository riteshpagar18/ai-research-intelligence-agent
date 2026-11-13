import os
from pydantic import BaseModel, Field
from tavily import TavilyClient


class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    score: float = Field(default=0.0)


def search_web(query):

    client = TavilyClient(
        api_key=os.getenv("TAVILY_API_KEY")
    )

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=3
    )

    results = []

    for item in response.get("results", []):

        result = SearchResult(
            title=item.get("title") or "Untitled",
            url=item.get("url") or "",
            content=item.get("content") or "",
            score=item.get("score") or 0.0
        )

        results.append(result.model_dump())

    return results