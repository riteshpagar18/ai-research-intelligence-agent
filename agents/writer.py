import json

from agents.gemini import generate_text


def generate_report(question, sources, verification):

    source_text = ""

    for i, source in enumerate(sources, start=1):

        source_text += f"""
        [S{i}]

        Title: {source['title']}
        URL: {source['url']}

        Information:
        {source['content'][:2500]}

        """

    prompt = f"""
    You are an expert research report writer.

    Research question:
    {question}

    Research sources:
    {source_text}

    Verification assessment and source-linked evidence:
    {json.dumps(verification, ensure_ascii=False, indent=2)}

    Generate a professional research report.

    Include these sections:

    1. Executive Summary
    2. Introduction
    3. Key Findings
    4. Detailed Analysis
    5. Limitations and Unverified Claims
    6. Conclusion
    7. References

    IMPORTANT:

    - Use only information supported by the supplied sources.
    - Add inline citations such as [S1], [S2].
    - Do not invent statistics or quotations.
    - Clearly identify uncertainty.
    - Do not cite a source as supporting a claim unless verification marks it supported.
    - If evidence is insufficient, say so.
    - Keep the report professional and readable.
    """

    return generate_text(prompt)
