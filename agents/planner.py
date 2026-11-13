from agents.gemini import generate_text


def create_research_plan(question, conversation_history=None):

    context = ""
    if conversation_history:
        recent = conversation_history[-3:]
        context = "\n\nRecent conversation context (use only if relevant):\n" + "\n".join(
            f"Question: {item['question']}\nSummary: {item.get('summary', '')[:500]}"
            for item in recent
        )

    prompt = f"""
    You are an expert AI research planner.

    Analyze the following research question:

    {question}

    {context}

    Break this question into 4 specific research tasks.

    Rules:
    - Each task should investigate a different aspect.
    - Tasks should be clear and searchable.
    - Return only a numbered list.
    """

    plan = generate_text(prompt)

    tasks = [
        line.strip()
        for line in plan.split("\n")
        if line.strip()
    ]

    return tasks
