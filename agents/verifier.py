import json

from agents.gemini import generate_text


def validate_evidence(result, sources):
    """Keep only evidence quotes that occur in the source text."""
    source_text_by_id = {
        source.get("source_id", f"S{i}"): " ".join(source["content"].split())
        for i, source in enumerate(sources, start=1)
    }
    for claim in result.get("claims", []):
        if not isinstance(claim, dict):
            continue
        valid_evidence = []
        for item in claim.get("evidence", []):
            if not isinstance(item, dict):
                continue
            source_id = item.get("source_id")
            source_content = source_text_by_id.get(source_id, "")
            quote = " ".join(str(item.get("quote", "")).split())
            if quote and quote.casefold() in source_content.casefold():
                valid_evidence.append({**item, "quote": quote})
        claim["evidence"] = valid_evidence
        claim["source_ids"] = list(dict.fromkeys(
            item["source_id"] for item in valid_evidence
        ))
        status = str(claim.get("status", "unsupported")).lower().replace(" ", "_")
        if status not in {"supported", "partially_supported", "unsupported", "conflicting"}:
            status = "unsupported"
        if status in {"supported", "partially_supported", "conflicting"} and not valid_evidence:
            status = "unsupported"
        claim["status"] = status
    return result


def _parse_json_response(raw):
    payload = raw.strip()
    if payload.startswith("```"):
        payload = payload.split("\n", 1)[1].rsplit("```", 1)[0]
    result = json.loads(payload)
    if not isinstance(result, dict):
        raise ValueError("Gemini returned unexpected verification data.")
    return result


def verify_sources(question, sources):

    source_text = ""

    for i, source in enumerate(sources, start=1):
        source_id = source.get("source_id", f"S{i}")

        source_text += f"""
        SOURCE {source_id}

        Title: {source['title']}
        URL: {source['url']}

        Content:
        {source['content'][:2500]}

        """

    prompt = f"""
    You are a research verification assistant.

    Original research question:
    {question}

    Review the following sources:

    {source_text}

    Your tasks:

    1. Identify the claims supported by the sources.
    2. Identify claims that are not sufficiently supported.
    3. Identify conflicting information.
    4. Explain limitations in the available evidence.

    Compare every source for each claim. If one source supports a claim and
    another contradicts it, label the claim conflicting and cite evidence from
    both sides. Do not mark it supported just because one source agrees.

    Do not invent facts.
    Do not assume that a source is reliable just because it appears in search results.

    Clearly state that this is an AI-assisted assessment,
    not independent proof of factual accuracy.
    """

    prompt += """

    Return valid JSON only, using this schema:
    {
      "claims": [
        {
          "claim": "short factual claim",
          "status": "supported|partially_supported|unsupported|conflicting",
          "source_ids": ["S1"],
          "evidence": [{"source_id": "S1", "quote": "exact supporting words from source content"}]
        }
      ],
      "limitations": ["evidence limitation"]
    }
    Quote source text exactly. Never fabricate a quotation. Use an empty evidence
    list when a claim is unsupported. This is an AI-assisted assessment, not proof.
    """

    try:
        raw = generate_text(prompt)
        result = _parse_json_response(raw)
        if not isinstance(result.get("claims"), list):
            raise ValueError("Gemini returned unexpected verification data.")
        return validate_evidence(result, sources)
    except (json.JSONDecodeError, ValueError, IndexError):
        return {"claims": [], "limitations": [], "raw_assessment": raw}


def verify_claim(question, claim_text, sources):
    """Evaluate one known claim against a supplied source set (benchmark use)."""
    source_lines = "\n\n".join(
        f"[{source.get('source_id', f'S{i}')}] {source['title']}\n"
        f"URL: {source['url']}\nText: {source['content']}"
        for i, source in enumerate(sources, start=1)
    )
    prompt = f"""
    Assess exactly this claim for the question: {question}

    Claim: {claim_text}

    Source evidence:
    {source_lines}

    Return valid JSON only with this shape:
    {{"status":"supported|partially_supported|unsupported|conflicting",
      "evidence":[{{"source_id":"S1","quote":"exact source words"}}]}}
    Use exact quotations. If evidence is missing, return unsupported and an empty list.
    Examine every source for contradictions. If sources disagree, choose conflicting
    and include an exact quote from each side; one supporting source does not cancel
    a contradictory source. Lack of mention alone is not a contradiction.
    """
    raw = ""
    try:
        raw = generate_text(prompt)
        result = _parse_json_response(raw)
        result["claim"] = claim_text
        checked = validate_evidence({"claims": [result]}, sources)
        return checked["claims"][0]
    except (json.JSONDecodeError, ValueError, IndexError):
        return {"claim": claim_text, "status": "unsupported", "source_ids": [], "evidence": [], "error": "Invalid verifier response"}
