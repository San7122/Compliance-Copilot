"""Calls Claude to generate a grounded, structured answer from retrieved chunks.

Structured output is enforced via Anthropic's tool-use with `tool_choice` forced to
that tool, rather than asking the model to "please respond in JSON" and hoping. The
tool's input_schema mirrors the required response schema exactly (answer, citations,
confidence, answerable), so the SDK gives us back a schema-validated dict directly from
`tool_use.input` -- no brittle regex/markdown-fence parsing needed.
"""

import json

import anthropic

from app.config import settings
from app.retrieval import RetrievedChunk

SYSTEM_PROMPT = """You are Compliance Copilot, an internal assistant that answers questions \
about a company's policy documents.

Rules you must follow:
1. Only use the provided document excerpts as your source of truth. Do not use outside \
knowledge, do not guess, and do not fill gaps with plausible-sounding details.
2. Every citation you return must be an excerpt that actually appears in the provided \
context, attributed to the correct document and section.
3. If the excerpts do not contain enough information to answer the question, you MUST \
set answerable to false, return an empty citations list, set confidence to "low", and \
set answer to a short statement that the documents do not address this topic. Do not \
partially answer from outside knowledge in this case.
4. Set confidence based on how directly and completely the excerpts answer the question:
   - "high": the excerpts directly and completely answer the question.
   - "medium": the excerpts partially answer it, or answer it indirectly.
   - "low": the excerpts barely relate to the question, or answerable is false.
5. Always call the submit_answer tool with your response. Never respond in plain text."""

ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit the structured answer to the user's compliance question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the user's question, in plain English, grounded only in the provided excerpts.",
            },
            "citations": {
                "type": "array",
                "description": "Citations supporting the answer. Empty if answerable is false.",
                "items": {
                    "type": "object",
                    "properties": {
                        "document": {"type": "string"},
                        "section": {"type": "string"},
                        "excerpt": {
                            "type": "string",
                            "description": "A short excerpt (1-2 sentences) copied from the source chunk that supports the answer.",
                        },
                    },
                    "required": ["document", "section", "excerpt"],
                },
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "answerable": {
                "type": "boolean",
                "description": "False if the provided excerpts do not contain the answer.",
            },
        },
        "required": ["answer", "citations", "confidence", "answerable"],
    },
}

UNANSWERABLE_RESPONSE = {
    "answer": "The provided documents do not address this topic.",
    "citations": [],
    "confidence": "low",
    "answerable": False,
}


def _build_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(
            f"[Document: {c.document} | Section: {c.section} | similarity: {c.similarity:.2f}]\n{c.content}"
        )
    return "\n\n---\n\n".join(blocks)


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> dict:
    if not chunks:
        # No chunks cleared the relevance bar at all -- don't even call the LLM,
        # it has nothing to ground an answer in.
        return dict(UNANSWERABLE_RESPONSE)

    if not settings.anthropic_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add a real key."
        )

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    context = _build_context(chunks)

    user_message = (
        f"Document excerpts:\n\n{context}\n\n---\n\nQuestion: {question}\n\n"
        "Call submit_answer with your structured response."
    )

    response = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        tools=[ANSWER_TOOL],
        tool_choice={"type": "tool", "name": "submit_answer"},
        messages=[{"role": "user", "content": user_message}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "submit_answer":
            return block.input

    # Defensive fallback: forced tool_choice should always return the tool, but if the
    # SDK/API ever returns something unexpected, fail safe rather than guessing.
    return dict(UNANSWERABLE_RESPONSE)
