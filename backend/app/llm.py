"""Calls the LLM to generate a grounded, structured answer from retrieved chunks.

Structured output is enforced via function calling with `tool_choice` forced to a single
tool, rather than asking the model to "please respond in JSON" and hoping. The function's
parameter schema mirrors the required response schema exactly (answer, citations,
confidence, answerable), so the shape is constrained by the API rather than by regex or
markdown-fence stripping.

One provider-specific detail worth knowing: OpenAI returns tool arguments as a **JSON
string** (`function.arguments`), not a parsed object. It must be `json.loads`-ed, and a
malformed payload raises rather than being coerced into something plausible -- the same
principle applied to `ModelAnswer` validation downstream.

Everything below `generate_answer` consumes the provider-neutral `Generation` dataclass,
so swapping providers touches this module and nothing else in the pipeline.
"""

import json
from dataclasses import dataclass

from openai import OpenAI

from app.config import settings
from app.retrieval import RetrievedChunk


@dataclass
class Generation:
    """A model response plus what it cost to produce.

    Refusals that short-circuit before the API call carry zero usage -- that's the
    point of the short-circuit, and it should be visible in the logs rather than
    indistinguishable from a call that happened to be cheap.
    """

    content: dict
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * settings.price_per_mtok_input
            + self.output_tokens * settings.price_per_mtok_output
        ) / 1_000_000

SYSTEM_PROMPT = """You are Compliance Copilot, an internal assistant that answers questions \
about the policy documents of Northwind, a financial services group.

Rules you must follow:
1. Only use the provided document excerpts as your source of truth. Do not use outside \
knowledge, do not guess, and do not fill gaps with plausible-sounding details.
2. To cite, return the `chunk_id` shown in the header of the excerpt you are relying on, \
plus a short quote copied verbatim from that same chunk. Do not write out document \
names, section titles, clause numbers or page numbers yourself — those are filled in \
from our records. Only cite chunk_ids that appear in the provided context.
3. If the excerpts do not contain enough information to answer the question, you MUST \
set answerable to false, return an empty citations list, set confidence to "low", and \
set answer to exactly: "I don't know based on the provided documents." Do not partially \
answer from outside knowledge in this case, and do not offer a guess with a caveat.
4. Set confidence based on how directly and completely the excerpts answer the question:
   - "high": the excerpts directly and completely answer the question.
   - "medium": the excerpts partially answer it, or answer it indirectly.
   - "low": the excerpts barely relate to the question, or answerable is false.
5. Northwind is a group of separate legal entities, and their policies differ on purpose. \
Each excerpt is labelled with the entity it binds:
   - If an "ANSWERING SCOPE" line appears below the excerpts, it is authoritative. The \
user has selected that entity, so answer for it and lead with ITS requirement, even if \
the question text does not name the entity. Fall back to group policy only where that \
entity has no applicable document on the topic, and say so when you do.
   - With no ANSWERING SCOPE line, "Northwind Financial Services Pvt. Ltd." is the \
group: answer from group policy unless the question names a specific subsidiary.
   - "Northwind Capital Markets Ltd" and "Northwind Payments (Singapore) Pte Ltd" are \
separate entities whose staff follow their own documents instead of the group policy.
   - If the excerpts show a subsidiary imposing a DIFFERENT requirement on the same \
topic, name both and never merge conflicting figures into one number. Lead with the \
entity named in the ANSWERING SCOPE; where there is NO ANSWERING SCOPE, lead with the \
group position. Then note that the other differs, naming the entity and the figure. \
Never present a subsidiary's requirement as if it applied group-wide.
6. Where one document states that another governs on a topic (for example, a retention \
period governed by the Records Retention Schedule), follow that precedence and say which \
document governs. Do not apply any general ranking of document types beyond what the \
documents themselves state.
7. Some excerpts are labelled "Status: SUPERSEDED". A superseded document does NOT state \
current requirements. Use it only to describe what was previously required, and say \
explicitly that it is superseded when you do. Never present superseded text as a current \
obligation.
8. An excerpt labelled "Type: guidance" is a plain-language handbook summary that \
paraphrases policy loosely. Where it differs from an applicable authoritative policy, \
the policy controls; prefer citing the policy, and do not resolve a conflict in the \
handbook's favour.
9. If the excerpts genuinely conflict and no governing rule resolves it, say so and cite \
both, rather than silently choosing one.
10. Always call the submit_answer tool with your response. Never respond in plain text."""

TOOL_NAME = "submit_answer"

# The JSON Schema of the structured answer, kept separate from the provider wrapper so
# the contract stays readable independently of whose function-calling format encloses it.
# This body is provider-neutral: it is the same schema Anthropic's `input_schema` carried.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer to the user's question, in plain English, grounded only in the provided excerpts.",
        },
        # Only chunk_id and excerpt. Document, section, clause and page are looked
        # up from the retrieved record by app/citations/mapper.py, so there is no
        # field here for the model to get wrong.
        "citations": {
            "type": "array",
            "description": "Citations supporting the answer. Empty if answerable is false.",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {
                        "type": "integer",
                        "description": "The chunk_id shown in the header of the excerpt you are citing. Must be one of the provided chunk_ids.",
                    },
                    "excerpt": {
                        "type": "string",
                        "description": "A short excerpt (1-2 sentences) copied verbatim from that chunk's text, supporting the answer.",
                    },
                },
                "required": ["chunk_id", "excerpt"],
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
}

# OpenAI function-calling wrapper.
ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Submit the structured answer to the user's compliance question.",
        "parameters": ANSWER_SCHEMA,
    },
}

UNANSWERABLE_RESPONSE = {
    "answer": "The provided documents do not address this topic.",
    "citations": [],
    "confidence": "low",
    "answerable": False,
}


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Label every excerpt with the identity that decides how it may be used.

    Entity and clause are in the label rather than left implicit in the prose: the model
    cannot tell a group policy from a near-identical subsidiary one by reading the clause
    text, because the corpus is written so they look the same apart from the numbers.
    """
    blocks = []
    for c in chunks:
        # chunk_id comes first because it is the only identifier the model has to hand
        # back; everything else in this label is context for reasoning, not for citing.
        label = [f"chunk_id: {c.chunk_id}", f"Document: {c.document}"]
        if c.doc_id:
            label.append(f"ID: {c.doc_id}" + (f" v{c.version}" if c.version else ""))
        if c.entity:
            label.append(f"Entity: {c.entity}")
        if c.document_type:
            label.append(f"Type: {c.document_type}")
        # Only surfaced when it isn't the default, so it reads as a warning rather than
        # boilerplate the model learns to skim past.
        if c.is_superseded:
            label.append("Status: SUPERSEDED")
        if c.clause:
            label.append(f"Clause: {c.clause}")
        if c.page:
            label.append(f"Page: {c.page}")
        label.append(f"Section: {c.section}")
        label.append(f"similarity: {c.similarity:.2f}")
        blocks.append("[" + " | ".join(label) + "]\n" + c.content)
    return "\n\n---\n\n".join(blocks)


def scope_directive(scope) -> str:
    """Tell the model which entity it is answering for.

    Retrieval and reranking already know the scope -- it is what filters other
    subsidiaries out and promotes the reader's own entity. The model did not, and that
    gap produced a real wrong answer: asked "how long must KYC records be retained?"
    with the entity selector set to Capital Markets, the correct source (NFS-SUB-002,
    seven years) was ranked first, but the model still led with the group's five years.
    From its point of view that was right -- the question text named no subsidiary, and
    the prompt only treated a subsidiary as selected when the question said so.

    So the scope is stated explicitly, and the two ways it can arise are distinguished,
    because they carry different weight: a selector choice is a fact about the user,
    while a phrase in the question is an inference from wording.

    Note this deliberately does NOT say "prefer the top-ranked chunk". Ranking order is
    not evidence about which entity binds the reader, and collapsing the two would make
    the answer follow retrieval noise whenever scores are close.
    """
    if scope is None or getattr(scope, "entity", None) is None:
        return ""

    entity = scope.entity
    if getattr(scope, "entity_source", None) == "explicit":
        origin = (
            "The user selected this entity for this request; the question text does not "
            "have to name it."
        )
    else:
        origin = "This entity was named in the question."

    return (
        f"\n\n---\n\nANSWERING SCOPE: {entity}\n{origin}\n"
        f"Answer for {entity}. Lead with the requirement that binds it. If another "
        "entity's document states a different figure, mention that difference after the "
        f"answer rather than in place of it. If {entity} has no applicable document on "
        "this topic, say so and answer from group policy instead."
    )


def generate_answer(question: str, chunks: list[RetrievedChunk], scope=None) -> Generation:
    if not chunks:
        # No chunks cleared the relevance bar at all -- don't even call the LLM,
        # it has nothing to ground an answer in.
        return Generation(content=dict(UNANSWERABLE_RESPONSE))

    if not settings.llm_key:
        raise RuntimeError(
            "No LLM API key is set. Set LLM_API_KEY (or OPENAI_API_KEY) in .env."
        )

    client = OpenAI(api_key=settings.llm_key)
    context = _build_context(chunks)

    user_message = (
        f"Document excerpts:\n\n{context}"
        f"{scope_directive(scope)}"
        f"\n\n---\n\nQuestion: {question}\n\n"
        "Call submit_answer with your structured response."
    )

    response = client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=1024,
        # The system prompt is a message here rather than a top-level parameter.
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        tools=[ANSWER_TOOL],
        tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
    )

    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }

    message = response.choices[0].message
    for call in message.tool_calls or []:
        if call.function.name != TOOL_NAME:
            continue
        try:
            arguments = json.loads(call.function.arguments)
        except json.JSONDecodeError as exc:
            # Do NOT fall back to a refusal here. Coercing unparseable output into a
            # plausible-looking "I don't know" would hide a broken integration behind a
            # normal-looking answer -- the same trap as swallowing an invalid enum.
            raise ValueError(
                f"model returned malformed tool arguments: {exc}"
            ) from exc
        return Generation(content=arguments, **usage)

    # Defensive fallback: forced tool_choice should always return the tool, but if the
    # SDK/API ever returns something unexpected, fail safe rather than guessing. The
    # call still happened, so the usage is still real and still gets logged.
    return Generation(content=dict(UNANSWERABLE_RESPONSE), **usage)
