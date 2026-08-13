"""The zero-chunk short-circuit is a cost and safety guarantee, not an optimisation.

If nothing cleared the relevance floor, calling the model would mean asking it to
answer with no grounding at all — the exact setup that produces confident invention.
It must refuse locally, without an API call.
"""

import pytest

from app.llm import UNANSWERABLE_RESPONSE, Generation, _build_context, generate_answer
from app.retrieval import RetrievedChunk


def chunk(content="Records are kept for seven years.", similarity=0.8):
    return RetrievedChunk(
        chunk_id=1,
        document="Data Retention Policy",
        section="Data Retention Policy > 2. Retention Periods",
        content=content,
        similarity=similarity,
    )


def test_no_chunks_refuses_without_an_api_call(monkeypatch):
    def explode(*a, **kw):
        raise AssertionError("the LLM client must not be constructed with zero chunks")

    monkeypatch.setattr("app.llm.OpenAI", explode)

    assert generate_answer("anything", []).content == UNANSWERABLE_RESPONSE


def test_short_circuited_refusal_reports_zero_cost():
    """A refusal that never reached the API must not look like a cheap API call."""
    generation = generate_answer("anything", [])

    assert generation.input_tokens == 0
    assert generation.output_tokens == 0
    assert generation.cost_usd == 0.0


def test_no_chunks_refusal_is_a_copy_not_the_shared_constant():
    """Callers mutate the result (citation stripping); the constant must stay pristine."""
    result = generate_answer("anything", []).content
    result["answer"] = "mutated"

    assert UNANSWERABLE_RESPONSE["answer"] == "The provided documents do not address this topic."


def test_cost_is_computed_from_configured_per_million_rates():
    from app.config import settings

    generation = Generation(content={}, input_tokens=1_000_000, output_tokens=1_000_000)

    expected = settings.price_per_mtok_input + settings.price_per_mtok_output
    assert generation.cost_usd == pytest.approx(expected)


def test_missing_api_key_raises_a_configuration_error(monkeypatch):
    monkeypatch.setattr("app.config.Settings.llm_key", property(lambda self: ""))

    with pytest.raises(RuntimeError, match="LLM API key"):
        generate_answer("anything", [chunk()])


def test_context_block_labels_document_and_section():
    """The model can only attribute citations correctly if the labels are in context."""
    context = _build_context([chunk()])

    assert "Data Retention Policy" in context
    assert "2. Retention Periods" in context
    assert "Records are kept for seven years." in context


def test_context_separates_multiple_chunks():
    context = _build_context([chunk("first"), chunk("second")])

    assert "first" in context and "second" in context
    assert "---" in context


# =====================================================================================
# OpenAI integration layer
#
# These are the tests that would have caught a bad provider swap. The rest of the suite
# stubs `generate_answer` entirely, so it passes whether or not the real API wiring is
# correct -- that gap is exactly why the first real call is risky. Here the OpenAI client
# itself is faked, so the request shape and the response parsing are both exercised.
# =====================================================================================

import json as _json

import openai
import pytest as _pytest

from app.llm import ANSWER_TOOL, TOOL_NAME, generate_answer


class _Function:
    def __init__(self, name, arguments):
        self.name, self.arguments = name, arguments


class _ToolCall:
    def __init__(self, name, arguments):
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class _Usage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens, self.completion_tokens = prompt_tokens, completion_tokens


class _Response:
    def __init__(self, tool_calls, prompt_tokens=1000, completion_tokens=200):
        self.choices = [type("C", (), {"message": _Message(tool_calls)})()]
        self.usage = _Usage(prompt_tokens, completion_tokens)


def _install(monkeypatch, *, response=None, error=None, captured=None):
    """Replace app.llm.OpenAI with a fake client, and give it a usable key."""
    monkeypatch.setattr("app.config.Settings.llm_key", property(lambda self: "test-key"))

    class _Completions:
        def create(self, **kwargs):
            if captured is not None:
                captured.update(kwargs)
            if error:
                raise error
            return response

    class _Client:
        def __init__(self, *a, **kw):
            self.chat = type("Chat", (), {"completions": _Completions()})()

    monkeypatch.setattr("app.llm.OpenAI", _Client)


VALID_ARGS = {
    "answer": "Within four (4) hours of discovery.",
    "citations": [{"chunk_id": 7, "excerpt": "within four (4) hours"}],
    "confidence": "high",
    "answerable": True,
}


# --- (a) OpenAI tool-call response ---------------------------------------------------


def test_openai_tool_call_is_parsed_into_generation(monkeypatch):
    _install(monkeypatch, response=_Response([_ToolCall(TOOL_NAME, _json.dumps(VALID_ARGS))]))

    generation = generate_answer("How quickly must a breach be reported?", [chunk()])

    assert generation.content == VALID_ARGS
    assert generation.content["citations"][0]["chunk_id"] == 7


def test_request_uses_forced_function_calling(monkeypatch):
    """Structured output is enforced by the API, not by asking nicely for JSON."""
    captured = {}
    _install(
        monkeypatch,
        response=_Response([_ToolCall(TOOL_NAME, _json.dumps(VALID_ARGS))]),
        captured=captured,
    )

    generate_answer("q", [chunk()])

    assert captured["tool_choice"] == {"type": "function", "function": {"name": TOOL_NAME}}
    assert captured["tools"] == [ANSWER_TOOL]
    assert ANSWER_TOOL["type"] == "function"
    # The system prompt is a message for OpenAI, not a top-level parameter.
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][1]["role"] == "user"


# --- (b) arguments arrive as a JSON STRING -------------------------------------------


def test_arguments_are_a_json_string_and_are_decoded(monkeypatch):
    """The single biggest provider difference: OpenAI returns a string, not a dict."""
    raw = _json.dumps(VALID_ARGS)
    _install(monkeypatch, response=_Response([_ToolCall(TOOL_NAME, raw)]))

    generation = generate_answer("q", [chunk()])

    assert isinstance(raw, str)
    assert isinstance(generation.content, dict)  # decoded, not passed through
    assert generation.content["answerable"] is True


# --- (c) malformed JSON ----------------------------------------------------------------


def test_malformed_json_arguments_raise_rather_than_degrade(monkeypatch):
    """Must not be coerced into a plausible refusal -- that would hide a broken integration."""
    _install(monkeypatch, response=_Response([_ToolCall(TOOL_NAME, "{not valid json")]))

    with _pytest.raises(ValueError, match="malformed tool arguments"):
        generate_answer("q", [chunk()])


# --- (d) missing tool call --------------------------------------------------------------


def test_missing_tool_call_falls_back_safely_but_keeps_usage(monkeypatch):
    _install(monkeypatch, response=_Response(None, prompt_tokens=50, completion_tokens=10))

    generation = generate_answer("q", [chunk()])

    assert generation.content["answerable"] is False
    assert generation.input_tokens == 50  # the call happened; usage is real
    assert generation.output_tokens == 10


def test_unexpected_tool_name_is_ignored(monkeypatch):
    _install(monkeypatch, response=_Response([_ToolCall("some_other_tool", "{}")]))

    assert generate_answer("q", [chunk()]).content["answerable"] is False


# --- (e/f/g) API errors propagate for the route to map ----------------------------------


def _http_response(status):
    import httpx

    return httpx.Response(status_code=status, request=httpx.Request("POST", "https://x"))


def test_rate_limit_error_propagates(monkeypatch):
    _install(monkeypatch, error=openai.RateLimitError(
        "rate limited", response=_http_response(429), body=None))

    with _pytest.raises(openai.RateLimitError):
        generate_answer("q", [chunk()])


def test_api_connection_error_propagates(monkeypatch):
    _install(monkeypatch, error=openai.APIConnectionError(request=_http_response(0).request))

    with _pytest.raises(openai.APIConnectionError):
        generate_answer("q", [chunk()])


def test_api_status_error_propagates(monkeypatch):
    _install(monkeypatch, error=openai.APIStatusError(
        "server error", response=_http_response(500), body=None))

    with _pytest.raises(openai.APIStatusError):
        generate_answer("q", [chunk()])


# --- (h) token + cost accounting --------------------------------------------------------


def test_openai_usage_fields_are_mapped_correctly(monkeypatch):
    """OpenAI names them prompt_tokens/completion_tokens, not input_/output_tokens."""
    _install(
        monkeypatch,
        response=_Response(
            [_ToolCall(TOOL_NAME, _json.dumps(VALID_ARGS))],
            prompt_tokens=1234,
            completion_tokens=567,
        ),
    )

    generation = generate_answer("q", [chunk()])

    assert generation.input_tokens == 1234
    assert generation.output_tokens == 567


def test_cost_is_computed_from_the_configured_openai_rates(monkeypatch):
    from app.config import settings

    _install(
        monkeypatch,
        response=_Response(
            [_ToolCall(TOOL_NAME, _json.dumps(VALID_ARGS))],
            prompt_tokens=1_000_000,
            completion_tokens=1_000_000,
        ),
    )

    generation = generate_answer("q", [chunk()])

    expected = settings.price_per_mtok_input + settings.price_per_mtok_output
    assert generation.cost_usd == _pytest.approx(expected)
