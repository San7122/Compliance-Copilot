import logging
import time

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.llm import generate_answer
from app.models import QueryLog
from app.retrieval import filter_by_relevance, retrieve
from app.schemas import AnswerResponse, HistoryItem, QueryRequest
from app.verification import enforce_citation_grounding

logger = logging.getLogger(__name__)

app = FastAPI(title="Compliance Copilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # no auth / internal tool, per assignment scope
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=AnswerResponse)
def query(req: QueryRequest, db: Session = Depends(get_db)):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    start = time.time()

    # Retrieval can fail independently of the LLM -- most often because the embedding
    # model can't be loaded. That's an availability problem, not a bad request, so it
    # gets its own handler rather than surfacing as an opaque 500.
    try:
        all_chunks = retrieve(db, question)
    except Exception:
        logger.exception("retrieval failed for question=%r", question)
        raise HTTPException(
            status_code=503,
            detail="Search is unavailable (the embedding model could not be loaded).",
        )

    relevant_chunks = filter_by_relevance(all_chunks)

    # Ordered most-specific first: RateLimitError is a subclass of APIStatusError, so
    # the reverse order would swallow it.
    try:
        result = generate_answer(question, relevant_chunks)
    except RuntimeError as e:
        # Configuration error (e.g. missing API key) -- the message is safe to show.
        raise HTTPException(status_code=500, detail=str(e))
    except anthropic.RateLimitError:
        raise HTTPException(
            status_code=429, detail="The language model is rate limited. Try again shortly."
        )
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=504, detail="Could not reach the language model.")
    except anthropic.APIStatusError as e:
        logger.exception("Anthropic API returned %s", e.status_code)
        raise HTTPException(
            status_code=502, detail=f"The language model returned an error ({e.status_code})."
        )
    except Exception:
        # Never leak a traceback to the caller; the details are in the server log.
        logger.exception("answer generation failed for question=%r", question)
        raise HTTPException(status_code=500, detail="Failed to generate an answer.")

    # Drop any citation whose excerpt isn't actually in the chunks the model was given.
    result, rejected = enforce_citation_grounding(result, relevant_chunks)
    if rejected:
        logger.warning(
            "dropped %d ungrounded citation(s) for question=%r", len(rejected), question
        )

    latency_ms = int((time.time() - start) * 1000)

    # Validate before writing: if the model returned something off-schema we want the
    # request to fail without having already committed a malformed row to query_log.
    try:
        response = AnswerResponse(**result)
    except Exception:
        logger.exception("model returned an off-schema response: %r", result)
        raise HTTPException(status_code=500, detail="The language model returned a malformed answer.")

    log = QueryLog(
        question=question,
        answer=response.answer,
        citations=[c.model_dump() for c in response.citations],
        confidence=response.confidence,
        answerable=response.answerable,
        retrieved_chunk_ids=[c.chunk_id for c in relevant_chunks],
        latency_ms=latency_ms,
    )
    db.add(log)
    db.commit()

    return response


@app.get("/history", response_model=list[HistoryItem])
def history(
    limit: int = Query(default=20, ge=1, le=settings.history_max_limit),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(QueryLog)
        .order_by(QueryLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        HistoryItem(
            id=r.id,
            question=r.question,
            answer=r.answer,
            citations=r.citations,
            confidence=r.confidence,
            answerable=r.answerable,
            latency_ms=r.latency_ms,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]
