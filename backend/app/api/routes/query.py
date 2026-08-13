import logging

import openai
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.models import QueryLog
from app.query.pipeline import (
    MalformedAnswer,
    QueryPipeline,
    QueryResult,
    QuestionInvalid,
    RetrievalUnavailable,
)
from app.schemas import AnswerResponse, HistoryItem, QueryRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=AnswerResponse)
def query(req: QueryRequest, db: Session = Depends(get_db)):
    try:
        result = QueryPipeline(db).run(req.question, entity=req.entity)
    except QuestionInvalid as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RetrievalUnavailable:
        raise HTTPException(
            status_code=503,
            detail="Search is unavailable (retrieval or the embedding API failed).",
        )
    except MalformedAnswer:
        # Nothing is logged: a malformed response must not land in query_log as though
        # it were an answer.
        raise HTTPException(
            status_code=500, detail="The language model returned a malformed answer."
        )
    except RuntimeError as e:
        # Configuration error (e.g. missing API key) — the message is safe to show.
        raise HTTPException(status_code=500, detail=str(e))
    # Ordered most-specific first: RateLimitError is a subclass of APIStatusError, so
    # the reverse order would swallow it.
    except openai.RateLimitError:
        raise HTTPException(
            status_code=429, detail="The language model is rate limited. Try again shortly."
        )
    except openai.APIConnectionError:
        raise HTTPException(status_code=504, detail="Could not reach the language model.")
    except openai.APIStatusError as e:
        logger.exception("LLM API returned %s", e.status_code)
        raise HTTPException(
            status_code=502, detail=f"The language model returned an error ({e.status_code})."
        )
    except Exception:
        # Never leak a traceback to the caller; the details are in the server log.
        logger.exception("query failed for question=%r", req.question)
        raise HTTPException(status_code=500, detail="Failed to generate an answer.")

    _log_query(db, req.question.strip(), result)
    return result.response


def _log_query(db: Session, question: str, result: QueryResult) -> None:
    response = result.response
    logger.info(
        "query answered grounded=%s confidence=%.2f citations=%d retrieved=%d "
        "latency_ms=%d model=%s tokens_in=%d tokens_out=%d cost_usd=%.6f",
        response.grounded,
        response.confidence,
        len(response.citations),
        len(result.retrieved_chunk_ids),
        result.latency_ms,
        settings.llm_model,
        result.input_tokens,
        result.output_tokens,
        result.cost_usd,
    )
    db.add(
        QueryLog(
            question=question,
            answer=response.answer,
            citations=[c.model_dump() for c in response.citations],
            confidence=response.confidence,
            grounded=response.grounded,
            refusal_reason=response.refusal_reason,
            retrieved_chunk_ids=result.retrieved_chunk_ids,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost_usd=result.cost_usd,
        )
    )
    db.commit()


@router.get("/history", response_model=list[HistoryItem])
def history(
    limit: int = Query(default=20, ge=1, le=settings.history_max_limit),
    db: Session = Depends(get_db),
):
    rows = db.query(QueryLog).order_by(QueryLog.created_at.desc()).limit(limit).all()
    return [
        HistoryItem(
            id=r.id,
            question=r.question,
            answer=r.answer,
            citations=r.citations,
            confidence=r.confidence,
            grounded=r.grounded,
            refusal_reason=r.refusal_reason,
            latency_ms=r.latency_ms,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            cost_usd=r.cost_usd,
            created_at=r.created_at.isoformat(),
        )
        for r in rows
    ]
