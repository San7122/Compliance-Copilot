import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.ingestion.pipeline import run_ingest
from app.schemas import IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
def ingest(db: Session = Depends(get_db)):
    """Re-run ingestion over the docs directory without restarting the container.

    Idempotent: documents whose content hash is unchanged are skipped rather than
    re-embedded, and a document that is re-processed only replaces its own chunks —
    adding new files never disturbs what is already indexed.
    """
    try:
        result = run_ingest(db)
    except Exception:
        # Individual document failures are handled inside the pipeline; reaching here
        # means something structural (database, embedding API) is wrong.
        logger.exception("ingestion failed")
        raise HTTPException(
            status_code=503,
            detail="Ingestion failed. Check the embedding API key and database connection.",
        )

    return IngestResponse(**result)
