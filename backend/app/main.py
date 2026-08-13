"""Application assembly.

Deliberately thin: routing lives in `app/api/routes/`, the offline pipeline in
`app/ingestion/`, and the online pipeline in `app/query/`. Keeping this file free of
business logic is what makes the ingestion/query separation visible at a glance rather
than something you have to trace through call sites to confirm.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, ingestion, query

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="Compliance Copilot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # no auth / internal tool, per assignment scope
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(query.router)
app.include_router(ingestion.router)
