"""Run with: python -m scripts.run_ingest
Waits briefly for Postgres to be ready (useful on `docker compose up` cold start,
where the backend container can start before Postgres finishes init), then ingests
every .md file in app.config.settings.docs_dir.
"""

import sys
import time

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.db import SessionLocal, engine
from app.ingest import run_ingest


def wait_for_db(max_wait_seconds: int = 30):
    start = time.time()
    while time.time() - start < max_wait_seconds:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except OperationalError:
            time.sleep(1)
    print("ERROR: could not connect to database after waiting", file=sys.stderr)
    sys.exit(1)


def main():
    wait_for_db()
    db = SessionLocal()
    try:
        result = run_ingest(db)
        print(f"Ingestion complete: {result}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
