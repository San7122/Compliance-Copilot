FROM python:3.11-slim

WORKDIR /app

# No build toolchain and no ML stack. Embeddings come from a hosted API, so torch and
# sentence-transformers are gone; every remaining dependency installs from a wheel.
# This is what takes the image from ~3.3GB to a few hundred MB, and it also removes the
# build-time model download that used to dominate build time.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
