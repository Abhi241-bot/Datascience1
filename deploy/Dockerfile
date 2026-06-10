# Hugging Face Space (Docker SDK) image: a single self-contained container that
# trains the model at build time and serves the API + drift dashboard on 7860.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_PYTHON_REFRESH=quiet

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --timeout 180 --retries 10 -r requirements.txt

COPY src ./src

# Train at build time so the image ships with the model + reference window.
RUN python -m src.serve.build_model

# Hugging Face Spaces routes traffic to port 7860.
EXPOSE 7860
CMD ["uvicorn", "src.serve.deploy:app", "--host", "0.0.0.0", "--port", "7860"]
