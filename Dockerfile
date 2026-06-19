FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml ./
COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./

RUN pip install --no-cache-dir .

# ponytail: single image, command chosen at runtime (api now; worker entrypoint added in M4).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
