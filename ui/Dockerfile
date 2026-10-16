# The screens' image. Only Streamlit + httpx + pandas: no LLM keys, no database --
# the screens only talk to the API (see ui/client.py).
# Built from the project root by docker-compose.yml (service "ui").
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.28 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --only-group ui --no-install-project

COPY ui ./ui
COPY .streamlit ./.streamlit

EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0", "--server.port", "8501", \
     "--server.headless", "true"]
