# Pinned image for ground truth (docs/DESIGN.md, Environment): linux/amd64, single thread.
# Build: docker build --platform linux/amd64 -t physical-intervene .
# The base image is pinned by digest; record the built image's digest at freeze 1.
FROM --platform=linux/amd64 python:3.12-slim-bookworm@sha256:82836454a795d9d75549a9e157fd08dbac6c383292cf89e99f4248a97ce6b2ff
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONHASHSEED=0 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY scripts ./scripts
RUN uv sync --locked --no-dev

CMD ["python", "scripts/replay_check.py", "--help"]
