# One image for every role; DSEARCH_ROLE etc. pick what a container runs (D-028).
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml ./
COPY proto ./proto
COPY scripts ./scripts
COPY src ./src
# Stubs are generated at build time, never committed (D-001).
RUN pip install "grpcio-tools>=1.62" \
    && python scripts/gen_proto.py \
    && pip install . \
    && pip uninstall -y grpcio-tools
COPY config ./config

CMD ["python", "-m", "dsearch.shard"]
