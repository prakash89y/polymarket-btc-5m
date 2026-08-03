# syntax=docker/dockerfile:1
#
# Two stages: a builder that compiles wheels, and a slim runtime that carries
# no toolchain. Keeps the image small and the attack surface minimal for a
# container that will eventually hold a funded private key.

FROM python:3.14-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

# EXTRAS is overridable so a data-collector container does not pull torch:
#   docker build --build-arg EXTRAS=".[data]" .
ARG EXTRAS=".[data]"
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install "${EXTRAS}"


FROM python:3.14-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PMBTC_CONFIG=/app/config/config.yaml \
    PMBTC_LOGGING__JSON_LOGS=true

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates tini \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 pmbtc

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=pmbtc:pmbtc config ./config
COPY --chown=pmbtc:pmbtc src ./src
COPY --chown=pmbtc:pmbtc pyproject.toml README.md ./

# data/ and logs/ are volumes: the decision log and the local history must
# outlive the container.
RUN mkdir -p /app/data /app/logs /app/artifacts && chown -R pmbtc:pmbtc /app

USER pmbtc

HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 \
    CMD pmbtc show-config --section app > /dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "pmbtc"]
CMD ["doctor"]
