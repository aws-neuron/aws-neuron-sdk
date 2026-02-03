FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    make enchant-2 git pandoc \
    && rm -rf /var/lib/apt/lists/* \
    && pandoc --version

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /docs

COPY requirements.txt .
RUN uv pip install --system -r requirements.txt --extra-index-url=https://pypi.org/simple

ENTRYPOINT ["/bin/bash"]
