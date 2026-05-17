FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir -e ".[embedding,vector]" \
    && pip install --no-cache-dir loguru

# Prompt files and data directories are mounted as volumes at runtime
RUN mkdir -p data/sqlite data/lance prompts

ENTRYPOINT ["hyporeddit"]
CMD ["--help"]
