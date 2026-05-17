# hyporeddit Validator

A local Python tool that scrapes r/hausbau (German homebuilding subreddit), stores posts and comments in a vector-searchable database, and evaluates user-authored hypotheses against the real experiences of homebuilders — producing confidence-weighted validation scores backed by cited evidence.

---

## What It Does

You write a hypothesis like:

> *"Homebuilders perceive the planning process as too slow."*

The system retrieves the most relevant discussions from its corpus, classifies each piece of evidence as supporting, contradicting, or neutral, and returns:

- A **validation score** (0–1)
- A **confidence rating** based on sample size and agreement
- A **stance distribution** (47 support / 8 contradict / 12 neutral)
- **Top evidence quotes** in German and English, per stance
- A **prose synthesis** summarizing key themes and notable dissent
- A **persistent run record** so scores can be tracked as the corpus grows

---

## Quick Start (Docker, Full Workflow)

### Prerequisites

- Docker and Docker Compose
- An Anthropic API key (for Claude Haiku + Sonnet)

### Setup + First End-to-End Run

```bash
git clone https://github.com/yourname/hyporeddit-validator
cd hyporeddit-validator

cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Build the image
docker compose build

# 1) Fetch historical Reddit data (choose a smaller limit first if you want a quick smoke run)
 --limit 1000

# 2) Process fetched posts/comments into chunks, translations, and embeddings
docker compose run --rm app process

# 3) Verify SQLite and LanceDB are in sync
docker compose run --rm app verify-stores

# 4) Check corpus stats
docker compose run --rm app stats

# 5) Evaluate a hypothesis
docker compose run --rm app evaluate "Homebuilders find the planning process too slow"
```

### Whole Application Runbook (Recommended Order)

Use this exact sequence to run the complete system lifecycle:

```bash
# One-time bootstrap
docker compose run --rm app backfill --limit 1000
docker compose run --rm app process
docker compose run --rm app verify-stores
docker compose run --rm app stats

# Hypothesis evaluation loop
docker compose run --rm app evaluate "Homebuilders find the planning process too slow"
docker compose run --rm app evaluate "Hausbauer empfinden den Planungsprozess als zu langsam"

# Ongoing daily refresh loop
docker compose run --rm app ingest
docker compose run --rm app process
docker compose run --rm app verify-stores
```

Quick health checks after each stage:
- `verify-stores` should report no missing/orphaned vectors.
- `stats` should show non-zero posts/comments/chunks after processing.
- `evaluate` should return `score`, `confidence`, stance distribution, and evidence items.

### Evaluating a Hypothesis

```bash
# English hypothesis
docker compose run --rm app evaluate "Homebuilders find the planning process too slow"

# German hypothesis
docker compose run --rm app evaluate "Hausbauer empfinden den Planungsprozess als zu langsam"

# Output as JSON (for scripting or notebooks)
docker compose run --rm app evaluate "Planning is too complex" --json

# Force re-run (ignore cache)
docker compose run --rm app evaluate "Planning is too complex" --force-rerun
```

### Daily Ingestion

Add to your host crontab to keep the corpus fresh:

```
0 6 * * * cd /path/to/hyporeddit-validator && docker compose run --rm app ingest && docker compose run --rm app process
```

### Local Python Run (Without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,embedding,vector]"

cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

hyporeddit backfill --limit 1000
hyporeddit process
hyporeddit verify-stores
hyporeddit stats
hyporeddit evaluate "Homebuilders find the planning process too slow"
```

---

## CLI Reference

```
hyporeddit backfill [--limit 1000]       Fetch historical posts (one-time)
hyporeddit ingest                         Fetch new posts since last run (daily)
hyporeddit process [--reprocess]          Build chunks + embeddings from fetched posts
hyporeddit process --make-translation     Also translate each chunk DE→EN via LLM (requires API key)
hyporeddit process --train-adapter        Force domain-adapter training after storing, ignoring threshold

hyporeddit evaluate "<text>"              Evaluate a hypothesis against the corpus
hyporeddit evaluate "<text>" --json       Output evaluation result as JSON
hyporeddit evaluate "<text>" --force-rerun  Recompute even if cached
hyporeddit evaluate "<text>" --top-k 150    Retrieve a different evidence set size
hyporeddit show-evaluation <run_id>       Re-display a past evaluation
hyporeddit history <hypothesis_id>        Show score trend for a hypothesis over time
hyporeddit list-hypotheses                List all stored hypotheses with latest scores

hyporeddit verify-stores                  Check SQLite and LanceDB are in sync
hyporeddit stats                          Corpus statistics
```

---

## Using from a Notebook

The evaluation pipeline is a first-class Python API. Import it directly in a Jupyter notebook:

```python
from hyporeddit.evaluation.pipeline import evaluate_hypothesis
from hyporeddit.storage.sqlite import get_all_hypotheses

# Evaluate interactively
result = evaluate_hypothesis("Homebuilders find the planning process too slow")

print(f"Score: {result.score:.2f} | Confidence: {result.confidence:.2f}")
print(f"Stances: {result.stance_distribution}")
print(f"Sample size: {result.sample_size} evidence chunks")
print()
print(result.synthesis)
print()
for item in result.evidence[:5]:
    print(f"[{item.stance.upper()}] {item.text_en}")
    print(f"  → {item.source_url}")
```

---

## Architecture Overview

```
r/{REDDIT_SUBREDDIT} (*.json)
       │
  RedditJsonAdapter          ← swappable (SourceAdapter interface)
       │
  Filter + Chunk             ← noise removal, sub-chunking
       │
  Translate (Haiku)          ← DE → EN per chunk, stored alongside original
       │
  Embed (BGE-M3)             ← cross-lingual, self-hosted
       │
  ┌────┴────┐
SQLite    LanceDB             ← dual-store, single write path
  └────┬────┘
       │
  Retrieval + Stance Classification (Haiku, batched)
       │
  Confidence-Weighted Aggregation
       │
  Synthesis (Sonnet)
       │
  EvaluationResult + persisted run history
```

---

## Configuration

All configuration is via environment variables (`.env` file):

| Variable | Description | Default |
|----------|-------------|---------|
| `REDDIT_SUBREDDIT` | Subreddit to ingest (without `r/` prefix) | `hausbau` |
| `ANTHROPIC_API_KEY` | Anthropic API key | required |
| `LLM_PROVIDER` | `anthropic` or `openai` | `anthropic` |
| `LLM_CLASSIFICATION_MODEL` | Model for stance classification | `claude-haiku-4-5-20251001` |
| `LLM_SYNTHESIS_MODEL` | Model for synthesis | `claude-sonnet-4-6` |
| `SQLITE_PATH` | Path to SQLite database | `/app/data/sqlite/hyporeddit.db` |
| `LANCE_PATH` | Path to LanceDB directory | `/app/data/lance` |
| `BGE_M3_DEVICE` | `cpu` or `cuda` | `cpu` |
| `LOG_LEVEL` | Logging verbosity | `INFO` |

---

## Data Storage

All data is stored locally in `./data/`:

```
data/
├── sqlite/
│   └── hyporeddit.db        ← posts, comments, chunks, hypotheses, evaluation runs
└── lance/
    └── chunks.lance/     ← BGE-M3 vector embeddings, keyed by chunk_id
```

No data leaves your machine except for LLM API calls (classification prompts + chunks are sent to Anthropic). Run `hyporeddit stats` to see what's stored.

---

## Limitations & Known Trade-offs

- **Reddit access**: uses unauthenticated public JSON endpoints. Rate-limited (~10 req/min), may be intermittently blocked by Cloudflare. The system handles this with backoff and retries but is not guaranteed to be stable long-term. Migrating to authenticated PRAW access (when app registration is possible) is a one-file change.
- **Translation quality**: German→English translations are done via Claude Haiku. Domain-specific terms (Bauträger, GEG, KfW) may not translate idiomatically. The German original is always preserved and used for embedding/classification.
- **Confidence weighting**: the weighting formula (recency decay, engagement, depth penalty) is heuristic and was not tuned on labelled data. Expect to revisit the formula once real evaluation runs accumulate.
- **BGE-M3 on CPU**: embedding a corpus of ~30k chunks takes 15–30 minutes on CPU. This only happens once at backfill. Daily delta is fast (<5 minutes).
- **English-language bias in r/hausbau**: the subreddit is ~95% German but occasionally contains English posts. These are handled correctly by BGE-M3 but are a small minority of the evidence base.

---

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run a single unit test
pytest tests/unit/test_filters.py

# Format
ruff format src/
ruff check src/
```

