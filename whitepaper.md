# hyporeddit Validator — Technical Whitepaper

> A local hypothesis validation engine that mines lived experience from r/hausbau (a German homebuilding subreddit) and produces confidence-weighted evidence scores for product and research hypotheses.

---

## Table of Contents

1. [Core Requirements](#1-core-requirements)
2. [Assumptions](#2-assumptions)
3. [Other Considerations](#3-other-considerations)
4. [Limitations](#4-limitations)
5. [Architecture](#5-architecture)
6. [Features & CLI Flags](#6-features--cli-flags)
7. [Design Decisions](#7-design-decisions)
8. [Technology Stack](#8-technology-stack)
9. [Database Schema](#9-database-schema)
10. [Instructions to Run](#10-instructions-to-run)

---

## 1. Core Requirements

### Functional Requirements

| # | Requirement |
|---|-------------|
| FR-1 | Ingest posts and all their comments from r/hausbau via Reddit's public JSON API |
| FR-4 | Retrieve semantically relevant evidence chunks using cross-lingual vector search |
| FR-5 | Classify each retrieved chunk as `supports`, `contradicts`, `neutral`, or `irrelevant` |
| FR-6 | Compute a weighted validation score (0–1) and a confidence rating (0–1) |
| FR-7 | Return per-chunk evidence quotes in both German (original) and English (translation) |
| FR-8 | Generate a prose synthesis summarizing key themes and notable dissent |
| FR-9 | Persist every evaluation run so scores can be tracked as the corpus grows over time |
| FR-10 | Expose all functionality as a CLI and as a Python library callable from Jupyter notebooks |

### Non-Functional Requirements

| # | Requirement |
|---|-------------|
| NFR-1 | Fully local storage — no external database servers required |
| NFR-8 | Run a daily incremental delta to pick up new posts without re-fetching the entire corpus |
| NFR-9 | Accept hypothesis text in English or German; auto-detect language |
| NFR-2 | Polite HTTP access — ≥6 s inter-request delay, exponential backoff, circuit breaker |
| NFR-3 | Resumable ingestion — interrupted jobs never restart from scratch |
| NFR-4 | Single unified write path — SQLite and LanceDB are always kept in sync |
| NFR-5 | Source-agnostic ingestion layer — adding a new data source is a single-file change |
| NFR-6 | Versioned prompt files — prompt iteration requires no code changes |
| NFR-7 | Dockerized deployment — reproducible environment on any machine |

---

## 2. Assumptions

| # | Assumption | Consequence if Wrong |
|---|------------|----------------------|
| A-1 | r/hausbau represents a meaningful sample of German homebuilder sentiment | Hypothesis scores may not generalize to the broader homebuilder population |
| A-2 | Reddit's unauthenticated public JSON endpoints remain accessible | Ingestion fails until fallback (PRAW, Arctic Shift) is integrated — this is a one-file change |
| A-3 | BGE-M3 cross-lingual alignment is sufficient to retrieve German content with English queries | Retrieval recall may be lower than expected; hybrid retrieval (sparse + dense) can compensate |
| A-4 | ~5–20 new posts per day is a reasonable estimate for delta volume | If growth is faster, comment-tree fetching may become a bottleneck |
| A-5 | Any API (Claude / OpenAI based) produces acceptable German→English translations for domain terms | Translations of highly technical/legal German terms (KfW, GEG, Bauantrag) may lose nuance |
| A-6 | The confidence weighting formula reflects actual user authority without labeled calibration data | Weighting may need tuning once real evaluation data accumulates |
| A-7 | A single user drives all hypothesis authoring and evaluation | The system has no multi-user auth, conflict resolution, or concurrent-write handling |

---

## 3. Other Considerations

### Translation
Translation is a enforced as a feature flag during processing. We currently use BGE-M3 vector embeddings which allows cross-language contextual embeddings for 100+ languages.

### Reproducibility

Hypothesis scores are non-deterministic across evaluation runs for two reasons: (1) LLM stance classification has nonzero temperature and may return different labels for borderline chunks, and (2) the corpus grows daily, changing the evidence pool. The `--force-rerun` flag exists to recompute with the current corpus; stored run history allows trend analysis over time. For reproducible point-in-time scores, snapshot the SQLite + LanceDB files before rerunning.

### Use of Reddit API

Due to time constraints, I was not able to get authenticated PRAW access. Therefore, for fundamental understanding, Reddit's unauthenticated JSON endpoints have been used. Due to rate limiting, you may experience significant wait times between api calls. This tool is designed for low-volume research access and includes polite-access safeguards (delays, backoff, circuit breaker). Commercial or high-volume use should migrate to authenticated PRAW access (a one-file change) and review Reddit's current Developer Terms of Service.
**NOte:** Reddit scraping was also considered. However, since API existed, it was a much easier way to implement.

### Use of LanceDB for vector databases
Originally, images were also considered as a viable source of information. For example, screenshot from a website. Due to time, budget and computational limitations (I only had a Mac Air lol to run LLMs locally), the scope was narrowed to posts and comments only.

---

## 4. Limitations

### Data Access

- **Rate-limited scraping**: Unauthenticated Reddit access is capped. The daily delta is low-volume enough to stay within limits, but aggressive backfill pacing is intentionally slowed.
- **Corpus ceiling**: Reddit's listing API returns at most 1,000 posts per endpoint. Historical depth beyond this requires third-party archives (Arctic Shift, Pushshift), which were deprioritized for demo.

### Quality

- **Translation drift**: German construction domain terms (Bauträger, Bodenplatte, KfW-Förderung, GEG) carry precise legal/technical meanings that LLMs may not translate idiomatically. The German original is always preserved and used for all embedding and classification.
- **Heuristic weighting**: The confidence weighting formula (recency decay, engagement, depth penalty, karma) was designed analytically, not trained on labeled data. Scores should be treated as directionally useful, not numerically precise, until calibration data is available.
- **English bias in synthesis**: The prose synthesis pass runs in English. Nuances expressed only in German and not fully captured in translation may be underrepresented in the final narrative.

### Performance

- **API fetches**: API fetches take signiofcant amount of time especially during the backfill, with `O(n.m)` time complexity. (n -> number of posts, m -> number of comments per post).
- **CPU embedding**: Embedding takes significant amount of time. But the code allows switching from cpu to cuda (nvidia gpu) or mps (apple acceleration) for gpu acceleration. 
- **LLM**: Evaluating a hypothesis with top_k=100 chunks requires ⌈100/15⌉ = 7 stance-classification calls plus one synthesis call. Testing for the entire application was done with `deepseek-v4-flash`.
- **No streaming output**: The evaluation pipeline returns results after all steps complete. There is no incremental/streaming output during a run.

### Scope

- **Single source**: v1 ingests only 1 depth of comments per post from Reddit. Other German homebuilding communities (Bauforum24, r/de, YouTube comments) are architecturally supported but not yet implemented.
- **No web UI**: The system is a CLI + Python library. A web frontend was deliberately excluded to keep v1 scope manageable.
- **No multi-user support**: There is no authentication, user isolation, or concurrent-write handling.

---

## 5. Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                      User Interface Layer                           │
│                                                                     │
│         CLI (Typer)              Jupyter Notebooks                  │
│    hyporeddit <command>      from hyporeddit.evaluation.pipeline    │
│                                    import evaluate_hypothesis       │
└──────────────────┬──────────────────────────┬───────────────────────┘
                   │                          │
       ┌───────────▼──────────┐   ┌───────────▼──────────────┐
       │   Ingestion Pipeline  │   │   Evaluation Pipeline    │
       │                      │   │                          │
       │  SourceAdapter        │   │  Retriever               │
       │  HTTP Client          │   │  Stance Classifier       │
       │  Filters + Chunker    │   │  Aggregator              │
       │  Translator           │   │  Synthesizer             │
       │  Embedder             │   │                          │
       └───────────┬──────────┘   └───────────┬──────────────┘
                   │                          │
       ┌───────────▼──────────────────────────▼──────────────┐
       │                    Storage Layer                     │
       │                                                     │
       │   SQLite (relational data)    LanceDB (vectors)     │
       │   posts, comments, chunks     chunk embeddings      │
       │   hypotheses, eval runs       1024-dim BGE-M3       │
       │   evidence, jobs              ANN search            │
       │                                                     │
       │   ← unified.store_chunk() is the only write path → │
       └─────────────────────────────────────────────────────┘
```

---

### Ingestion Data Flow

```
Reddit *.json endpoints
        │  (unauthenticated, public)
        ▼
  PolitHttpClient
  ├── 6s inter-request delay
  ├── Exponential backoff: 30s → 2min → 10min
  ├── Circuit breaker: pause 1h if >50% error rate
  └── Max 3 retries per request
        │
        ▼
  RedditJsonAdapter                   ← implements SourceAdapter ABC
  ├── fetch_posts(limit, after)       ← paginated listing
  └── fetch_comments(post_id)        ← full comment tree
        │
        ▼
  filters.apply(raw_comments)
  ├── Hard drop: [deleted], [removed], AutoModerator, known bots
  └── Agreement-token drop: "Danke", "+1", "👍", "Stimmt", ...
        │
        ▼
  chunker.chunk(post | comment)
  ├── Posts  → title + body, one chunk
  ├── Comments → body only, one chunk
  └── Sub-chunking: if >400 words → overlapping 350-word windows (50-word overlap)
        │
        ├──────────────────────┐
        ▼                      ▼
  translator.translate()   bge_m3.encode()     ─ or ─    AdaptiveEncoder.encode()
  LLM: DE → EN           BGE-M3 (1024-dim)              BGE-M3 + linear adapter
  stored as text_en        dense embedding                domain-adapted embedding
        │                      │
        └──────────┬───────────┘
                   ▼
        unified.store_chunk(chunk, embedding)
        ├── sqlite.insert_chunk(chunk)          ← transactional, first
        └── lance.upsert_vector(chunk_id, emb)  ← keyed by chunk_id
```

---

### Evaluation Data Flow

```
User: hyporeddit evaluate "Homebuilders find planning too slow"
        │
        ▼
  [Language Detection]
  "en" | "de" — auto-detected
        │
        ▼
  [Step 1 — Embed Hypothesis]
  BGE_M3_Encoder.encode_query(hypothesis)
  → query_vector (1024-dim, with BGE-M3 query prefix)
        │
        ▼
  [Step 2 — Vector Search]
  lance.search(query_vector, top_k=100)
  → [(chunk_id, cosine_score), ...]     ← ANN search
        │
        ▼
  [Step 3 — Hydrate]
  sqlite.get_chunks(chunk_ids)
  → [Chunk(text_de, text_en, parent_post_title, metadata)]
        │
        ▼
  [Step 4 — Batch Stance Classification]
  Batches of 15 chunks per LLM call
  LLM → [{"chunk_id", "stance", "rationale"}]
  Stances: supports | contradicts | neutral | irrelevant
        │
        ▼
  [Step 5 — Confidence-Weighted Aggregation]
  Per-chunk weight = relevance × recency × engagement × depth_penalty × karma_factor
  score     = Σ(weight_i × stance_value_i) / Σ(weight_i)
  confidence = agreement_rate × min(n/50, 1.0)
        │
        ▼
  [Step 6 — Synthesis]
  LLM — one call
  Input: hypothesis + top evidence per stance + statistics
  Output: prose summary, key themes, notable dissent
        │
        ▼
  [Step 7 — Persist + Return]
  sqlite.insert_evaluation_run(run)
  sqlite.insert_evidence_classifications(evidence)
  → EvaluationResult returned to caller
```

---

### Domain Adapter (Optional Fine-Tuning Layer)

```
Existing chunks in SQLite (is_filtered=0)
        │
        ▼
  _sample_pairs(db)
  → positive pairs: two chunks from the same parent_post_id
        │
        ▼
  BGE_M3_Encoder.encode(all_texts)      ← frozen base model, one batched call
  → base_embeddings (n, 1024)
        │
        ▼
  _AdapterLayer training (identity-initialized Linear 1024→1024)
  ├── InfoNCE loss with in-batch negatives, temperature=0.07
  ├── AdamW optimizer, lr=1e-3
  └── 3 epochs (default)
        │
        ▼
  Checkpoint saved → data/model/adapter.pt
  AdaptiveEncoder loaded on next process run
  (acts as drop-in replacement for BGE_M3_Encoder)
```

---

### Component Map

```
src/hyporeddit/
│
├── cli.py                    ← Typer entrypoints (thin wrappers only)
├── config.py                 ← All env-var access via pydantic-settings
│
├── sources/
│   ├── base.py               ← SourceAdapter ABC
│   └── reddit_json.py        ← Unauthenticated Reddit JSON adapter
│
├── ingestion/
│   ├── scheduler.py          ← Backfill + daily delta orchestration
│   ├── http_client.py        ← Polite HTTP: delay, backoff, circuit breaker
│   ├── filters.py            ← Hard filters + agreement-token heuristic
│   ├── chunker.py            ← Post/comment → Chunk objects, sub-chunking
│   ├── processor.py          ← process_all(): translate + embed + store
│   └── stats.py              ← Corpus statistics printer
│
├── embedding/
│   ├── bge_m3.py             ← BGE-M3 encoder wrapper (dense, 1024-dim)
│   └── adapter.py            ← Linear adapter + InfoNCE training loop
│
├── translation/
│   └── translator.py         ← Batched DE→EN via Haiku
│
├── storage/
│   ├── sqlite.py             ← Schema, repositories, all SQL
│   ├── lance.py              ← LanceDB vector store wrapper
│   ├── unified.py            ← store_chunk(): the only write path
│   └── migrations/           ← Schema versioning
│
├── llm/
│   ├── base.py               ← LLMClient ABC
│   ├── anthropic.py          ← Haiku (classify) + Sonnet (synthesize)
│   └── openai_compat.py      ← OpenAI / LM Studio fallback
│
├── evaluation/
│   ├── pipeline.py           ← evaluate_hypothesis() — full orchestration
│   ├── retriever.py          ← Vector search + chunk hydration
│   ├── stance.py             ← Batched stance classification
│   ├── aggregator.py         ← Confidence-weighted scoring
│   ├── synthesizer.py        ← Prose summary via Sonnet
│   └── display.py            ← Rich terminal output
│
└── models/
    ├── ingestion.py          ← Chunk, ChunkMetadata dataclasses
    └── evaluation.py         ← EvaluationResult, EvidenceItem Pydantic models
```

---

## 6. Features & CLI Flags

### Ingestion Commands

#### `hyporeddit backfill [--limit N]`

One-time historical fetch. Pages through `/r/{REDDIT_SUBREDDIT}/new.json` collecting up to N posts (default 1,000) and their full comment trees. Job state is persisted in SQLite after each page — safe to interrupt and resume.

```bash
hyporeddit backfill                 # fetch up to 1,000 posts
hyporeddit backfill --limit 200     # quick smoke-test with 200 posts
```

#### `hyporeddit ingest`

Daily delta fetch. Retrieves the first page of `/r/{REDDIT_SUBREDDIT}/new.json` (100 posts), diffs against stored post IDs, and fetches comment trees only for unseen posts. Designed for daily cron scheduling.

```bash
hyporeddit ingest
```

#### `hyporeddit process [--reprocess] [--make-translation] [--train-adapter]`

Converts fetched raw posts and comments into embedded chunks.

| Flag | Effect |
|------|--------|
| _(no flags)_ | Chunk, embed, and store all unprocessed content |
| `--reprocess` | Re-process already-processed posts (useful after config changes) |
| `--make-translation` | Also translate each chunk DE→EN via any LLM (requires API key) |
| `--train-adapter` | Force adapter training after storing, ignoring the chunk-count threshold |

```bash
hyporeddit process
hyporeddit process --make-translation
hyporeddit process --train-adapter
hyporeddit process --reprocess --make-translation
```

---

### Evaluation Commands

#### `hyporeddit evaluate "<text>" [--json] [--force-rerun] [--top-k N]`

Evaluates a hypothesis. Accepts English or German text.

| Flag | Effect |
|------|--------|
| _(no flags)_ | Rich terminal output with score, confidence, evidence table, synthesis |
| `--json` | Output full `EvaluationResult` as JSON (for piping, scripting, notebooks) |
| `--force-rerun` | Bypass the evaluation cache and recompute from scratch |
| `--top-k N` | Override the number of candidate chunks to retrieve (default: 100) |

```bash
hyporeddit evaluate "Homebuilders find the planning process too slow"
hyporeddit evaluate "Bauherren empfinden den Planungsprozess als zu langsam"
hyporeddit evaluate "Planning is too complex" --json | jq '.score'
hyporeddit evaluate "Planning is too complex" --force-rerun
hyporeddit evaluate "Planning is too complex" --top-k 200
```

#### `hyporeddit show-evaluation <run_id>`

Re-displays the full output of any past evaluation run by its UUID.

```bash
hyporeddit show-evaluation 3f7a1d2e-...
```

#### `hyporeddit history <hypothesis_id>`

Shows the score trend for a hypothesis across all its evaluation runs, ordered by date. Useful for tracking whether community sentiment changes as the corpus grows.

```bash
hyporeddit history 9c1b4f88-...
```

#### `hyporeddit list-hypotheses`

Lists all stored hypotheses with their hypothesis ID and most recent evaluation score.

```bash
hyporeddit list-hypotheses
```

---

### Maintenance Commands

#### `hyporeddit verify-stores [--fix]`

Detects orphaned chunks (in SQLite but not in LanceDB) and orphaned vectors (in LanceDB but not SQLite). With `--fix`, re-embeds missing chunks to restore sync.

```bash
hyporeddit verify-stores
hyporeddit verify-stores --fix
```

#### `hyporeddit stats`

Prints corpus statistics: post count, comment count, chunk count, date range, hypothesis count, evaluation run count.

```bash
hyporeddit stats
```

---

### Python API Use Cases

The evaluation pipeline is a first-class Python function — the CLI is a thin wrapper.

```python
from hyporeddit.evaluation.pipeline import evaluate_hypothesis

# Basic evaluation
result = evaluate_hypothesis("Homebuilders find the planning process too slow")
print(f"Score: {result.score:.2f} | Confidence: {result.confidence:.2f}")
print(result.synthesis)

# Increase evidence set
result = evaluate_hypothesis("Planning is complex", top_k=200)

# Force fresh computation (ignore cache)
result = evaluate_hypothesis("Planning is complex", force_rerun=True)

# Iterate evidence
for item in result.evidence:
    print(f"[{item.stance.upper()}] {item.text_en}")
    print(f"  Rationale: {item.rationale}")
    print(f"  Source: {item.source_url}")
    print(f"  Weight: {item.weight:.3f} | Retrieval score: {item.retrieval_score:.3f}")

# Export to JSON
import json
print(json.dumps(result.model_dump(), indent=2, default=str))
```

---

## 7. Design Decisions

### D-1: Unauthenticated Reddit Access

Reddit app registration was attempted but blocked by Reddit's anti-abuse system. The public unauthenticated JSON endpoints return identical data for public subreddits and are sufficient for this access pattern (low volume, daily batch).

**Mitigations built in**: 1-second inter-request delay, exponential backoff (30s → 2min → 10min), circuit breaker (pause 1 hour when >50% error rate), idempotent upserts, resumable job state.

**Migration path**: Switching to authenticated PRAW access is a one-file change in `sources/reddit_json.py` — all downstream pipeline logic is unaffected.

---

### D-2: Source-Agnostic Ingestion Layer

All ingestion logic is written against the `SourceAdapter` ABC, not against Reddit directly. Adding a new source (Bauforum24, r/de, YouTube comments) means implementing the three-method interface — not touching the pipeline.

This decision was made in v1 rather than deferred because retrofitting the abstraction after tight coupling is established costs significantly more than building it correctly from the start.

---

### D-3: Stance Classification over Direct Scoring

Three scoring approaches were evaluated:

| Approach | Decision |
|----------|----------|
| Direct LLM scoring (reads top-k, outputs 0–1) | Rejected — black box, no audit trail, biased by retrieval order |
| Stance classification → aggregate | Selected — interpretable, per-chunk audit trail, counter-evidence surfaced explicitly |
| Confidence-weighted classification | Selected (extension) — incorporates engagement, recency, and source quality signals |

The stance-classification approach was chosen specifically because each classified chunk is a citable piece of evidence. Stakeholders can inspect the `47 supports / 8 contradicts / 12 neutral` distribution rather than trusting a single unexplained score.

---

### D-4: Two-Tier LLM Strategy

Stance classification runs ~7 times per evaluation (batches of 15 chunks, 100 chunks total). At Sonnet pricing this is expensive. Haiku/mini-class models are 10–20x cheaper and are adequate for a structured 4-way classification task.

The synthesis pass is one call per evaluation. Output quality matters here (this is what stakeholders read). Sonnet-class is worth the cost for this single call.

---

### D-5: Store German Original, Embed German Only

- **Store both**: English translations serve human readability in evidence quotes. German originals are the ground truth for embedding and classification.
- **Embed German only (not both)**: Embedding both would double vector storage and cause duplicate retrieval. BGE-M3 is cross-lingual — an English query retrieves semantically similar German content natively, without translation.
- **Never embed English-only**: Translation introduces semantic drift on domain-specific terms (Bauträger, GEG, KfW-Förderung, Bodenplatte). These terms carry precise legal/technical meaning. The German original must be the embedding source.

---

### D-6: SQLite + LanceDB Dual-Store

| Option | Verdict |
|--------|---------|
| Postgres + pgvector | Rejected — requires a running server; overkill for single-user local use |
| SQLite + LanceDB | **Selected** — fully file-based, no server process, portable, fast |
| SQLite + Chroma | Rejected — Chroma's persistence has historically been unstable across versions |
| Postgres + Qdrant | Rejected — two servers, most complex setup |

LanceDB was chosen over Chroma specifically because it is file-based (a directory on disk, like SQLite), has better metadata filtering, and is more actively maintained as of 2024–2025.

**Sync integrity**: A single unified write path (`unified.store_chunk()`) is the only way to write chunks — direct access to `sqlite.py` or `lance.py` for chunk writes is prohibited. Shared key (`chunk_id`) allows cross-store verification.

---

### D-7: Chunking at Comment Level, Not Thread Level

Thread-level embedding averages across all comments, burying specific evidence in noise. r/hausbau's value is in individual comments — personal experience reports, specific timelines, named contractors and problems. Retrieving at thread level retrieves noise.

Sub-chunking triggers above 400 words with 50-word overlap to preserve context across sub-chunk boundaries.

---

### D-8: Keep Short German Comments

A `< 20 words` length filter was initially considered but rejected after observing that German comments frequently compress significant meaning into very short phrases:

> *"Wir haben 14 Monate auf den Bauantrag gewartet."*  
> (7 words = "We waited 14 months for the building permit.")

This is perfect evidence for a planning-speed hypothesis. The stance classifier handles genuine irrelevance — it will mark noise as `irrelevant`. The filter's job is only to eliminate structurally non-informative content (bots, deletions, pure acknowledgments).

---

### D-9: Full Evaluation History, Not Just Latest Score

Every evaluation run is fully persisted — not just the latest score per hypothesis. This decision cannot be retrofitted: if run #1's evidence classifications are not stored, the score history can never be reconstructed. The corpus grows daily; tracking how hypothesis scores evolve over time is the core long-term value proposition.

---

### D-10: Versioned Prompt Files

Prompts are stored in `prompts/stance_classification_v1.txt`, `prompts/synthesis_v1.txt`, `prompts/translation_v1.txt` — decoupled from code. Prompt iteration (A/B testing, rollback, audit) requires no code changes and the iteration history is explicit via version suffixes.

---

## 8. Technology Stack

| Technology | Version | Role | Why This Specifically |
|------------|---------|------|----------------------|
| **Python** | ≥ 3.11 | Runtime | Match/case, tomllib, `str \| None` syntax; stable ecosystem for ML tooling |
| **Typer** | ≥ 0.12 | CLI framework | Type-annotated commands, auto-generated `--help`, minimal boilerplate |
| **Pydantic v2** | ≥ 2.7 | Data validation & models | Fast Rust-backed validation, clean `model_dump()` / JSON serialization |
| **pydantic-settings** | ≥ 2.3 | Config from env vars | `.env` file loading, type coercion, central env-var access point |
| **SQLite** (stdlib) | — | Relational storage | Zero-config, file-based, fully transactional, built into Python stdlib |
| **LanceDB** | ≥ 0.10 | Vector store | File-based (no server), active maintenance, native Python, good metadata filtering |
| **BGE-M3** (`FlagEmbedding`) | ≥ 1.2 | Embeddings | Cross-lingual (100 langs), excellent German quality, dense+sparse+ColBERT in one pass, self-hosted |
| **PyTorch** | ≥ 2.0 | Adapter training | Required by FlagEmbedding; enables linear adapter fine-tuning with InfoNCE |
| **httpx** | ≥ 0.27 | HTTP client | Async-ready, clean API, timeout/retry control, better than `requests` for new projects |
| **Anthropic SDK** | ≥ 0.28 | Claude API | Claude Haiku (classification + translation), Claude Sonnet (synthesis) |
| **OpenAI SDK** | ≥ 1.0 | LLM fallback | OpenAI-compatible endpoint support (LM Studio, local models) |
| **Rich** | ≥ 13.7 | Terminal output | Tables, panels, colored text for readable CLI output |
| **Loguru** | ≥ 0.7 | Logging | Structured, colorized, one-line setup vs stdlib `logging` boilerplate |
| **NumPy** | ≥ 1.26 | Embedding arrays | Standard ndarray handling for embeddings; required by LanceDB and BGE-M3 |
| **Pandas** | ≥ 2.0 | DataFrame utilities | LanceDB results come as DataFrames; used for batch processing |
| **pytest** | ≥ 8.2 | Testing | Standard Python testing; `pytest-mock` for unit isolation |
| **Ruff** | ≥ 0.4 | Linting + formatting | 10–100x faster than flake8/black, single tool, same rules |
| **Docker Compose** | v3.9 | Deployment | Reproducible environment, pinned BGE-M3 model dependencies, mounted data volumes |

---

## 9. Database Schema

### SQLite — Full Schema

```sql
-- ─────────────────────────────────────────────
-- Ingestion Tables
-- ─────────────────────────────────────────────

CREATE TABLE sources (
    id          TEXT PRIMARY KEY,    -- e.g. 'reddit:r/hausbau'
    name        TEXT,
    config      JSON,                -- adapter-specific config blob
    created_at  TEXT
);

CREATE TABLE posts (
    id            TEXT PRIMARY KEY,  -- Reddit post ID (e.g. 't3_abc123' prefix stripped)
    source_id     TEXT REFERENCES sources(id),
    title         TEXT,
    body          TEXT,
    author        TEXT,
    created_utc   INTEGER,           -- Unix timestamp
    score         INTEGER,           -- upvote count
    upvote_ratio  REAL,              -- 0.0–1.0
    num_comments  INTEGER,
    flair         TEXT,
    url           TEXT,
    is_self       INTEGER,           -- 1 = text post, 0 = link post
    edited        INTEGER,           -- Unix timestamp or 0
    fetched_at    TEXT               -- ISO-8601 UTC
);

CREATE TABLE comments (
    id            TEXT PRIMARY KEY,  -- Reddit comment ID
    post_id       TEXT REFERENCES posts(id),
    parent_id     TEXT,              -- parent comment or post ID (t1_/t3_ prefix stripped)
    author        TEXT,
    body          TEXT,
    created_utc   INTEGER,
    score         INTEGER,
    depth         INTEGER,           -- 0 = top-level comment
    is_submitter  INTEGER,           -- 1 = OP
    edited        INTEGER,
    fetched_at    TEXT
);

CREATE TABLE authors (
    username             TEXT PRIMARY KEY,
    link_karma           INTEGER,
    comment_karma        INTEGER,
    account_created_utc  INTEGER,
    captured_at          TEXT
);

CREATE TABLE chunks (
    chunk_id          TEXT PRIMARY KEY,  -- stable UUID (source_id + char_offset)
    source_type       TEXT,              -- 'post' | 'comment'
    source_id         TEXT,              -- post or comment ID
    parent_post_id    TEXT,
    parent_post_title TEXT,
    parent_post_body  TEXT,
    text_de           TEXT,              -- German original
    text_en           TEXT,              -- English translation (NULL until translated)
    char_offset       INTEGER,           -- byte offset into source text (for sub-chunks)
    is_filtered       INTEGER DEFAULT 0, -- 1 = excluded by filter; kept for audit
    created_at        TEXT
);

-- ─────────────────────────────────────────────
-- Evaluation Tables
-- ─────────────────────────────────────────────

CREATE TABLE hypotheses (
    id          TEXT PRIMARY KEY,    -- UUID
    text        TEXT,
    language    TEXT,                -- 'de' | 'en'
    created_at  TEXT
);

CREATE TABLE evaluation_runs (
    id                    TEXT PRIMARY KEY,  -- UUID
    hypothesis_id         TEXT REFERENCES hypotheses(id),
    run_at                TEXT,              -- ISO-8601 UTC
    score                 REAL,             -- 0.0–1.0
    confidence            REAL,             -- 0.0–1.0
    sample_size           INTEGER,          -- total chunks classified
    stance_supports       INTEGER,
    stance_contradicts    INTEGER,
    stance_neutral        INTEGER,
    stance_irrelevant     INTEGER,
    synthesis             TEXT,             -- prose summary
    model_classification  TEXT,             -- model ID used for classification
    model_synthesis       TEXT              -- model ID used for synthesis
);

CREATE TABLE evidence_classifications (
    id               TEXT PRIMARY KEY,  -- UUID
    run_id           TEXT REFERENCES evaluation_runs(id),
    chunk_id         TEXT REFERENCES chunks(chunk_id),
    stance           TEXT,             -- 'supports' | 'contradicts' | 'neutral' | 'irrelevant'
    rationale        TEXT,             -- one-sentence rationale from LLM
    weight           REAL,             -- confidence weight computed by aggregator
    retrieval_score  REAL,             -- cosine similarity from vector search
    created_at       TEXT
);

-- ─────────────────────────────────────────────
-- Operational Tables
-- ─────────────────────────────────────────────

CREATE TABLE ingestion_jobs (
    id             TEXT PRIMARY KEY,  -- UUID
    mode           TEXT,              -- 'backfill' | 'delta'
    source_id      TEXT,
    status         TEXT,              -- 'running' | 'complete' | 'failed' | 'paused'
    cursor         TEXT,              -- Reddit pagination cursor (for resumption)
    posts_fetched  INTEGER DEFAULT 0,
    started_at     TEXT,
    updated_at     TEXT,
    error          TEXT               -- last error message if status='failed'
);

-- ─────────────────────────────────────────────
-- Indexes
-- ─────────────────────────────────────────────

CREATE INDEX idx_chunks_source_id        ON chunks(source_id);
CREATE INDEX idx_chunks_parent_post_id   ON chunks(parent_post_id);
CREATE INDEX idx_evidence_run_id         ON evidence_classifications(run_id);
CREATE INDEX idx_evaluation_runs_hyp     ON evaluation_runs(hypothesis_id);
CREATE INDEX idx_posts_created           ON posts(created_utc);
CREATE INDEX idx_comments_post_id        ON comments(post_id);
```

---

### LanceDB — Vector Table Schema

LanceDB stores minimal metadata alongside each vector — only fields needed for filtered retrieval. Full metadata lives in SQLite and is hydrated after retrieval.

```python
class ChunkVector(LanceModel):
    chunk_id:       str           # matches SQLite chunks.chunk_id
    vector:         Vector(1024)  # BGE-M3 dense embedding (L2-normalized)
    source_type:    str           # 'post' | 'comment' — enables type-filtered search
    parent_post_id: str           # enables thread-scoped retrieval
    created_utc:    int           # enables recency-filtered search
    score:          int           # upvote score — enables quality-filtered search
```

---

### Confidence Weight Formula

The per-chunk weight used in score aggregation:

```
weight(chunk, retrieval_score) =
    retrieval_score                              [cosine similarity, 0–1]
  × exp(-0.693 × age_days / 180)               [recency: 180-day half-life]
  × log1p(max(chunk.score, 1)) / log1p(1000)   [engagement: log-normalized upvotes]
  × 1 / (1 + chunk.depth × 0.2)               [depth penalty: top-level preferred]
  × (0.8 + 0.2 × log1p(max(karma,1))/log1p(100000))  [author karma signal]
```

All constants are configurable in `config.py` and in the `.env` file.

---

### Score Aggregation Formula

```
score      = Σ(weight_i × stance_value_i) / Σ(weight_i)
             where stance_values = {supports: 1.0, contradicts: 0.0, neutral: 0.5}
             irrelevant chunks are excluded from the sum

confidence = agreement_rate × min(n / 50, 1.0)
             where agreement_rate = max(supports, n-supports) / n
             and n = number of relevant chunks (supports + contradicts + neutral)
```

---

## 10. Instructions to Run

### Prerequisites

- Docker and Docker Compose
- An Anthropic API key (`sk-ant-...`)
- (Optional) An OpenAI API key or local LM Studio instance for LLM fallback

---

### Option A — Docker (Recommended)

#### Step 1: Clone and Configure

```bash
git clone https://github.com/yourname/hyporeddit-validator
cd hyporeddit-validator

cp .env.example .env
# Open .env and set ANTHROPIC_API_KEY=sk-ant-...
```

#### Step 2: Build the Image

```bash
docker compose build
```

#### Step 3: Bootstrap (One-Time)

```bash
# Fetch up to 1,000 historical posts and all their comments
docker compose run --rm app hyporeddit backfill --limit 1000

# Chunk, embed, and store all fetched content
# Add --make-translation to also translate chunks DE→EN (uses Haiku API)
docker compose run --rm app hyporeddit process --make-translation

# Verify SQLite and LanceDB are in sync (should report 0 orphans)
docker compose run --rm app hyporeddit verify-stores

# Check corpus statistics
docker compose run --rm app hyporeddit stats
```

Expected output from `stats` after a full backfill:
```
Posts:     ~1,000
Comments:  ~8,000–15,000
Chunks:    ~10,000–30,000
```

#### Step 4: Evaluate a Hypothesis

```bash
# English hypothesis
docker compose run --rm app hyporeddit evaluate "Homebuilders find the planning process too slow"

# German hypothesis
docker compose run --rm app hyporeddit evaluate "Bauherren empfinden den Planungsprozess als zu langsam"

# JSON output for scripting
docker compose run --rm app hyporeddit evaluate "Planning is too complex" --json

# Larger evidence set
docker compose run --rm app hyporeddit evaluate "Planning is too complex" --top-k 200

# Force recompute (bypass cache)
docker compose run --rm app hyporeddit evaluate "Planning is too complex" --force-rerun
```

#### Step 5: Daily Refresh (Set Up Cron)

Add to your host crontab to keep the corpus current:

```cron
0 6 * * * cd /path/to/hyporeddit-validator && \
  docker compose run --rm app hyporeddit ingest && \
  docker compose run --rm app hyporeddit process --make-translation
```

---

### Option B — Local Python (Without Docker)

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1       # Windows PowerShell

# Install the package with all dependencies
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY

# Bootstrap
hyporeddit backfill --limit 1000
hyporeddit process --make-translation
hyporeddit verify-stores
hyporeddit stats

# Evaluate
hyporeddit evaluate "Homebuilders find the planning process too slow"
```

> **Note**: BGE-M3 will be downloaded from Hugging Face on first run (~2 GB). Set `BGE_M3_DEVICE=cuda` in `.env` if a CUDA GPU is available — embedding will be 10–20x faster.

---

### Option C — Jupyter Notebook

```python
# In a notebook cell, after installing the package:
from hyporeddit.evaluation.pipeline import evaluate_hypothesis

result = evaluate_hypothesis("Homebuilders find the planning process too slow")

print(f"Score:      {result.score:.2f}")
print(f"Confidence: {result.confidence:.2f}")
print(f"Stances:    {result.stance_distribution}")
print(f"Samples:    {result.sample_size} chunks")
print()
print(result.synthesis)
print()
for item in result.evidence[:5]:
    print(f"[{item.stance.upper():12}] {item.text_en or item.text_de}")
    print(f"  Weight: {item.weight:.3f}  |  Source: {item.source_url}")
```

---

### Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | _(required)_ | Anthropic API key for Claude Haiku + Sonnet |
| `OPENAI_API_KEY` | `lm-studio` | OpenAI API key (or dummy value for LM Studio) |
| `LLM_BASE_URL` | `http://localhost:1234/v1` | Base URL for OpenAI-compatible API |
| `LLM_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `LLM_CLASSIFICATION_MODEL` | `claude-haiku-4-5-20251001` | Model for stance classification + translation |
| `LLM_SYNTHESIS_MODEL` | `claude-sonnet-4-6` | Model for synthesis |
| `OPENAI_CLASSIFICATION_MODEL` | `deepseek-r1-distill-qwen-7b` | Model when `LLM_PROVIDER=openai` |
| `SQLITE_PATH` | `data/sqlite/hyporeddit.db` | SQLite database file path |
| `LANCE_PATH` | `data/lance` | LanceDB directory path |
| `BGE_M3_DEVICE` | `cpu` | `cpu` or `cuda` |
| `BGE_M3_BATCH_SIZE` | `32` | BGE-M3 encoding batch size |
| `ADAPTER_PATH` | `data/model/adapter.pt` | Linear adapter checkpoint path |
| `ADAPTER_TRAIN_THRESHOLD` | `200` | New chunks in a process run that triggers adapter training |
| `REQUEST_DELAY_SECONDS` | `1.0` | Seconds between HTTP requests |
| `RETRIEVAL_TOP_K` | `100` | Default evidence chunks per evaluation |
| `CLASSIFICATION_BATCH_SIZE` | `15` | Chunks per LLM classification call |
| `RECENCY_HALF_LIFE_DAYS` | `180.0` | Half-life for recency decay in confidence weighting |
| `LOG_LEVEL` | `INFO` | Loguru log verbosity |

---

### Troubleshooting

| Symptom | Likely Cause | Resolution |
|---------|-------------|------------|
| `verify-stores` reports orphaned chunks | LanceDB write failed after SQLite write | Run `hyporeddit verify-stores --fix` |
| `evaluate` returns cached result unexpectedly | Evaluation cache hit | Use `--force-rerun` to recompute |
| 403 errors during backfill | Reddit/Cloudflare rate limiting | Wait 1 hour; the circuit breaker handles this automatically |
| BGE-M3 download hangs | Hugging Face connectivity | Set `HF_ENDPOINT` or pre-download the model |
| `process` produces no chunks | Posts were fetched but not processed | Check `hyporeddit stats`; ensure `backfill` completed before `process` |
| Empty synthesis text | No relevant evidence found in corpus | Increase `--top-k` or verify corpus size with `hyporeddit stats` |

---