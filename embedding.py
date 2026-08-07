# pip install litellm python-dotenv numpy
import os
import json
import asyncio
import re
import random
import time
import numpy as np
from dotenv import load_dotenv
from litellm import aembedding

load_dotenv()

def _normalize_base_url(raw):
    """Accept plain URL or markdown-link-like values and return a clean URL."""
    if not raw:
        return ""
    val = raw.strip().strip('"').strip("'")
    match = re.match(r"^\[[^\]]+\]\(([^)]+)\)$", val)
    if match:
        val = match.group(1).strip()
    return val

# Prefer OpenAI-style vars first, then compatibility fallbacks.
API_KEY = (
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("API_KEY")
    or os.environ.get("ANTHROPIC_AUTH_TOKEN")
)
API_BASE = _normalize_base_url(
    os.environ.get("OPENAI_API_BASE")
    or os.environ.get("OPENAI_BASE_URL")
    or os.environ.get("ANTHROPIC_BASE_URL")
    or ""
)

if not API_KEY:
    raise EnvironmentError(
        "No API key found. Set OPENAI_API_KEY (preferred) or API_KEY in .env"
    )

EMBED_MODEL = "text-embedding-3-large"
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "25"))
EMBED_MAX_CONCURRENCY = int(os.environ.get("EMBED_MAX_CONCURRENCY", "1"))
EMBED_MAX_RETRIES = int(os.environ.get("EMBED_MAX_RETRIES", "8"))
EMBED_REQUEST_TIMEOUT = float(os.environ.get("EMBED_REQUEST_TIMEOUT", "120"))
EMBED_CHECKPOINT_PATH = os.environ.get("EMBED_CHECKPOINT_PATH", "llis_embeddings.partial.npy")
EMBED_STATE_PATH = os.environ.get("EMBED_STATE_PATH", "llis_embeddings.state.json")
RUN_DEMO_SEARCH = os.environ.get("RUN_DEMO_SEARCH", "0") == "1"

# ── Text builder ──────────────────────────────────────────────
def build_search_text(rec):
    parts = [
        rec.get("Lesson Title**", "") or rec.get("title", ""),
        rec.get("Driving Event", "") or rec.get("driving_event", ""),
        rec.get("Lesson Learned", "") or rec.get("lesson_learned", ""),
        rec.get("Recommendation(s)", "") or rec.get("recommendation", ""),
    ]
    return "\n".join(p for p in parts if p).strip()

def load_records(path="llis_clean.json"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [data] if isinstance(data, dict) else data

def _fmt_duration(seconds):
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"

def _load_checkpoint(total_texts, batch_size):
    if not (os.path.exists(EMBED_CHECKPOINT_PATH) and os.path.exists(EMBED_STATE_PATH)):
        return 0, []

    try:
        with open(EMBED_STATE_PATH, encoding="utf-8") as f:
            state = json.load(f)
        if state.get("total_texts") != total_texts or state.get("batch_size") != batch_size:
            return 0, []

        embeddings = np.load(EMBED_CHECKPOINT_PATH)
        completed_batches = int(state.get("completed_batches", 0))
        expected_vectors = min(total_texts, completed_batches * batch_size)
        if embeddings.shape[0] != expected_vectors:
            return 0, []
        return completed_batches, [embeddings[i] for i in range(embeddings.shape[0])]
    except Exception:
        return 0, []

def _save_checkpoint(all_vecs, completed_batches, total_batches, total_texts, batch_size):
    arr = np.asarray(all_vecs, dtype="float32")
    np.save(EMBED_CHECKPOINT_PATH, arr)
    with open(EMBED_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "completed_batches": completed_batches,
                "total_batches": total_batches,
                "total_texts": total_texts,
                "batch_size": batch_size,
            },
            f,
            indent=2,
        )

# ── Async batched embedding ───────────────────────────────────
async def embed_one_batch(batch):
    kwargs = {
        "model": EMBED_MODEL,
        "input": batch,
        "api_key": API_KEY,
        "timeout": EMBED_REQUEST_TIMEOUT,
    }
    if API_BASE:
        kwargs["api_base"] = API_BASE
    for attempt in range(EMBED_MAX_RETRIES + 1):
        try:
            resp = await asyncio.wait_for(
                aembedding(**kwargs),
                timeout=EMBED_REQUEST_TIMEOUT + 10,
            )
            return [d["embedding"] for d in resp["data"]]
        except Exception as exc:
            msg = str(exc).lower()
            is_rate_limited = "429" in msg or "ratelimit" in msg or "throttl" in msg
            is_timeout = "timeout" in msg or isinstance(exc, asyncio.TimeoutError)
            if (not is_rate_limited and not is_timeout) or attempt >= EMBED_MAX_RETRIES:
                raise

            # Exponential backoff with jitter to reduce synchronized retry bursts.
            delay = min(60.0, (2 ** attempt)) + random.uniform(0, 0.5)
            reason = "rate limit" if is_rate_limited else "timeout"
            print(
                f"  {reason}; retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{EMBED_MAX_RETRIES})"
            )
            await asyncio.sleep(delay)

async def embed_batch_adaptive(batch, min_chunk_size=1):
    """Embed a batch; on persistent timeout, split into smaller chunks."""
    try:
        return await embed_one_batch(batch)
    except Exception as exc:
        msg = str(exc).lower()
        is_timeout_like = "timeout" in msg or isinstance(exc, asyncio.TimeoutError)
        if not is_timeout_like or len(batch) <= min_chunk_size:
            raise

        mid = max(1, len(batch) // 2)
        left = batch[:mid]
        right = batch[mid:]
        print(
            f"  adaptive split: timeout on size {len(batch)} "
            f"-> {len(left)} + {len(right)}"
        )
        left_vecs = await embed_batch_adaptive(left, min_chunk_size=min_chunk_size)
        right_vecs = await embed_batch_adaptive(right, min_chunk_size=min_chunk_size)
        return left_vecs + right_vecs

async def embed_texts_async(
    texts,
    batch_size=EMBED_BATCH_SIZE,
    max_concurrency=EMBED_MAX_CONCURRENCY,
):
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    start = time.monotonic()
    completed_batches, all_vecs = _load_checkpoint(len(texts), batch_size)

    if completed_batches:
        elapsed = time.monotonic() - start
        print(
            f"  resuming from batch {completed_batches + 1}/{len(batches)}"
            f" | completed {completed_batches}/{len(batches)}"
            f" | elapsed {_fmt_duration(elapsed)}"
        )

    sem = asyncio.Semaphore(max_concurrency)  # cap concurrent requests
    for idx in range(completed_batches, len(batches)):
        async with sem:
            vecs = await embed_batch_adaptive(batches[idx])
        all_vecs.extend(vecs)

        completed_batches = idx + 1
        elapsed = time.monotonic() - start
        rate = completed_batches / elapsed if elapsed > 0 else 0
        remaining = len(batches) - completed_batches
        eta = (remaining / rate) if rate > 0 else float("inf")
        print(
            f"  batch {idx + 1}/{len(batches)} done ({len(vecs)} vecs)"
            f" | completed {completed_batches}/{len(batches)}"
            f" | elapsed {_fmt_duration(elapsed)}"
            f" | eta {_fmt_duration(eta)}"
        )
        _save_checkpoint(all_vecs, completed_batches, len(batches), len(texts), batch_size)

    arr = np.asarray(all_vecs, dtype="float32")
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12  # L2 normalize
    return arr

# ── Search ────────────────────────────────────────────────────
def search(query, records, embeddings, k=5):
    q = asyncio.run(embed_texts_async([query]))[0]
    scores = embeddings @ q
    top = np.argsort(-scores)[:k]
    return [{
        "score":     round(float(scores[i]), 4),
        "lesson_id": records[i].get("Lesson ID") or records[i].get("lesson_id"),
        "title":     records[i].get("Lesson Title**") or records[i].get("title"),
        "center":    records[i].get("Center") or records[i].get("center"),
    } for i in top]

if __name__ == "__main__":
    records = load_records()
    print(f"Loaded {len(records)} records.")

    texts = [build_search_text(r) for r in records]
    embeddings = asyncio.run(embed_texts_async(texts))
    np.save("llis_embeddings.npy", embeddings)
    for path in (EMBED_CHECKPOINT_PATH, EMBED_STATE_PATH):
        if os.path.exists(path):
            os.remove(path)
    print(f"Saved embeddings: shape {embeddings.shape}")

    if RUN_DEMO_SEARCH:
        for hit in search("electrical connector torque failure", records, embeddings):
            print(f"[{hit['score']}] {hit['lesson_id']} — {hit['title']} ({hit['center']})")