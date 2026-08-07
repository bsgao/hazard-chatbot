# pip install litellm python-dotenv numpy
import os, json, asyncio, re
import numpy as np
from dotenv import load_dotenv
from litellm import aembedding

load_dotenv()

def _normalize_base_url(raw):
    if not raw: return ""
    val = raw.strip().strip('"').strip("'")
    m = re.match(r"^\[[^\]]+\]\(([^)]+)\)$", val)
    return m.group(1).strip() if m else val

API_KEY  = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
API_BASE = _normalize_base_url(os.environ.get("OPENAI_API_BASE")
                               or os.environ.get("OPENAI_BASE_URL") or "")
EMBED_MODEL = "text-embedding-3-large"

def load_records(path="llis_clean.json"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [data] if isinstance(data, dict) else data

def get(rec, *keys):
    for k in keys:
        if rec.get(k):
            return rec[k]
    return ""

async def _embed_query(text):
    kwargs = {"model": EMBED_MODEL, "input": [text], "api_key": API_KEY, "timeout": 60}
    if API_BASE:
        kwargs["api_base"] = API_BASE
    resp = await aembedding(**kwargs)
    v = np.asarray(resp["data"][0]["embedding"], dtype="float32")
    return v / (np.linalg.norm(v) + 1e-12)   # normalize to match corpus

def search(query, records, embeddings, k=5):
    q = asyncio.run(_embed_query(query))
    scores = embeddings @ q                    # cosine sim (both normalized)
    top = np.argsort(-scores)[:k]
    results = []
    for i in top:
        r = records[i]
        results.append({
            "score":     round(float(scores[i]), 4),
            "lesson_id": get(r, "Lesson ID", "lesson_id"),
            "title":     get(r, "Lesson Title**", "title"),
            "center":    get(r, "Center", "center"),
            "snippet":   get(r, "Lesson Learned", "lesson_learned")[:200],
        })
    return results

if __name__ == "__main__":
    records = load_records()
    embeddings = np.load("llis_embeddings.npy")   # (2166, 3072), already normalized
    assert embeddings.shape[0] == len(records), \
        f"Mismatch: {embeddings.shape[0]} vectors vs {len(records)} records"
    print(f"Loaded {len(records)} records + embeddings {embeddings.shape}")

    # Try a few queries
    for query in [
        "electrical connector torque failure",
        "software testing gaps before launch",
        "contamination during assembly",
        "Overheating of components"
    ]:
        print(f"\n=== {query} ===")
        for hit in search(query, records, embeddings):
            print(f"[{hit['score']}] {hit['lesson_id']} — {hit['title']} ({hit['center']})")
            print(f"      {hit['snippet']}...")