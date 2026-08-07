# pip install litellm python-dotenv numpy
import os, json, re, asyncio
import numpy as np
from dotenv import load_dotenv
from litellm import completion, aembedding

load_dotenv()

# Silence LiteLLM async logging warning
import litellm
litellm.success_callback = []
litellm._async_success_callback = []

def _normalize_base_url(raw):
    if not raw: return ""
    val = raw.strip().strip('"').strip("'")
    m = re.match(r"^\[[^\]]+\]\(([^)]+)\)$", val)
    return m.group(1).strip() if m else val

API_KEY  = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY")
API_BASE = _normalize_base_url(os.environ.get("OPENAI_API_BASE")
                               or os.environ.get("OPENAI_BASE_URL") or "")
EMBED_MODEL = "text-embedding-3-large"
CHAT_MODEL  = "gpt-4.1-mini"

TOP_K = int(os.environ.get("TOP_K", "5"))
CONTEXT_CHARS = int(os.environ.get("CONTEXT_CHARS", "1200"))  # per-lesson cap in context

def get(rec, *keys):
    for k in keys:
        if rec.get(k):
            return rec[k]
    return ""

def load_records(path="llis_clean.json"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [data] if isinstance(data, dict) else data

# ── Retrieval ─────────────────────────────────────────────────
async def _embed_query(text):
    kwargs = {"model": EMBED_MODEL, "input": [text], "api_key": API_KEY, "timeout": 60}
    if API_BASE:
        kwargs["api_base"] = API_BASE
    resp = await aembedding(**kwargs)
    v = np.asarray(resp["data"][0]["embedding"], dtype="float32")
    return v / (np.linalg.norm(v) + 1e-12)

def retrieve(query, records, embeddings, k=TOP_K):
    q = asyncio.run(_embed_query(query))
    scores = embeddings @ q
    top = np.argsort(-scores)[:k]
    hits = []
    for i in top:
        r = records[i]
        hits.append({
            "score":     float(scores[i]),
            "lesson_id": get(r, "Lesson ID", "lesson_id"),
            "title":     get(r, "Lesson Title**", "title"),
            "center":    get(r, "Center", "center"),
            "driving_event":  get(r, "Driving Event", "driving_event"),
            "lesson_learned": get(r, "Lesson Learned", "lesson_learned"),
            "recommendation": get(r, "Recommendation(s)", "recommendation"),
        })
    return hits

# ── Build context block from retrieved lessons ────────────────
def build_context(hits):
    blocks = []
    for h in hits:
        body = (
            f"Driving Event: {h['driving_event']}\n"
            f"Lesson Learned: {h['lesson_learned']}\n"
            f"Recommendation: {h['recommendation']}"
        )[:CONTEXT_CHARS]
        blocks.append(
            f"[Lesson {h['lesson_id']}] {h['title']} ({h['center']})\n{body}"
        )
    return "\n\n---\n\n".join(blocks)

# ── Generate a grounded answer ────────────────────────────────
SYSTEM_PROMPT = (
    "You are a NASA Lessons Learned assistant. Answer the user's question using ONLY "
    "the provided lessons. Cite the specific lessons you use by their ID in the form "
    "[Lesson <id>]. If the provided lessons do not contain enough information to answer, "
    "say so clearly rather than guessing. Be concise and practical."
)

def answer(query, hits):
    context = build_context(hits)
    user_msg = (
        f"QUESTION:\n{query}\n\n"
        f"RELEVANT LESSONS:\n{context}\n\n"
        "Answer the question, citing lessons by [Lesson <id>]."
    )
    kwargs = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "api_key": API_KEY,
        "temperature": 0.2,
        "timeout": 90,
    }
    if API_BASE:
        kwargs["api_base"] = API_BASE
    resp = completion(**kwargs)
    return resp["choices"][0]["message"]["content"].strip()

# ── Interactive loop ──────────────────────────────────────────
if __name__ == "__main__":
    records = load_records()
    embeddings = np.load("llis_embeddings.npy")
    assert embeddings.shape[0] == len(records), "records/embeddings mismatch"
    print(f"Loaded {len(records)} lessons + embeddings {embeddings.shape}")
    print(f"Hazard Chatbot ready (top_k={TOP_K}). Type your question, or 'quit' to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not query:
            continue
        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        hits = retrieve(query, records, embeddings)
        reply = answer(query, hits)

        print(f"\nAssistant:\n{reply}\n")
        print("Sources:")
        for h in hits:
            print(f"  [Lesson {h['lesson_id']}] {h['title']} "
                  f"({h['center']})  score={h['score']:.3f}")
        print()