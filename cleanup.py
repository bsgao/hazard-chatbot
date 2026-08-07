import csv
import re
import json
from html import unescape

# ── Column order (renamed as requested) ───────────────────────
COLUMNS = [
    "Lesson Title**", "Center", "Lesson Date**", "Publish Date**", "Categories",
    "Driving Event", "Lesson Learned", "Recommendation(s)", "Abstract",
    "Lesson ID", "Lesson Documented", "Related Policy**",
    "Attachments", "Submitters",
]

# ── Text cleaning ─────────────────────────────────────────────
def clean_text(raw):
    """Strip HTML, unescape entities, collapse whitespace,
       and normalize 'Not Provided' to empty string."""
    if raw is None:
        return ""
    txt = re.sub(r"<[^>]+>", " ", raw)     # remove HTML tags
    txt = unescape(txt)                     # &amp; -> &, etc.
    txt = re.sub(r"\s+", " ", txt).strip()  # collapse whitespace
    if txt.lower() in {"not provided", "none", ""}:
        return ""
    return txt

def clean_date(raw):
    """Keep just YYYY-MM-DD from an ISO timestamp."""
    if not raw:
        return ""
    return raw.strip()[:10]

def split_categories(raw):
    if not raw:
        return []
    return [c.strip() for c in re.split(r"[;,]", raw) if c.strip()]

# ── Parse one raw row (list of columns) into a clean record ───
def parse_record(cols):
    # Pad short rows so indexing never fails
    cols = list(cols) + [""] * (len(COLUMNS) - len(cols))
    raw = dict(zip(COLUMNS, cols))

    return {
        "Lesson ID":       raw["Lesson ID"].strip(),
        "Lesson Title**":  clean_text(raw["Lesson Title**"]),
        "Center":          raw["Center"].strip(),
        "Lesson Date**":   clean_date(raw["Lesson Date**"]),
        "Publish Date**":  clean_date(raw["Publish Date**"]),
        "Categories":      split_categories(raw["Categories"]),
        "Driving Event":   clean_text(raw["Driving Event"]),
        "Lesson Learned":  clean_text(raw["Lesson Learned"]),
        "Recommendation(s)": clean_text(raw["Recommendation(s)"]),
        "Abstract":        clean_text(raw["Abstract"]),
    }

# ── Load the whole dataset ────────────────────────────────────
def load_dataset(path):
    records, skipped = [], 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        # This export has a metadata preamble line, then comma-delimited rows.
        reader = csv.reader(f, delimiter=",", quotechar='"')
        for i, row in enumerate(reader):
            if not row or all(not c.strip() for c in row):
                continue  # skip blank lines

            # Skip SharePoint export metadata preamble line.
            if i == 0 and row and row[0].startswith("ListSchema="):
                continue

            rec = parse_record(row)
            # keep only records that have usable hazard signal
            if rec["Driving Event"] or rec["Lesson Learned"]:
                records.append(rec)
            else:
                skipped += 1
    print(f"Parsed {len(records)} records, skipped {skipped} empty.")
    return records

# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    recs = load_dataset("LLIS General Access Database.csv")   # <- your file
    # Save clean structured output for the next step
    with open("llis_clean.json", "w", encoding="utf-8") as out:
        json.dump(recs, out, indent=2, ensure_ascii=False)
    # Peek at the first record when available
    if recs:
        print(json.dumps(recs[0], indent=2, ensure_ascii=False))
    else:
        print("No usable records found after cleanup.")