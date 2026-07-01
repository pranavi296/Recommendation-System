"""
Real-time recommendation engine demo.

Architecture:
  1. Offline layer  -> TF-IDF embeddings built once from item category+tags
  2. Online layer    -> a per-session vector that updates on every click
                         (exponentially-weighted average of clicked item vectors)
  3. Serving layer   -> cosine similarity retrieval blended with popularity,
                         plus epsilon-greedy exploration so the feed doesn't
                         collapse into one category

No page reload is needed: the frontend calls POST /api/interact after every
click and re-renders the "Recommended for you" rail with the response.
"""

import json
import random
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "items.json"
STATIC_DIR = BASE_DIR / "static"

# ---- Tunable knobs -----------------------------------------------------
SESSION_DECAY = 0.65      # weight kept from the old session vector on each click
EXPLORATION_EPSILON = 0.15  # chance a slot is filled with a random unseen item
TOP_N = 6

app = Flask(__name__, static_folder=None)

# ---- Load catalog & build offline embeddings ---------------------------
with open(DATA_PATH) as f:
    ITEMS = json.load(f)

ITEM_BY_ID = {item["id"]: item for item in ITEMS}
ITEM_IDS = [item["id"] for item in ITEMS]

def item_text(item):
    # category is repeated so it carries a bit more weight than a single tag
    return f"{item['category']} {item['category']} " + " ".join(item["tags"])

corpus = [item_text(item) for item in ITEMS]
vectorizer = TfidfVectorizer()
ITEM_MATRIX = vectorizer.fit_transform(corpus).toarray()  # shape (n_items, n_features)
ITEM_INDEX = {item_id: i for i, item_id in enumerate(ITEM_IDS)}

# popularity as a fallback ranking signal, normalized to [0, 1]
POPULARITY = np.array([item["popularity"] for item in ITEMS])

# ---- In-memory session store (fine for a demo; swap for Redis in prod) --
SESSIONS = {}

def new_session():
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {
        "vector": np.zeros(ITEM_MATRIX.shape[1]),
        "clicked": [],       # ordered list of item ids
        "seen_categories": {},
    }
    return sid

def get_session(sid):
    if not sid or sid not in SESSIONS:
        sid = new_session()
    return sid, SESSIONS[sid]


def update_session_vector(session, item_id):
    idx = ITEM_INDEX[item_id]
    item_vec = ITEM_MATRIX[idx]
    old = session["vector"]
    if np.linalg.norm(old) == 0:
        new = item_vec
    else:
        new = SESSION_DECAY * old + (1 - SESSION_DECAY) * item_vec
    session["vector"] = new
    session["clicked"].append(item_id)
    cat = ITEM_BY_ID[item_id]["category"]
    session["seen_categories"][cat] = session["seen_categories"].get(cat, 0) + 1


def recommend(session, top_n=TOP_N):
    clicked_set = set(session["clicked"])
    candidates = [iid for iid in ITEM_IDS if iid not in clicked_set]

    if np.linalg.norm(session["vector"]) == 0:
        # cold start: pure popularity ranking
        scored = sorted(
            candidates,
            key=lambda iid: POPULARITY[ITEM_INDEX[iid]],
            reverse=True,
        )
        results = []
        for iid in scored[:top_n]:
            results.append({**ITEM_BY_ID[iid], "score": round(float(POPULARITY[ITEM_INDEX[iid]]), 3), "reason": "popular"})
        return results

    cand_idx = [ITEM_INDEX[iid] for iid in candidates]
    sims = cosine_similarity(session["vector"].reshape(1, -1), ITEM_MATRIX[cand_idx])[0]
    # blend similarity with a light popularity prior so ties favor known-good content
    blended = 0.85 * sims + 0.15 * POPULARITY[cand_idx]

    ranked = sorted(zip(candidates, blended, sims), key=lambda t: t[1], reverse=True)

    top_dominant_category = max(session["seen_categories"], key=session["seen_categories"].get)

    results = []
    used_ids = set()
    for iid, score, sim in ranked:
        if len(results) >= top_n:
            break
        results.append({
            **ITEM_BY_ID[iid],
            "score": round(float(score), 3),
            "reason": "similar to your recent clicks" if sim > 0.05 else "popular",
        })
        used_ids.add(iid)

    # epsilon-greedy exploration: swap in a random item from a category the
    # session hasn't focused on, so the feed keeps some diversity
    if random.random() < EXPLORATION_EPSILON and results:
        other_candidates = [
            iid for iid in candidates
            if iid not in used_ids and ITEM_BY_ID[iid]["category"] != top_dominant_category
        ]
        if other_candidates:
            explore_id = random.choice(other_candidates)
            results[-1] = {**ITEM_BY_ID[explore_id], "score": None, "reason": "exploring something new"}

    return results


# ---- Routes -------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)


@app.route("/api/session", methods=["POST"])
def create_session():
    sid = new_session()
    session = SESSIONS[sid]
    return jsonify({
        "session_id": sid,
        "catalog": ITEMS,
        "recommended": recommend(session),
    })


@app.route("/api/interact", methods=["POST"])
def interact():
    body = request.get_json(force=True)
    sid = body.get("session_id")
    item_id = body.get("item_id")

    sid, session = get_session(sid)
    if item_id not in ITEM_BY_ID:
        return jsonify({"error": "unknown item_id"}), 400

    update_session_vector(session, item_id)
    recs = recommend(session)

    return jsonify({
        "session_id": sid,
        "clicked_item": ITEM_BY_ID[item_id],
        "recommended": recs,
        "click_history": [ITEM_BY_ID[i]["title"] for i in session["clicked"][-5:]],
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    body = request.get_json(force=True) or {}
    sid = body.get("session_id")
    if sid in SESSIONS:
        del SESSIONS[sid]
    new_sid = new_session()
    return jsonify({"session_id": new_sid, "recommended": recommend(SESSIONS[new_sid])})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
