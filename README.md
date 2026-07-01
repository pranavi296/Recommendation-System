AI RECOMMENDATION SYSTEM
Real-time recommendation engine that re-ranks suggestions on every user click using session-vector embeddings, cosine similarity, and epsilon-greedy exploration — built with Flask, NumPy, and scikit-learn.

```
click on a card
      |
      v
POST /api/interact  ---->  update session vector (exp. weighted avg)
      |                            |
      |                            v
      |                    cosine similarity vs. all items
      |                            |
      |                            v
      |                blend with popularity + epsilon-greedy exploration
      |                            |
      v                            v
frontend re-renders  <----  top-N recommendations (JSON)
"Recommended for you"
```

**Offline layer** — `app.py` builds a TF-IDF vector for every catalog item
from its category + tags, once, at startup. This stands in for what would
normally be a trained embedding model (two-tower network, matrix
factorization, etc.) in a production system.

**Online layer** — each browser session gets a vector in memory
(`SESSIONS` dict). Every click blends the clicked item's vector into the
session vector with exponential decay (`SESSION_DECAY`), so recent clicks
matter more than older ones without needing to store full history for
scoring.

**Serving layer** — on every request, cosine similarity between the session
vector and all unclicked items gives a similarity-based ranking. This is
blended 85/15 with a popularity prior (so cold, sparse sessions don't return
noisy results), and one slot is occasionally swapped for a random item from
an under-explored category (`EXPLORATION_EPSILON`) — a simple stand-in for
a multi-armed bandit exploration strategy.

Because everything is in-memory numpy/sklearn math over ~40 items, a full
re-rank takes low single-digit milliseconds — the "real-time" feel comes
from the architecture, not from brute compute.

## Running it

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in a browser. Click items in the catalog
at the bottom — watch the "Recommended for you" rail at the top update
immediately, and start skewing toward whatever category/tags you've been
clicking.

## Project structure

```
reco-engine/
├── app.py              Flask backend: embeddings, session logic, API routes
├── data/items.json     40-item synthetic content catalog (7 categories)
├── static/
│   ├── index.html      Page layout
│   └── script.js        Talks to the API, re-renders on every click
└── requirements.txt
```


