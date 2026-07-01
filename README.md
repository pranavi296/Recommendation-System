# Live Feed — a real-time recommendation engine demo

A minimal but complete example of a recommendation system that updates its
suggestions **as the user interacts**, with no page reload and no retraining.
Built for a final-year AI/CS project — small enough to read in one sitting,
but demonstrates the same architecture pattern real recommender systems use.

## How it works

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

## Extending this for a full semester project

This is intentionally the smallest version that proves the architecture.
Good next steps, roughly in order of effort:

1. **Swap the toy dataset** for something real — MovieLens, Retailrocket, or
   H&M Personalized Fashion (Kaggle) all work well and let you compute real
   offline metrics (precision@k, recall@k, NDCG) against held-out data.
2. **Replace TF-IDF with learned embeddings** — sentence-transformers for
   text-heavy items, or a two-tower neural model trained on
   (user, item, interaction) triples for a classic collaborative-filtering
   setup.
3. **Add a real event stream** — replace the direct REST call with
   Redis Streams or Kafka between the frontend and the recommender, so you
   can show a proper streaming pipeline diagram in your report/demo.
4. **Persist sessions** — move `SESSIONS` out of a Python dict into Redis
   so the demo survives a server restart and can scale past one process.
5. **Formalize the exploration strategy** — replace the epsilon-greedy stub
   with a proper multi-armed bandit (Thompson sampling via the `river` or
   `mabwiser` library) and log regret over a simulated session to show it
   actually improves diversity/engagement over pure exploitation.
6. **Push instead of poll** — add WebSockets (Flask-SocketIO or FastAPI) so
   updates are pushed to the client, useful if you want multiple tabs/users
   to see live updates without a click, e.g. "3 people are viewing this now."

## Demo tip

For a presentation: click 3-4 items from one category (e.g. sci-fi), then
narrate what's happening in the rail as it visibly shifts — that's the
single moment that sells the whole project to evaluators.
