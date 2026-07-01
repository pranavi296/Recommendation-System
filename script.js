let sessionId = null;

const recGrid = document.getElementById("rec-grid");
const catalogGrid = document.getElementById("catalog-grid");
const historyEl = document.getElementById("history");
const resetBtn = document.getElementById("reset-btn");

function cardHTML(item, { isRec } = {}) {
  const reason = isRec && item.reason ? `<div class="reason">${item.reason}</div>` : "";
  return `
    <div class="card ${isRec ? "rec-card" : ""}" data-id="${item.id}">
      <span class="cat">${item.category}</span>
      <p class="title">${item.title}</p>
      <p class="meta">${item.tags.join(" · ")}</p>
      ${reason}
    </div>
  `;
}

function renderRecommended(items) {
  recGrid.innerHTML = items.map(item => cardHTML(item, { isRec: true })).join("");
}

function renderCatalog(items) {
  catalogGrid.innerHTML = items.map(item => cardHTML(item)).join("");
}

function renderHistory(history) {
  if (!history || history.length === 0) {
    historyEl.textContent = "No clicks yet — try clicking something in the catalog below.";
    return;
  }
  historyEl.textContent = "Recently clicked: " + history.join(" → ");
}

function attachClickHandlers() {
  document.querySelectorAll(".card").forEach(card => {
    card.addEventListener("click", () => onItemClick(Number(card.dataset.id)));
  });
}

async function onItemClick(itemId) {
  const res = await fetch("/api/interact", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, item_id: itemId })
  });
  const data = await res.json();
  sessionId = data.session_id;
  renderRecommended(data.recommended);
  renderHistory(data.click_history);
  attachClickHandlers();
}

async function init() {
  const res = await fetch("/api/session", { method: "POST" });
  const data = await res.json();
  sessionId = data.session_id;
  renderRecommended(data.recommended);
  renderCatalog(data.catalog);
  renderHistory([]);
  attachClickHandlers();
}

resetBtn.addEventListener("click", async () => {
  const res = await fetch("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId })
  });
  const data = await res.json();
  sessionId = data.session_id;
  renderRecommended(data.recommended);
  renderHistory([]);
  attachClickHandlers();
});

init();
