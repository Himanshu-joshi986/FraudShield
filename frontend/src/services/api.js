const BASE = "/api";

async function handle(res) {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

export const api = {
  importCsv: (filename, domain, csvText) =>
    fetch(`${BASE}/data/import-csv`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename, domain, csv_text: csvText }),
    }).then(handle),
  createCase: (payload) =>
    fetch(`${BASE}/cases`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }).then(handle),

  seedSynthetic: () => fetch(`${BASE}/cases/seed-synthetic`, { method: "POST" }).then(handle),

  listCases: (params = {}) => {
    const qs = new URLSearchParams(params).toString();
    return fetch(`${BASE}/cases${qs ? `?${qs}` : ""}`).then(handle);
  },

  caseDetail: (caseId) => fetch(`${BASE}/cases/${caseId}`).then(handle),

  investigate: (caseId) => fetch(`${BASE}/cases/${caseId}/investigate`, { method: "POST" }).then(handle),

  caseGraph: (caseId) => fetch(`${BASE}/cases/${caseId}/graph`).then(handle),

  reportPdfUrl: (caseId) => `${BASE}/cases/${caseId}/report/pdf`,

  analytics: () => fetch(`${BASE}/analytics`).then(handle),

  knowledgeBase: () => fetch(`${BASE}/knowledge`).then(handle),

  knowledgeQuery: (question, topK = 3) =>
    fetch(`${BASE}/knowledge/query?question=${encodeURIComponent(question)}&top_k=${topK}`, {
      method: "POST",
    }).then(handle),
};
