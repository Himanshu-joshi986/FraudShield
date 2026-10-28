// In development, call the API directly. This avoids a stale or separately
// launched Vite server silently serving index.html for /api requests.
const BASE = import.meta.env.DEV
  ? "http://127.0.0.1:8000/api"
  : (import.meta.env.VITE_API_BASE || "/api");

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

  deleteCase: (caseId) => fetch(`${BASE}/cases/${encodeURIComponent(caseId)}`, { method: "DELETE" }).then((res) => {
    if (!res.ok) return handle(res);
    return null;
  }),

  investigate: (caseId) => fetch(`${BASE}/cases/${caseId}/investigate`, { method: "POST" }).then(handle),

  caseGraph: (caseId) => fetch(`${BASE}/cases/${caseId}/graph`).then(handle),

  // Signal scorecard, policy trace, counterfactuals and case timeline.
  caseInsights: (caseId) => fetch(`${BASE}/cases/${caseId}/insights`).then(handle),

  // The case's document dossier: OCR fields + the invoices that filed the
  // same invoice number, for the side-by-side comparison.
  caseDocuments: (caseId) => fetch(`${BASE}/cases/${caseId}/documents`).then(handle),

  // Document images live at /api/documents/<file>, so they resolve from
  // the same base as every other request -- absolute in dev, proxied in
  // a production build.
  documentImageUrl: (fileName) => `${BASE}/documents/${encodeURIComponent(fileName)}`,

  reportPdfUrl: (caseId) => `${BASE}/cases/${caseId}/report/pdf`,

  analytics: () => fetch(`${BASE}/analytics`).then(handle),

  knowledgeBase: () => fetch(`${BASE}/knowledge`).then(handle),

  knowledgeQuery: (question, topK = 3, caseId = "") =>
    fetch(`${BASE}/knowledge/query?question=${encodeURIComponent(question)}&top_k=${topK}${caseId ? `&case_id=${encodeURIComponent(caseId)}` : ""}`, {
      method: "POST",
    }).then(handle),

  uploadKnowledgeDocument: (formData) =>
    fetch(`${BASE}/knowledge/documents`, { method: "POST", body: formData }).then(handle),

  linkKnowledgeDocument: (documentId, caseId) =>
    fetch(`${BASE}/knowledge/documents/${encodeURIComponent(documentId)}/link`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ case_id: caseId || null }),
    }).then(handle),

  removeKnowledgeDocument: (documentId) =>
    fetch(`${BASE}/knowledge/documents/${encodeURIComponent(documentId)}`, { method: "DELETE" }).then(handle),
};
