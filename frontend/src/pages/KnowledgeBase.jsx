import React, { useEffect, useState } from "react";
import { api } from "../services/api";

export default function KnowledgeBase() {
  const [kb, setKb] = useState(null);
  const [question, setQuestion] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingKb, setLoadingKb] = useState(true);
  const [error, setError] = useState("");

  async function loadKnowledge() {
    setLoadingKb(true);
    setError("");
    try { setKb(await api.knowledgeBase()); }
    catch (e) { setError(e.message || "Could not load the knowledge base."); }
    finally { setLoadingKb(false); }
  }

  useEffect(() => { loadKnowledge(); }, []);

  async function handleQuery(e) {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true); setError("");
    try {
      const response = await api.knowledgeQuery(question);
      setResults(response.results);
    } catch (e) {
      setError(e.message || "Knowledge search failed.");
    } finally { setLoading(false); }
  }

  return (
    <main className="knowledge-page">
      <header className="page-heading">
        <div><span className="page-kicker">REFERENCE LIBRARY / POLICY</span><h1>Knowledge Base</h1><p>Regulatory guidance and internal policy used to support investigations.</p></div>
        {kb && <div className="knowledge-count"><strong>{kb.documents.length}</strong><span>DOCUMENTS<br />{kb.stats.n_chunks} indexed passages</span></div>}
      </header>

      {error && <div className="page-error" role="alert"><span>{error}</span>{!kb && <button onClick={loadKnowledge}>Retry</button>}</div>}

      <div className="knowledge-layout">
        <section className="knowledge-panel documents-panel">
          <div className="panel-heading"><div><span className="page-kicker">SOURCE MATERIAL</span><h2>Indexed documents</h2></div><span className="panel-total">{kb?.documents.length ?? "—"}</span></div>
          {loadingKb ? <div className="knowledge-state">Loading policy documents…</div> : (
            <div className="document-list">
              {(kb?.documents || []).map((doc) => (
                <article className="document-item" key={doc.document_id}>
                  <span className="document-id">{doc.document_id}</span>
                  <div className="document-copy"><h3>{doc.title}</h3><p>{doc.source}</p></div>
                  <span className="document-jurisdiction">{doc.jurisdiction}</span>
                </article>
              ))}
              {kb?.documents.length === 0 && <div className="knowledge-state">No documents have been indexed yet.</div>}
            </div>
          )}
        </section>

        <section className="knowledge-panel search-panel">
          <div className="panel-heading"><div><span className="page-kicker">ASK THE LIBRARY</span><h2>Search guidance</h2></div><span className="search-symbol" aria-hidden="true">⌕</span></div>
          <p className="search-description">Search the indexed policies for relevant passages and source references.</p>
          <form onSubmit={handleQuery} className="knowledge-search-form">
            <label htmlFor="knowledge-question">Your question</label>
            <textarea id="knowledge-question" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="For example: when should a suspicious activity report be filed?" rows={4} />
            <button type="submit" disabled={loading || !question.trim()}>{loading ? "Searching…" : "Search documents"}<span aria-hidden="true">→</span></button>
          </form>
          {results && <div className="search-results">
            <div className="results-heading">{results.length ? `${results.length} relevant passage${results.length === 1 ? "" : "s"}` : "No matching passages"}</div>
            {results.map((result, i) => <article className="search-result" key={`${result.document_id}-${i}`}><div><strong>{result.title || result.document_id}</strong><span>{result.document_id} · {result.jurisdiction} · match {Math.max(0, 1 - result.distance).toFixed(2)}</span></div><p>{result.snippet}</p></article>)}
          </div>}
        </section>
      </div>
      <p className="knowledge-footnote">Search results provide reference material for analysts. Verify applicable requirements and use your organization’s review process before taking action.</p>
    </main>
  );
}
