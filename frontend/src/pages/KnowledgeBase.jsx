import React, { useEffect, useRef, useState } from "react";
import { api } from "../services/api";

const EMPTY_FORM = { title: "", jurisdiction: "Unspecified", case_id: "" };

export default function KnowledgeBase() {
  const [kb, setKb] = useState(null);
  const [cases, setCases] = useState([]);
  const [question, setQuestion] = useState("");
  const [searchCase, setSearchCase] = useState("");
  const [results, setResults] = useState(null);
  const [file, setFile] = useState(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [linkingDocument, setLinkingDocument] = useState("");
  const [removeTarget, setRemoveTarget] = useState(null);
  const [removingDocument, setRemovingDocument] = useState(false);
  const [loadingKb, setLoadingKb] = useState(true);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const fileInput = useRef(null);

  async function loadKnowledge() {
    setLoadingKb(true);
    try { setKb(await api.knowledgeBase()); }
    catch (e) { setError(e.message || "Could not load the knowledge base."); }
    finally { setLoadingKb(false); }
  }

  useEffect(() => {
    loadKnowledge();
    api.listCases({ limit: 100 }).then(setCases).catch(() => {});
  }, []);

  async function handleUpload(event) {
    event.preventDefault();
    if (!file) return;
    setUploading(true); setError(""); setNotice("");
    const data = new FormData();
    data.append("file", file);
    data.append("title", form.title.trim() || file.name.replace(/\.[^.]+$/, ""));
    data.append("jurisdiction", form.jurisdiction);
    data.append("case_id", form.case_id);
    try {
      const response = await api.uploadKnowledgeDocument(data);
      setNotice(`“${response.document.title}” is saved and ready to search.`);
      setFile(null); setForm(EMPTY_FORM);
      if (fileInput.current) fileInput.current.value = "";
      await loadKnowledge();
    } catch (e) { setError(e.message || "The document could not be added."); }
    finally { setUploading(false); }
  }

  async function handleQuery(e) {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true); setError("");
    try {
      const response = await api.knowledgeQuery(question, 5, searchCase);
      setResults(response.results);
    } catch (e) { setError(e.message || "Knowledge search failed."); }
    finally { setLoading(false); }
  }

  async function handleLink(doc, caseId) {
    setLinkingDocument(doc.document_id); setError(""); setNotice("");
    try {
      await api.linkKnowledgeDocument(doc.document_id, caseId);
      setNotice(caseId ? `“${doc.title}” is now linked to ${caseId}. Investigations for that case can retrieve it.` : `“${doc.title}” is now in the shared library.`);
      await loadKnowledge();
    } catch (e) { setError(e.message || "Could not update the case link."); }
    finally { setLinkingDocument(""); }
  }

  async function handleRemove() {
    if (!removeTarget) return;
    setRemovingDocument(true); setError(""); setNotice("");
    try {
      await api.removeKnowledgeDocument(removeTarget.document_id);
      setResults(current => current?.filter(result => result.document_id !== removeTarget.document_id) ?? current);
      setNotice(`“${removeTarget.title}” and its searchable passages were removed from the library.`);
      setRemoveTarget(null);
      await loadKnowledge();
    } catch (e) { setError(e.message || "The document could not be removed."); }
    finally { setRemovingDocument(false); }
  }

  const userDocuments = (kb?.documents || []).filter(doc => doc.category === "user");
  const regulatoryDocuments = (kb?.documents || []).filter(doc => doc.category !== "user");

  return (
    <main className="knowledge-page">
      <header className="knowledge-hero">
        <div>
          <span className="page-kicker">REFERENCE LIBRARY / SHARED EVIDENCE</span>
          <h1>Knowledge Base</h1>
          <p>Bring policies, analyst notes, and reference documents together. Search them here or attach a source to a case so its investigators can retrieve it.</p>
        </div>
        <div className="knowledge-count"><strong>{kb?.documents.length ?? "—"}</strong><span>DOCUMENTS<br />{kb?.stats.n_chunks ?? "—"} searchable passages</span></div>
      </header>

      {error && <div className="page-error" role="alert"><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error">×</button></div>}
      {notice && <div className="kb-notice" role="status"><span>✓</span>{notice}<button onClick={() => setNotice("")} aria-label="Dismiss message">×</button></div>}

      <div className="kb-workspace">
        <section className="knowledge-panel kb-upload-panel">
          <div className="panel-heading"><div><span className="page-kicker">CONTRIBUTE A SOURCE</span><h2>Add to the library</h2></div><span className="kb-step">01</span></div>
          <p className="kb-intro">Upload material you already have. Files stay in this project’s data folder and are indexed locally for future searches and investigations.</p>
          <form onSubmit={handleUpload} className="kb-upload-form">
            <label className={`kb-dropzone ${file ? "has-file" : ""}`} htmlFor="kb-file">
              <span className="kb-upload-icon">↑</span>
              <strong>{file ? file.name : "Choose a document to add"}</strong>
              <span>{file ? `${(file.size / 1024).toFixed(0)} KB · ready to index` : "PDF, Markdown, or plain text · up to 10 MB"}</span>
              <input ref={fileInput} id="kb-file" type="file" accept=".pdf,.md,.txt,application/pdf,text/markdown,text/plain" onChange={e => { setFile(e.target.files?.[0] || null); setNotice(""); }} />
            </label>
            <div className="kb-form-grid">
              <label>Document title<input value={form.title} onChange={e => setForm({...form, title:e.target.value})} placeholder="e.g. Internal escalation guidance" maxLength={160} /></label>
              <label>Jurisdiction or source<input value={form.jurisdiction} onChange={e => setForm({...form, jurisdiction:e.target.value})} placeholder="e.g. Internal · US · State" maxLength={120} /></label>
            </div>
            <label className="kb-case-label">Link to a case <span>Optional · linked case investigators can retrieve this material</span>
              <select value={form.case_id} onChange={e => setForm({...form, case_id:e.target.value})}>
                <option value="">Shared library · available to all cases</option>
                {cases.map(c => <option key={c.case_id} value={c.case_id}>{c.case_id}{c.customer_name ? ` · ${c.customer_name}` : ""}</option>)}
              </select>
            </label>
            <button className="kb-primary-button" type="submit" disabled={!file || uploading}>{uploading ? <><span className="kb-spinner" />Saving and indexing…</> : <>Save document to library <span>→</span></>}</button>
            <p className="kb-upload-footnote">Searchable text PDFs only. Scanned image PDFs need OCR before upload. The original file and extracted searchable text are kept locally.</p>
          </form>
        </section>

        <section className="knowledge-panel search-panel kb-search-panel">
          <div className="panel-heading"><div><span className="page-kicker">ASK THE LIBRARY</span><h2>Find useful passages</h2></div><span className="search-symbol" aria-hidden="true">⌕</span></div>
          <p className="search-description">Search regulations and contributed material. Choose a case to include its linked sources alongside shared references.</p>
          <form onSubmit={handleQuery} className="knowledge-search-form">
            <label htmlFor="knowledge-question">What are you looking for?</label>
            <textarea id="knowledge-question" value={question} onChange={e => setQuestion(e.target.value)} placeholder="Try: What evidence should be documented before escalating a suspicious claim?" rows={4} />
            <div className="kb-search-controls">
              <select aria-label="Search within case sources" value={searchCase} onChange={e => { setSearchCase(e.target.value); setResults(null); }}>
                <option value="">Entire library</option>
                {cases.map(c => <option key={c.case_id} value={c.case_id}>{c.case_id} · linked + shared sources</option>)}
              </select>
              <button type="submit" disabled={loading || !question.trim()}>{loading ? "Searching…" : "Search sources"}<span aria-hidden="true">→</span></button>
            </div>
          </form>
          {results && <div className="search-results">
            <div className="results-heading">{results.length ? `${results.length} relevant passage${results.length === 1 ? "" : "s"}` : "No matching passages"}</div>
            {results.map((result, i) => <article className="search-result" key={`${result.document_id}-${i}`}><div><strong>{result.title || result.document_id}</strong><span>{result.document_id} · {result.jurisdiction || "Source"}{result.case_id ? ` · linked to ${result.case_id}` : " · shared"} · match {Math.max(0, 1 - result.distance).toFixed(2)}</span></div><p>{result.snippet}</p></article>)}
          </div>}
        </section>
      </div>

      <section className="knowledge-panel kb-sources-panel">
        <div className="panel-heading"><div><span className="page-kicker">LIBRARY CATALOG</span><h2>Available sources</h2></div><span className="panel-total">{kb?.documents.length ?? "—"}</span></div>
        {loadingKb ? <div className="knowledge-state">Loading library…</div> : <>
          {userDocuments.length > 0 && <><div className="kb-group-label">CONTRIBUTED MATERIAL <span>{userDocuments.length}</span></div><div className="kb-document-grid">{userDocuments.map(doc => <article className="kb-document-card" key={doc.document_id}>
            <span className="kb-file-mark">{doc.original_filename?.split(".").pop()?.toUpperCase() || "DOC"}</span>
            <div className="kb-document-copy"><span className="document-id">{doc.document_id}</span><h3>{doc.title}</h3><p>{doc.source} · {doc.jurisdiction || "Unspecified"}</p></div>
            {doc.case_id ? <span className="kb-linked-badge">↗ {doc.case_id}</span> : <span className="kb-shared-badge">SHARED</span>}
            <div className="kb-link-control"><select aria-label={`Link ${doc.title} to a case`} value={doc.case_id || ""} onChange={e => handleLink(doc, e.target.value)} disabled={linkingDocument === doc.document_id}>
              <option value="">Shared · all cases</option>{cases.map(c => <option key={c.case_id} value={c.case_id}>{c.case_id}{c.customer_name ? ` · ${c.customer_name}` : ""}</option>)}
            </select><span>{linkingDocument === doc.document_id ? "Saving link…" : "Set availability"}</span><button className="kb-remove-button" type="button" onClick={() => setRemoveTarget(doc)} disabled={removingDocument}>Remove</button></div>
          </article>)}</div></>}
          <div className="kb-group-label">REGULATORY REFERENCE <span>{regulatoryDocuments.length}</span></div>
          <div className="document-list">{regulatoryDocuments.map(doc => <article className="document-item" key={doc.document_id}><span className="document-id">{doc.document_id}</span><div className="document-copy"><h3>{doc.title}</h3><p>{doc.source}</p></div><span className="document-jurisdiction">{doc.jurisdiction}</span></article>)}
            {kb?.documents.length === 0 && <div className="knowledge-state">No documents have been indexed yet.</div>}
          </div>
        </>}
      </section>
      <p className="knowledge-footnote">Search results are reference material for analysts. Confirm applicability and follow your organization’s review process before taking action.</p>
      {removeTarget && <div className="kb-confirm-backdrop" role="presentation" onMouseDown={event => { if (event.target === event.currentTarget && !removingDocument) setRemoveTarget(null); }}>
        <section className="kb-confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="kb-remove-title" aria-describedby="kb-remove-description">
          <span className="page-kicker">REMOVE CONTRIBUTED SOURCE</span>
          <h2 id="kb-remove-title">Remove “{removeTarget.title}”?</h2>
          <p id="kb-remove-description">This permanently deletes the uploaded file, its extracted searchable text, and all indexed passages. Regulatory references are protected. This cannot be undone.</p>
          <div className="kb-confirm-actions"><button type="button" onClick={() => setRemoveTarget(null)} disabled={removingDocument}>Keep document</button><button className="kb-confirm-remove" type="button" onClick={handleRemove} disabled={removingDocument}>{removingDocument ? "Removing…" : "Remove source"}</button></div>
        </section>
      </div>}
    </main>
  );
}
