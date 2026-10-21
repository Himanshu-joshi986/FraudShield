import React, { useEffect, useState } from "react";
import { api } from "../services/api";

export default function KnowledgeBase() {
  const [kb, setKb] = useState(null);
  const [question, setQuestion] = useState("");
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => { api.knowledgeBase().then(setKb); }, []);

  async function handleQuery(e) {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    const r = await api.knowledgeQuery(question);
    setResults(r.results);
    setLoading(false);
  }

  return (
    <div className="max-w-3xl mx-auto p-10">
      <h1 className="font-serif text-2xl text-ink-text mb-1">Knowledge Base</h1>
      <p className="text-sm text-ink-dim mb-8">
        Regulatory and policy documents the Investigation Agent draws on.
      </p>

      <div className="border border-ink-750 p-6 mb-6">
        <h2 className="font-serif text-base text-ink-text mb-4">
          Indexed documents {kb ? `(${kb.stats.n_chunks} chunks)` : ""}
        </h2>
        <div className="space-y-3">
          {kb?.documents.map((d) => (
            <div key={d.document_id} className="border-l-2 border-forensic pl-4">
              <div className="text-xs font-mono text-ink-dim">{d.document_id} &middot; {d.jurisdiction}</div>
              <div className="font-serif text-ink-text">{d.title}</div>
              <div className="text-[11px] text-ink-faint mt-0.5">{d.source}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="border border-ink-750 p-6">
        <h2 className="font-serif text-base text-ink-text mb-4">Search</h2>
        <form onSubmit={handleQuery} className="flex gap-2 mb-4">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. how long do we have to file a SAR?"
            className="flex-1 bg-ink-850 border border-ink-750 px-3 py-2 text-sm text-ink-text font-serif focus:outline-none focus:border-forensic"
          />
          <button
            type="submit" disabled={loading}
            className="text-sm font-mono px-4 py-2 border border-forensic text-forensic hover:bg-forensic/10 disabled:opacity-50"
          >
            {loading ? "Searching…" : "Search"}
          </button>
        </form>
        {results && (
          <div className="space-y-3">
            {results.length === 0 && <p className="text-sm text-ink-dim font-serif italic">No relevant passages found.</p>}
            {results.map((r, i) => (
              <div key={i} className="border-l-2 border-ink-750 pl-4">
                <div className="text-xs font-mono text-ink-faint">{r.document_id} &middot; distance {r.distance}</div>
                <p className="text-sm font-serif text-ink-text mt-1">{r.snippet}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
