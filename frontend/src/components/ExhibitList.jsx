import React from "react";

const TYPE_COLORS = {
  graph: "#5B8DBE",
  temporal: "#B98530",
  document: "#8B6FB5",
  regulatory: "#7C8CA3",
  model_explanation: "#4B7A63",
};

const TYPE_LABELS = {
  graph: "Graph",
  temporal: "Temporal",
  document: "Document",
  regulatory: "Regulatory",
  model_explanation: "Model",
};

/**
 * Evidence rendered as numbered exhibits, each with a thin left border
 * colored by evidence type, a monospace evidence ID (a real identifier,
 * not decoration), and its source -- the visual encoding of "every claim
 * is traceable" that runs through the whole dashboard.
 */
export default function ExhibitList({ evidence }) {
  if (!evidence || evidence.length === 0) {
    return (
      <p className="text-sm text-ink-dim font-serif italic">
        No evidence was gathered for this case — it did not meet the threshold for investigation.
      </p>
    );
  }

  const sorted = [...evidence].sort((a, b) => b.confidence - a.confidence);

  return (
    <div className="space-y-3">
      {sorted.map((item, i) => (
        <div
          key={item.evidence_id}
          className="border-l-2 pl-4 py-1"
          style={{ borderColor: TYPE_COLORS[item.type] || "#4B5A70" }}
        >
          <div className="flex items-baseline justify-between gap-3 mb-1">
            <span className="text-xs font-mono text-ink-dim">
              Exhibit {i + 1} &middot; {item.evidence_id}
            </span>
            <span className="text-[11px] font-mono" style={{ color: TYPE_COLORS[item.type] }}>
              {TYPE_LABELS[item.type] || item.type} &middot; confidence {item.confidence.toFixed(2)}
            </span>
          </div>
          <p className="text-sm text-ink-text font-serif leading-relaxed">{item.finding}</p>
          <p className="text-[11px] text-ink-faint mt-1">Source: {item.source}</p>
          {item.source_reference && Object.keys(item.source_reference).length > 0 && (
            <details className="mt-2 text-[11px] font-mono text-ink-faint">
              <summary className="cursor-pointer hover:text-forensic">Trace to source record</summary>
              <pre className="mt-2 overflow-x-auto whitespace-pre-wrap border-l border-ink-750 pl-3">
                {JSON.stringify(item.source_reference, null, 2)}
              </pre>
            </details>
          )}
        </div>
      ))}
    </div>
  );
}
