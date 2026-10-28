import React, { useMemo, useState } from "react";

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
 *
 * The type filter exists because a dossier with twelve exhibits across
 * five types is not reviewable as one list; a reviewer usually wants
 * "show me the document findings" or "show me the model reasoning".
 */
export default function ExhibitList({ evidence, showFilters = false, focusIds = [], onFocusComplete }) {
  const [activeType, setActiveType] = useState("all");
  const [collapsed, setCollapsed] = useState(false);

  const sorted = useMemo(
    () => [...(evidence || [])].sort((a, b) => b.confidence - a.confidence),
    [evidence]
  );

  const counts = useMemo(() => {
    const map = new Map();
    for (const item of sorted) map.set(item.type, (map.get(item.type) || 0) + 1);
    return map;
  }, [sorted]);

  const focusSet = useMemo(() => new Set(focusIds || []), [focusIds]);

  const visible = useMemo(
    () => (activeType === "all" ? sorted : sorted.filter((item) => item.type === activeType)),
    [sorted, activeType]
  );

  if (!evidence || evidence.length === 0) {
    return (
      <p className="text-sm text-ink-dim font-serif italic">
        No evidence was gathered for this case — it did not meet the threshold for investigation.
      </p>
    );
  }

  const types = [...counts.keys()];

  return (
    <div className="exhibits">
      {showFilters && types.length > 1 && (
        <div className="exhibit-filters" role="group" aria-label="Filter exhibits by type">
          <button
            type="button"
            className={`exhibit-chip ${activeType === "all" ? "is-active" : ""}`}
            onClick={() => setActiveType("all")}
          >
            All <span>{sorted.length}</span>
          </button>
          {types.map((type) => (
            <button
              key={type}
              type="button"
              className={`exhibit-chip ${activeType === type ? "is-active" : ""}`}
              style={activeType === type ? { borderColor: TYPE_COLORS[type], color: TYPE_COLORS[type] } : undefined}
              onClick={() => setActiveType(type)}
            >
              {TYPE_LABELS[type] || type} <span>{counts.get(type)}</span>
            </button>
          ))}
        </div>
      )}

      <div className="exhibit-list">
        {visible.map((item) => {
          const isFocused = focusSet.has(item.evidence_id);
          return (
            <article
              key={item.evidence_id}
              id={`exhibit-${item.evidence_id}`}
              className={`exhibit ${isFocused ? "is-focused" : ""}`}
              style={{ borderColor: TYPE_COLORS[item.type] || "#4B5A70" }}
            >
              <div className="exhibit-head">
                <span className="text-xs font-mono text-ink-dim">
                  Exhibit {sorted.indexOf(item) + 1} &middot; {item.evidence_id}
                </span>
                <span className="text-[11px] font-mono" style={{ color: TYPE_COLORS[item.type] }}>
                  {TYPE_LABELS[item.type] || item.type} &middot; confidence {item.confidence.toFixed(2)}
                </span>
              </div>
              <p className="text-sm text-ink-text font-serif leading-relaxed">{item.finding}</p>
              <p className="text-[11px] text-ink-faint mt-1">Source: {item.source}</p>

              {item.source_reference && Object.keys(item.source_reference).length > 0 && (
                <details
                  className="mt-2 text-[11px] font-mono text-ink-faint"
                  onToggle={(e) => {
                    if (e.currentTarget.open && onFocusComplete) onFocusComplete(item);
                  }}
                >
                  <summary className="cursor-pointer hover:text-forensic">Trace to source record</summary>
                  <pre className="mt-2 overflow-x-auto whitespace-pre-wrap border-l border-ink-750 pl-3">
                    {JSON.stringify(item.source_reference, null, 2)}
                  </pre>
                </details>
              )}
            </article>
          );
        })}
      </div>

      {activeType !== "all" && (
        <button type="button" className="exhibit-clear" onClick={() => setActiveType("all")}>
          Show all {sorted.length} exhibits
        </button>
      )}
    </div>
  );
}
