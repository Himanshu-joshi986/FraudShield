import React from "react";
import { riskColor, caseStatusLabel, investigationComplete } from "./RiskStamp";

export default function CaseFileList({ cases, activeCaseId, onSelect, onGoHome }) {
  return (
    <aside className="case-sidebar w-72 shrink-0 h-screen sticky top-0 flex flex-col">
      <button onClick={onGoHome} className="brand-lockup text-left px-5 py-5">
        <span className="brand-mark" aria-hidden="true">F</span>
        <span className="brand-copy"><span className="brand-name">FraudShield</span><span className="brand-caption">INVESTIGATION DESK</span></span>
      </button>
      <div className="case-count"><span>CASE REGISTER</span><b>{String(cases.length).padStart(2, "0")}</b></div>
      <div className="case-list flex-1 overflow-y-auto">
        {cases.map((c) => (
          <button key={c.case_id} onClick={() => onSelect(c.case_id)}
            className={`case-list-row w-full text-left px-5 py-3 border-b border-l-2 transition-colors ${activeCaseId === c.case_id ? "is-active" : ""}`}
            style={{ borderLeftColor: c.status !== "new" ? riskColor(c.risk_level) : "transparent" }}>
            <div className="flex items-center justify-between"><span className="case-row-id">{c.case_id}</span><span className="case-row-amount">${Number(c.amount).toLocaleString()}</span></div>
            <div className="case-row-name text-sm font-serif mt-0.5 truncate">{c.customer_name || "Unattributed"}</div>
            <div className="case-row-status text-[10px] font-mono mt-1" style={{ color: riskColor(c.risk_level) }}>
              {investigationComplete(c.status) ? `${caseStatusLabel(c.status)} · ${c.risk_level} · ${Number(c.risk_score).toFixed(2)}` : caseStatusLabel(c.status)}
            </div>
          </button>
        ))}
        {cases.length === 0 && <p className="px-5 py-6 text-sm font-serif italic text-ink-dim">No cases yet.</p>}
      </div>
      <div className="sidebar-footer"><span className="status-light" /> LOCAL WORKSPACE <span className="sidebar-version">v1.0</span></div>
    </aside>
  );
}
