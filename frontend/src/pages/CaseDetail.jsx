import React, { useEffect, useState } from "react";
import { api } from "../services/api";
import { RiskStamp, DecisionStamp, caseStatusLabel, investigationComplete } from "../components/RiskStamp";
import ExhibitList from "../components/ExhibitList";
import FraudGraph from "../components/FraudGraph";

const TABS = ["Overview", "Evidence", "Fraud graph", "Timeline", "Report"];

export default function CaseDetail({ caseId, onBack, onCaseUpdated }) {
  const [detail, setDetail] = useState(null);
  const [graph, setGraph] = useState(null);
  const [investigating, setInvestigating] = useState(false);
  const [error, setError] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [liveEvents, setLiveEvents] = useState([]);
  const [tab, setTab] = useState("Overview");
  const wsRef = React.useRef(null);

  async function load() {
    try {
      const d = await api.caseDetail(caseId);
      setDetail(d);
      setLoadError(null);
      if (d.case.customer_id) {
        const g = await api.caseGraph(caseId).catch(() => null);
        setGraph(g);
      }
    } catch (e) {
      setLoadError(e.message || "Could not load this case.");
    }
  }

  useEffect(() => { load(); }, [caseId]);

  async function handleInvestigate() {
    setInvestigating(true);
    setError(null);
    setLiveEvents([]);

    // Connect the live agent-activity WebSocket BEFORE triggering the
    // investigation, so no early events are missed.
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/cases/${caseId}`);
    wsRef.current = ws;
    ws.onmessage = (msg) => {
      try {
        const event = JSON.parse(msg.data);
        setLiveEvents((prev) => [...prev, event]);
      } catch (e) { /* ignore malformed frames */ }
    };

    try {
      await new Promise((resolve) => { ws.onopen = resolve; ws.onerror = resolve; });
      await api.investigate(caseId);
      await load();
      await onCaseUpdated?.();
    } catch (e) {
      setError(e.message || "Investigation failed. Please try again.");
    } finally {
      setInvestigating(false);
      ws.close();
    }
  }

  if (!detail) return (
    <div className="p-10 text-ink-dim font-serif">
      {loadError ? (
        <p role="alert" className="text-risk-high">{loadError}</p>
      ) : "Opening case file…"}
    </div>
  );

  const { case: c, agent_executions, evidence, reports, decisions } = detail;
  const decision = decisions[0];
  const report = reports[0];

  return (
    <div className="max-w-4xl mx-auto p-10">
      <button onClick={onBack} className="text-xs font-mono text-ink-faint hover:text-forensic mb-6">
        &larr; back to command center
      </button>

      <div className="flex items-start justify-between mb-2">
        <div>
          <h1 className="font-serif text-2xl text-ink-text">{c.customer_name || "Unattributed case"}</h1>
          <div className="text-xs font-mono text-ink-dim mt-1">
            {c.case_id} &middot; ${c.amount.toLocaleString(undefined, { minimumFractionDigits: 2 })} &middot; {c.domain}
          </div>
          <div className="text-[11px] font-mono text-ink-faint mt-1">
            {c.customer_id && <>Entity {c.customer_id}</>}
            {c.transaction_id && <> &middot; Transaction {c.transaction_id}</>}
            {c.claim_id && <> &middot; Claim {c.claim_id}</>}
          </div>
        </div>
        <div className="flex flex-col items-end gap-2">
          <RiskStamp level={c.risk_level} />
          {decision && <DecisionStamp decision={decision.decision} size="sm" />}
        </div>
      </div>

      {investigationComplete(c.status) && (
        <div className="case-completion-banner">
          <span className="completion-check">✓</span>
          <div><strong>Investigation complete</strong><span>{caseStatusLabel(c.status)} · Risk score {Number(c.risk_score).toFixed(2)}</span></div>
          <span className="completion-date">RESULT SAVED</span>
        </div>
      )}

      {c.status === "new" && (
        <div className="mt-6 border border-ink-750 p-5">
          <p className="text-sm font-serif text-ink-dim mb-3">
            This case has not yet been investigated. Running the pipeline will dispatch Sentinel,
            Investigation, Evidence Analyst, Report Architect, and Case Commander in sequence.
          </p>
          <button
            onClick={handleInvestigate}
            disabled={investigating}
            className="text-sm font-mono px-4 py-2 border border-forensic text-forensic hover:bg-forensic/10 transition-colors disabled:opacity-50"
          >
            {investigating ? "Investigating…" : "Run investigation"}
          </button>
          {error && (
            <p className="text-xs font-mono text-risk-high mt-3">
              {error} — the case remains unscored; you can retry.
            </p>
          )}
          {investigating && liveEvents.length > 0 && (
            <div className="mt-4 border-t border-ink-750 pt-4 space-y-1.5">
              {liveEvents.map((e, i) => (
                <div key={i} className="text-xs font-mono flex items-center gap-2">
                  <span className={e.event === "agent_completed" ? "text-risk-low" : "text-ink-faint"}>
                    {e.event === "agent_started" ? "○" : e.event === "agent_completed" ? "●" : "✓"}
                  </span>
                  <span className="text-ink-dim">{e.agent || "Pipeline"}</span>
                  {e.summary && <span className="text-ink-faint truncate">— {e.summary.slice(0, 70)}</span>}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {c.status !== "new" && (
        <>
          <div className="flex gap-1 mt-8 border-b border-ink-750">
            {TABS.map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={`text-sm font-mono px-4 py-2 -mb-px border-b-2 transition-colors ${
                  tab === t ? "border-forensic text-forensic" : "border-transparent text-ink-dim hover:text-ink-text"
                }`}
              >
                {t}
              </button>
            ))}
          </div>

          <div className="pt-6">
            {tab === "Overview" && (
              <div className="space-y-6">
                <div>
                  <SectionLabel>Why this case was flagged</SectionLabel>
                  <ExhibitList evidence={evidence.slice().sort((a, b) => b.confidence - a.confidence).slice(0, 3)} />
                </div>
                {decision && (
                  <div>
                    <SectionLabel>Recommended disposition</SectionLabel>
                    <p className="text-sm font-serif text-ink-text">{decision.reason}</p>
                    <p className="text-[11px] font-mono text-ink-faint mt-2">
                      policy version {decision.policy_version}
                    </p>
                  </div>
                )}
              </div>
            )}

            {tab === "Evidence" && (
              <div>
                <SectionLabel>Evidence dossier ({evidence.length} exhibit{evidence.length !== 1 ? "s" : ""})</SectionLabel>
                <ExhibitList evidence={evidence} />
                <p className="text-[11px] text-ink-faint font-serif italic mt-4">
                  This explanation describes model and evidence signals. It does not establish that
                  fraud actually occurred.
                </p>
              </div>
            )}

            {tab === "Fraud graph" && (
              <div>
                <SectionLabel>Connected entities</SectionLabel>
                <FraudGraph nodes={graph?.nodes} edges={graph?.edges} />
              </div>
            )}

            {tab === "Timeline" && (
              <div>
                <SectionLabel>Agent execution timeline</SectionLabel>
                <div className="space-y-3">
                  {agent_executions.map((log) => (
                    <div key={log.execution_id} className="flex gap-4 text-sm">
                      <span className="text-[11px] font-mono text-ink-faint w-40 shrink-0">{log.agent_name}</span>
                      <div>
                        <p className="font-serif text-ink-text">{log.output_summary}</p>
                        <p className="text-[10px] font-mono text-ink-faint mt-0.5">{log.duration_ms?.toFixed(1)}ms</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {tab === "Report" && (
              <div>
                <SectionLabel>Investigation report</SectionLabel>
                {report ? (
                  <>
                    <a
                      href={api.reportPdfUrl(caseId)}
                      target="_blank" rel="noreferrer"
                      className="inline-block text-sm font-mono px-4 py-2 border border-forensic text-forensic hover:bg-forensic/10 transition-colors mb-4"
                    >
                      Download PDF report
                    </a>
                    <pre className="text-[11px] font-mono text-ink-dim whitespace-pre-wrap border-l-2 border-ink-750 pl-4">
                      {report.content}
                    </pre>
                  </>
                ) : (
                  <p className="text-sm text-ink-dim font-serif italic">
                    No report was generated — this case was auto-closed without a full investigation.
                  </p>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function SectionLabel({ children }) {
  return <h2 className="font-serif text-base text-ink-text mb-3">{children}</h2>;
}
