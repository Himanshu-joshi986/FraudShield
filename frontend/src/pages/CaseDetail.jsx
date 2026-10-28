import React, { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../services/api";
import { RiskStamp, DecisionStamp, caseStatusLabel, investigationComplete } from "../components/RiskStamp";
import ExhibitList from "../components/ExhibitList";
import FraudGraph from "../components/FraudGraph";
import SignalScorecard from "../components/SignalScorecard";
import PolicyTrace from "../components/PolicyTrace";
import DocumentCompare from "../components/DocumentCompare";
import PipelineStepper from "../components/PipelineStepper";
import CaseActivityTimeline from "../components/CaseActivityTimeline";
import ReportView from "../components/ReportView";

const TABS = ["Overview", "Evidence", "Fraud graph", "Timeline", "Report"];

export default function CaseDetail({ caseId, onBack, onCaseUpdated, onCaseRemoved }) {
  const [detail, setDetail] = useState(null);
  const [graph, setGraph] = useState(null);
  const [insights, setInsights] = useState(null);
  const [dossier, setDossier] = useState(null);
  const [investigating, setInvestigating] = useState(false);
  const [error, setError] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [liveEvents, setLiveEvents] = useState([]);
  const [tab, setTab] = useState("Overview");
  const [focusEvidence, setFocusEvidence] = useState([]);
  const [confirmation, setConfirmation] = useState(null);
  const [removing, setRemoving] = useState(false);
  const wsRef = React.useRef(null);

  const load = useCallback(async () => {
    try {
      // One round trip's worth of latency instead of four sequential ones:
      // the case record, the entity graph, the scorecard and the document
      // dossier are independent, so they are fetched in parallel. Each
      // one is allowed to fail on its own without taking the page down.
      const [detailResult, graphResult, insightsResult, dossierResult] = await Promise.allSettled([
        api.caseDetail(caseId),
        api.caseGraph(caseId).catch(() => null),
        api.caseInsights(caseId).catch(() => null),
        api.caseDocuments(caseId).catch(() => null),
      ]);

      if (detailResult.status === "rejected") {
        throw detailResult.reason;
      }
      setDetail(detailResult.value);
      setGraph(graphResult.status === "fulfilled" ? graphResult.value : null);
      setInsights(insightsResult.status === "fulfilled" ? insightsResult.value : null);
      setDossier(dossierResult.status === "fulfilled" ? dossierResult.value : null);
      setLoadError(null);
    } catch (e) {
      setLoadError(e.message || "Could not load this case.");
    }
  }, [caseId]);

  useEffect(() => { load(); }, [load]);

  async function handleInvestigate() {
    setInvestigating(true);
    setError(null);
    setLiveEvents([]);

    // Connect the live agent-activity WebSocket BEFORE triggering the
    // investigation, so no early events are missed.
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const wsHost = import.meta.env.DEV ? "127.0.0.1:8000" : window.location.host;
    const ws = new WebSocket(`${proto}://${wsHost}/ws/cases/${caseId}`);
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

  function handleReinvestigate() {
    setConfirmation("rerun");
  }

  async function confirmCaseAction() {
    if (confirmation === "rerun") {
      setConfirmation(null);
      await handleInvestigate();
      return;
    }
    if (confirmation === "remove") {
      setRemoving(true);
      setError(null);
      try {
        await api.deleteCase(caseId);
        setConfirmation(null);
        setRemoving(false);
        await onCaseRemoved?.();
      } catch (e) {
        setError(e.message || "Could not remove this case.");
        setRemoving(false);
      }
    }
  }

  useEffect(() => {
    if (!confirmation || investigating || removing) return undefined;
    function onKeyDown(event) {
      if (event.key === "Escape") setConfirmation(null);
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [confirmation, investigating, removing]);

  // Jumping to an exhibit from the report: switch to the Evidence tab and
  // scroll the referenced exhibit into view.
  const openEvidence = useCallback((evidenceId) => {
    setTab("Evidence");
    setFocusEvidence([evidenceId]);
    window.setTimeout(() => {
      const node = document.getElementById(`exhibit-${evidenceId}`);
      if (node) node.scrollIntoView({ behavior: "smooth", block: "center" });
    }, 60);
  }, []);

  const highlightedNodes = useMemo(() => {
    // graph evidence names connected entities; surface them in the graph
    // tab so the two views agree on what the case touched.
    if (!detail?.evidence?.length) return [];
    const ids = new Set();
    for (const item of detail.evidence) {
      if (item.type !== "graph") continue;
      const connections = item.source_reference?.connections || [];
      for (const connection of connections) {
        if (connection?.connected_customer) ids.add(connection.connected_customer);
        if (connection?.shared_entity) ids.add(connection.shared_entity);
      }
    }
    return [...ids];
  }, [detail]);

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
      <div className="case-detail-actions">
        <button onClick={onBack} className="text-xs font-mono text-ink-faint hover:text-forensic">
          &larr; back to command center
        </button>
        <button type="button" className="case-remove-trigger" onClick={() => setConfirmation("remove")} disabled={investigating || removing}>
          Remove case
        </button>
      </div>

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
          <button type="button" className="completion-rerun" onClick={handleReinvestigate} disabled={investigating}>
            {investigating ? "RE-RUNNING…" : "RE-RUN INVESTIGATION"}
          </button>
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
                {insights ? (
                  <SignalScorecard insights={insights} />
                ) : (
                  <p className="text-sm text-ink-dim font-serif italic">
                    The signal breakdown is unavailable for this case.
                  </p>
                )}

                {insights?.policy && (
                  <PolicyTrace
                    policy={insights.policy}
                    counterfactuals={insights.counterfactuals}
                    scoreBelowMedium={insights.score_below_medium}
                    riskScore={insights.risk_score}
                    riskAdjustment={insights.fusion?.risk_adjustment}
                  />
                )}

                <div>
                  <SectionLabel>Top evidence</SectionLabel>
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
              <div className="space-y-6">
                {dossier?.has_document && (
                  <div>
                    <SectionLabel>Supporting document</SectionLabel>
                    <DocumentCompare dossier={dossier} />
                  </div>
                )}

                <div>
                  <SectionLabel>Evidence dossier ({evidence.length} exhibit{evidence.length !== 1 ? "s" : ""})</SectionLabel>
                  <ExhibitList
                    evidence={evidence}
                    showFilters
                    focusIds={focusEvidence}
                    onFocusComplete={(item) => setFocusEvidence([item.evidence_id])}
                  />
                  <p className="text-[11px] text-ink-faint font-serif italic mt-4">
                    This explanation describes model and evidence signals. It does not establish that
                    fraud actually occurred.
                  </p>
                </div>
              </div>
            )}

            {tab === "Fraud graph" && (
              <div>
                <SectionLabel>Connected entities</SectionLabel>
                <FraudGraph nodes={graph?.nodes} edges={graph?.edges} highlightIds={highlightedNodes} />
                <p className="text-[11px] text-ink-faint font-serif italic mt-4">
                  Entities highlighted in the graph are the ones named in this case's graph evidence.
                </p>
              </div>
            )}

            {tab === "Timeline" && (
              <div className="space-y-8">
                <div>
                  <SectionLabel>Case activity</SectionLabel>
                  <CaseActivityTimeline
                    events={insights?.events || []}
                    velocityClusterSize={insights?.velocity_cluster_size || 0}
                    velocityWindowSeconds={insights?.velocity_window_seconds || 0}
                  />
                </div>

                <div>
                  <SectionLabel>Agent execution</SectionLabel>
                  <PipelineStepper agents={agent_executions} />
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
                    <ReportView content={report.content} onOpenEvidence={openEvidence} />
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

      {confirmation && (
        <div className="case-dialog-backdrop" onMouseDown={(event) => {
          if (event.target === event.currentTarget && !investigating && !removing) setConfirmation(null);
        }}>
          <section
            className={`case-confirm-dialog ${confirmation === "remove" ? "is-destructive" : ""}`}
            role={confirmation === "remove" ? "alertdialog" : "dialog"}
            aria-modal="true"
            aria-labelledby="case-confirm-title"
            aria-describedby="case-confirm-description"
          >
            <span className="case-dialog-kicker">CASE ACTION · {c.case_id}</span>
            <h2 id="case-confirm-title">
              {confirmation === "rerun" ? "Run this investigation again?" : "Remove this case?"}
            </h2>
            <p id="case-confirm-description">
              {confirmation === "rerun"
                ? "A new run will replace the saved report, evidence dossier, and routing decision for this case. The current result remains saved if you cancel."
                : "This removes the case and its saved investigation activity, evidence, report, and decision from the case register. Linked source transactions or claims and supporting source files are retained."}
            </p>
            {error && <p className="case-dialog-error" role="alert">{error}</p>}
            <div className="case-dialog-actions">
              <button type="button" className="case-dialog-cancel" onClick={() => setConfirmation(null)} disabled={investigating || removing} autoFocus={confirmation === "remove"}>
                {confirmation === "remove" ? "Keep case" : "Cancel"}
              </button>
              <button
                type="button"
                className="case-dialog-confirm"
                onClick={confirmCaseAction}
                disabled={investigating || removing}
                autoFocus={confirmation === "rerun"}
              >
                {removing ? "Removing…" : confirmation === "rerun" ? "Run again" : "Remove case"}
              </button>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}

function SectionLabel({ children }) {
  return <h2 className="font-serif text-base text-ink-text mb-3">{children}</h2>;
}
