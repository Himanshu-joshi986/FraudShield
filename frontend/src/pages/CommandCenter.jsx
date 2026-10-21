import React from "react";
import { riskColor, caseStatusLabel, investigationComplete } from "../components/RiskStamp";

export default function CommandCenter({ cases, analytics, onSeed, onOpenCase, loading }) {
  const stats = analytics || { total_cases: 0, auto_close_monitor: 0, human_review: 0, escalated: 0, unscored: 0, avg_risk_score: 0 };
  const recent = cases.slice(0, 10);

  return (
    <main className="command-page">
      <div className="command-overline"><span className="overline-rule" /> FRAUD OPERATIONS <span className="overline-divider">/</span> CASE OVERVIEW</div>
      <header className="command-heading">
        <div>
          <h1>Investigation desk</h1>
          <p>Review incoming cases, follow connected evidence, and record a clear outcome.</p>
        </div>
        <div className="desk-date"><span>WORKSPACE</span><b>LIVE CASE REGISTER</b></div>
      </header>

      {cases.length === 0 ? (
        <section className="welcome-panel">
          <div className="welcome-copy"><span className="welcome-kicker">YOUR CASE FILE STARTS HERE</span><h2>Bring the record.<br />Follow the evidence.</h2><p>Load the project case file to explore investigations, or add your own records from the Data Intake workspace.</p>
            <div className="welcome-actions"><button onClick={onSeed} disabled={loading}>{loading ? "Loading case file…" : "Load project case file  →"}</button><span>30 prepared investigation records</span></div>
          </div>
          <div className="welcome-art" aria-hidden="true"><div className="art-orbit orbit-one" /><div className="art-orbit orbit-two" /><div className="art-core">F<span>+</span></div><i className="art-node node-a" /><i className="art-node node-b" /><i className="art-node node-c" /><span className="art-caption">LINKED<br />SIGNALS</span></div>
        </section>
      ) : (
        <section className="desk-banner"><div><span className="banner-kicker">CASE REGISTER</span><h2>{stats.total_cases} cases in the workspace</h2><p>Each case keeps its source record, evidence trail, and investigation outcome together.</p></div><div className="banner-count"><strong>{String(stats.unscored).padStart(2, "0")}</strong><span>AWAITING ANALYSIS</span></div></section>
      )}

      <section className="workload-section">
        <div className="section-heading"><div><span className="section-kicker">AT A GLANCE</span><h2>Workload</h2></div><span className="section-caption">Updated from the case register</span></div>
        <div className="workload-grid">
          <Stat label="Total cases" value={stats.total_cases} tone="neutral" detail="All records" />
          <Stat label="Investigated" value={Math.max(0, stats.total_cases - stats.unscored)} tone="complete" detail="Pipeline complete" />
          <Stat label="Escalated" value={stats.escalated} tone="high" detail="Priority review" />
          <Stat label="Human review" value={stats.human_review} tone="medium" detail="Analyst decision" />
          <Stat label="Monitoring" value={stats.auto_close_monitor} tone="low" detail="Routine follow-up" />
          <Stat label="Unscored" value={stats.unscored} tone="unscored" detail="Awaiting analysis" />
        </div>
      </section>

      <section className="register-section">
        <div className="section-heading"><div><span className="section-kicker">CASE REGISTER</span><h2>{cases.length ? "Recently added" : "Ready when you are"}</h2></div>{cases.length > 10 && <span className="section-caption">Showing 10 of {cases.length} · use the register to browse all</span>}</div>
        {recent.length > 0 ? <div className="case-table-wrap"><table className="case-table"><thead><tr><th>CASE</th><th>CUSTOMER</th><th>TYPE</th><th>AMOUNT</th><th>INVESTIGATION STATUS</th><th></th></tr></thead><tbody>{recent.map(c => <tr key={c.case_id} onClick={() => onOpenCase(c.case_id)} tabIndex="0" onKeyDown={e => e.key === "Enter" && onOpenCase(c.case_id)}>
          <td><span className="table-case-id">{c.case_id}</span><span className="table-date">{new Date(c.timestamp).toLocaleDateString()}</span></td>
          <td className="table-customer">{c.customer_name || "Unattributed"}</td>
          <td><span className="domain-tag">{c.domain === "insurance_claim" ? "Insurance claim" : "Transaction"}</span></td>
          <td className="table-amount">${Number(c.amount).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
          <td><span className={`status-tag ${investigationComplete(c.status) ? "status-complete" : "status-new"}`} style={investigationComplete(c.status) ? { color: riskColor(c.risk_level) } : undefined}><i />{caseStatusLabel(c.status)}</span>{investigationComplete(c.status) && <span className="table-risk-score">{c.risk_level} risk · {Number(c.risk_score).toFixed(2)}</span>}</td>
          <td className="table-arrow">↗</td>
        </tr>)}</tbody></table></div> : <div className="empty-register"><span className="empty-register-icon">＋</span><div><b>No cases have been filed yet</b><p>Load the project case file above or use “File new case” to begin.</p></div></div>}
      </section>
    </main>
  );
}

function Stat({ label, value, tone, detail }) {
  return <article className={`workload-card tone-${tone}`}><span className="workload-label">{label}</span><strong>{value}</strong><span className="workload-detail">{detail}</span><i className="workload-accent" /></article>;
}
