import React from "react";
import { riskColor } from "../components/RiskStamp";

export default function Analytics({ analytics, error }) {
  if (!analytics) return <main className="page-state"><span className="page-kicker">FRAUD OPERATIONS / PERFORMANCE</span><h1>Analytics</h1><p>{error || "Loading case analytics…"}</p></main>;

  const total = analytics.total_cases || 0;
  const investigated = Math.max(0, total - analytics.unscored);
  const bars = [
    { label: "Escalated", value: analytics.escalated, color: riskColor("HIGH"), description: "Priority investigation" },
    { label: "Human review", value: analytics.human_review, color: riskColor("MEDIUM"), description: "Analyst decision required" },
    { label: "Monitoring", value: analytics.auto_close_monitor, color: riskColor("LOW"), description: "Routine follow-up" },
    { label: "Awaiting analysis", value: analytics.unscored, color: "#87958c", description: "Not yet investigated" },
  ];

  return (
    <main className="analytics-page">
      <header className="page-heading">
        <div><span className="page-kicker">FRAUD OPERATIONS / PERFORMANCE</span><h1>Analytics</h1><p>Live status and risk summaries from the case register.</p></div>
        <div className="analytics-updated"><i /> LIVE CASE DATA</div>
      </header>

      <section className="analytics-metrics" aria-label="Case summary">
        <Metric label="Cases in register" value={total} detail="All sources" />
        <Metric label="Investigations complete" value={investigated} detail={`${total ? Math.round((investigated / total) * 100) : 0}% of cases`} tone="green" />
        <Metric label="Awaiting analysis" value={analytics.unscored} detail="Queued for review" tone="neutral" />
        <Metric label="Average risk score" value={Number(analytics.avg_risk_score || 0).toFixed(2)} detail="Scored cases only" tone="gold" />
      </section>

      <div className="analytics-content">
        <section className="analytics-panel disposition-panel">
          <div className="panel-heading"><div><span className="page-kicker">CURRENT WORKLOAD</span><h2>Case disposition</h2></div><span className="panel-total">{total} total</span></div>
          <div className="disposition-list">
            {bars.map((bar) => (
              <div className="disposition-row" key={bar.label}>
                <div className="disposition-copy"><span className="disposition-mark" style={{ background: bar.color }} /><div><strong>{bar.label}</strong><small>{bar.description}</small></div></div>
                <div className="disposition-track" aria-label={`${bar.value} ${bar.label}`}><i style={{ width: `${total ? (bar.value / total) * 100 : 0}%`, background: bar.color }} /></div>
                <b className="disposition-value">{bar.value}</b>
              </div>
            ))}
          </div>
        </section>

        <aside className="analytics-panel quality-panel">
          <span className="page-kicker">DEMO SCENARIO CHECK</span>
          <h2>Known scenario misses</h2>
          <strong className="quality-value">{analytics.known_fraud_scenarios_missed}</strong>
          <p>Cases where the current disposition differs from the expected synthetic scenario outcome.</p>
          <div className="quality-footnote">Demonstration measure · not a real-world performance estimate</div>
        </aside>
      </div>

      <p className="analytics-note">These figures describe the current local case database. The synthetic scenario library is for demonstration; the project documentation contains benchmark measurements on IEEE-CIS and carclaims data.</p>
    </main>
  );
}

function Metric({ label, value, detail, tone = "" }) {
  return <article className={`analytics-metric ${tone ? `metric-${tone}` : ""}`}><span>{label}</span><strong>{value}</strong><small>{detail}</small></article>;
}
