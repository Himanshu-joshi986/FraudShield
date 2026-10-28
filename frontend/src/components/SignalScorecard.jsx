import React, { useMemo } from "react";

/**
 * SIGNAL SCORECARD
 *
 * A decomposition of the available signal average. Evidence-policy risk
 * floors are shown separately so the bar contributions remain auditable.
 *
 * Two rules this component refuses to break:
 *
 *   1. An unavailable model is never given a fabricated score. A separate
 *      display-only reference can be shown when clearly labeled as such.
 *   2. A signal is never shown as "the reason" for the score on its own.
 *      The policy trace below says what actually drove the routing.
 */

const SIGNAL_META = {
  tabular_ml_risk: { label: "Tabular model", short: "Model", tone: "#5B8DBE" },
  graph_risk: { label: "Graph connectivity", short: "Graph", tone: "#8B6FB5" },
  temporal_risk: { label: "Transaction velocity", short: "Velocity", tone: "#B98530" },
  document_risk: { label: "Document check", short: "Document", tone: "#4B7A63" },
};

const ORDER = ["tabular_ml_risk", "graph_risk", "temporal_risk", "document_risk"];

function levelTone(level) {
  if (level === "HIGH") return "#B54B42";
  if (level === "MEDIUM") return "#B98530";
  if (level === "LOW") return "#4B7A63";
  return "#7C8CA3";
}

export default function SignalScorecard({ insights }) {
  const signals = useMemo(() => {
    if (!insights?.signals?.length) return [];
    const byName = new Map(insights.signals.map((s) => [s.name, s]));
    return ORDER.map((name) => byName.get(name)).filter(Boolean);
  }, [insights]);

  if (!insights || !signals.length) return null;

  const thresholds = insights.thresholds || {};
  const high = thresholds.HIGH ?? 0.65;
  const medium = thresholds.MEDIUM ?? 0.35;
  const riskScore = insights.risk_score ?? 0;
  const baseRiskScore = insights.base_risk_score ?? insights.fusion?.base_risk_score ?? riskScore;
  const adjustment = insights.fusion?.risk_adjustment;
  const level = insights.risk_level;
  const tone = levelTone(level);
  const availableCount = signals.filter((s) => s.available).length;

  // Band background: where this case's score sits relative to the policy
  // thresholds, drawn to scale so the position is honest at a glance.
  const scaleMax = 1;
  const pct = (value) => `${Math.max(0, Math.min(100, value * 100))}%`;

  return (
    <section className="scorecard" aria-label="Signal scorecard">
      <header className="scorecard-head">
        <div>
          <h3 className="scorecard-title">Why this case scored {riskScore.toFixed(2)}</h3>
          <p className="scorecard-sub">
            {availableCount} of {signals.length} signals available for this case
            {insights.fusion?.degraded ? " · scoring degraded, manual review required" : ""}
          </p>
        </div>
        <div className="scorecard-total" style={{ borderColor: tone, color: tone }}>
          <span>risk score</span>
          <strong>{riskScore.toFixed(2)}</strong>
        </div>
      </header>

      <div className="scorecard-scale" aria-hidden="true">
        <div className="scorecard-scale-track">
          <div className="scorecard-scale-band band-low" style={{ width: pct(medium) }} />
          <div className="scorecard-scale-band band-medium" style={{ width: pct(high - medium) }} />
          <div className="scorecard-scale-band band-high" style={{ left: pct(high), right: 0 }} />
          <div className="scorecard-scale-marker" style={{ left: pct(riskScore), background: tone }} />
        </div>
        <div className="scorecard-scale-legend">
          <span>0.00</span>
          <span className="tick-medium">MEDIUM {medium.toFixed(2)}</span>
          <span className="tick-high">HIGH {high.toFixed(2)}</span>
          <span>1.00</span>
        </div>
      </div>

      {adjustment?.type === "evidence_floor" && (
        <div className="scorecard-adjustment" role="note">
          <strong>{adjustment.signal === "duplicate_invoice_across_cases" ? "High-risk assessment: duplicate invoice across claims" : `Human-review assessment: ${adjustment.signal === "claim_amount_mismatch" ? "invoice amount mismatch" : "high transaction velocity"}`}</strong>
          <span>
            {adjustment.applied
              ? `The signal average was ${baseRiskScore.toFixed(2)}. This evidence raises the assessment to ${riskScore.toFixed(2)} (${level}) under the review policy.`
              : `The signal average is ${baseRiskScore.toFixed(2)}. The ${adjustment.signal === "duplicate_invoice_across_cases" ? "duplicate-invoice evidence independently requires escalation" : "review trigger independently requires human review"}; the assessment remains ${riskScore.toFixed(2)} (${level}).`}
            {" "}This is a review priority, not a finding that fraud is proven.
          </span>
        </div>
      )}

      <div className="signal-grid">
        {signals.map((signal) => {
          const meta = SIGNAL_META[signal.name] || { label: signal.name, tone: "#4B5A70" };
          if (!signal.available) {
            const hasBaseline = signal.baseline_value != null;
            return (
              <div key={signal.name} className="signal-card is-unavailable">
                <div className="signal-card-head">
                  <span className="signal-name">{meta.label}</span>
                  <span className="signal-tag">{hasBaseline ? "reference only" : "not scored"}</span>
                </div>
                <div className="signal-card-body">
                  <span className={`signal-value ${hasBaseline ? "is-reference" : "muted"}`}>
                    {hasBaseline ? signal.baseline_value.toFixed(2) : "N/A"}
                  </span>
                  <span className="signal-weight">{hasBaseline ? "baseline reference" : `${(signal.weight * 100).toFixed(0)}% nominal weight`}</span>
                </div>
                <div className="signal-bar" role="presentation">
                  <div className="signal-bar-fill is-empty" style={{ width: "0%" }} />
                </div>
                <p className="signal-note">{signal.note}</p>
                <div className="signal-card-foot">
                  <span>{hasBaseline ? "No model score available" : "No score available"}</span>
                  <span className="signal-foot-note">not included in the average</span>
                </div>
              </div>
            );
          }
          const contributionPct = signal.contribution_pct ?? 0;
          const weightShare = signal.weight_share_pct ?? 0;
          return (
            <div key={signal.name} className="signal-card">
              <div className="signal-card-head">
                <span className="signal-name">{meta.label}</span>
                <span className="signal-tag" style={{ color: meta.tone, borderColor: `${meta.tone}55` }}>
                  {contributionPct.toFixed(0)}% of score
                </span>
              </div>
              <div className="signal-card-body">
                <span className="signal-value">
                  {signal.name === "document_risk" && signal.value === 0 ? "Clear" : signal.value.toFixed(2)}
                </span>
                <span className="signal-weight">x {signal.weight.toFixed(2)} weight</span>
              </div>
              <div className="signal-bar" role="presentation">
                <div
                  className="signal-bar-fill"
                  style={{ width: `${Math.max(0, contributionPct)}%`, background: meta.tone }}
                />
              </div>
              {signal.name === "temporal_risk" && signal.note && (
                <p className="signal-note">{signal.note}</p>
              )}
              {signal.name === "document_risk" && signal.value === 0 ? (
                <div className="signal-card-foot">
                  <span>No document risk indicators detected</span>
                  <span className="signal-foot-note">document was checked</span>
                </div>
              ) : <div className="signal-card-foot">
                <span>
                  contributes {signal.contribution.toFixed(3)} of {baseRiskScore.toFixed(3)} base score
                </span>
                <span className="signal-foot-note">
                  {weightShare > 0 ? `${weightShare.toFixed(0)}% of the weight` : "excluded from the average"}
                </span>
              </div>}
            </div>
          );
        })}
      </div>
    </section>
  );
}
