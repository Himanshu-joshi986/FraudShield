import React from "react";
import { riskColor } from "../components/RiskStamp";

export default function Analytics({ analytics }) {
  if (!analytics) return <div className="p-10 text-ink-dim font-serif">Loading…</div>;

  const total = analytics.total_cases || 1;
  const investigated = Math.max(0, analytics.total_cases - analytics.unscored);
  const bars = [
    { label: "Escalated", value: analytics.escalated, color: riskColor("HIGH") },
    { label: "Human review", value: analytics.human_review, color: riskColor("MEDIUM") },
    { label: "Auto-closed", value: analytics.auto_close_monitor, color: riskColor("LOW") },
    { label: "Not yet investigated", value: analytics.unscored, color: "#4B5A70" },
  ];

  return (
    <div className="max-w-3xl mx-auto p-10">
      <h1 className="font-serif text-2xl text-ink-text mb-1">Analytics</h1>
      <p className="text-sm text-ink-dim mb-8">
        Computed live from the case database. Demonstration scale — see the note below.
      </p>

      <div className="border border-ink-750 p-6 mb-6">
        <div className="analytics-completion-summary"><div><strong>{investigated}</strong><span>investigations complete</span></div><div><strong>{analytics.unscored}</strong><span>awaiting analysis</span></div></div>
        <h2 className="font-serif text-base text-ink-text mb-4">Case disposition</h2>
        <div className="space-y-3">
          {bars.map((b) => (
            <div key={b.label}>
              <div className="flex justify-between text-xs font-mono text-ink-dim mb-1">
                <span>{b.label}</span>
                <span>{b.value}</span>
              </div>
              <div className="h-2 bg-ink-850 border border-ink-750">
                <div className="h-full" style={{ width: `${(b.value / total) * 100}%`, background: b.color }} />
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="border border-ink-750 p-5">
          <div className="text-[11px] font-mono text-ink-faint uppercase tracking-wide mb-1">Average risk score</div>
          <div className="font-serif text-3xl text-ink-text">{analytics.avg_risk_score.toFixed(2)}</div>
          <p className="text-xs text-ink-dim font-serif mt-1">across all investigated cases</p>
        </div>
        <div className="border border-ink-750 p-5">
          <div className="text-[11px] font-mono text-ink-faint uppercase tracking-wide mb-1">
            Scenario ground-truth misses
          </div>
          <div className="font-serif text-3xl text-ink-text">{analytics.known_fraud_scenarios_missed}</div>
          <p className="text-xs text-ink-dim font-serif mt-1">
            demonstration-only, checked against the synthetic scenario library
          </p>
        </div>
      </div>

      <p className="text-[11px] text-ink-faint font-serif italic mt-8 border-t border-ink-750 pt-4">
        These figures reflect the current case database, which in a demo/development environment
        is small (the 30-case synthetic scenario library). They are not a claim about real-world
        detection performance — see docs/EXPERIMENTS.md for the actual measured model metrics on
        IEEE-CIS and carclaims.
      </p>
    </div>
  );
}
