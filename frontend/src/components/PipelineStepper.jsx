import React from "react";

/**
 * PIPELINE STEPPER
 *
 * The five agents as they actually ran, with the time each one took.
 * The flat list this replaces showed agent names and durations but no
 * shape -- you could not see where the time went or which stage carried
 * the decision.
 */

const AGENT_ORDER = [
  { match: "sentinel", label: "Sentinel", role: "Signal collection" },
  { match: "investigation", label: "Investigation", role: "Graph + RAG" },
  { match: "evidence", label: "Evidence", role: "Dossier build" },
  { match: "report", label: "Report", role: "Narrative" },
  { match: "commander", label: "Commander", role: "Routing" },
];

const STATUS_TONE = {
  completed: "#4B7A63",
  running: "#B98530",
  failed: "#B54B42",
  skipped: "#7C8CA3",
  pending: "#9AA69B",
};

export default function PipelineStepper({ agents = [] }) {
  if (!agents.length) {
    return <p className="timeline-empty">No pipeline execution is recorded for this case yet.</p>;
  }

  const byName = new Map(agents.map((a) => [a.agent_name, a]));
  const steps = AGENT_ORDER.map((step) => {
    // matched on "contains", not "starts with": the agents are named
    // "Sentinel Agent", "Case Commander" and so on, and a prefix match
    // silently missed the Case Commander entirely.
    const match = agents.find((a) => (a.agent_name || "").toLowerCase().includes(step.match));
    return { ...step, run: match || byName.get(step.label) };
  });

  const durations = agents.map((a) => a.duration_ms || 0);
  const total = durations.reduce((sum, d) => sum + d, 0);
  const slowest = Math.max(...durations, 0);

  return (
    <div className="stepper">
      <ol className="stepper-track">
        {steps.map((step, i) => {
          const run = step.run;
          const tone = STATUS_TONE[run?.status] || "#7C8CA3";
          const duration = run?.duration_ms || 0;
          return (
            <li key={step.match} className={`step ${run ? "has-run" : "is-missing"}`}>
              <div className="step-marker" style={{ borderColor: tone }}>
                <span style={{ background: run ? tone : "transparent" }} />
              </div>
              <div className="step-body">
                <div className="step-title">
                  <strong>{step.label}</strong>
                  <span className="step-role">{step.role}</span>
                </div>
                {run ? (
                  <>
                    <p className="step-summary">{run.output_summary || run.summary}</p>
                    <div className="step-meta">
                      <span className="step-status" style={{ color: tone }}>
                        {run.status}
                      </span>
                      {duration > 0 && (
                        <span className="step-duration">
                          {duration.toFixed(0)}ms
                          {slowest > 0 && duration === slowest && (
                            <em> · slowest stage</em>
                          )}
                        </span>
                      )}
                    </div>
                    {slowest > 0 && (
                      <div className="step-bar" role="presentation">
                        <div
                          className="step-bar-fill"
                          style={{ width: `${Math.max(2, (duration / slowest) * 100)}%`, background: tone }}
                        />
                      </div>
                    )}
                  </>
                ) : (
                  <p className="step-summary is-missing">did not run</p>
                )}
              </div>
            </li>
          );
        })}
      </ol>
      {total > 0 && (
        <p className="stepper-total">Total pipeline time {total.toFixed(0)}ms across {agents.length} agents</p>
      )}
    </div>
  );
}
