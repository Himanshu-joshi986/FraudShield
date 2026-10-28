import React from "react";

/**
 * POLICY TRACE
 *
 * The routing decision is only useful if a reviewer can see WHICH rule
 * produced it. This panel shows the policy conditions that fired, in the
 * order the Case Commander evaluated them, and states plainly whether the
 * routing was driven by the score or by a policy override.
 *
 * A case that escalates because of a ring override is a DIFFERENT
 * decision from a case that escalates because its score crossed 0.65,
 * and the two need different follow-up. The panel never collapses them
 * into one number.
 */

const DECISION_TONE = {
  ESCALATE: "#B54B42",
  HUMAN_REVIEW: "#B98530",
  AUTO_CLOSE_MONITOR: "#4B7A63",
};

export default function PolicyTrace({ policy, counterfactuals = [], scoreBelowMedium = false, riskScore = null, riskAdjustment = null }) {
  if (!policy) return null;

  const tone = DECISION_TONE[policy.decision] || "#7C8CA3";
  const overrides = policy.overrides || [];
  const scoreDriven = policy.score_driven !== false;

  const actionable = (counterfactuals || []).filter((c) => c.achievable);
  const structural = (counterfactuals || []).filter((c) => !c.achievable);
  const noCounterfactual = (counterfactuals || []).length === 0;

  return (
    <section className="policy-trace" aria-label="Routing policy trace">
      <header className="policy-head">
        <span className="policy-kicker">Policy trace</span>
        <h3 className="policy-title">
          Routed to <span style={{ color: tone }}>{policy.decision?.replaceAll("_", " ")}</span>
          <span className="policy-version">policy {policy.policy_version}</span>
        </h3>
      </header>

      <div className={`policy-driver ${scoreDriven ? "is-score" : "is-override"}`}>
        <span className="policy-driver-label">
          {scoreDriven ? "Driven by the risk score" : "Driven by a policy override"}
        </span>
        <p>
          {scoreDriven
            ? "No override applied — the fused score alone decided the disposition."
            : `${overrides.join(", ")} applied alongside the score, so this case routes for review regardless of the score band.`}
        </p>
      </div>

      {policy.conditions_triggered?.length > 0 && (
        <ol className="policy-conditions">
          {policy.conditions_triggered.map((condition, i) => (
            <li key={i}>
              <span className="policy-conditions-index">{i + 1}</span>
              <span>{condition}</span>
            </li>
          ))}
        </ol>
      )}

      {policy.policy_source && (
        <p className="policy-source">Source: {policy.policy_source}</p>
      )}

      {(actionable.length > 0 || structural.length > 0 || noCounterfactual) && (
        <div className="counterfactual">
          <h4 className="counterfactual-title">What would change the outcome</h4>
          <p className="counterfactual-note">
            Computed from the underlying weighted signal average; evidence-policy adjustments are shown separately.
          </p>

          {noCounterfactual && (
            <div className="counterfactual-empty">
              <p>
                {riskAdjustment?.applied
                  ? `The underlying signal average was ${riskAdjustment.base_score.toFixed(2)}. The HIGH assessment is maintained by the duplicate-invoice evidence floor; changing unrelated signals will not remove that finding.`
                  : scoreBelowMedium
                  ? `This case scored ${riskScore != null ? riskScore.toFixed(2) : "below the MEDIUM band"} — already under the MEDIUM threshold, so there is no score change that would lower it further.`
                  : "No usable signal values were recorded for this case, so no score counterfactual can be computed."}
              </p>
              {overrides.length > 0 && (
                <p className="counterfactual-empty-override">
                  The disposition was set by the {overrides.join(" and ").toLowerCase()} instead —
                  that is what a reviewer should act on.
                </p>
              )}
            </div>
          )}

          {actionable.length > 0 && (
            <ul className="counterfactual-list">
              {actionable.map((c) => (
                <li key={`${c.signal}-${c.target_label}`} className="is-actionable">
                  <span className="counterfactual-target">{c.target_label}</span>
                  <span className="counterfactual-text">{c.explanation}</span>
                </li>
              ))}
            </ul>
          )}

          {structural.length > 0 && (
            <details className="counterfactual-more">
              <summary>
                {structural.length} further condition{structural.length === 1 ? "" : "s"} that would not change the score
              </summary>
              <ul className="counterfactual-list">
                {structural.map((c) => (
                  <li key={`${c.signal}-${c.target_label}`}>
                    <span className="counterfactual-target muted">{c.target_label}</span>
                    <span className="counterfactual-text">{c.explanation}</span>
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </section>
  );
}
