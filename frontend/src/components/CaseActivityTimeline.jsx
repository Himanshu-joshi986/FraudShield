import React from "react";

/**
 * CASE ACTIVITY TIMELINE
 *
 * The business timeline of a case: what was filed, when, and which
 * filings landed close enough together to look like a burst.
 *
 * This is the investigator's view of the case -- as opposed to the
 * pipeline's view (how long each agent took). The two are deliberately
 * separate: one says "what happened to this customer", the other says
 * "how the machine looked at it".
 */

const KIND_META = {
  transaction: { label: "Transaction", tone: "#5B8DBE", glyph: "T" },
  claim: { label: "Claim", tone: "#B98530", glyph: "C" },
  document: { label: "Document", tone: "#8B6FB5", glyph: "D" },
};

const LEVEL_TONE = {
  HIGH: "#B54B42",
  MEDIUM: "#B98530",
  LOW: "#4B7A63",
  UNSCORED: "#7C8CA3",
};

function formatStamp(value) {
  if (!value) return "no timestamp";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toISOString().replace("T", " ").slice(0, 16) + " UTC";
}

export default function CaseActivityTimeline({ events = [], velocityClusterSize = 0, velocityWindowSeconds = 0 }) {
  if (!events.length) {
    return <p className="timeline-empty">No recorded activity for this case yet.</p>;
  }

  const windowMinutes = Math.round((velocityWindowSeconds || 0) / 60);

  return (
    <div className="activity">
      {velocityClusterSize > 1 && (
        <div className="activity-note">
          <strong>
            {velocityClusterSize} filings within {windowMinutes} minutes
          </strong>
          <span>
            Highlighted entries fall inside the {windowMinutes}-minute velocity window the Sentinel
            agent measures. A burst of filings from one entity is a known pattern.
          </span>
        </div>
      )}

      <ol className="activity-list">
        {events.map((event, i) => {
          const meta = KIND_META[event.kind] || { label: event.kind, tone: "#7C8CA3", glyph: "•" };
          const levelTone = LEVEL_TONE[event.risk_level] || LEVEL_TONE.UNSCORED;
          const classes = [
            "activity-item",
            event.is_current ? "is-current" : "",
            event.in_velocity_window ? "in-window" : "",
            event.flagged ? "is-flagged" : "",
          ]
            .filter(Boolean)
            .join(" ");

          return (
            <li key={`${event.kind}-${event.case_id}-${i}`} className={classes}>
              <span className="activity-rail" aria-hidden="true">
                <span className="activity-dot" style={{ borderColor: meta.tone }}>
                  {meta.glyph}
                </span>
              </span>
              <div className="activity-body">
                <div className="activity-head">
                  <span className="activity-title">{event.title}</span>
                  <span className="activity-time">{formatStamp(event.timestamp)}</span>
                </div>
                <p className="activity-detail">{event.detail}</p>
                <div className="activity-tags">
                  <span className="activity-tag" style={{ color: meta.tone, borderColor: `${meta.tone}55` }}>
                    {meta.label}
                  </span>
                  {event.is_current && <span className="activity-tag is-current">this case</span>}
                  {event.in_velocity_window && (
                    <span className="activity-tag is-window">in velocity window</span>
                  )}
                  {event.flagged && (
                    <span className="activity-tag is-flagged" style={{ color: levelTone, borderColor: `${levelTone}55` }}>
                      flagged
                    </span>
                  )}
                </div>
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
