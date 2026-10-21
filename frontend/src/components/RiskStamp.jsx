import React from "react";

const RISK_STYLES = {
  HIGH: { color: "#B54B42", label: "HIGH RISK" },
  MEDIUM: { color: "#B98530", label: "MEDIUM RISK" },
  LOW: { color: "#4B7A63", label: "LOW RISK" },
  UNSCORED: { color: "#4B5A70", label: "UNSCORED" },
};

const DECISION_STYLES = {
  ESCALATE: { color: "#B54B42", label: "DONE · ESCALATED" },
  HUMAN_REVIEW: { color: "#B98530", label: "DONE · HUMAN REVIEW" },
  AUTO_CLOSE_MONITOR: { color: "#4B7A63", label: "DONE · MONITOR" },
  new: { color: "#4B5A70", label: "NEW" },
};

export function RiskStamp({ level, size = "md" }) {
  const s = RISK_STYLES[level] || RISK_STYLES.UNSCORED;
  const sizeClass = size === "sm" ? "text-[10px] px-2 py-0.5" : "text-xs px-3 py-1";
  return (
    <span className={`risk-stamp ${sizeClass}`} style={{ color: s.color, borderColor: s.color }}>
      {s.label}
    </span>
  );
}

export function DecisionStamp({ decision, size = "md" }) {
  const s = DECISION_STYLES[decision] || DECISION_STYLES.new;
  const sizeClass = size === "sm" ? "text-[10px] px-2 py-0.5" : "text-xs px-3 py-1";
  return (
    <span className={`risk-stamp ${sizeClass}`} style={{ color: s.color, borderColor: s.color }}>
      {s.label}
    </span>
  );
}

export function riskColor(level) {
  return (RISK_STYLES[level] || RISK_STYLES.UNSCORED).color;
}

const CASE_STATUS_LABELS = {
  new: "Awaiting analysis",
  AUTO_CLOSE_MONITOR: "Done · monitoring",
  HUMAN_REVIEW: "Done · review required",
  ESCALATE: "Done · escalated",
};

export function caseStatusLabel(status) {
  return CASE_STATUS_LABELS[status] || status?.replaceAll("_", " ") || "Status unavailable";
}

export function investigationComplete(status) {
  return ["AUTO_CLOSE_MONITOR", "HUMAN_REVIEW", "ESCALATE"].includes(status);
}
