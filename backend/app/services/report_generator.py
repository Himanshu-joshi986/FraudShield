"""
Phase 11 -- PDF report generation (Section 42).

Renders the Report Architect's evidence-backed content (Phase 9) plus
the "Explain This Case" ranking (Phase 10) into a downloadable PDF.
Nothing here invents content: every section pulls directly from
state['report']['content'] / state['evidence'] / explain_case() output,
which are themselves traceable back to real tool calls (Section 28's
evidence chain).

Uses reportlab's Platypus API (SimpleDocTemplate + Paragraph/Table),
which is the right tool for a genuinely multi-section, multi-page
document per the project's own PDF skill guidance -- not raw canvas
drawing, which doesn't handle pagination/wrapping for a document this
structured.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RISK_COLORS = {"LOW": colors.HexColor("#4ADE80"), "MEDIUM": colors.HexColor("#FBBF24"),
              "HIGH": colors.HexColor("#FB7185"), "UNSCORED": colors.grey}


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("FSTitle", parent=styles["Title"], fontSize=18, spaceAfter=4))
    styles.add(ParagraphStyle("FSDisclaimer", parent=styles["Normal"], fontSize=9,
                              textColor=colors.HexColor("#666666"), spaceAfter=10))
    styles.add(ParagraphStyle("FSSectionHeading", parent=styles["Heading2"],
                              spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#1E293B")))
    styles.add(ParagraphStyle("FSEvidence", parent=styles["Normal"], fontSize=9,
                              leftIndent=14, spaceAfter=4, textColor=colors.HexColor("#334155")))
    styles.add(ParagraphStyle("FSMono", parent=styles["Normal"], fontName="Courier", fontSize=8,
                              textColor=colors.HexColor("#475569")))
    return styles


def _evidence_table(evidence: list[dict], styles):
    if not evidence:
        return Paragraph("No findings of this type for this case.", styles["FSEvidence"])

    data = [["ID", "Finding", "Confidence"]]
    for e in evidence:
        finding = e["finding"]
        if len(finding) > 140:
            finding = finding[:137] + "..."
        data.append([e["evidence_id"], Paragraph(finding, styles["FSEvidence"]), f"{e['confidence']:.2f}"])

    table = Table(data, colWidths=[1.3 * inch, 4.3 * inch, 0.7 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    return table


def generate_report_pdf(final_state: dict, explanation: dict) -> Path:
    """
    final_state: a completed FraudCaseState (post Case Commander).
    explanation: output of explainability.explain_case.explain_case(final_state)

    Returns the path to the generated PDF.
    """
    case_id = final_state["case_id"]
    case = final_state["input_data"]
    report = final_state.get("report", {})
    routing = final_state.get("routing_decision", {})
    evidence = final_state.get("evidence", [])
    risk_level = final_state.get("risk_level", "UNSCORED")

    styles = _styles()
    out_path = REPORTS_DIR / f"FraudShield_Report_{case_id}.pdf"
    doc = SimpleDocTemplate(str(out_path), pagesize=LETTER,
                            topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                            leftMargin=0.6 * inch, rightMargin=0.6 * inch)
    story = []

    # --- header ---
    story.append(Paragraph("FraudShield Investigation Report", styles["FSTitle"]))
    story.append(Paragraph(
        "Demonstration draft &mdash; NOT an official regulatory filing. See Section 6 for uncertainty notes.",
        styles["FSDisclaimer"]))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#CBD5E1")))
    story.append(Spacer(1, 8))

    header_data = [
        ["Case ID", case_id, "Risk Level", risk_level],
        ["Domain", case.get("domain", "unknown"), "Risk Score", f"{final_state.get('risk_score', 0):.2f}"],
        ["Amount", f"${case.get('amount', 0):,.2f}", "Generated", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")],
    ]
    header_table = Table(header_data, colWidths=[1.1 * inch, 2.3 * inch, 1.1 * inch, 1.8 * inch])
    header_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("BACKGROUND", (3, 0), (3, 0), RISK_COLORS.get(risk_level, colors.grey)),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # --- 1. executive summary / explain this case ---
    story.append(Paragraph("1. Executive Summary &mdash; Why This Case Was Flagged", styles["FSSectionHeading"]))
    story.append(Paragraph(explanation.get("headline", ""), styles["Normal"]))
    story.append(Spacer(1, 6))
    for r in explanation.get("reasons", []):
        story.append(Paragraph(
            f"<b>{r['rank']}.</b> {r['reason']} "
            f"<font color='#64748B'>[{r['evidence_id']}, confidence {r['confidence']:.2f}, "
            f"source: {r['source']}]</font>",
            styles["FSEvidence"],
        ))

    # --- 2-5: evidence by type ---
    by_type: dict[str, list] = {}
    for e in evidence:
        by_type.setdefault(e["type"], []).append(e)

    section_titles = [
        ("graph", "2. Graph / Network Analysis"),
        ("temporal", "3. Temporal Analysis"),
        ("document", "4. Document Analysis"),
        ("regulatory", "5. Regulatory / Policy References"),
        ("model_explanation", "6. Model Explanation"),
    ]
    for key, title in section_titles:
        story.append(Paragraph(title, styles["FSSectionHeading"]))
        story.append(_evidence_table(by_type.get(key, []), styles))
        story.append(Spacer(1, 4))

    # --- uncertainty ---
    story.append(Paragraph("7. Uncertainty &amp; Responsible Use", styles["FSSectionHeading"]))
    story.append(Paragraph(explanation.get("disclaimer", ""), styles["Normal"]))
    story.append(Spacer(1, 6))

    # --- recommended disposition ---
    story.append(Paragraph("8. Recommended Disposition", styles["FSSectionHeading"]))
    story.append(Paragraph(
        f"<b>{routing.get('decision', 'UNKNOWN')}</b> &mdash; {routing.get('reason', '')}",
        styles["Normal"],
    ))
    story.append(Paragraph(
        f"Policy conditions triggered: {'; '.join(routing.get('policy_conditions_triggered', []))}",
        styles["FSMono"],
    ))
    story.append(Paragraph(f"Policy version: {routing.get('policy_version', '')} "
                          f"(source: {routing.get('policy_source', '')})", styles["FSMono"]))

    doc.build(story)
    return out_path
