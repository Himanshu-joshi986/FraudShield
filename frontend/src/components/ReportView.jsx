import React, { useMemo } from "react";

/**
 * REPORT VIEW
 *
 * The report agent produces a structured, indented plain-text document.
 * Rendering it in a <pre> threw away that structure: every heading, every
 * field, every evidence reference looked like the same grey block.
 *
 * This renders it as a document instead -- titled sections, readable
 * fields, and evidence references that are clickable, so a reviewer
 * reading a claim in the report can jump straight to the exhibit that
 * supports it.
 *
 * Nothing here changes the report's content; it only presents it.
 */

const SECTION_PATTERN = /^(\d+)\.\s+(.+)$/;
const REFERENCE_PATTERN = /\[([A-Za-z0-9\-_]+)\]/;

function splitSections(lines) {
  const sections = [];
  let current = null;

  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    const match = line.match(SECTION_PATTERN);
    if (match) {
      current = { title: match[2].trim(), number: match[1], lines: [] };
      sections.push(current);
      continue;
    }
    if (!current) continue;
    current.lines.push(line);
  }
  return sections;
}

function isFieldLine(line) {
  const trimmed = line.trim();
  if (!trimmed) return false;
  if (trimmed.startsWith("[")) return false;
  return /^[A-Za-z][A-Za-z ()/-]{2,40}:/.test(trimmed);
}

function Reference({ id, onOpen }) {
  return (
    <button type="button" className="report-reference" onClick={() => onOpen?.(id)} title={`Open exhibit ${id}`}>
      {id}
    </button>
  );
}

export default function ReportView({ content, onOpenEvidence }) {
  const parsed = useMemo(() => {
    const lines = (content || "").split("\n");
    const title = lines[0]?.trim() || "Investigation report";
    const disclaimer = lines[1]?.trim() || "";
    const header = [];
    let index = 2;
    while (index < lines.length && lines[index].trim() && !SECTION_PATTERN.test(lines[index])) {
      header.push(lines[index].trim());
      index += 1;
    }
    return { title, disclaimer, header, sections: splitSections(lines.slice(index)) };
  }, [content]);

  if (!content) {
    return <p className="text-sm text-ink-dim font-serif italic">This case has no report content.</p>;
  }

  return (
    <article className="report">
      <header className="report-head">
        <h3 className="report-title">{parsed.title}</h3>
        {parsed.disclaimer && <p className="report-disclaimer">{parsed.disclaimer}</p>}
        {parsed.header.length > 0 && (
          <dl className="report-meta">
            {parsed.header.map((line, i) => {
              const [label, ...rest] = line.split(":");
              if (!rest.length) return <div key={i} className="report-meta-row"><dt>{line}</dt></div>;
              return (
                <div key={i} className="report-meta-row">
                  <dt>{label}</dt>
                  <dd>{rest.join(":").trim()}</dd>
                </div>
              );
            })}
          </dl>
        )}
      </header>

      {parsed.sections.map((section) => {
        const body = section.lines.filter((line) => line.trim());
        if (!body.length) return null;
        return (
          <section key={section.number} className="report-section">
            <h4 className="report-section-title">
              <span className="report-section-number">{section.number}</span>
              {section.title}
            </h4>
            <div className="report-section-body">
              {body.map((line, i) => {
                const trimmed = line.trim();

                if (trimmed.startsWith("[")) {
                  const idMatch = trimmed.match(REFERENCE_PATTERN);
                  const before = trimmed.slice(0, idMatch ? idMatch.index : trimmed.length);
                  const after = idMatch ? trimmed.slice(idMatch.index + idMatch[1].length + 2) : "";
                  return (
                    <p key={i} className="report-exhibit">
                      <span className="report-exhibit-text">{before.trim()}</span>
                      {idMatch && <Reference id={idMatch[1]} onOpen={onOpenEvidence} />}
                      <span className="report-exhibit-text">{after.replace(/^\s*\]/, "").trim()}</span>
                    </p>
                  );
                }

                if (isFieldLine(trimmed)) {
                  const [label, ...rest] = trimmed.split(":");
                  return (
                    <p key={i} className="report-field">
                      <span className="report-field-label">{label}:</span>{" "}
                      <span className="report-field-value">{rest.join(":").trim()}</span>
                    </p>
                  );
                }

                return <p key={i} className="report-line">{trimmed}</p>;
              })}
            </div>
          </section>
        );
      })}
    </article>
  );
}
