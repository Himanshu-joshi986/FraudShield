import React, { useState } from "react";
import { api } from "../services/api";

/**
 * DOCUMENT COMPARISON
 *
 * Shows the actual supporting document(s) next to the fields read from
 * them, so a reviewer can check the machine's reading against the pixels
 * rather than taking the OCR on faith.
 *
 * When the same invoice number was filed on more than one case, every
 * document in that group is shown side by side and the fields that
 * DIFFER between them are highlighted. That comparison is the finding:
 * one invoice, three different claimants.
 */

const FLAG_LABELS = {
  claim_amount_mismatch: "Amount does not match the claim record",
  duplicate_invoice_across_cases: "Invoice number also filed on another case",
  claimant_name_not_found_on_document: "Claimant name not found on the document",
  low_ocr_confidence: "Low OCR confidence",
  no_amount_found_on_document: "No amount found on the document",
  ocr_unavailable: "OCR unavailable for this document",
};

export default function DocumentCompare({ dossier }) {
  const [zoom, setZoom] = useState(null);

  if (!dossier || !dossier.has_document) {
    return (
      <p className="doc-empty">
        {dossier?.summary || "No supporting document is attached to this case."}
      </p>
    );
  }

  const documents = dossier.documents || [];
  const rows = dossier.rows || [];

  return (
    <div className="doc-compare">
      <div className={`doc-banner ${dossier.duplicate_across_cases ? "is-duplicate" : "is-single"}`}>
        <span className="doc-banner-mark">{dossier.duplicate_across_cases ? "!" : "i"}</span>
        <p>{dossier.summary}</p>
      </div>

      <div className="doc-grid" style={{ "--doc-count": documents.length }}>
        {documents.map((doc) => (
          <figure key={doc.case_id} className={`doc-card ${doc.is_current_case ? "is-current" : ""}`}>
            <figcaption className="doc-card-head">
              <span className="doc-card-case">{doc.case_id}</span>
              {doc.is_current_case && <span className="doc-card-tag">this case</span>}
              <span className="doc-card-claimant">{doc.customer_name || "Unnamed claimant"}</span>
            </figcaption>

            {doc.degraded ? (
              <div className="doc-image is-missing">
                <span>Image available, but it could not be read</span>
                <small>{doc.degraded_reason}</small>
              </div>
            ) : (
              <button
                type="button"
                className="doc-image-button"
                onClick={() => setZoom(doc)}
                title="Open the full-size document"
              >
                <img
                  className="doc-image"
                  src={doc.image_name ? api.documentImageUrl(doc.image_name) : undefined}
                  alt={`Supporting document for case ${doc.case_id}`}
                  loading="lazy"
                  decoding="async"
                />
                <span className="doc-image-hint">click to enlarge</span>
              </button>
            )}

            <div className="doc-card-meta">
              <span>
                {doc.ocr_backend ? `read by ${doc.ocr_backend}` : "not read"}
                {doc.ocr_confidence != null && ` · confidence ${doc.ocr_confidence.toFixed(2)}`}
              </span>
              {doc.flags?.length > 0 && (
                <ul className="doc-flags">
                  {doc.flags.map((flag) => (
                    <li key={flag}>{FLAG_LABELS[flag] || flag}</li>
                  ))}
                </ul>
              )}
            </div>
          </figure>
        ))}
      </div>

      {rows.length > 0 && (
        <table className="doc-table">
          <thead>
            <tr>
              <th scope="col">Field</th>
              {documents.map((doc) => (
                <th key={doc.case_id} scope="col" className={doc.is_current_case ? "is-current" : ""}>
                  {doc.case_id}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.label} className={row.differs ? "is-different" : ""}>
                <th scope="row">{row.label}</th>
                {row.cells.map((cell, i) => (
                  <td key={i} className={cell.highlight ? "is-highlight" : ""}>
                    <span className="doc-cell-value">{cell.value}</span>
                    {cell.sub && <span className="doc-cell-sub">{cell.sub}</span>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {zoom && (
        <div className="doc-lightbox" role="dialog" aria-modal="true" onClick={() => setZoom(null)}>
          <button type="button" className="doc-lightbox-close" onClick={() => setZoom(null)}>
            close
          </button>
          <figure onClick={(e) => e.stopPropagation()}>
            <img src={api.documentImageUrl(zoom.image_name)} alt={`Document for ${zoom.case_id}`} />
            <figcaption>
              {zoom.case_id} · {zoom.customer_name || "Unnamed claimant"} ·{" "}
              {zoom.invoice_number || "invoice number unreadable"}
            </figcaption>
          </figure>
        </div>
      )}
    </div>
  );
}
