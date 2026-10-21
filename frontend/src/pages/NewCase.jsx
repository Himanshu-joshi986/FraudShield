import React, { useState } from "react";
import { api } from "../services/api";

export default function NewCase({ onCancel, onCreated }) {
  const [domain, setDomain] = useState("financial_transaction");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");

  async function submit(event) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    const form = new FormData(event.currentTarget);
    const payload = {
      source: "manual",
      domain,
      customer_name: form.get("customer_name").trim() || null,
      customer_id: form.get("customer_id").trim() || null,
      amount: Number(form.get("amount")),
      data_source: "user_submitted",
    };
    const recordId = form.get("record_id").trim();
    if (domain === "financial_transaction") payload.transaction_id = recordId || null;
    else payload.claim_id = recordId || null;
    const timestamp = form.get("timestamp");
    if (timestamp) payload.timestamp = new Date(timestamp).toISOString();

    try {
      const created = await api.createCase(payload);
      await onCreated(created.case_id);
    } catch (err) {
      setError(err.message || "Could not create this case. Check the fields and try again.");
    } finally {
      setSubmitting(false);
    }
  }

  const inputClass = "w-full bg-ink-950 border border-ink-750 px-3 py-2 text-sm text-ink-text focus:border-forensic focus:outline-none";
  const labelClass = "block text-xs font-mono text-ink-dim mb-1";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="new-case-title"
        className="w-full max-w-3xl max-h-[90vh] overflow-y-auto border border-ink-750 bg-ink-950 p-6 md:p-10 shadow-2xl"
      >
        <div className="flex items-start justify-between gap-4 mb-6">
          <div>
            <h1 id="new-case-title" className="font-serif text-2xl text-ink-text">File a new case</h1>
            <p className="text-sm text-ink-dim mt-2">
              Enter the available case facts. The case will be saved before you start its investigation.
            </p>
          </div>
          <button
            type="button"
            onClick={onCancel}
            disabled={submitting}
            aria-label="Close case intake"
            className="text-sm font-mono px-3 py-2 border border-ink-750 text-ink-dim hover:text-ink-text disabled:opacity-50"
          >
            Close
          </button>
        </div>

      <form onSubmit={submit} className="space-y-5">
        <div>
          <label className={labelClass} htmlFor="domain">Case type</label>
          <select id="domain" className={inputClass} value={domain} onChange={(e) => setDomain(e.target.value)}>
            <option value="financial_transaction">Financial transaction</option>
            <option value="insurance_claim">Insurance claim</option>
          </select>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className={labelClass} htmlFor="customer_name">Customer name</label>
            <input id="customer_name" name="customer_name" className={inputClass} maxLength={200} autoComplete="name" />
          </div>
          <div>
            <label className={labelClass} htmlFor="customer_id">Customer / entity ID</label>
            <input id="customer_id" name="customer_id" className={inputClass} maxLength={200} />
          </div>
          <div>
            <label className={labelClass} htmlFor="record_id">
              {domain === "financial_transaction" ? "Transaction ID (optional)" : "Claim ID (optional)"}
            </label>
            <input id="record_id" name="record_id" className={inputClass} maxLength={200} />
          </div>
          <div>
            <label className={labelClass} htmlFor="amount">Amount</label>
            <input id="amount" name="amount" type="number" min="0.01" step="0.01" required className={inputClass} />
          </div>
          <div>
            <label className={labelClass} htmlFor="timestamp">Event time (optional)</label>
            <input id="timestamp" name="timestamp" type="datetime-local" className={inputClass} />
          </div>
        </div>
        <p className="text-xs font-serif text-ink-faint">
          Investigation uses only signals available for the entered case and linked records. The separately
          trained benchmark GNN and LSTM are not applied to manually entered cases.
        </p>
        {error && <p role="alert" className="text-xs font-mono text-risk-high">{error}</p>}
        <div className="flex gap-3 pt-2">
          <button type="submit" disabled={submitting} className="text-sm font-mono px-4 py-2 border border-forensic text-forensic hover:bg-forensic/10 disabled:opacity-50">
            {submitting ? "Saving case…" : "Save and review case"}
          </button>
          <button type="button" onClick={onCancel} disabled={submitting} className="text-sm font-mono px-4 py-2 border border-ink-750 text-ink-dim hover:text-ink-text disabled:opacity-50">
            Cancel
          </button>
        </div>
      </form>
      </section>
    </div>
  );
}
