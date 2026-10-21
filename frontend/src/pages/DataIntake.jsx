import React, { useState } from "react";
import { api } from "../services/api";

const TEMPLATE = "transaction_id,customer_id,customer_name,amount,timestamp,account_id,device_id,merchant_id,is_fraud\nTX-1001,C-204,Jordan Lee,840.50,2026-10-20T09:35:00Z,AC-55,DV-14,MER-8,\nTX-1002,C-319,Sam Rivera,129.99,2026-10-20T10:12:00Z,AC-82,DV-14,MER-3,\n";

export default function DataIntake({ onImported }) {
  const [domain, setDomain] = useState("financial_transaction");
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  async function submit(event) {
    event.preventDefault();
    if (!file) return;
    setBusy(true); setError(""); setResult(null);
    try {
      const imported = await api.importCsv(file.name, domain, await file.text());
      setResult(imported);
      await onImported();
    } catch (err) {
      setError(err.message || "The file could not be imported.");
    } finally { setBusy(false); }
  }

  function downloadTemplate() {
    const blob = new Blob([TEMPLATE], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a"); link.href = url; link.download = "fraudshield-transactions-template.csv";
    link.click(); URL.revokeObjectURL(url);
  }

  return <main className="data-page">
    <div className="data-eyebrow">DATA WORKSPACE <span> / </span> INTAKE</div>
    <div className="data-heading">
      <div><h1>Bring your records in.</h1><p>Build a case file from your own transaction or claims data.</p></div>
      <button className="template-button" onClick={downloadTemplate}>↓ <span>Download CSV template</span></button>
    </div>
    <div className="intake-grid">
      <section className="intake-card">
        <div className="step-label"><span>01</span> Configure import</div>
        <form onSubmit={submit}>
          <label className="field-label" htmlFor="record-domain">Record type</label>
          <select id="record-domain" value={domain} onChange={e => setDomain(e.target.value)}>
            <option value="financial_transaction">Financial transactions</option>
            <option value="insurance_claim">Insurance claims</option>
          </select>
          <label className="field-label file-label" htmlFor="record-file">Choose a CSV file</label>
          <label className={`file-drop ${file ? "has-file" : ""}`} htmlFor="record-file">
            <span className="file-icon">↥</span>
            <strong>{file ? file.name : "Drop a file here or browse"}</strong>
            <span>{file ? `${(file.size / 1024).toFixed(1)} KB · CSV` : "CSV format · up to 5,000 rows"}</span>
            <input id="record-file" type="file" accept=".csv,text/csv" onChange={e => { setFile(e.target.files?.[0] || null); setResult(null); }} />
          </label>
          <button className="import-button" type="submit" disabled={!file || busy}>{busy ? "Importing records…" : "Import records  →"}</button>
        </form>
        {error && <p role="alert" className="intake-error">{error}</p>}
        {result && <div className="import-result" role="status"><span>✓</span><div><strong>{result.imported} records added to your case file</strong><p>Each row is saved as a case and linked to its customer and source record.{result.graph_warning ? " Relationship analysis may be delayed while the graph store reconnects." : " Shared accounts and devices are ready for relationship analysis."}</p></div></div>}
      </section>
      <aside className="format-card">
        <div className="step-label"><span>02</span> Shape your data</div>
        <p className="format-intro">One row becomes one case. Only <b>amount</b> is required; add the fields you have.</p>
        <div className="field-list">
          <div><b>Required</b><span>amount</span></div>
          <div><b>Record</b><span>transaction_id or claim_id</span></div>
          <div><b>Identity</b><span>customer_id, customer_name</span></div>
          <div><b>Connections</b><span>account_id, device_id, address_id, broker_id, email</span></div>
          <div><b>Context</b><span>timestamp, merchant_id, policy_id</span></div>
          <div><b>Optional label</b><span>is_fraud or fraud_label</span></div>
        </div>
        <div className="data-note"><span>i</span><p>Extra columns are retained with the source record. Shared customer, account and device IDs help investigators find links across your uploads.</p></div>
      </aside>
    </div>
  </main>;
}
