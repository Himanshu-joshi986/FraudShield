# FraudShield classroom walkthrough

This fictional case pack demonstrates the complete user workflow: import case records, add and associate reference material in the Knowledge Base, investigate cases, and review the saved outcome. Names, IDs, merchants, amounts, and documents are synthetic training material. A risk signal is a lead for review, not proof of fraud.

## Before you begin

- Run FraudShield locally and open the dashboard.
- Use a fresh or empty case database for the stable `C-DOC-*` case IDs. Import `document_check_cases.csv` only once; importing it again is correctly rejected as a duplicate.
- The receipt evidence is already registered for these case IDs in `data/documents/manifest.json`. The app reads the OCR-ready PNG copies in `data/documents/`; the PDFs in this bundle are readable presentation copies.

## Complete walkthrough

### 1. Import four case records

1. Open **Data intake**.
2. Choose **Financial transactions**.
3. Upload `document_check_cases.csv` and choose **Import records**.
4. Confirm that four cases were added. The register should include `C-DOC-101` through `C-DOC-104`.

The CSV connects three customers through shared device `D-DEMO-77` and merchant `M-DEMO-14`. The fraud-label column is intentionally blank. The rows create the case and source transaction records; their case-linked receipt images are already present in the project’s document manifest.

Velocity is evaluated against earlier events for the same customer in the preceding hour. Later events do not retroactively create a burst for an earlier case, and exact duplicate ingestions of the same customer, timestamp, amount, and domain count once.

The `INV-7003` receipt on `C-DOC-103` and `C-DOC-104` is cross-checked across cases. That finding raises both cases to a HIGH review assessment and escalation; the scorecard keeps the underlying signal average separate from this evidence-based floor. This is a strong duplicate-filing indicator, not a final determination of fraud. The tabular model has no score for these rows because they do not include the 69-feature vector its trained model requires. The scorecard shows `0.05` as a display-only baseline reference, clearly marked as not scored and excluded from the average.

### 2. Add supporting material to the Knowledge Base

1. Open **Knowledge Base**.
2. Add these PDFs from `evidence/` as separate documents:
   - `device_activity_summary.pdf`
   - `merchant_activity_note.pdf`
   - `analyst_review_referral.pdf`
3. Give each a clear title. Keep the device summary and merchant note in **Shared library · available to all cases**.
4. In **Available sources**, set the analyst referral’s availability to `C-DOC-102`. The app saves the link immediately. That case can retrieve the referral during investigation; other cases cannot.
5. Search for “shared device across several customers” and then search within `C-DOC-102` to demonstrate the linked source.

These reference PDFs become searchable knowledge. The receipt PDFs are presentation copies; don’t upload them as a substitute for the case evidence. The case’s **Evidence** tab shows the receipt and OCR comparison from the registered document images.

### 3. Investigate and complete the cases

Open each case from the register and choose **Run investigation**. The five agents run in sequence; review the live agent activity, then the saved outcome:

| Case | Customer | Receipt check expected |
|---|---|---|
| `C-DOC-101` | Maya Chen | `INV-6001`, $1,840.00 matches the case |
| `C-DOC-102` | Luis Ortega | `INV-6002`; receipt is $2,875.50 versus case amount $2,275.50 |
| `C-DOC-103` | Priya Shah | `INV-7003`; same invoice number is filed on `C-DOC-104` |
| `C-DOC-104` | Maya Chen | `INV-7003`; duplicate filing and $3,190.00 receipt versus $1,295.75 case amount |

After each run, the case gets a completed investigation status, risk score, policy disposition, evidence dossier, and generated report. **Done · human review** and **Done · escalated** are completed pipeline outcomes routed for analyst action; they don’t mean a human decision has been made. For the clearest walkthrough, open **Overview** for the scorecard and policy trace, **Evidence** for the receipt comparison and exhibits, then **Report** for the generated report.

### 4. Optional: demonstrate the standalone graph CSV

`transactions_demo.csv` is an alternate representation of the same four fictional events, using generated case IDs. Use it **instead of** `document_check_cases.csv` only when you want to show graph relationships without the stable receipt-check IDs. Do not import both into one demo database: the app will create duplicate case rows for the same events. Velocity analysis now uses earlier events only and collapses exact duplicate event records, but duplicate cases still clutter the register. If both files were already imported, use a fresh demo database for a clean presentation.

## Bundle contents

- `document_check_cases.csv` — stable case records for the end-to-end workflow
- `transactions_demo.csv` — alternate CSV intake for the shared-device graph scenario
- `evidence/` — readable receipts and contextual reference PDFs
- `document_check_presenter_guide.pdf` — short case comparison and presentation notes
