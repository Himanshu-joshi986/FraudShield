"""
Phase 6a -- Synthetic claim/invoice document generator.

Renders the document records from data/synthetic/synthetic_documents.csv
(Phase 2) into actual PNG images that look like claim forms and invoices,
so the Document AI layer has real images to OCR rather than being handed
pre-digitized text (which would make the OCR step meaningless theatre).

Section 8 provenance: every generated document is a synthetic artifact
and is stamped as such in its own filename and metadata. These are not
real claim documents and must never be presented as such.

Deliberate realism: documents are rendered with slightly varying layout,
font size, and an optional light rotation/noise, because a Document AI
pipeline that only works on pixel-perfect renders isn't testing anything.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

random.seed(11)

REPO_ROOT = Path(__file__).resolve().parents[3]
SYNTHETIC_DIR = REPO_ROOT / "data" / "synthetic"
DOCS_OUT = REPO_ROOT / "data" / "documents"
DOCS_OUT.mkdir(parents=True, exist_ok=True)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]
FONT_BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
]


def _load_font(candidates, size):
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_invoice(record: dict, out_path: Path, add_noise: bool = True) -> dict:
    """
    Renders one invoice/claim document as a PNG.

    Font size is kept >=22px because OCR accuracy degrades badly on small
    text -- verified during development: at default (tiny) font size,
    tesseract read "1234.56" as "123458", dropping the decimal point.
    Real scanned invoices are ~200-300 DPI, so rendering at a realistic
    size is the honest test, not an easier one.
    """
    W, H = 1000, 700
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    title_font = _load_font(FONT_BOLD_CANDIDATES, 34)
    label_font = _load_font(FONT_BOLD_CANDIDATES, 24)
    body_font = _load_font(FONT_CANDIDATES, 24)
    small_font = _load_font(FONT_CANDIDATES, 18)

    doc_type = str(record["doc_type"]).upper()
    y = 45

    d.text((60, y), f"{doc_type}", font=title_font, fill="black")
    d.text((640, y + 6), "SYNTHETIC - DEMO ONLY", font=small_font, fill="#888888")
    y += 60
    d.line([(60, y), (W - 60, y)], fill="black", width=2)
    y += 35

    fields = [
        ("Invoice Number:", str(record["invoice_number"])),
        ("Claim Reference:", str(record["case_id"])),
        ("Scenario:", str(record["scenario_id"])),
        ("Claimant:", str(record.get("customer_name", "N/A"))),
        ("Date of Loss:", str(record.get("doc_date", "2026-05-14"))),
    ]
    for label, value in fields:
        d.text((60, y), label, font=label_font, fill="black")
        d.text((360, y), value, font=body_font, fill="black")
        y += 42

    y += 20
    d.line([(60, y), (W - 60, y)], fill="#cccccc", width=1)
    y += 30

    d.text((60, y), "Description:", font=label_font, fill="black")
    d.text((360, y), "Vehicle repair services rendered", font=body_font, fill="black")
    y += 55

    # The amount printed on the document. In mismatch scenarios this is
    # deliberately DIFFERENT from the claim amount on file -- that
    # discrepancy is what the validator must detect.
    d.text((60, y), "TOTAL AMOUNT:", font=label_font, fill="black")
    d.text((360, y), f"$ {float(record['invoice_amount']):,.2f}", font=title_font, fill="black")
    y += 70

    d.text((60, y), "Authorized Signature: ______________________", font=small_font, fill="#444444")

    if add_noise:
        angle = random.uniform(-0.6, 0.6)
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor="white")

    img.save(out_path, "PNG")

    dup_raw = record.get("duplicate_of_invoice")
    # pandas reads empty CSV cells as NaN; str(nan) == "nan" which is
    # truthy and would mark every document a duplicate. Normalize properly.
    duplicate_of = "" if (dup_raw is None or pd.isna(dup_raw)) else str(dup_raw).strip()

    return {
        "case_id": record["case_id"],
        "scenario_id": record["scenario_id"],
        "doc_type": record["doc_type"],
        "image_path": str(out_path.relative_to(REPO_ROOT)),
        "printed_invoice_amount": float(record["invoice_amount"]),
        "invoice_number": str(record["invoice_number"]),
        "claim_amount_on_file": float(record["claim_amount"]),
        "expected_amount_mismatch": bool(record["amount_mismatch"]),
        "expected_duplicate_of": duplicate_of,
        "data_source": "synthetic",
    }


def main():
    docs_df = pd.read_csv(SYNTHETIC_DIR / "synthetic_documents.csv")
    cases_df = pd.read_csv(SYNTHETIC_DIR / "synthetic_cases.csv")
    name_by_case = dict(zip(cases_df["case_id"], cases_df["customer_name"]))
    date_by_case = dict(zip(cases_df["case_id"], cases_df["timestamp"].str[:10]))

    manifest = []
    for _, row in docs_df.iterrows():
        record = row.to_dict()
        record["customer_name"] = name_by_case.get(record["case_id"], "Unknown Claimant")
        record["doc_date"] = date_by_case.get(record["case_id"], "2026-05-14")

        filename = f"SYNTHETIC_{record['case_id']}_{record['invoice_number']}.png"
        meta = render_invoice(record, DOCS_OUT / filename)
        meta["customer_name"] = record["customer_name"]
        manifest.append(meta)

    with open(DOCS_OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Generated {len(manifest)} synthetic documents -> {DOCS_OUT}")
    mismatches = sum(1 for m in manifest if m["expected_amount_mismatch"])
    duplicates = sum(1 for m in manifest if m["expected_duplicate_of"])
    print(f"  with deliberate amount mismatch: {mismatches}")
    print(f"  marked as duplicate invoices:    {duplicates}")


if __name__ == "__main__":
    main()
