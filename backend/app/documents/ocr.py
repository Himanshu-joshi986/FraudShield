"""
Phase 6b -- OCR adapter.

Pluggable OCR backends behind one interface, so the rest of the Document
AI layer never knows or cares which engine produced the text.

Backends:
  - "tesseract"  -- VERIFIED WORKING in this environment (tesseract 5.3.4
                    + pytesseract). This is what the measured Phase 6
                    results in docs/EXPERIMENTS.md §7 were produced with.
  - "paddleocr"  -- the engine named in the project spec. NOT installed
                    here (~500MB+ of model weights and a paddlepaddle
                    dependency that doesn't fit this environment's disk).
                    The adapter is written against PaddleOCR's real API
                    and is selected by setting OCR_BACKEND=paddleocr once
                    you've installed it -- see docs/RUN_DOCUMENT_AI.md.

Per the project's rule against silent substitution: PaddleOCR is not
faked here. If you select it without installing it, this module raises a
clear error telling you what to install, rather than quietly falling back
and letting you believe PaddleOCR ran.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OCRResult:
    text: str
    backend: str
    confidence: float | None = None
    word_boxes: list = field(default_factory=list)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and bool(self.text.strip())


class OCRBackend:
    name = "base"

    def extract(self, image_path: str | Path) -> OCRResult:
        raise NotImplementedError


class TesseractBackend(OCRBackend):
    """Verified working in the build environment (tesseract 5.3.4)."""

    name = "tesseract"

    def extract(self, image_path: str | Path) -> OCRResult:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as e:
            return OCRResult(text="", backend=self.name, error=f"missing dependency: {e}")

        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)

            # per-word confidence, so downstream evidence can carry a real
            # confidence value rather than an invented one
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and int(c) >= 0]
            mean_conf = (sum(confs) / len(confs) / 100.0) if confs else None

            boxes = []
            for i, word in enumerate(data.get("text", [])):
                if word.strip():
                    boxes.append({
                        "text": word,
                        "left": data["left"][i], "top": data["top"][i],
                        "width": data["width"][i], "height": data["height"][i],
                        "conf": data["conf"][i],
                    })

            return OCRResult(text=text, backend=self.name, confidence=mean_conf, word_boxes=boxes)
        except Exception as e:  # pragma: no cover - depends on local tesseract install
            return OCRResult(text="", backend=self.name, error=str(e))


class PaddleOCRBackend(OCRBackend):
    """
    NOT INSTALLED / NOT RUN in the build environment.
    Written against PaddleOCR's real API. Install with:
        pip install paddlepaddle paddleocr
    then set OCR_BACKEND=paddleocr.
    """

    name = "paddleocr"

    def __init__(self, lang: str = "en"):
        self.lang = lang
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as e:
                raise ImportError(
                    "PaddleOCR is not installed. Install it with:\n"
                    "    pip install paddlepaddle paddleocr\n"
                    "Or use the tesseract backend (OCR_BACKEND=tesseract), "
                    "which is what the measured Phase 6 results used."
                ) from e
            self._engine = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
        return self._engine

    def extract(self, image_path: str | Path) -> OCRResult:
        try:
            engine = self._get_engine()
            raw = engine.ocr(str(image_path), cls=True)
        except Exception as e:
            return OCRResult(text="", backend=self.name, error=str(e))

        lines, confs, boxes = [], [], []
        for page in (raw or []):
            for entry in (page or []):
                box, (text, conf) = entry[0], entry[1]
                lines.append(text)
                confs.append(float(conf))
                boxes.append({"text": text, "box": box, "conf": float(conf)})

        return OCRResult(
            text="\n".join(lines),
            backend=self.name,
            confidence=(sum(confs) / len(confs)) if confs else None,
            word_boxes=boxes,
        )


_BACKENDS = {"tesseract": TesseractBackend, "paddleocr": PaddleOCRBackend}


def get_ocr_backend(name: str | None = None) -> OCRBackend:
    """
    Selects the OCR backend. Defaults to the OCR_BACKEND env var, then
    tesseract (the verified one). Raises on an unknown name rather than
    silently defaulting, so a typo never causes a wrong-engine result to
    be reported as if it came from the intended engine.
    """
    name = (name or os.environ.get("OCR_BACKEND") or "tesseract").lower()
    if name not in _BACKENDS:
        raise ValueError(f"Unknown OCR backend '{name}'. Available: {sorted(_BACKENDS)}")
    return _BACKENDS[name]()
