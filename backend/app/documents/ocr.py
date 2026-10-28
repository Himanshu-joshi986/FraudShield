"""
Phase 6b -- OCR adapter.

Pluggable OCR backends behind one interface, so the rest of the Document
AI layer never knows or cares which engine produced the text.

Backends:
  - "tesseract"  -- VERIFIED WORKING in this environment (tesseract 5.3.4
                    + pytesseract). This is what the measured Phase 6
                    results in docs/EXPERIMENTS.md §7 were produced with.
  - "rapidocr"   -- PP-OCRv4 running on onnxruntime. A pure pip install:
                    no system binary, no admin rights, ~15MB of models.
                    Selected automatically when tesseract is not present
                    on the machine (e.g. a Windows dev box), which is what
                    the Document AI layer falls back to there. The engine
                    that actually ran is always reported in the findings,
                    never hidden.
  - "paddleocr"  -- the engine named in the project spec. NOT installed
                    here (~500MB+ of model weights and a paddlepaddle
                    dependency that doesn't fit this environment's disk).
                    The adapter is written against PaddleOCR's real API
                    and is selected by setting OCR_BACKEND=paddleocr once
                    you've installed it -- see docs/RUN_DOCUMENT_AI.md.

Per the project's rule against silent substitution: PaddleOCR is not
faked here. If you select it without installing it, this module raises a
clear error telling you what to install, rather than quietly falling back
and letting you believe PaddleOCR ran. The same rule applies to every
backend when it is selected EXPLICITLY (via OCR_BACKEND or an argument):
an explicit choice that cannot run is an error, not a silent downgrade.

When NO engine is selected explicitly, the first AVAILABLE engine is
used, in the order tesseract -> rapidocr -> paddleocr. That is a
deliberate, reported fallback -- every finding records the backend that
produced it, so the result is never misattributed.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

# Auto-selection order: the engine the measured Phase-6 results used
# comes first, so nothing changes on a machine that has it.
_AUTO_ORDER = ["tesseract", "rapidocr", "paddleocr"]


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

    def available(self) -> bool:
        """Can this engine actually run on this machine?"""
        return False

    def extract(self, image_path: str | Path) -> OCRResult:
        raise NotImplementedError


class TesseractBackend(OCRBackend):
    """Verified working in the build environment (tesseract 5.3.4)."""

    name = "tesseract"
    _available_cache: bool | None = None

    def available(self) -> bool:
        if TesseractBackend._available_cache is None:
            try:
                import pytesseract
                # pytesseract imports fine even when the tesseract binary
                # is missing, so ask the binary itself.
                pytesseract.get_tesseract_version()
                TesseractBackend._available_cache = True
            except Exception:
                TesseractBackend._available_cache = False
        return TesseractBackend._available_cache

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


class RapidOCRBackend(OCRBackend):
    """
    PP-OCRv4 on onnxruntime -- a pure pip install, with no system binary
    and no admin rights required.

    Added because the project's own document features (field extraction,
    duplicate-invoice detection, the dashboard's side-by-side invoice
    comparison) are unusable on a machine without the tesseract binary,
    and a feature that cannot run is not a feature. The engine that ran
    is recorded in every finding, so the evidence always states which
    OCR produced it.
    """

    name = "rapidocr"
    _engine = None
    _engine_lock = threading.Lock()
    _available_cache: bool | None = None

    def available(self) -> bool:
        if RapidOCRBackend._available_cache is None:
            try:
                import rapidocr_onnxruntime  # noqa: F401
                RapidOCRBackend._available_cache = True
            except Exception:
                RapidOCRBackend._available_cache = False
        return RapidOCRBackend._available_cache

    def _get_engine(self):
        # one shared engine: the model is loaded once, and the onnxruntime
        # session is safe to use from several threads.
        if RapidOCRBackend._engine is None:
            with RapidOCRBackend._engine_lock:
                if RapidOCRBackend._engine is None:
                    from rapidocr_onnxruntime import RapidOCR
                    RapidOCRBackend._engine = RapidOCR()
        return RapidOCRBackend._engine

    def extract(self, image_path: str | Path) -> OCRResult:
        try:
            engine = self._get_engine()
        except ImportError as e:
            return OCRResult(text="", backend=self.name, error=f"missing dependency: {e}")

        try:
            raw, _elapse = engine(str(image_path))
        except Exception as e:
            return OCRResult(text="", backend=self.name, error=str(e))

        if not raw:
            return OCRResult(text="", backend=self.name, error="OCR produced no readable text")

        lines, confs, boxes = [], [], []
        for entry in raw:
            polygon, text, score = entry[0], entry[1], float(entry[2])
            lines.append(text)
            confs.append(score)
            # normalised to the same left/top/width/height shape the
            # tesseract backend returns, so consumers do not care which
            # engine ran.
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            boxes.append({
                "text": text,
                "left": min(xs), "top": min(ys),
                "width": max(xs) - min(xs), "height": max(ys) - min(ys),
                "conf": score,
            })

        return OCRResult(
            text="\n".join(lines),
            backend=self.name,
            confidence=(sum(confs) / len(confs) if confs else None),
            word_boxes=boxes,
        )


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

    def available(self) -> bool:
        try:
            import paddleocr  # noqa: F401
            return True
        except ImportError:
            return False

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
        except ImportError as e:
            return OCRResult(text="", backend=self.name, error=str(e))

        try:
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
            confidence=(sum(confs) / len(confs) if confs else None),
            word_boxes=boxes,
        )


_BACKENDS = {"tesseract": TesseractBackend, "rapidocr": RapidOCRBackend, "paddleocr": PaddleOCRBackend}

_INSTALL_HINTS = {
    "tesseract": "install the tesseract binary (apt install tesseract-ocr / winget install UB-Mannheim.TesseractOCR)",
    "rapidocr": "pip install rapidocr-onnxruntime",
    "paddleocr": "pip install paddlepaddle paddleocr",
}


def get_ocr_backend(name: str | None = None) -> OCRBackend:
    """
    Selects the OCR backend.

    EXPLICIT selection (an argument, or the OCR_BACKEND environment
    variable) is honoured strictly: if that engine cannot run, this
    raises a clear error rather than quietly using a different one and
    letting the result be misattributed.

    With NO explicit selection, the first available engine is used, in
    the order tesseract -> rapidocr -> paddleocr.
    """
    explicit = (name or os.environ.get("OCR_BACKEND") or "").strip().lower()

    if explicit:
        if explicit not in _BACKENDS:
            raise ValueError(f"Unknown OCR backend '{explicit}'. Available: {sorted(_BACKENDS)}")
        backend = _BACKENDS[explicit]()
        if not backend.available():
            raise RuntimeError(
                f"OCR_BACKEND={explicit} was selected explicitly but that engine is not "
                f"available on this machine. Install it with: {_INSTALL_HINTS.get(explicit, '...')}"
            )
        return backend

    for candidate in _AUTO_ORDER:
        backend = _BACKENDS[candidate]()
        if backend.available():
            return backend

    raise RuntimeError(
        "No OCR engine is available on this machine. Install one of: "
        + ", ".join(f"{name} ({_INSTALL_HINTS[name]})" for name in _AUTO_ORDER)
    )
