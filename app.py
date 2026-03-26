"""
CivicTrack ML Service
---------------------
Serves three YOLO detection models + NLP text classifier via REST API.

Models (place trained .pt files in ./models/):
  models/pothole.pt   – Roads / pothole detection       (YOLO)
  models/garbage.pt   – Garbage / waste detection        (YOLO)
  models/crack.pt     – Road crack detection             (YOLO)

NLP model:
  models/classifier.pkl – trained by train_nlp.py

Usage:
  python app.py                         (dev, localhost:5000)
  gunicorn app:app --timeout 120        (production on Render)
"""

import os, io, base64, pickle, math, logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)-8s] %(message)s")
log = logging.getLogger("civictrack")

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB max payload
CORS(app)  # Allow cross-origin requests from Vercel frontend

# ── Model paths (override via env vars on Render dashboard) ───────────────────
MODEL_PATHS = {
    "pothole":  os.environ.get("MODEL_POTHOLE",  "models/pothole.pt"),
    "garbage":  os.environ.get("MODEL_GARBAGE",  "models/garbage.pt"),
    "crack":    os.environ.get("MODEL_CRACK",    "models/crack.pt"),
}
NLP_PATH = os.environ.get("NLP_MODEL", "models/classifier.pkl")

# Map UI category names → YOLO model key
CATEGORY_MODEL_MAP = {
    "Roads":         "pothole",
    "Garbage":       "garbage",
    "Road Cracks":   "crack",
    "Other":         None,
}

# ── Lazy caches (models load on first request to save startup RAM) ────────────
_yolo_cache = {}
_nlp_model  = None


# ── YOLOv12 AAttn compatibility patch ────────────────────────────────────────
def _patch_aattn():
    """
    Patches YOLOv12's AAttn module so it works whether the checkpoint
    stores a single qkv conv *or* separate qk + v convs.  Runs once.
    """
    try:
        import torch                                 # noqa: F811
        from ultralytics.nn.modules.block import AAttn
    except Exception:
        return  # ultralytics not installed or different version

    if getattr(AAttn, "_ct_patched", False):
        return

    _orig_forward = AAttn.forward

    def _safe_forward(self, x):
        try:
            return _orig_forward(self, x)
        except AttributeError as exc:
            if not (hasattr(self, "qk") and hasattr(self, "v")
                    and "qkv" in str(exc)):
                raise
            b, c, h, w = x.shape
            n = h * w
            if hasattr(self, "pe"):
                x = x + self.pe(x)
            qk = self.qk(x).flatten(2).transpose(1, 2)
            v  = self.v(x).flatten(2).transpose(1, 2)
            q, k = qk.chunk(2, dim=-1)
            heads    = int(getattr(self, "num_heads", 8))
            head_dim = c // heads if heads else c
            scale    = 1.0 / math.sqrt(head_dim) if head_dim else 1.0
            q = q.view(b, n, heads, head_dim).transpose(1, 2)
            k = k.view(b, n, heads, head_dim).transpose(1, 2)
            v = v.view(b, n, heads, head_dim).transpose(1, 2)
            import torch as _t
            attn = _t.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
            out  = (attn @ v).transpose(1, 2).contiguous().view(b, n, c)
            out  = out.transpose(1, 2).view(b, c, h, w)
            if hasattr(self, "proj"):
                out = self.proj(out)
            return out

    AAttn.forward     = _safe_forward
    AAttn._ct_patched = True
    log.info("[Patch] AAttn forward patched for YOLOv12 compat")

_patch_aattn()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _get_yolo(name: str):
    """Load & cache a YOLO model by key.  Returns None when .pt missing."""
    if name in _yolo_cache:
        return _yolo_cache[name]
    path = MODEL_PATHS.get(name, "")
    if path and os.path.exists(path):
        from ultralytics import YOLO
        model = YOLO(path)
        _yolo_cache[name] = model
        log.info(f"Loaded YOLO model: {name} ← {path}")
    else:
        _yolo_cache[name] = None
        log.warning(f"Model file not found: {path}")
    return _yolo_cache[name]


def _get_nlp():
    """Load & cache the NLP pipeline."""
    global _nlp_model
    if _nlp_model is None:
        if os.path.exists(NLP_PATH):
            with open(NLP_PATH, "rb") as f:
                _nlp_model = pickle.load(f)
            log.info(f"Loaded NLP classifier ← {NLP_PATH}")
        else:
            log.warning(f"NLP model not found at {NLP_PATH}")
    return _nlp_model


def _decode_image(b64: str) -> Image.Image:
    """Decode a base64 image string (with or without data-URI prefix)."""
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _run_yolo(model_name: str, img: Image.Image, conf: float = 0.25) -> dict:
    """Run one YOLO model and return a clean result dict."""
    yolo = _get_yolo(model_name)
    if yolo is None:
        return {"has_detection": False, "count": 0, "confidence": 0.0,
                "error": f"Model '{model_name}' not loaded"}
    results = yolo(img, conf=conf, verbose=False)[0]
    boxes   = results.boxes
    count   = len(boxes)
    max_conf = float(boxes.conf.max().item()) if count > 0 else 0.0
    return {"has_detection": count > 0, "count": count,
            "confidence": round(max_conf, 3)}


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════════════════

# ── Health / readiness check ──────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    loaded = {k: os.path.exists(v) for k, v in MODEL_PATHS.items()}
    loaded["nlp"] = os.path.exists(NLP_PATH)
    return jsonify({"status": "ok", "models_present": loaded})


# ── NLP text classification ──────────────────────────────────────────────────────
@app.route("/classify-text", methods=["POST"])
def classify_text():
    """
    POST { "text": "Large pothole near market..." }
    → { category, confidence, needs_review }
    """
    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    nlp = _get_nlp()
    if nlp is None:
        return jsonify({"category": "Other", "confidence": 0.0,
                        "needs_review": True,
                        "error": "NLP model not loaded"})

    probas   = nlp.predict_proba([text])[0]
    idx      = probas.argmax()
    category = nlp.classes_[idx]
    conf     = float(probas[idx])

    return jsonify({
        "category":     category,
        "confidence":   round(conf, 3),
        "needs_review": conf < 0.65
    })


# ── Single-category image detection ────────────────────────────────────────
@app.route("/detect", methods=["POST"])
def detect():
    """
    POST { "category": "Roads", "image_base64": "..." }
    → { has_detection, count, confidence }
    """
    data       = request.json or {}
    category   = data.get("category", "Roads")
    image_b64  = data.get("image_base64", "")

    model_name = CATEGORY_MODEL_MAP.get(category)

    if not image_b64:
        return jsonify({"error": "No image provided"}), 400
    if not model_name:
        return jsonify({"has_detection": False, "count": 0,
                        "confidence": 0.0,
                        "info": f"No vision model for '{category}'"})

    try:
        img    = _decode_image(image_b64)
        result = _run_yolo(model_name, img)

        # For Roads, also run the crack model and merge results
        if category == "Roads":
            crack = _run_yolo("crack", img)
            if crack.get("has_detection"):
                result["has_detection"]    = True
                result["crack_count"]      = crack["count"]
                result["crack_confidence"] = crack["confidence"]

        return jsonify(result)
    except Exception as e:
        log.exception("Detection error")
        return jsonify({"has_detection": False, "count": 0,
                        "confidence": 0.0, "error": str(e)}), 500


# ── Model-to-category display mapping ────────────────────────────────────────
MODEL_DISPLAY_MAP = {
    "pothole": "Roads",
    "garbage": "Garbage",
    "crack":   "Road Cracks",
}


# ── Full analysis (text + image) ─────────────────────────────────────────────
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main endpoint called by the frontend on issue submission.

    POST { "text": "...", "image_base64": "..." }
    → { text_result, image_result }
    """
    data    = request.json or {}
    text    = data.get("text", "").strip()
    img_b64 = data.get("image_base64", "")

    response = {"text_result": None, "image_result": None}

    # 1 — Text classification
    if text:
        nlp = _get_nlp()
        if nlp:
            probas   = nlp.predict_proba([text])[0]
            idx      = probas.argmax()
            category = nlp.classes_[idx]
            conf     = float(probas[idx])
            response["text_result"] = {
                "category":     category,
                "confidence":   round(conf, 3),
                "needs_review": conf < 0.65
            }

    # 2 — Image detection: run ALL models and pick best
    if img_b64:
        try:
            img = _decode_image(img_b64)
            best_model = None
            best_conf  = 0.0
            best_count = 0
            all_results = {}

            for name in MODEL_PATHS:
                res = _run_yolo(name, img)
                all_results[name] = res
                if res.get("has_detection") and res["confidence"] > best_conf:
                    best_conf  = res["confidence"]
                    best_model = name
                    best_count = res["count"]

            if best_model:
                response["image_result"] = {
                    "has_detection":    True,
                    "detected_category": MODEL_DISPLAY_MAP.get(best_model, "Other"),
                    "model":            best_model,
                    "count":            best_count,
                    "confidence":       round(best_conf, 3),
                    "all_detections":   all_results,
                }
            else:
                response["image_result"] = {
                    "has_detection":    False,
                    "detected_category": "Other",
                    "count":            0,
                    "confidence":       0.0,
                    "all_detections":   all_results,
                }
        except Exception as e:
            log.exception("Image analysis error")
            response["image_result"] = {"error": str(e)}

    return jsonify(response)


# ── Detect all models on one image ───────────────────────────────────────────
@app.route("/detect-all", methods=["POST"])
def detect_all():
    """
    Run all 3 YOLO models on a single image and return combined results
    plus the auto-detected category.
    POST { "image_base64": "..." }
    → { pothole: {...}, garbage: {...}, crack: {...}, detected_category: "..." }
    """
    data    = request.json or {}
    img_b64 = data.get("image_base64", "")
    if not img_b64:
        return jsonify({"error": "No image provided"}), 400

    try:
        img        = _decode_image(img_b64)
        results    = {}
        best_model = None
        best_conf  = 0.0

        for name in MODEL_PATHS:
            res = _run_yolo(name, img)
            results[name] = res
            if res.get("has_detection") and res["confidence"] > best_conf:
                best_conf  = res["confidence"]
                best_model = name

        results["detected_category"] = MODEL_DISPLAY_MAP.get(best_model, "Other")
        return jsonify(results)
    except Exception as e:
        log.exception("detect-all error")
        return jsonify({"error": str(e)}), 500


# ── Annotated image ──────────────────────────────────────────────────────────
@app.route("/annotate", methods=["POST"])
def annotate():
    """
    Returns the annotated image as base64 JPEG.
    POST { "image_base64": "...", "category": "Roads" }
    """
    data       = request.json or {}
    img_b64    = data.get("image_base64", "")
    category   = data.get("category", "Roads")
    model_name = CATEGORY_MODEL_MAP.get(category, "pothole")

    if not img_b64:
        return jsonify({"error": "No image provided"}), 400

    try:
        import cv2, numpy as np
        img_pil   = _decode_image(img_b64)
        frame     = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        yolo = _get_yolo(model_name)
        if yolo is None:
            return jsonify({"error": f"Model {model_name} not loaded"}), 404

        results   = yolo(frame, conf=0.25, verbose=False)[0]
        annotated = results.plot()

        _, buf  = cv2.imencode(".jpg", annotated)
        b64_out = base64.b64encode(buf.tobytes()).decode()
        return jsonify({"annotated_image": b64_out,
                        "count": len(results.boxes)})
    except Exception as e:
        log.exception("Annotation error")
        return jsonify({"error": str(e)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _get_nlp()  # pre-load NLP (small, fast)
    # YOLO models load lazily on first request to save memory
    port = int(os.environ.get("PORT", 5000))
    log.info(f"Starting CivicTrack ML service on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
