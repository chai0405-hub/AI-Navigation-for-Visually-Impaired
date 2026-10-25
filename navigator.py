"""
Core logic for AI Navigation for Visually Impaired.

Takes an image, detects objects with YOLOv8, works out where each object is
(left / ahead / right) and roughly how close it is, then produces short
spoken-style navigation guidance in English, Hindi or Kannada.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image

# Objects that matter most for safe walking get priority in the spoken output.
HAZARD_CLASSES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "train",
    "traffic light", "fire hydrant", "stop sign", "bench", "dog", "cow",
    "horse", "chair", "couch", "potted plant", "dining table", "bed",
    "suitcase", "parking meter", "toilet", "refrigerator", "tv",
}

# ---------------------------------------------------------------------------
# Translations
# ---------------------------------------------------------------------------
OBJECT_NAMES = {
    "hi": {
        "person": "व्यक्ति", "bicycle": "साइकिल", "car": "कार",
        "motorcycle": "मोटरसाइकिल", "bus": "बस", "truck": "ट्रक",
        "train": "ट्रेन", "traffic light": "ट्रैफिक लाइट",
        "stop sign": "स्टॉप साइन", "bench": "बेंच", "dog": "कुत्ता",
        "cow": "गाय", "horse": "घोड़ा", "chair": "कुर्सी",
        "couch": "सोफा", "dining table": "मेज़", "bed": "बिस्तर",
        "potted plant": "गमला", "door": "दरवाज़ा", "bottle": "बोतल",
        "cup": "कप", "laptop": "लैपटॉप", "cell phone": "मोबाइल",
        "backpack": "बैग", "handbag": "हैंडबैग", "suitcase": "सूटकेस",
        "tv": "टीवी", "book": "किताब", "clock": "घड़ी",
    },
    "kn": {
        "person": "ವ್ಯಕ್ತಿ", "bicycle": "ಸೈಕಲ್", "car": "ಕಾರು",
        "motorcycle": "ಬೈಕ್", "bus": "ಬಸ್", "truck": "ಲಾರಿ",
        "train": "ರೈಲು", "traffic light": "ಟ್ರಾಫಿಕ್ ಲೈಟ್",
        "stop sign": "ಸ್ಟಾಪ್ ಫಲಕ", "bench": "ಬೆಂಚ್", "dog": "ನಾಯಿ",
        "cow": "ಹಸು", "horse": "ಕುದುರೆ", "chair": "ಕುರ್ಚಿ",
        "couch": "ಸೋಫಾ", "dining table": "ಮೇಜು", "bed": "ಹಾಸಿಗೆ",
        "potted plant": "ಹೂಕುಂಡ", "bottle": "ಬಾಟಲಿ", "cup": "ಕಪ್",
        "laptop": "ಲ್ಯಾಪ್‌ಟಾಪ್", "cell phone": "ಮೊಬೈಲ್",
        "backpack": "ಬ್ಯಾಗ್", "handbag": "ಕೈಚೀಲ", "suitcase": "ಸೂಟ್‌ಕೇಸ್",
        "tv": "ಟಿವಿ", "book": "ಪುಸ್ತಕ", "clock": "ಗಡಿಯಾರ",
    },
}

PHRASES = {
    "en": {
        "left": "on your left", "center": "ahead", "right": "on your right",
        "very_close": "very close", "close": "close", "medium": "a few steps away", "far": "far",
        "item": "{name} {pos}, {dist}",
        "stop": "Stop. Obstacle directly ahead.",
        "go_left": "Path ahead is blocked. Move slightly left.",
        "go_right": "Path ahead is blocked. Move slightly right.",
        "careful": "Obstacle ahead. Walk slowly.",
        "clear": "Path ahead looks clear.",
        "nothing": "No objects detected. Path looks clear.",
        "seen": "I can see",
    },
    "hi": {
        "left": "आपके बाईं ओर", "center": "सामने", "right": "आपके दाईं ओर",
        "very_close": "बहुत पास", "close": "पास", "medium": "कुछ कदम दूर", "far": "दूर",
        "item": "{name} {pos}, {dist}",
        "stop": "रुकिए। ठीक सामने रुकावट है।",
        "go_left": "सामने रास्ता बंद है। थोड़ा बाईं ओर चलिए।",
        "go_right": "सामने रास्ता बंद है। थोड़ा दाईं ओर चलिए।",
        "careful": "सामने रुकावट है। धीरे चलिए।",
        "clear": "सामने का रास्ता साफ़ है।",
        "nothing": "कोई वस्तु नहीं दिखी। रास्ता साफ़ है।",
        "seen": "मुझे दिख रहा है",
    },
    "kn": {
        "left": "ನಿಮ್ಮ ಎಡಕ್ಕೆ", "center": "ಮುಂದೆ", "right": "ನಿಮ್ಮ ಬಲಕ್ಕೆ",
        "very_close": "ತುಂಬಾ ಹತ್ತಿರ", "close": "ಹತ್ತಿರ", "medium": "ಕೆಲವು ಹೆಜ್ಜೆ ದೂರ", "far": "ದೂರ",
        "item": "{name} {pos}, {dist}",
        "stop": "ನಿಲ್ಲಿ. ನೇರವಾಗಿ ಮುಂದೆ ಅಡಚಣೆ ಇದೆ.",
        "go_left": "ಮುಂದಿನ ದಾರಿ ಮುಚ್ಚಿದೆ. ಸ್ವಲ್ಪ ಎಡಕ್ಕೆ ಸರಿಯಿರಿ.",
        "go_right": "ಮುಂದಿನ ದಾರಿ ಮುಚ್ಚಿದೆ. ಸ್ವಲ್ಪ ಬಲಕ್ಕೆ ಸರಿಯಿರಿ.",
        "careful": "ಮುಂದೆ ಅಡಚಣೆ ಇದೆ. ನಿಧಾನವಾಗಿ ನಡೆಯಿರಿ.",
        "clear": "ಮುಂದಿನ ದಾರಿ ಸ್ಪಷ್ಟವಾಗಿದೆ.",
        "nothing": "ಯಾವುದೇ ವಸ್ತು ಕಾಣುತ್ತಿಲ್ಲ. ದಾರಿ ಸ್ಪಷ್ಟವಾಗಿದೆ.",
        "seen": "ನನಗೆ ಕಾಣುತ್ತಿದೆ",
    },
}

DIST_RANK = {"very_close": 0, "close": 1, "medium": 2, "far": 3}
APPROX_METERS = {"very_close": "< 1 m", "close": "1–2 m", "medium": "2–4 m", "far": "> 4 m"}


@dataclass
class Detection:
    name: str
    confidence: float
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    position: str  # left / center / right
    distance: str  # very_close / close / medium / far
    in_path: bool


def load_model(weights: str = "yolov8n.pt"):
    """Load YOLOv8. The small 'n' model is downloaded automatically on first run."""
    from ultralytics import YOLO
    return YOLO(weights)


def _position(cx: float, width: int) -> str:
    rel = cx / width
    if rel < 0.35:
        return "left"
    if rel > 0.65:
        return "right"
    return "center"


def _distance(box_h: float, box_w: float, img_h: int, img_w: int) -> str:
    """Rough proximity from how much of the frame the object fills."""
    h_frac = box_h / img_h
    area_frac = (box_h * box_w) / (img_h * img_w)
    if h_frac > 0.7 or area_frac > 0.35:
        return "very_close"
    if h_frac > 0.4 or area_frac > 0.12:
        return "close"
    if h_frac > 0.18 or area_frac > 0.03:
        return "medium"
    return "far"


def detect(model, image: Image.Image, conf: float = 0.4) -> Tuple[List[Detection], np.ndarray]:
    """Run detection. Returns detections and an annotated RGB image."""
    img = image.convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    result = model.predict(arr, conf=conf, verbose=False)[0]

    detections: List[Detection] = []
    for b in result.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        name = result.names[int(b.cls[0])]
        cx = (x1 + x2) / 2
        pos = _position(cx, w)
        dist = _distance(y2 - y1, x2 - x1, h, w)
        # In the walking path if the box overlaps the middle 30% of the frame.
        in_path = x2 > 0.35 * w and x1 < 0.65 * w
        detections.append(Detection(name, float(b.conf[0]), (x1, y1, x2, y2), pos, dist, in_path))

    # Closest and most hazardous first.
    detections.sort(key=lambda d: (DIST_RANK[d.distance], d.name not in HAZARD_CLASSES, -d.confidence))
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB
    return detections, annotated


def _side_load(dets: List[Detection], side: str) -> int:
    """How crowded a side is: closer objects count more."""
    weight = {"very_close": 4, "close": 3, "medium": 2, "far": 1}
    return sum(weight[d.distance] for d in dets if d.position == side)


def build_guidance(dets: List[Detection], lang: str = "en", max_items: int = 4) -> Tuple[str, str]:
    """Return (spoken_text, decision_key)."""
    p = PHRASES[lang]
    names = OBJECT_NAMES.get(lang, {})

    if not dets:
        return p["nothing"], "clear"

    blocking = [d for d in dets if d.in_path and d.distance in ("very_close", "close")]
    caution = [d for d in dets if d.in_path and d.distance == "medium"]

    if blocking:
        left, right = _side_load(dets, "left"), _side_load(dets, "right")
        very_close = any(d.distance == "very_close" for d in blocking)
        if very_close and left > 0 and right > 0:
            decision = "stop"          # blocked ahead and both sides busy
        else:
            decision = "go_left" if left <= right else "go_right"
    elif caution:
        decision = "careful"
    else:
        decision = "clear"

    items = []
    seen = set()
    for d in dets:
        key = (d.name, d.position)
        if key in seen:
            continue
        seen.add(key)
        items.append(p["item"].format(
            name=names.get(d.name, d.name), pos=p[d.position], dist=p[d.distance]))
        if len(items) >= max_items:
            break

    text = f"{p[decision]} {p['seen']}: " + "; ".join(items) + "."
    return text, decision


def speak(text: str, lang: str = "en") -> bytes:
    """Convert text to MP3 bytes with Google Text-to-Speech."""
    from io import BytesIO
    from gtts import gTTS
    buf = BytesIO()
    gTTS(text=text, lang=lang, slow=False).write_to_fp(buf)
    return buf.getvalue()
