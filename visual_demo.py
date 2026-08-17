"""
AI Navigation for Visually Impaired — Visual Demo
Shows live camera feed with YOLOv8 detection boxes and speaks voice guidance.

Run: python visual_demo.py
Press Q to quit.

JSON output mode (used by ObjectDetectorService):
    python visual_demo.py --json-output
When --json-output is set, detection events are written as JSON lines to
stdout instead of calling speak().  speak() calls are suppressed entirely.
"""

import argparse
import json
import threading
import time
import subprocess

# ---------------------------------------------------------------------------
# Argument parsing — done before heavy imports so --help exits quickly
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="AI Navigation Visual Demo")
_parser.add_argument(
    "--json-output",
    action="store_true",
    default=False,
    help="Write detection events as JSON lines to stdout instead of speaking.",
)
_args, _unknown = _parser.parse_known_args()
JSON_OUTPUT: bool = _args.json_output

# Heavy imports deferred to after argument parsing
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "models/yolov8s.onnx"
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.25  # lower = catches more objects, higher = fewer false positives
FOCAL_LENGTH_PX = 600
SPEAK_COOLDOWN_SEC = 3.0  # minimum seconds between voice messages

# All COCO labels YOLOv8 can detect — no filtering, show everything
SUPPORTED_LABELS = None  # None = detect all classes

# Reference heights in metres for distance estimation
REF_HEIGHTS = {
    "person": 1.7, "bicycle": 1.0, "car": 1.5, "motorcycle": 1.1,
    "bus": 3.0, "truck": 2.5, "traffic_light": 0.8, "fire hydrant": 0.6,
    "stop sign": 0.9, "bench": 0.5, "cat": 0.3, "dog": 0.5,
    "chair": 0.9, "couch": 0.9, "dining table": 0.75, "toilet": 0.7,
    "tv": 0.6, "laptop": 0.3, "bottle": 0.25, "cup": 0.12,
    "book": 0.22, "clock": 0.3, "vase": 0.3, "scissors": 0.15,
    "backpack": 0.5, "umbrella": 1.0, "handbag": 0.3, "suitcase": 0.7,
    "sports ball": 0.22, "skateboard": 0.1, "bed": 0.6, "refrigerator": 1.8,
    "door": 2.0, "stairs": 0.3, "wall": 2.5,
}

# Colours per priority
COLOR_URGENT = (0, 0, 255)    # red
COLOR_NORMAL = (0, 165, 255)  # orange
COLOR_FAR    = (0, 255, 0)    # green

# ---------------------------------------------------------------------------
# TTS (Windows PowerShell SAPI)
# ---------------------------------------------------------------------------
last_spoken_time = 0.0
speak_lock = threading.Lock()

def speak(text: str):
    """Speak *text* via Windows SAPI.  No-op when --json-output is active."""
    if JSON_OUTPUT:
        # In JSON output mode all audio is handled by VoiceGuidanceEngine.
        return
    global last_spoken_time
    now = time.time()
    with speak_lock:
        if now - last_spoken_time < SPEAK_COOLDOWN_SEC:
            return
        last_spoken_time = now
    print(f"[VOICE] {text}")
    escaped = text.replace("'", "''")
    script = f"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 2; $s.Speak('{escaped}')"
    threading.Thread(
        target=lambda: subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True),
        daemon=True
    ).start()


# ---------------------------------------------------------------------------
# JSON output helpers (used when --json-output is active)
# ---------------------------------------------------------------------------
_json_output_lock = threading.Lock()

def _emit_json(event: dict) -> None:
    """Write *event* as a JSON line to stdout (thread-safe)."""
    line = json.dumps(event)
    with _json_output_lock:
        print(line, flush=True)


def _emit_obstacle(label: str, distance: float, direction: str, confidence: float) -> None:
    """Emit an obstacle detection event as a JSON line."""
    _emit_json({
        "type": "obstacle",
        "label": label,
        "distance": round(distance, 3),
        "direction": direction,
        "confidence": round(confidence, 4),
        "timestamp": time.time(),
    })


def _emit_clear() -> None:
    """Emit a 'path clear' event as a JSON line."""
    _emit_json({"type": "clear", "timestamp": time.time()})

# ---------------------------------------------------------------------------
# Distance estimation
# ---------------------------------------------------------------------------
def estimate_distance(label: str, bbox_height_px: float, frame_height: int) -> float:
    ref_h = REF_HEIGHTS.get(label, 1.5)
    if bbox_height_px <= 0:
        return 99.0
    dist = (FOCAL_LENGTH_PX * ref_h) / bbox_height_px
    return max(0.1, min(50.0, dist))

# ---------------------------------------------------------------------------
# Guidance message — gives actionable navigation instructions per object
# ---------------------------------------------------------------------------
def make_guidance(label: str, distance: float, cx: float, frame_width: int) -> str:
    third = frame_width / 3

    if cx < third:
        position = "to your left"
        avoid = "move right"
    elif cx > 2 * third:
        position = "to your right"
        avoid = "move left"
    else:
        position = "directly ahead"
        avoid = "stop or move sideways"

    if distance <= 0.5:
        return f"Stop! {label} {position}, {distance:.1f} metres. Do not move forward."
    elif distance <= 1.0:
        return f"Warning! {label} {position}, {distance:.1f} metres. {avoid}."
    elif distance <= 2.0:
        return f"{label} {position}, {distance:.1f} metres. Slow down and {avoid}."
    else:
        return f"{label} {position}, {distance:.1f} metres."


def make_multi_guidance(detections_info: list) -> str:
    """
    detections_info: list of (label, distance, cx, frame_width)
    Groups by label and describes each one with position + action.
    """
    if not detections_info:
        return ""

    # Group persons separately, number them
    persons = [(d, cx, fw) for (lbl, d, cx, fw) in detections_info if lbl == "person"]
    others  = [(lbl, d, cx, fw) for (lbl, d, cx, fw) in detections_info if lbl != "person"]

    parts = []

    # Describe persons individually with numbers
    if len(persons) == 1:
        d, cx, fw = persons[0]
        third = fw / 3
        pos = "left" if cx < third else ("right" if cx > 2*third else "ahead")
        avoid = "move right" if cx < third else ("move left" if cx > 2*third else "stop or move sideways")
        if d <= 1.0:
            parts.append(f"Warning! Person {pos} {d:.1f} metres. {avoid}.")
        else:
            parts.append(f"Person {pos} {d:.1f} metres.")
    elif len(persons) > 1:
        persons.sort(key=lambda x: x[0])  # sort by distance
        desc = []
        for i, (d, cx, fw) in enumerate(persons, 1):
            third = fw / 3
            pos = "left" if cx < third else ("right" if cx > 2*third else "ahead")
            desc.append(f"person {i} {pos} {d:.1f} metres")
        parts.append(f"{len(persons)} people detected: {', '.join(desc)}.")
        # Warn about closest
        d0, cx0, fw0 = persons[0]
        third = fw0 / 3
        avoid = "move right" if cx0 < third else ("move left" if cx0 > 2*third else "move sideways")
        if d0 <= 2.0:
            parts.append(f"Closest is {d0:.1f} metres ahead. {avoid}.")

    # Describe other obstacles
    for lbl, d, cx, fw in sorted(others, key=lambda x: x[1]):
        third = fw / 3
        pos = "left" if cx < third else ("right" if cx > 2*third else "ahead")
        if d <= 2.0:
            parts.append(f"{lbl} {pos} {d:.1f} metres.")

    return " ".join(parts)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("[AI Navigation] Loading YOLOv8n model...")
    model = YOLO(MODEL_PATH, task="detect")
    print("[AI Navigation] Model loaded.")

    print(f"[AI Navigation] Opening camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open camera. Check it is connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("[AI Navigation] Camera ready. Press Q to quit.\n")

    speak("Navigation assistance started. Point camera at your surroundings.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        h, w = frame.shape[:2]

        # Run YOLOv8 inference
        results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

        detections_info = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            if SUPPORTED_LABELS is not None and label not in SUPPORTED_LABELS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bbox_h = y2 - y1
            cx = (x1 + x2) / 2
            distance = estimate_distance(label, bbox_h, h)

            # Pick colour based on distance
            if distance <= 1.0:
                color = COLOR_URGENT
            elif distance <= 2.0:
                color = COLOR_NORMAL
            else:
                color = COLOR_FAR

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            tag = f"{label} {distance:.1f}m ({conf:.0%})"
            cv2.putText(frame, tag, (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if distance <= 3.0:
                detections_info.append((label, distance, cx, w))

        # Generate and speak multi-object guidance
        guidance = make_multi_guidance(detections_info)
        if guidance:
            speak(guidance)

        # In JSON output mode, emit per-detection obstacle events and a
        # 'clear' event when no obstacles are within 3 m.
        if JSON_OUTPUT:
            if detections_info:
                for label, distance, cx, fw in detections_info:
                    third = fw / 3
                    if cx < third:
                        direction = "left"
                    elif cx > 2 * third:
                        direction = "right"
                    else:
                        direction = "ahead"
                    # Retrieve confidence from the last matching box
                    conf_val = 0.0
                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        if model.names[cls_id] == label:
                            conf_val = float(box.conf[0])
                            break
                    _emit_obstacle(label, distance, direction, conf_val)
            else:
                _emit_clear()

        # HUD overlay
        if detections_info:
            closest = min(detections_info, key=lambda x: x[1])
            status = f"Closest: {closest[0]} {closest[1]:.1f}m"
        else:
            status = "Path clear"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "AI Navigation | Press Q to quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("AI Navigation for Visually Impaired", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    speak("Navigation assistance stopped.")
    print("[AI Navigation] Stopped.")

if __name__ == "__main__":
    main()
