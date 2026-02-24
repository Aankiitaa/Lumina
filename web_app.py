import atexit
import os
import pickle
import threading
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

# ----------------------------
# Load model
# ----------------------------
try:
    model_dict = pickle.load(open("model.p", "rb"))
    model = model_dict["model"]
except Exception as e:
    raise RuntimeError("Could not load model.p. Ensure train_model.py has saved it.") from e

# ----------------------------
# MediaPipe Hands
# ----------------------------
mp_hands = None
mp_drawing = None
hands = None
mediapipe_ready = False
mediapipe_error = ""

try:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    mediapipe_ready = True
except Exception as e:
    mediapipe_error = str(e)


# ----------------------------
# Utils
# ----------------------------
def normalize_landmarks(landmarks):
    arr = np.array(landmarks).reshape(-1, 3)
    base = arr[0]
    arr -= base
    max_value = np.max(np.abs(arr))
    if max_value != 0:
        arr /= max_value
    return arr.flatten()


def extract_landmarks(frame_bgr):
    if not mediapipe_ready or hands is None:
        return None, None
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks:
        hl = results.multi_hand_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark])
        return coords, hl
    return None, None


# ----------------------------
# Runtime state
# ----------------------------
class RuntimeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.cap = None
        self.running = False
        self.frame_jpeg = None

        self.video_width = 640
        self.video_height = 420

        self.sentence = ""
        self.prev_prediction = ""
        self.stable_start_time = time.time()
        self.prediction_history = deque(maxlen=20)
        self.prev_frame_time = time.time()
        self.most_common = "-"
        self.fps = 0
        self.frame_index = 0


state = RuntimeState()


# ----------------------------
# Camera thread
# ----------------------------
def camera_loop():
    while state.running:
        ok, frame = state.cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (state.video_width, state.video_height))
        hand_landmarks = None
        state.frame_index += 1

        # Run heavy model inference every other frame to reduce lag.
        if state.frame_index % 2 == 0:
            landmarks_raw, hand_landmarks = extract_landmarks(frame)
            if landmarks_raw is not None:
                normalized = normalize_landmarks(landmarks_raw)
                pred = model.predict([normalized])[0]
                state.prediction_history.append(pred)
                state.most_common = Counter(state.prediction_history).most_common(1)[0][0]

                if state.most_common == state.prev_prediction:
                    if time.time() - state.stable_start_time >= 2.5:
                        if str(state.most_common).lower() == "space":
                            state.sentence += " "
                        else:
                            if state.sentence == "" or state.sentence[-1] != state.most_common:
                                state.sentence += str(state.most_common).upper()
                        state.stable_start_time = time.time()
                else:
                    state.prev_prediction = state.most_common
                    state.stable_start_time = time.time()

                if mp_drawing is not None and mp_hands is not None:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        now = time.time()
        fps = 1.0 / max(1e-6, (now - state.prev_frame_time))
        state.prev_frame_time = now
        state.fps = int(fps)

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if ok:
            with state.lock:
                state.frame_jpeg = buf.tobytes()

    if state.cap is not None:
        state.cap.release()


def start_camera():
    state.cap = cv2.VideoCapture(0)
    if not state.cap.isOpened():
        raise RuntimeError("Unable to open webcam.")
    state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, state.video_width)
    state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, state.video_height)
    state.cap.set(cv2.CAP_PROP_FPS, 30)
    state.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    state.running = True
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()


def stop_camera():
    state.running = False
    time.sleep(0.2)


# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", mediapipe_ready=mediapipe_ready, mediapipe_error=mediapipe_error)


@app.route("/video_feed")
def video_feed():
    def gen():
        while state.running:
            frame = None
            with state.lock:
                frame = state.frame_jpeg
            if frame is None:
                time.sleep(0.01)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify(
        {
            "prediction": str(state.most_common).upper(),
            "sentence": state.sentence,
            "fps": state.fps,
        }
    )


@app.post("/action/speak")
def speak():
    text = state.sentence.strip()
    if not text:
        return jsonify({"ok": False, "message": "Sentence is empty."}), 400
    os.system(f'say "{text}"')
    return jsonify({"ok": True})


@app.post("/action/clear")
def clear():
    state.sentence = ""
    state.prediction_history.clear()
    state.prev_prediction = ""
    state.stable_start_time = time.time()
    return jsonify({"ok": True})


@app.post("/action/save")
def save():
    text = state.sentence
    if not text.strip():
        return jsonify({"ok": False, "message": "Sentence is empty."}), 400
    with open("output_sentence.txt", "w") as f:
        f.write(text)
    return jsonify({"ok": True})


atexit.register(stop_camera)


if __name__ == "__main__":
    start_camera()
    app.run(host="127.0.0.1", port=5001, debug=False)
