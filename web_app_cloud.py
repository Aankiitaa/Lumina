import atexit
import os
import pickle
import random
import sqlite3
import string
import threading
import time
from collections import Counter, deque
from datetime import datetime, timedelta
from functools import wraps
import base64

import cv2
import mediapipe as mp
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from flask import Flask, Response, abort, jsonify, redirect, render_template, request, send_from_directory, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from flask_cors import CORS

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

def init_mediapipe_lazy():
    global mp_hands, mp_drawing, hands, mediapipe_ready, mediapipe_error
    if mediapipe_ready and hands is not None:
        return
    
    try:
        import mediapipe as mp
        print(f"DEBUG: MediaPipe lazy-init version: {getattr(mp, '__version__', 'unknown')}")
        try:
            from mediapipe.python.solutions import hands as mp_hands_module
            from mediapipe.python.solutions import drawing_utils as mp_drawing_module
            mp_hands = mp_hands_module
            mp_drawing = mp_drawing_module
        except ImportError:
            mp_hands = mp.solutions.hands
            mp_drawing = mp.solutions.drawing_utils
        
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        mediapipe_ready = True
        print("DEBUG: MediaPipe lazy-init successful.")
    except Exception as e:
        mediapipe_ready = False
        mediapipe_error = str(e)
        print("CRITICAL: MediaPipe lazy-init failed! Error:", mediapipe_error)
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
    init_mediapipe_lazy()
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
# Runtime state for Cloud (per-user)
# ----------------------------
class RuntimeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.sentence = ""
        self.prev_prediction = ""
        self.stable_start_time = time.time()
        self.last_commit_time = 0.0
        self.prediction_changed_since_commit = True
        self.prediction_history = deque(maxlen=12)
        self.prev_frame_time = time.time()
        self.most_common = "-"
        self.hold_progress = 0.0
        self.fps = 0
        self.stable_hold_seconds = 2.0

user_states = {}

def get_user_state(user_id):
    if user_id not in user_states:
        user_states[user_id] = RuntimeState()
    return user_states[user_id]


# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)
# Enable CORS for rendering from Vercel if needed
CORS(app)
app.secret_key = os.environ.get("LUMINA_SECRET_KEY", "change-this-in-production")
DB_PATH = os.environ.get("LUMINA_AUTH_DB", "auth.db")


def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_auth_db():
    conn = get_db_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_verified BOOLEAN DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS otp_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            otp_code TEXT NOT NULL,
            otp_expiry DATETIME NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_stats (
            user_id INTEGER PRIMARY KEY,
            xp INTEGER DEFAULT 0,
            streak INTEGER DEFAULT 0,
            last_active_date TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            lesson_key TEXT,
            step_index INTEGER,
            completed BOOLEAN DEFAULT 0,
            UNIQUE(user_id, lesson_key, step_index),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_achievements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            achievement_id TEXT,
            awarded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        """
    )
    try:
        conn.execute("ALTER TABLE users ADD COLUMN email TEXT UNIQUE")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE users ADD COLUMN is_verified BOOLEAN DEFAULT 0")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

init_auth_db()

def generate_otp():
    return "".join(random.choices(string.digits, k=6))

def send_otp_email(email, otp):
    sender_email = os.environ.get("LUMINA_EMAIL")
    sender_password = os.environ.get("LUMINA_EMAIL_PASS")

    if not sender_email or not sender_password:
        return True

    html_content = f"""
    <html>
    <body style="font-family: 'Inter', sans-serif; background-color: #121212; color: #ffffff; padding: 40px; text-align: center;">
        <div style="max-width: 500px; margin: 0 auto; background-color: #1e1e1e; border-radius: 24px; padding: 40px; border: 1px solid #333;">
            <h1 style="color: #f0e54f; font-size: 32px; letter-spacing: 0.1em; margin-bottom: 8px;">LUMINA</h1>
            <p style="color: #9a9a9a; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 32px;">Verification Code</p>
            <p style="color: #ffffff; font-size: 16px; line-height: 1.5; margin-bottom: 24px;">
                Welcome to Lumina! Use the code below to verify your account and start your journey.
            </p>
            <div style="background-color: #2a2a2a; color: #f0e54f; font-size: 40px; font-weight: 700; padding: 20px; border-radius: 16px; letter-spacing: 8px; margin-bottom: 24px;">
                {otp}
            </div>
            <p style="color: #9a9a9a; font-size: 13px;">
                This code expires in 10 minutes. If you didn't request this, please ignore this email.
            </p>
        </div>
    </body>
    </html>
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Your Lumina Verification Code"
    msg["From"] = f"Lumina <{sender_email}>"
    msg["To"] = email
    msg.attach(MIMEText(html_content, "html"))

    try:
        # Add a 10-second timeout to prevent Gunicorn workers from hanging on Render
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception as e:
        print(f"Failed to send email to {email}. Error: {e}")
        print(f"--- FALLBACK OTP for {email} is: {otp} ---")
        return False

def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            if request.path.startswith("/action/") or request.path.startswith("/api/"):
                return jsonify({"ok": False, "message": "Unauthorized"}), 401
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapper

@app.route("/")
@login_required
def dashboard():
    return render_template("dashboard.html", username=session.get("username", ""))

@app.route("/translator")
@login_required
def translator():
    return render_template("index_cloud.html", mediapipe_ready=mediapipe_ready, mediapipe_error=mediapipe_error)

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"): return redirect(url_for("dashboard"))
    error = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            error = "Username and password are required"
        else:
            conn = get_db_conn()
            user = conn.execute("SELECT id, username, password_hash, is_verified FROM users WHERE username = ?", (username,)).fetchone()
            conn.close()
            if user and check_password_hash(user["password_hash"], password):
                if not user["is_verified"]:
                    session["temp_user_id"] = user["id"]
                    return redirect(url_for("verify_otp"))
                session["user_id"] = user["id"]
                session["username"] = user["username"]
                return redirect(url_for("dashboard"))
            error = "Invalid username or password"
    return render_template("login.html", error=error)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("user_id"): return redirect(url_for("dashboard"))
    error = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password or not confirm_password: error = "All fields are required"
        elif len(username) < 3: error = "Username must be at least 3 characters"
        elif "@" not in email or "." not in email: error = "Invalid email address"
        elif len(password) < 6: error = "Password must be at least 6 characters"
        elif password != confirm_password: error = "Passwords do not match"
        else:
            conn = get_db_conn()
            existing = conn.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email)).fetchone()
            if existing: error = "Username or email already exists"
            else:
                user_cursor = conn.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)", (username, email, generate_password_hash(password)))
                user_id = user_cursor.lastrowid
                otp = generate_otp()
                expiry = (datetime.now() + timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
                conn.execute("INSERT INTO otp_codes (user_id, otp_code, otp_expiry) VALUES (?, ?, ?)", (user_id, otp, expiry))
                conn.commit()
                conn.close()
                send_otp_email(email, otp)
                session["temp_user_id"] = user_id
                return redirect(url_for("verify_otp"))
            conn.close()
    return render_template("signup.html", error=error)

@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    temp_user_id = session.get("temp_user_id")
    if not temp_user_id: return redirect(url_for("signup"))
    error = ""
    if request.method == "POST":
        otp_input = request.form.get("otp", "").strip()
        conn = get_db_conn()
        otp_record = conn.execute("SELECT otp_code, otp_expiry FROM otp_codes WHERE user_id = ? ORDER BY id DESC LIMIT 1", (temp_user_id,)).fetchone()
        if otp_record:
            if otp_input == otp_record["otp_code"]:
                expiry = datetime.strptime(otp_record["otp_expiry"], "%Y-%m-%d %H:%M:%S")
                if datetime.now() < expiry:
                    conn.execute("UPDATE users SET is_verified = 1 WHERE id = ?", (temp_user_id,))
                    user = conn.execute("SELECT username FROM users WHERE id = ?", (temp_user_id,)).fetchone()
                    conn.commit()
                    conn.close()
                    session.pop("temp_user_id", None)
                    session["user_id"] = temp_user_id
                    session["username"] = user["username"]
                    return redirect(url_for("dashboard"))
                else: error = "OTP has expired. Please sign up again."
            else: error = "Invalid OTP code."
        else: error = "No OTP found. Please sign up again."
        conn.close()
    return render_template("otp_verify.html", error=error)

@app.route("/academy")
@login_required
def academy():
    return render_template("academy_cloud.html", username=session.get("username", ""))

@app.route("/sign_image/<label>")
@login_required
def sign_image(label):
    label = os.path.basename(label).strip()
    signs_dir = os.path.join(os.path.dirname(__file__), "static", "signs")
    for var in [label, label.lower(), label.upper()]:
        for ext in ("png", "jpg", "jpeg", "webp"):
            filename = f"{var}.{ext}"
            filepath = os.path.join(signs_dir, filename)
            if os.path.exists(filepath):
                return send_from_directory(signs_dir, filename)
    abort(404)

@app.route("/api/gamification/stats")
@login_required
def gamification_stats():
    user_id = session["user_id"]
    conn = get_db_conn()
    stats = conn.execute("SELECT xp, streak, last_active_date FROM user_stats WHERE user_id = ?", (user_id,)).fetchone()
    if not stats:
        conn.execute("INSERT INTO user_stats (user_id, xp, streak) VALUES (?, 0, 0)", (user_id,))
        conn.commit()
        stats = {"xp": 0, "streak": 0, "last_active_date": None}
    order = ["alphabet", "numbers", "greetings", "family", "questions"]
    progress_rows = conn.execute("SELECT lesson_key, step_index, completed FROM user_progress WHERE user_id = ?", (user_id,)).fetchall()
    conn.close()
    completed_lessons = {row["lesson_key"] for row in progress_rows if row["step_index"] == 2 and row["completed"]}
    progress_map = {}
    for lesson in order:
        steps_done = sum(1 for row in progress_rows if row["lesson_key"] == lesson and row["completed"])
        progress_map[lesson] = int((steps_done / 3) * 100)
    unlocked = {order[0]}
    for i in range(1, len(order)):
        if order[i-1] in completed_lessons:
            unlocked.add(order[i])
    return jsonify({
        "xp": stats["xp"], "streak": stats["streak"], "progress": progress_map, "unlocked": list(unlocked)
    })

@app.post("/api/gamification/update_progress")
@login_required
def update_progress():
    user_id = session["user_id"]
    data = request.json
    lesson_key = data.get("lesson_key")
    step_index = data.get("step_index")
    xp_award = data.get("xp", 0)
    if lesson_key is None or step_index is None: return jsonify({"ok": False}), 400
    conn = get_db_conn()
    conn.execute("INSERT INTO user_progress (user_id, lesson_key, step_index, completed) VALUES (?, ?, ?, 1) ON CONFLICT(user_id, lesson_key, step_index) DO UPDATE SET completed=1", (user_id, lesson_key, step_index))
    conn.execute("UPDATE user_stats SET xp = xp + ? WHERE user_id = ?", (xp_award, user_id))
    from datetime import date
    today = date.today().isoformat()
    stats = conn.execute("SELECT last_active_date, streak FROM user_stats WHERE user_id = ?", (user_id,)).fetchone()
    new_streak = stats["streak"]
    if stats["last_active_date"] != today:
        if stats["last_active_date"]:
            from datetime import datetime, timedelta
            last_date = datetime.fromisoformat(stats["last_active_date"]).date()
            if last_date == date.today() - timedelta(days=1): new_streak += 1
            elif last_date < date.today() - timedelta(days=1): new_streak = 1
        else: new_streak = 1
        conn.execute("UPDATE user_stats SET last_active_date = ?, streak = ? WHERE user_id = ?", (today, new_streak, user_id))
    conn.commit()
    conn.close()
    return jsonify({"ok": True, "new_xp": xp_award, "streak": new_streak})

@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ----------------------------
# Cloud Inference Endpoint
# ----------------------------
@app.post("/api/process_frame")
@login_required
def process_frame():
    user_id = session["user_id"]
    state = get_user_state(user_id)
    
    data = request.json.get("image")
    if not data:
        return jsonify({"ok": False, "message": "No image provided"}), 400
        
    try:
        header, encoded = data.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Calculate roughly FPS (request frequency)
        now = time.time()
        fps = 1.0 / max(1e-6, (now - state.prev_frame_time))
        state.prev_frame_time = now
        state.fps = int(fps)

        # Mirror the frame laterally to match the model's training orientation
        frame = cv2.flip(frame, 1)

        infer_frame = cv2.resize(frame, (320, 210))
        landmarks_raw, hand_landmarks = extract_landmarks(infer_frame)
        
        if landmarks_raw is not None:
            normalized = normalize_landmarks(landmarks_raw)
            pred = model.predict([normalized])[0]
            state.prediction_history.append(pred)

            if len(state.prediction_history) >= 5:
                state.most_common = Counter(state.prediction_history).most_common(1)[0][0]
                
                if state.most_common != state.prev_prediction:
                    state.prev_prediction = state.most_common
                    state.stable_start_time = now
                    state.prediction_changed_since_commit = True
                    state.hold_progress = 0.0
                else:
                    elapsed = now - state.stable_start_time
                    state.hold_progress = min(1.0, elapsed / state.stable_hold_seconds)
                    noise_floor_ok = (now - state.last_commit_time) >= 0.4

                    if state.hold_progress >= 1.0 and noise_floor_ok and state.prediction_changed_since_commit:
                        if str(state.most_common).lower() == "space":
                            state.sentence += " "
                        elif state.sentence == "" or state.sentence[-1] != state.most_common:
                            state.sentence += str(state.most_common).upper()
                        state.last_commit_time = now
                        state.stable_start_time = now
                        state.hold_progress = 0.0
                        state.prediction_changed_since_commit = False
        else:
            state.hold_progress = 0.0

        return jsonify({
            "ok": True,
            "prediction": str(state.most_common).upper(),
            "sentence": state.sentence,
            "fps": state.fps,
            "hold_progress": round(state.hold_progress, 3),
            "mp_error": mediapipe_error
        })
    except Exception as e:
        return jsonify({"ok": False, "message": str(e)}), 500


@app.post("/action/speak")
@login_required
def speak():
    # Speak might not work natively on Render (requires espeak/tts engine), but keep route structural parity
    return jsonify({"ok": True, "message": "Triggered speak"})

@app.post("/action/clear")
@login_required
def clear():
    user_id = session["user_id"]
    state = get_user_state(user_id)
    state.sentence = ""
    state.prediction_history.clear()
    state.prev_prediction = ""
    state.stable_start_time = time.time()
    return jsonify({"ok": True})

@app.post("/action/delete")
@login_required
def delete_last():
    user_id = session["user_id"]
    state = get_user_state(user_id)
    if state.sentence:
        state.sentence = state.sentence[:-1]
    return jsonify({"ok": True})

if __name__ == "__main__":
    init_auth_db()
    app.run(host="127.0.0.1", port=5002, debug=False)
