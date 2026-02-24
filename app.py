import cv2
import time
import pickle
import threading
import numpy as np
import mediapipe as mp
import pyttsx3
from collections import deque, Counter
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk  # pip install pillow
import os
from tkinter import messagebox
# ----------------------------
# Load model
# ----------------------------
try:
    model_dict = pickle.load(open('model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    raise RuntimeError("Could not load model.p. Ensure train_model.py has saved it.") from e

# ----------------------------
# TTS setup
# ----------------------------
# tts = pyttsx3.init()
# tts.setProperty('rate', 120)

# ----------------------------
# MediaPipe Hands
# ----------------------------
mp_hands = None
mp_drawing = None
hands = None
mediapipe_ready = False
mediapipe_error = ""

try:
    # API used by this app.
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
    """
    Input: landmarks as flat (63,) or shaped (21,3)
    Output: normalized flat (63,)
    """
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
# GUI App
# ----------------------------
class SignApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ASL Sentence Builder")
        self.root.geometry("1040x620")
        self.root.configure(bg="#f6f1eb")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Soft pastel background
        bg_canvas = tk.Canvas(self.root, highlightthickness=0, bd=0)
        bg_canvas.grid(row=0, column=0, sticky="nsew")
        bg_canvas.create_rectangle(0, 0, 2000, 2000, fill="#f6f1eb", outline="")
        bg_canvas.create_oval(-150, -150, 550, 420, fill="#f9d6c8", outline="")
        bg_canvas.create_oval(280, -200, 980, 360, fill="#efd7f9", outline="")
        bg_canvas.create_oval(620, 180, 1240, 780, fill="#f7e2cb", outline="")

        main_frame = tk.Frame(self.root, bg="#f6f1eb")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=28, pady=28)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=2)

        # Left card: video
        video_card = tk.Frame(main_frame, bg="#fbf8f5", bd=0, highlightthickness=1, highlightbackground="#eadfda")
        video_card.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        video_card.grid_rowconfigure(0, weight=1)
        video_card.grid_columnconfigure(0, weight=1)

        self.video_label = tk.Label(video_card, bg="#f1ebe6")
        self.video_label.grid(row=0, column=0, padx=14, pady=14, sticky="nsew")

        # Right card: info + buttons
        right_frame = tk.Frame(main_frame, bg="#fbf8f5", bd=0, highlightthickness=1, highlightbackground="#eadfda")
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.grid_columnconfigure(0, weight=1)

        # Prediction / sentence / fps
        self.pred_var = tk.StringVar(value="Prediction: —")
        self.sent_var = tk.StringVar(value="Sentence: ")
        self.fps_var  = tk.StringVar(value="FPS: —")

        tk.Label(
            right_frame,
            textvariable=self.pred_var,
            font=("Avenir Next", 24, "bold"),
            fg="#2f2a2a",
            bg="#fbf8f5"
        ).grid(row=0, column=0, padx=24, pady=(28, 8), sticky="w")

        tk.Label(
            right_frame,
            textvariable=self.sent_var,
            font=("Avenir Next", 16),
            fg="#4a4343",
            bg="#fbf8f5",
            wraplength=320,
            justify="left"
        ).grid(row=1, column=0, padx=24, pady=8, sticky="w")

        tk.Label(
            right_frame,
            textvariable=self.fps_var,
            font=("Avenir Next", 13),
            fg="#8c7f7a",
            bg="#fbf8f5"
        ).grid(row=2, column=0, padx=24, pady=(2, 24), sticky="w")

        # Buttons row
        btns = tk.Frame(right_frame, bg="#fbf8f5")
        btns.grid(row=3, column=0, padx=24, pady=(0, 24), sticky="w")

        btn_style = {
            "font": ("Avenir Next", 13),
            "fg": "#2f2a2a",
            "bg": "#f0e5de",
            "activeforeground": "#2f2a2a",
            "activebackground": "#e9d8ce",
            "relief": "flat",
            "bd": 0,
            "cursor": "hand2",
            "padx": 16,
            "pady": 8
        }

        tk.Button(btns, text="Speak (↵)", command=self.speak_sentence, **btn_style).pack(side="left", padx=(0, 8))
        tk.Button(btns, text="Clear", command=self.clear_sentence, **btn_style).pack(side="left", padx=8)
        tk.Button(btns, text="Save to File", command=self.save_sentence, **btn_style).pack(side="left", padx=(8, 0))

        # Webcam feed
        self.video_width = 620
        self.video_height = 520



        # State
        self.cap = None
        self.running = False
        self.sentence = ""
        self.prev_prediction = ""
        self.stable_start_time = time.time()
        self.prediction_history = deque(maxlen=20)
        self.prev_frame_time = time.time()
        self.most_common = "—"

        # Bind Enter for Speak
        self.root.bind("<Return>", lambda e: self.speak_sentence())

        if not mediapipe_ready:
            messagebox.showwarning(
                "MediaPipe Unavailable",
                "Hand tracking is disabled because MediaPipe solutions API is unavailable.\n"
                f"Details: {mediapipe_error}\n\n"
                "The camera window will still open."
            )

        # Start camera in a worker thread so UI stays responsive
        self.start_camera()

        # Safe close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------- Controls -------------
    

    def clear_sentence(self):
        self.sentence = ""
        self.prediction_history.clear()
        self.prev_prediction = ""
        self.stable_start_time = time.time()
        self.sent_var.set("Sentence: ")

    #TTS
    def speak_sentence(self):
        text = self.sentence.strip()
        if not text:
          messagebox.showinfo("Speak", "Sentence is empty.")
          return
        os.system(f'say "{text}"')

    def save_sentence(self):
        text = self.sentence
        if not text.strip():
            messagebox.showinfo("Save", "Sentence is empty.")
            return
        # mirror your run2.py behavior
        with open("output_sentence.txt", "w") as f:
            f.write(text)
        messagebox.showinfo("Saved", "Saved to output_sentence.txt")

    # ------------- Camera / Loop -------------
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Unable to open webcam.")
            return
        self.running = True
        threading.Thread(target=self.update_loop, daemon=True).start()

    def update_loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.video_width, self.video_height))
            # Get landmarks + prediction
            landmarks_raw, hand_landmarks = extract_landmarks(frame)
            if landmarks_raw is not None:
                normalized = normalize_landmarks(landmarks_raw)
                pred = model.predict([normalized])[0]
                self.prediction_history.append(pred)
                self.most_common = Counter(self.prediction_history).most_common(1)[0][0]

                # stable for ~2.5s
                if self.most_common == self.prev_prediction:
                    if time.time() - self.stable_start_time >= 2.5:
                        if self.most_common.lower() == "space":
                            self.sentence += " "
                            self.sent_var.set(f"Sentence: {self.sentence}")


                        else:
                            if self.sentence == "" or self.sentence[-1] != self.most_common:
                                self.sentence += self.most_common.upper()
                                self.sent_var.set(f"Sentence: {self.sentence}")
                        self.stable_start_time = time.time()
                else:
                    self.prev_prediction = self.most_common
                    self.stable_start_time = time.time()

                # draw landmarks
                if mp_drawing is not None and mp_hands is not None:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # compute FPS
            now = time.time()
            fps = 1.0 / max(1e-6, (now - self.prev_frame_time))
            self.prev_frame_time = now

            # push to Tkinter label
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk  # keep reference
            self.video_label.configure(image=imgtk)

            # update small labels
            self.pred_var.set(f"Prediction: {str(self.most_common).upper()}")
            self.fps_var.set(f"FPS: {int(fps)}")

        # loop ends: release cam
        if self.cap:
            self.cap.release()

    # ------------- Close -------------
    def on_close(self):
        self.running = False
        # tiny wait so the thread can exit gracefully
        self.root.after(200, self.root.destroy)

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = SignApp(root)
    root.mainloop()









 
