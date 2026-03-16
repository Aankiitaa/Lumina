import cv2
import mediapipe as mp
import numpy as np
import os
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base = landmarks[0]  # use wrist as origin
    landmarks -= base
    max_value = np.max(np.abs(landmarks))
    if max_value != 0:
        landmarks /= max_value  # scale to range [-1, 1]
    return landmarks.flatten()

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Create data directory
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Ask user for which letter to record
current_label = input("Enter the label for this gesture (e.g., A, B, space, clear): ").strip().lower()
if not current_label.isalnum():
    print("Invalid label. Please enter an alphanumeric name.")
    exit()


# Set number of samples to collect
samples_per_label = int(input("How many samples do you want to collect? (e.g., 200): "))
collected = 0

# Start webcam
cap = cv2.VideoCapture(0)
print(f"\n📸 Starting data collection for: {current_label}\nPress 'q' to quit early.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        # Normalize based on wrist position (landmark 0)
        wrist = landmarks[0]
        landmarks = np.array([[lm[0] - wrist[0], lm[1] - wrist[1], lm[2] - wrist[2]] for lm in landmarks])
        landmarks = landmarks.flatten()


        # Save sample
        label_dir = os.path.join(DATA_DIR, current_label)
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)
        normalized = normalize_landmarks(landmarks)
        np.save(os.path.join(label_dir, f'{collected}.npy'), normalized)


        collected += 1

        cv2.putText(frame, f'{current_label}: {collected}/{samples_per_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if collected >= samples_per_label:
            print(f"\n✅ Done collecting {samples_per_label} samples for letter: {current_label}")
            print("🔁 Close window and rerun script to collect for another letter.")
            break
    else:
        cv2.putText(frame, 'No hand detected', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Collecting Sign Data - Press Q to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"\n⚠️ Collection manually stopped at {collected} samples for {current_label}")
        break

cap.release()
cv2.destroyAllWindows()
