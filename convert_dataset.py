import os
import numpy as np
def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks).reshape(-1, 3)
    base = landmarks[0]  # use wrist as origin
    landmarks -= base
    max_value = np.max(np.abs(landmarks))
    if max_value != 0:
        landmarks /= max_value  # scale to range [-1, 1]
    return landmarks.flatten()

DATA_DIR = 'data'  # Folder where your collected data is saved
landmark_data = []
labels = []

for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    for file in os.listdir(label_dir):
        if file.endswith('.npy'):
            data = np.load(os.path.join(label_dir, file))
            normalized = normalize_landmarks(data)
            landmark_data.append(normalized)
            labels.append(label)

landmark_data = np.array(landmark_data)
labels = np.array(labels)

np.save('your_landmarks.npy', landmark_data)
np.save('your_labels.npy', labels)

print(f"✅ Saved {len(landmark_data)} samples for {len(set(labels))} classes.")
