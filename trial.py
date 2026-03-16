import os
import numpy as np

data_dirs = ['data', 'asl_alphabet_train']
total = 0

for data_dir in data_dirs:
    print(f"\n📁 Checking directory: {data_dir}")
    for label in os.listdir(data_dir):
        path = os.path.join(data_dir, label)
        if not os.path.isdir(path):
            continue
        files = [f for f in os.listdir(path) if f.endswith('.npy')]
        print(f"{label}: {len(files)} samples")
        total += len(files)

print(f"\n🔢 Total samples across both sources: {total}")
