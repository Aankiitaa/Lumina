# Lumina

This project predicts ASL hand signs from webcam input and builds a sentence.

## What to use
- Recommended app: `web_app.py` (runs on localhost)
- Optional desktop app: `app.py` (Tkinter)

## macOS

### 1. Clone

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create virtual environment (Python 3.10)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

If `python3.10` is missing:

```bash
brew install python@3.10
```

### 3. Install runtime dependencies

```bash
pip install --upgrade pip
pip install -r requirements-web.txt
```

### 4. Run web app

```bash
python web_app.py
```

Open:

- `http://127.0.0.1:5001`

## Windows

### 1. Clone

```powershell
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create virtual environment (Python 3.10)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
```

If Python 3.10 is missing, install it from the official installer and ensure `py` works.

### 3. Install runtime dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements-web.txt
```

### 4. Run web app

```powershell
python web_app.py
```

Open:

- `http://127.0.0.1:5001`


## Notes

- This repo includes a pre-trained model file: `model.p`.
- Raw training data is not required to run inference.
- Webcam permission is required.

## For retraining

Training-related scripts/data in this repo:

- `collect_data.py`
- `convert_dataset.py`
- `train_model.py`
- `data/`
- `your_landmarks.npy`, `your_labels.npy`

Retraining is optional and not needed for normal app usage.
