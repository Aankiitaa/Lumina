# Lumina: Sign Language Translator and Academy

Lumina is a web-based educational platform that combines real-time American Sign Language (ASL) translation with a gamified learning curriculum. It utilizes MediaPipe for hand landmark detection and a pre-trained machine learning model for sign classification.

## Technical Requirements
- Python 3.10
- Webcam
- Modern Web Browser (Chrome/Edge/Safari)

---

## Installation and Setup

### macOS
1. **Clone the repository**
   ```bash
   git clone https://github.com/Aankiitaa/Lumina.git
   cd Lumina
   ```
2. **Create and activate a virtual environment**
   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements-web.txt
   ```
4. **Run the application**
   ```bash
   python web_app.py
   ```
   Access the application at `http://127.0.0.1:5001`.

### Windows
1. **Clone the repository**
   ```powershell
   git clone https://github.com/Aankiitaa/Lumina.git
   cd Lumina
   ```
2. **Create and activate a virtual environment**
   ```powershell
   py -3.10 -m venv .venv
   .\.venv\Scripts\activate
   ```
3. **Install dependencies**
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements-web.txt
   ```
4. **Run the application**
   ```powershell
   python web_app.py
   ```
   Access the application at `http://127.0.0.1:5001`.

---

## File Registry

### Application Core
- **web_app.py**: The main Flask application server handling routes, authentication, and the MediaPipe inference pipeline.
- **model.p**: Pre-trained machine learning model for sign classification.
- **lumina_auth.db**: SQLite database for user accounts and progress tracking.

### User Interface (Templates)
- **templates/login.html**: Redesigned light-theme login page.
- **templates/signup.html**: Redesigned signup page with user onboarding.
- **templates/otp_verify.html**: Email verification interface.
- **templates/dashboard.html**: Home navigation providing access to the Academy and Translator.
- **templates/academy.html**: Gamified learning interface with interactive sign practice.
- **templates/index.html**: Live camera translator workspace.

### Assets and Data
- **static/**: Directory containing CSS, JavaScript, and decorative mascot illustrations.
- **data/**: Directory for training dataset storage.
- **requirements-web.txt**: List of Python dependencies for the web application.

### Developer Tools (Retraining)
- **collect_data.py**: Script for capturing new hand landmark data.
- **train_model.py**: Script for retraining the classifier model.
- **your_landmarks.npy / your_labels.npy**: Processed training data.

---

## Authentication Configuration
To enable live email verification, set the following environment variables:
- `LUMINA_EMAIL`: Your service Gmail address.
- `LUMINA_EMAIL_PASS`: Your Google App Password.
- `LUMINA_SECRET_KEY`: A unique string for session security.
