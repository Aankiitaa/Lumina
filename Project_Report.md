# Project Report: Lumina - Sign Language Translator and Academy

**Academic Project Submission**

**Team Members:**
- Ankita Mandal (12023052017025)
- Sanjana Hazra (12023052017016)
- Ledia Roy (12023052017008)
- Paramartha Ghosh (12023052017010)

---

## 1. Abstract

Communication is a fundamental bridge between communities, yet significant access barriers still exist for the deaf and hard-of-hearing populations in integrating with the broader public. **Lumina** is a comprehensive, web-based platform designed to democratize communication and education for American Sign Language (ASL). It achieves this by combining a real-time computer-vision-based sign language translator with a gamified ASL learning academy. By utilizing cutting-edge MediaPipe hand-tracking, a custom-trained Random Forest classifier, and a Flask-based web architecture, Lumina offers an accurate, low-latency translation service running directly through conventional webcams. Furthermore, its Academy module employs modern gamification strategies—such as experience points (XP), daily streaks, and progressive unlocking mechanisms—to motivate and structure the learning experience for new signers.

## 2. Introduction

### 2.1 Background
Sign language is the primary means of communication for millions of deaf and hard-of-hearing individuals globally. However, very few hearing individuals understand it, leading to a profound communication gap in everyday life—from educational institutions and healthcare to retail and social interactions.

### 2.2 Problem Statement
There is a lack of accessible, user-friendly tools that not only translate static or real-time sign language for the hearing but also provide a structured, engaging way for people to actively learn the language. Existing solutions usually require expensive proprietary hardware, offer high latency, or provide a dry learning experience that fails to retain users.

### 2.3 Objectives
- **Build a Real-Time Translation Engine:** Utilize standard webcams to recognize and translate ASL signs in real-time.
- **Develop an Interactive Learning Platform:** Create a structured Academy that teaches ASL basics through interactive modules and quizzes.
- **Implement Gamification:** Use progression systems (XP, Streaks, Unlocks) to drive user engagement and learning retention.
- **Ensure Web Accessibility:** Deliver the entire experience through a standard web browser without requiring heavy installations on the client side.

### 2.4 Scope
Currently, Lumina focuses on translating static American Sign Language (ASL) alphabets, numbers, and basic words. The Academy covers foundational topics (Alphabet, Numbers, Greetings, Family, Questions). The scope encompasses the development of the ML model, the web server back-end, database integration for user state management, and the interactive front-end.

## 3. Technology Stack

Lumina integrates several robust modern technologies to fulfill its objectives:

### 3.1 Backend & Architecture
- **Language:** Python
- **Framework:** Flask (handles routing, secure sessions, authentication, and HTTP responses)
- **Database:** SQLite (`lumina_auth.db`) managed natively for storing user accounts, OTP data, and gamification metrics.

### 3.2 Machine Learning & Computer Vision
- **MediaPipe Hands:** A highly optimized Google framework for detecting 21 3D hand landmarks in real-time.
- **OpenCV:** Handles basic image manipulation, video frame capture from webcams, and visual feedback drawing.
- **Scikit-Learn:** Provides the `RandomForestClassifier` used to map landmark features to ASL characters.
- **Numpy & Pickle:** Used for data manipulation, array storage (`your_labels.npy`, `your_landmarks.npy`), and serializing the trained model (`model.p`).

### 3.3 Frontend
- **Languages:** HTML5, CSS3, JavaScript
- **Templating Engine:** Jinja2 (used extensively within Flask for dynamic web view rendering)

### 3.4 Additional Functionalities
- **Email Delivery (SMTP):** Native Python `smtplib` used to dispatch secure One-Time Passwords (OTPs) for user verification during registration.
- **Text-to-Speech (TTS):** Leverages native OS capabilities to vocally announce translated text.

## 4. System Architecture & Methodology

### 4.1 Data Pipeline and Model Training
1. **Data Collection (`collect_data.py`):** Captures frames from a webcam, extracting the 21 `(x, y)` hand landmarks using MediaPipe for defined classes.
2. **Normalization:** The absolute coordinates are converted to relative coordinates corresponding to the bounding box of the hand. This normalization guarantees that the model learns the *pose* of the hand, maintaining accuracy regardless of distance from the camera or its position in the frame.
3. **Model Training (`train_model.py`):** The extracted features (`your_landmarks.npy` and `your_labels.npy`) are split into training and testing sets. A `RandomForestClassifier` is trained, typically reaching high accuracy for static hand poses. The model is exported via Pickle for real-time inference.

### 4.2 Application Flow & Database Schema
The server relies on `web_app.py` as its core controller. The SQLite database is structured to support the community and the learning platform safely:
- **`users` Table:** Stores username, email, securely hashed passwords (`werkzeug.security`), and signup timestamps.
- **`user_stats` Table:** Maintains state for `xp`, `streak`, `lessons_completed`, and access validation for the Academy.
- **Authentication Flow:** User registration forces an email-based OTP verification before the account is finalized, avoiding spam/bot interactions and securing user data.

### 4.3 Real-Time Inference Pipeline
The *Live Translator* is the flagship functional module of the app:
- Instead of uploading heavy video streams, the client uses their browser to capture webcam feed.
- Frames are continuously processed through OpenCV and MediaPipe. 
- Discovered normalized landmarks are passed to the `model.predict()` function.
- **Stability Logic:** To prevent jitter and misinterpretation during finger-movement, the application implements a threshold buffer. A prediction is only committed to the "sentence output string" if it is consistently recognized over a fixed duration (e.g., 20+ sequential frames).

## 5. Modules Description

### 5.1 Authentication & Gamification Module
Handles standard web-security protocols. Unique to Lumina is the gamified user state. Upon successfully completing an Academy quiz, backend routes update the `user_stats` table in SQLite, incrementing XP, extending daily streaks, and subsequently unlocking the next tier of lessons (e.g., clearing "Alphabets" unlocks "Numbers").

### 5.2 The Academy Module
A sequential learning environment (`academy.html`). It breaks down the vastness of ASL into manageable categories. Each category presents visual references (images/videos of the signs) and follows up with interactive quizzes to test the user's retention before marking the module as complete.

### 5.3 Live Translator Dashboard
User interface where the continuous webcam feed is displayed alongside real-time bounding boxes and hand-landmark skeletons. The translated text builds up in an output box at the bottom of the screen, which can be dynamically cleared, edited, or spoken aloud via TTS.

## 6. Testing & Evaluation

- **Model Accuracy:** The RandomForestClassifier was evaluated using a traditional 80/20 train/test split. Evaluation metrics (like `accuracy_score`) demonstrated robust recognition capabilities specifically for static ASL signs in varied lighting.
- **System Latency:** The decision to utilize positional landmarks instead of running CNNs over full image frames allows the backend to process inference in mere milliseconds per frame, enabling a smooth 30 FPS visual output.
- **Functional Testing:** Thorough testing was conducted on database integrity (foreign keys, atomic updates on session progress), session management, and routing restrictions (preventing unauthenticated users from accessing the Academy).

## 7. Future Enhancements

While Lumina establishes a firm foundation, future iterations aim to incorporate:
1. **Dynamic Word Recognition:** Shifting from static classifications to sequence models (like LSTMs or Transformers) capable of tracking hand trajectories over time to decode complex words and phrases.
2. **Client-Side Inference:** Migrating the machine learning execution from the Flask backend directly to the user's browser using `TensorFlow.js`. This will drastically reduce server costs and eliminate latency caused by network roundtrips.
3. **Social Leaderboards:** Enhancing the Academy module by introducing peer leaderboards, friend challenges, and detailed performance analytics to further drive engagement.
4. **Mobile Deployment:** Re-packaging the front end into a cross-platform mobile application (React Native or Flutter) to allow on-the-go real-time translation.

## 8. Conclusion

Lumina successfully demonstrates the intersection of Artificial Intelligence and Human-Computer Interaction in solving real-world accessibility issues. By delivering a fast, accurate real-time translation engine and pairing it with a comprehensive, gamified Academy platform, Lumina serves not just as a utility, but as a driving force toward widespread sign language literacy. The project is highly scalable, technically robust, and paves the way for further advancements in assistive communication technologies.
