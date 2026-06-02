"""
text_to_sign.py — Text-to-Sign Translation Module for Lumina

This module handles the complete Text-to-Sign pipeline:
1. Loads the Gemini API key from .env
2. Converts English sentences to ASL gloss (Time-Topic-Comment structure)
3. Resolves each ASL token to a visual sign (GIF/image) or fingerspelling fallback

Usage:
    from text_to_sign import init_gemini, translate_to_signs
    init_gemini()
    result = translate_to_signs("I am going to the store tomorrow")
"""

import os
import glob
from dotenv import load_dotenv

# ------------------------------------
# Load environment variables from .env
# ------------------------------------
load_dotenv()

# ------------------------------------
# Gemini API Setup
# ------------------------------------
_gemini_model = None
_gemini_ready = False
_gemini_error = ""

# Base directory for this project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths to sign assets
WORD_SIGNS_DIR = os.path.join(BASE_DIR, "static", "signs", "words")
ALPHABET_SIGNS_DIR = os.path.join(BASE_DIR, "static", "signs")


def init_gemini():
    """
    Initialize the Gemini API client.
    Must be called once at application startup.
    Returns True if successful, False otherwise.
    """
    global _gemini_model, _gemini_ready, _gemini_error

    api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key or api_key == "paste_your_full_api_key_here":
        _gemini_error = "GEMINI_API_KEY not set in .env file"
        print(f"[TEXT-TO-SIGN] Warning: {_gemini_error}")
        return False

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        _gemini_ready = True
        print("[TEXT-TO-SIGN] Gemini API initialized successfully.")
        return True
    except Exception as e:
        _gemini_error = str(e)
        print(f"[TEXT-TO-SIGN] Failed to initialize Gemini: {_gemini_error}")
        return False


def is_ready():
    """Check if the Gemini API is initialized and ready."""
    return _gemini_ready


def get_error():
    """Get the error message if initialization failed."""
    return _gemini_error


# ------------------------------------
# English → ASL Gloss Conversion
# ------------------------------------

# The prompt instructs Gemini to act as an ASL linguistics expert
_ASL_PROMPT = """You are an expert ASL (American Sign Language) linguist and translator.

Your task: Convert the given English sentence into ASL gloss notation.

ASL grammar rules you MUST follow:
1. TIME-TOPIC-COMMENT structure: Time references come first, then the topic, then the comment/action.
   Example: "I will go to the store tomorrow" → "TOMORROW STORE I GO"
2. Remove all English grammatical words that don't exist in ASL:
   - Remove articles: a, an, the
   - Remove linking verbs: is, am, are, was, were, be, being, been
   - Remove auxiliary verbs: do, does, did, will, would, shall, should, can, could, may, might, must
   - Remove prepositions when implied by context: to, at, in, on, for, with, of
   - Remove conjunctions: and, but, or (unless critical for meaning)
3. Use ROOT/BASE form of verbs: "going" → "GO", "eaten" → "EAT", "ran" → "RUN"
4. Keep the essential meaning — do NOT add words that aren't implied.
5. Pronouns are kept: I, YOU, HE, SHE, WE, THEY
6. For questions, put the question word at the END:
   "What is your name?" → "YOUR NAME WHAT"
   "Where do you live?" → "YOU LIVE WHERE"
7. Negation comes AFTER the verb: "I don't like coffee" → "COFFEE I LIKE NOT"

Output format:
- Return ONLY the ASL gloss words, separated by spaces
- ALL CAPS
- No punctuation, no explanation, no extra text
- Multi-word signs should use hyphens: THANK-YOU, ICE-CREAM
- If the input is a single word, just return that word in caps

Examples:
- "I am going to the store tomorrow" → "TOMORROW STORE I GO"
- "What is your name?" → "YOUR NAME WHAT"
- "She doesn't like coffee" → "COFFEE SHE LIKE NOT"
- "My family is very happy today" → "TODAY MY FAMILY VERY HAPPY"
- "Thank you for helping me" → "HELP ME YOU THANK-YOU"
- "Hello" → "HELLO"
- "I love you" → "I LOVE YOU"

Now convert this sentence:
"""


def english_to_asl_gloss(sentence):
    """
    Convert an English sentence to ASL gloss using Gemini API.

    Args:
        sentence (str): English sentence to convert

    Returns:
        dict: {
            "ok": bool,
            "original": str,        # Original English input
            "gloss": str,           # Full ASL gloss string
            "tokens": list[str],    # Individual ASL tokens
            "error": str|None       # Error message if failed
        }
    """
    if not _gemini_ready:
        return {
            "ok": False,
            "original": sentence,
            "gloss": "",
            "tokens": [],
            "error": _gemini_error or "Gemini API not initialized"
        }

    sentence = sentence.strip()
    if not sentence:
        return {
            "ok": False,
            "original": sentence,
            "gloss": "",
            "tokens": [],
            "error": "Empty sentence"
        }

    try:
        response = _gemini_model.generate_content(
            _ASL_PROMPT + f'"{sentence}"',
            generation_config={
                "temperature": 0.1,     # Low temperature for consistent output
                "max_output_tokens": 200,
            }
        )

        gloss = response.text.strip()

        # Clean up the response — remove any quotes, periods, extra whitespace
        gloss = gloss.strip('"\'.,!?')
        gloss = gloss.upper()

        # Split into individual tokens
        tokens = [t.strip() for t in gloss.split() if t.strip()]

        return {
            "ok": True,
            "original": sentence,
            "gloss": " ".join(tokens),
            "tokens": tokens,
            "error": None
        }

    except Exception as e:
        return {
            "ok": False,
            "original": sentence,
            "gloss": "",
            "tokens": [],
            "error": f"Gemini API error: {str(e)}"
        }


# ------------------------------------
# Sign Resolution (Word → Visual)
# ------------------------------------

def _find_sign_file(word):
    """
    Look for a sign image/GIF in the words directory.

    Searches for: word.gif, word.mp4, word.png, word.jpg, word.webp
    Also tries replacing hyphens with underscores and vice versa.

    Returns the web-accessible path (e.g., "/static/signs/words/hello.gif")
    or None if not found.
    """
    # Normalize: lowercase, replace hyphens with underscores
    normalized = word.lower().replace("-", "_").replace(" ", "_")

    # Try multiple naming conventions
    variations = [normalized, word.lower(), word.lower().replace("_", "-")]

    for name in variations:
        for ext in ("gif", "mp4", "png", "jpg", "jpeg", "webp"):
            filepath = os.path.join(WORD_SIGNS_DIR, f"{name}.{ext}")
            if os.path.exists(filepath):
                return f"/static/signs/words/{name}.{ext}"

    return None


def _fingerspell(word):
    """
    Break a word into individual letters and map each to an alphabet image.

    Uses existing alphabet images at static/signs/A.png ... Z.png

    Returns a list of dicts:
    [{"letter": "A", "src": "/static/signs/A.png"}, ...]
    """
    letters = []
    for char in word.upper():
        if char.isalpha():
            # Check if the alphabet image exists
            img_path = os.path.join(ALPHABET_SIGNS_DIR, f"{char.upper()}.png")
            if os.path.exists(img_path):
                letters.append({
                    "letter": char.upper(),
                    "src": f"/static/signs/{char.upper()}.png"
                })
            else:
                # Fallback: letter without image
                letters.append({
                    "letter": char.upper(),
                    "src": None
                })
    return letters


def resolve_signs(tokens):
    """
    Resolve a list of ASL gloss tokens into visual sign data.

    For each token:
    - If a word sign (GIF/image) exists → type: "word"
    - If not found → type: "fingerspell" (broken into individual letters)

    Args:
        tokens (list[str]): ASL gloss tokens from english_to_asl_gloss()

    Returns:
        list[dict]: Each dict has:
            - "type": "word" | "fingerspell"
            - "word": the original token
            - "src": path to GIF/image (for type "word")
            - "letters": list of letter dicts (for type "fingerspell")
    """
    result = []

    for token in tokens:
        sign_path = _find_sign_file(token)

        if sign_path:
            result.append({
                "type": "word",
                "word": token,
                "src": sign_path
            })
        else:
            letters = _fingerspell(token)
            result.append({
                "type": "fingerspell",
                "word": token,
                "letters": letters
            })

    return result


# ------------------------------------
# High-Level Translate Function
# ------------------------------------

def translate_to_signs(sentence):
    """
    Complete translation pipeline: English → ASL Gloss → Visual Signs.

    This is the main function called by the Flask route.

    Args:
        sentence (str): English sentence

    Returns:
        dict: {
            "ok": bool,
            "original": str,
            "gloss": str,
            "signs": list[dict],
            "error": str|None
        }
    """
    # Step 1: Convert English to ASL gloss
    gloss_result = english_to_asl_gloss(sentence)

    if not gloss_result["ok"]:
        return {
            "ok": False,
            "original": sentence,
            "gloss": "",
            "signs": [],
            "error": gloss_result["error"]
        }

    # Step 2: Resolve each token to a visual sign
    signs = resolve_signs(gloss_result["tokens"])

    return {
        "ok": True,
        "original": gloss_result["original"],
        "gloss": gloss_result["gloss"],
        "signs": signs,
        "error": None
    }


# ------------------------------------
# Utility: List available word signs
# ------------------------------------

def list_available_signs():
    """
    List all word signs currently available in static/signs/words/.
    Useful for debugging and checking which words are covered.
    """
    if not os.path.isdir(WORD_SIGNS_DIR):
        return []

    signs = []
    for f in sorted(os.listdir(WORD_SIGNS_DIR)):
        name, ext = os.path.splitext(f)
        if ext.lower() in (".gif", ".mp4", ".png", ".jpg", ".jpeg", ".webp"):
            signs.append(name)
    return signs
