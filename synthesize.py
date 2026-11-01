
import os
import base64
from pathlib import Path

from dotenv import load_dotenv
from google import genai

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY introuvable dans le fichier .env")

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.8-flash-tts"

# --------------------------------------------------
# 2. Texte à transformer en audio
# --------------------------------------------------

TEXT = """
Bonjour et bienvenue.
Ceci est un test de synthèse vocale avec Gemini.
La voix doit être naturelle, claire et agréable à écouter.
"""

# --------------------------------------------------
# 3. Génération audio avec Gemini
# --------------------------------------------------

response = client.interactions.create(
    model=MODEL,
    input=[
        {
            "type": "user_input",
            "content": [
                {
                    "type": "text",
                    "text": TEXT,
                    "annotations": [
                        {
                            "type": "speech_metadata",
                            "style": (
                                "Speak in natural French, with clear pronunciation, "
                                "a warm and professional tone, and a moderate pace."
                            )
                        }
                    ]
                }
            ]
        }
    ],
    response_format={"type": "audio"},
    generation_config={
        "speech_config": [
            {"voice": "Kore"}
        ]
    }
)

# --------------------------------------------------
# 4. Récupération et décodage de l'audio
# --------------------------------------------------

if not response.output_audio or not response.output_audio.data:
    raise RuntimeError("Gemini n'a renvoyé aucune donnée audio.")

# Les données audio sont encodées en Base64
audio_data = base64.b64decode(response.output_audio.data)

# --------------------------------------------------
# 5. Enregistrement du fichier WAV
# --------------------------------------------------

output_file = BASE_DIR / "sortie.wav"

with open(output_file, "wb") as f:
    f.write(audio_data)

print(f"Audio généré avec succès : {output_file}")
