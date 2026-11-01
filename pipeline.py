
import os
import base64
import torch

from pathlib import Path
from dotenv import load_dotenv
from transformers import (
    pipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)
from google import genai
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_AUDIO = BASE_DIR / "whats_test_audio_demba.ogg"
OUTPUT_AUDIO = BASE_DIR / "sortie.wav"

# Détection du matériel
torch_device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

hf_device = 0 if torch.cuda.is_available() else -1

print(f"Matériel utilisé : {torch_device}")


# ==================================================
# 2. TRANSCRIPTION WOLOF
# ==================================================

print("\n[1/3] Transcription en wolof...")

if not INPUT_AUDIO.exists():
    raise FileNotFoundError(f"Audio introuvable : {INPUT_AUDIO}")

asr = pipeline(
    "automatic-speech-recognition",
    model="AIHubSN/Kiriku-Wolof-ASR",
    device=hf_device
)

resultat = asr(str(INPUT_AUDIO))

texte_wolof = resultat["text"]

print("\nTranscription wolof :")
print(texte_wolof)


# ==================================================
# 3. TRADUCTION WOLOF -> FRANÇAIS
# ==================================================

# print("\n[2/3] Traduction en français...")

# checkpoint = "galsenai/wolofToFrenchTranslator_nllb"

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# model = AutoModelForSeq2SeqLM.from_pretrained(
#     checkpoint
# ).to(torch_device)

# model.eval()


# def get_translator(text, lang="wo"):

#     if lang.lower() == "wo":
#         prefix = "translate Wolof to French: "

#     elif lang.lower() == "fr":
#         prefix = "translate French to Wolof: "

#     else:
#         raise ValueError("Invalid language code")

#     inputs = tokenizer(
#         prefix + text,
#         return_tensors="pt",
#         truncation=True
#     ).to(torch_device)

#     with torch.no_grad():
#         translated_tokens = model.generate(
#             **inputs,
#             max_length=230
#         )

#     return tokenizer.batch_decode(
#         translated_tokens,
#         skip_special_tokens=True
#     )[0]


# texte_francais = get_translator(texte_wolof, lang="wo")

# print("\nTraduction française :")
# print(texte_francais)


## TRADUCTION ##

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = "galsenai/wolofToFrenchTranslator_nllb"

# Chargement direct du tokenizer NLLB
tokenizer = NllbTokenizer.from_pretrained(checkpoint)

# Chargement du modèle
model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint
).to(device)

model.eval()


def get_translator(text, lang="wo"):

    if lang.lower() == "wo":
        prefix = "translate Wolof to French: "

    elif lang.lower() == "fr":
        prefix = "translate French to Wolof: "

    else:
        raise ValueError("Invalid language code")

    inputs = tokenizer(
        prefix + text,
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.no_grad():
        translated_tokens = model.generate(
            **inputs,
            max_length=230
        )

    return tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]


texte_francais = get_translator(texte_wolof, lang="wo")

print("\nTraduction française :")
print(texte_francais)


# ==================================================
# 4. CONFIGURATION GEMINI TTS
# ==================================================

print("\n[3/3] Génération de la voix française...")

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY introuvable dans le fichier .env"
    )

client = genai.Client(api_key=API_KEY)

MODEL = "gemini-3.8-flash-tts"


# ==================================================
# 5. SYNTHÈSE VOCALE
# ==================================================

response = client.interactions.create(

    model=MODEL,

    input=[
        {
            "type": "user_input",
            "content": [
                {
                    "type": "text",
                    "text": texte_francais,
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


# ==================================================
# 6. ENREGISTREMENT DU FICHIER AUDIO
# ==================================================

if not response.output_audio or not response.output_audio.data:
    raise RuntimeError(
        "Gemini n'a renvoyé aucune donnée audio."
    )

audio_data = base64.b64decode(
    response.output_audio.data
)

OUTPUT_AUDIO.write_bytes(audio_data)

print("\n" + "=" * 50)
print("TRAITEMENT TERMINÉ")
print("=" * 50)

print(f"\nAudio original : {INPUT_AUDIO}")
print(f"Texte wolof : {texte_wolof}")
print(f"Traduction française : {texte_francais}")
print(f"Audio final : {OUTPUT_AUDIO}")
