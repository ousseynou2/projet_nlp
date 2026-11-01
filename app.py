from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
import os
import base64
import tempfile
import torch
import gradio as gr

from pathlib import Path
from dotenv import load_dotenv

from transformers import (
    pipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

from google import genai


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY introuvable dans .env")

client = genai.Client(api_key=API_KEY)

GEMINI_MODEL = "gemini-3.8-flash-tts"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

hf_device = 0 if torch.cuda.is_available() else -1

print(f"Matériel utilisé : {device}")


# =========================================================
# 2. CHARGEMENT DES MODELES
# =========================================================

print("\nChargement du modèle Kiriku ASR...")

asr = pipeline(
    "automatic-speech-recognition",
    model="AIHubSN/Kiriku-Wolof-ASR",
    device=hf_device
)

print("Modèle ASR chargé.")

print("\nChargement du modèle de traduction...")

checkpoint = "galsenai/wolofToFrenchTranslator_nllb"

# tokenizer = AutoTokenizer.from_pretrained(checkpoint)

tokenizer = NllbTokenizer.from_pretrained(checkpoint)


model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint
).to(device)

model.eval()

print("Modèle de traduction chargé.")

print("\nTous les modèles sont prêts.")


# =========================================================
# 3. TRANSCRIPTION WOLOF
# =========================================================

def transcribe_audio(audio_path):

    if audio_path is None:
        raise gr.Error("Veuillez importer ou enregistrer un audio.")

    resultat = asr(audio_path)

    texte_wolof = resultat["text"].strip()

    if not texte_wolof:
        raise gr.Error("Aucune parole détectée.")

    return texte_wolof


# =========================================================
# 4. TRADUCTION WOLOF - FRANÇAIS
# =========================================================

def get_translator(text, lang="wo"):

    if lang.lower() == "wo":
        prefix = "translate Wolof to French: "

    elif lang.lower() == "fr":
        prefix = "translate French to Wolof: "

    else:
        raise ValueError("Code langue invalide.")

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
    )[0].strip()


# =========================================================
# 5. SYNTHESE VOCALE GEMINI
# =========================================================

def generate_speech(text):

    if not text.strip():
        raise gr.Error("Aucun texte à synthétiser.")

    response = client.interactions.create(

        model=GEMINI_MODEL,

        input=[
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": text,
                        "annotations": [
                            {
                                "type": "speech_metadata",
                                "style": (
                                    "Speak in natural French, with clear "
                                    "pronunciation, a warm and professional "
                                    "tone, and a moderate pace."
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

    if not response.output_audio or not response.output_audio.data:
        raise gr.Error("Gemini n'a retourné aucune donnée audio.")

    audio_data = base64.b64decode(
        response.output_audio.data
    )

    with tempfile.NamedTemporaryFile(
        suffix=".wav",
        delete=False
    ) as audio_file:

        audio_file.write(audio_data)

        return audio_file.name


# =========================================================
# 6. PIPELINE COMPLET
# =========================================================

def process_audio(audio_path):

    # Etape 1 : transcription
    texte_wolof = transcribe_audio(audio_path)

    # Etape 2 : traduction
    texte_francais = get_translator(
        texte_wolof,
        lang="wo"
    )

    # Etape 3 : synthèse vocale
    audio_francais = generate_speech(
        texte_francais
    )

    return (
        texte_wolof,
        texte_francais,
        audio_francais
    )


# =========================================================
# 7. INTERFACE GRADIO
# =========================================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Traduction vocale Wolof - Français"
) as demo:

    gr.Markdown(
        """
        # Traduction vocale Wolof – Français

        **De la parole wolof à la voix française grâce à l'intelligence artificielle.**

        Importez un fichier audio ou utilisez votre microphone.
        """
    )

    with gr.Row():

        with gr.Column():

            gr.Markdown("### 1. Audio en wolof")

            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Importer ou enregistrer un audio"
            )

            process_button = gr.Button(
                "Transcrire et traduire",
                variant="primary"
            )

        with gr.Column():

            gr.Markdown("### 2. Transcription en wolof")

            transcription_output = gr.Textbox(
                label="Texte wolof",
                lines=5
            )

            gr.Markdown("### 3. Traduction française")

            translation_output = gr.Textbox(
                label="Texte français",
                lines=5
            )

    gr.Markdown("---")

    gr.Markdown("### 4. Synthèse vocale française")

    audio_output = gr.Audio(
        label="Audio traduit en français",
        type="filepath"
    )

    process_button.click(
        fn=process_audio,
        inputs=audio_input,
        outputs=[
            transcription_output,
            translation_output,
            audio_output
        ]
    )

    gr.Markdown(
        """
        ---
        **Technologies :** Kiriku ASR | GalsenAI NLLB | Gemini TTS
        """
    )


# =========================================================
# 8. LANCEMENT DE L'APPLICATION
# =========================================================

if __name__ == "__main__":

    demo.queue()

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
