from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

asr = pipeline(
    "automatic-speech-recognition",
    model="AIHubSN/Kiriku-Wolof-ASR",
    device=device
)

resultat = asr("whats_test_audio_demba.ogg")

texte_wolof = resultat["text"]

print("Transcription en wolof :", texte_wolof)