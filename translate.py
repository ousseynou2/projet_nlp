from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = "galsenai/wolofToFrenchTranslator_nllb"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)


def get_translator(text, lang="wo"):
    if lang.lower() == "wo":
        prefix = "translate Wolof to French: "
    elif lang.lower() == "fr":
        prefix = "translate French to Wolof: "
    else:
        raise ValueError("Invalid language code")
    inputs = tokenizer(prefix + text, return_tensors="pt").to(device)
    translated_tokens = model.generate(**inputs, max_length=230)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]