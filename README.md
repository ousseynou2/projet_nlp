## PROJET 03 – NLP / ASR / TTS

### 1. Présentation du projet

Ce projet consiste à développer une application permettant de traduire automatiquement un message vocal exprimé en wolof vers le français, puis de restituer la traduction sous forme de message audio.

L'application dispose d'une interface graphique développée avec Gradio, permettant à l'utilisateur d'importer un fichier audio ou d'enregistrer directement un message à partir de son microphone.

Le traitement repose sur trois principales briques d'intelligence artificielle :
    - Reconnaissance automatique de la parole (ASR) : conversion de la parole wolof en texte.
    - Traduction  : traduction du texte wolof en français.
    - Synthèse vocale (TTS) : génération d'un message audio en français à partir du texte français obtenu.


### 2. Architecture générale du projet

Le fonctionnement de l'application repose sur une chaîne de traitement composée de quatre étapes principales.

```text
        UTILISATEUR
             |
             v
    Importation / Enregistrement
          d'un audio
             |
             v
    1. RECONNAISSANCE VOCALE
       Kiriku-Wolof-ASR
             |
             v
       Texte en wolof
             |
             v
    2. TRADUCTION AUTOMATIQUE
       GalsenAI NLLB
             |
             v
       Texte en français
             |
             v
    3. SYNTHÈSE VOCALE
       Gemini TTS
             |
             v
       Audio en français
             |
             v
    4. RESTITUTION DU RÉSULTAT
       Interface Gradio
```

## 3. Technologies et modèles utilisés

| Composant | Technologie / Modèle | Fonction |
|---|---|---|
| Langage de programmation | Python | Développement de l'application |
| Interface utilisateur | Gradio | Interaction avec l'utilisateur |
| Reconnaissance vocale (ASR) | `AIHubSN/Kiriku-Wolof-ASR` | Transcription du wolof |
| Traduction automatique | `galsenai/wolofToFrenchTranslator_nllb` | Traduction wolof-français |
| Synthèse vocale (TTS) | Gemini TTS | Génération de la voix française |
| Traitement des modèles | Hugging Face Transformers | Chargement et exécution des modèles |
| Gestion des variables d'environnement | python-dotenv | Chargement de la clé API |

### 3.1. Reconnaissance automatique de la parole (ASR)

La première étape consiste à convertir le signal audio en texte.

Pour cela, le projet utilise le modèle **Kiriku-Wolof-ASR**, développé pour la reconnaissance automatique de la parole en wolof.

Le modèle reçoit le fichier audio en entrée et produit une transcription textuelle en wolof.

Cette opération permet de passer d'une information sonore à une représentation textuelle exploitable par le système.

### 3.2. Traduction 

Une fois la transcription obtenue, le texte wolof est transmis au modèle **GalsenAI NLLB**, basé sur l'architecture de traduction neuronale NLLB (*No Language Left Behind*).

Ce modèle assure la traduction du texte wolof vers le français.

La traduction est réalisée à partir d'une séquence textuelle fournie au modèle, qui génère ensuite une nouvelle séquence correspondant au texte traduit.

### 3.3. Synthèse vocale (TTS)

La troisième étape consiste à transformer le texte français en parole.

Le projet utilise le modèle **Gemini TTS** pour générer un fichier audio à partir de la traduction française.

Des instructions de génération sont transmises au modèle afin d'obtenir une voix française naturelle, avec une prononciation claire, un ton professionnel et un débit modéré.

Le résultat est un fichier audio qui peut être écouté directement depuis l'interface.

### 3.4. Interface utilisateur avec Gradio

L'application utilise Gradio pour proposer une interface simple et accessible.

Elle permet à l'utilisateur de :

- Importer un fichier audio contenant un message en wolof ;
- Enregistrer directement un message avec son microphone ;
- Visualiser la transcription du message en wolof ;
- Consulter la traduction française ;
- Écouter le résultat audio en français.

## 4. Structure du projet

Le projet est organisé de la manière suivante :

```text
Projet_NLP/
│
├── app.py                  # Application principale et interface Gradio
├── .env                    # Variables d'environnement et clé API Gemini
├── README.md               # Documentation du projet
├── requirements.txt        # Dépendances Python (à créer)
│
├── venv_nlp/               # Environnement virtuel Python
│
└── whats_test_audio_demba.ogg  # Audio de test
├── transcription.py/           # Script de transcription(audio vers texte)
├── translate.py/               # Script de traduction(Wolof vers français)
├── synthesize.py/               # Script de vocalitisation(texte français vers audio français)
├── pipeline.py/               # Script global (transcription, traduction, vocalisation)
│
└── whats_test_audio_demba.ogg  # Audio de test
```

## 5. Installation et configuration

### 5.1. Prérequis

Avant de lancer le projet, il est nécessaire de disposer de :

- Python ;
- pip ;
- Une connexion Internet pour télécharger les modèles et utiliser l'API Gemini ;
- Une clé API Gemini valide.

### 5.2. Création de l'environnement virtuel

Dans le terminal, se positionner dans le répertoire du projet puis exécuter :

```bash
python -m venv venv_nlp
```

Activer ensuite l'environnement virtuel sous Windows :

```bash
venv_nlp\Scripts\activate
```

### 5.3. Installation des dépendances

Installer les bibliothèques nécessaires :

```bash
python -m pip install --upgrade pip

pip install torch
pip install transformers==4.57.6
pip install sentencepiece
pip install gradio
pip install python-dotenv
pip install google-genai
```

### 5.4. Configuration de la clé API Gemini

Créer un fichier `.env` à la racine du projet et y renseigner la clé API :

```env
GEMINI_API_KEY=votre_cle_api
```

**Important :** le fichier `.env` contient une information confidentielle. Il ne doit pas être partagé ni publié sur GitHub.

### 5.5. Préparation du fichier requirements.txt

Pour enregistrer les dépendances effectivement installées dans l'environnement :

```bash
pip freeze > requirements.txt
```

Ce fichier permettra de faciliter la reproduction de l'environnement de travail.

## 6. Exécution de l'application

Une fois les dépendances installées et la clé API configurée, lancer l'application avec :

```bash
python app.py
```

Gradio démarre alors un serveur local.

Ouvrir l'adresse suivante dans un navigateur :

```text
http://127.0.0.1:7860
```

L'utilisateur peut ensuite importer un fichier audio ou utiliser son microphone pour démarrer le processus de traduction.

## 7. Déroulement d'une opération de traduction

Pour chaque message soumis, le système effectue les opérations suivantes :

1. **Réception de l'audio :** l'utilisateur importe ou enregistre un message en wolof.
2. **Transcription :** Kiriku ASR analyse le signal sonore et produit le texte correspondant.
3. **Traduction :** le modèle GalsenAI NLLB convertit le texte wolof en français.
4. **Génération vocale :** Gemini TTS transforme la traduction française en fichier audio.
5. **Restitution :** Gradio affiche la transcription, la traduction et le lecteur audio.
