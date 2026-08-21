"""Génère les embeddings des événements (Mistral) et construit l'index vectoriel FAISS.

Lit data/processed/events.json, transforme chaque événement en Document LangChain
(texte à vectoriser + métadonnées), et sauvegarde l'index FAISS résultant dans data/index/.

Appelle l'API Mistral (mistral-embed, payant) — nécessite MISTRAL_API_KEY dans .env.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings

# Chemin du fichier produit par preprocess_events.py (données déjà nettoyées/structurées).
PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "events.json"

# Dossier de destination de l'index FAISS (deux fichiers y seront écrits : index.faiss et index.pkl).
INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "index"


def load_processed_events() -> list[dict]:
    """Charger les événements structurés produits par preprocess_events.py."""
    return json.loads(PROCESSED_PATH.read_text(encoding="utf-8"))


def to_documents(events: list[dict]) -> list[Document]:
    """Convertir chaque événement en Document LangChain : texte à vectoriser + métadonnées associées."""
    documents = []
    for event in events:
        # Toutes les métadonnées sauf 'text', déjà utilisé comme contenu principal du Document
        # (page_content). Ces métadonnées resteront attachées au vecteur dans l'index, récupérables
        # au moment de la recherche (titre, dates, lieu, url...) sans avoir à revectoriser quoi que ce soit.
        metadata = {key: value for key, value in event.items() if key != "text"}
        documents.append(Document(page_content=event["text"], metadata=metadata))
    return documents


def main() -> None:
    """Point d'entrée : générer les embeddings et construire/sauvegarder l'index FAISS."""
    # Chargement du fichier .env dans les variables d'environnement du processus courant.
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    # Échec explicite et immédiat si la clé est absente, plutôt qu'une erreur réseau confuse plus tard.
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY manquant : renseignez-le dans .env (voir .env.example).")

    events = load_processed_events()
    documents = to_documents(events)
    print(f"{len(documents)} événements à vectoriser...")

    # Client d'embeddings Mistral : model="mistral-embed" par défaut (pas besoin de le préciser).
    embeddings = MistralAIEmbeddings(mistral_api_key=api_key)

    # Un seul appel qui fait tout : vectorise chaque Document (appels à l'API Mistral, par lots
    # en interne) ET construit l'index FAISS à partir des vecteurs obtenus.
    vector_store = FAISS.from_documents(documents, embeddings)

    # Création du dossier de destination s'il n'existe pas encore (premier lancement du script).
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    # Écriture de l'index sur disque : index.faiss (les vecteurs) + index.pkl (documents/métadonnées).
    vector_store.save_local(str(INDEX_PATH))

    print(f"Index FAISS sauvegardé dans {INDEX_PATH}")


if __name__ == "__main__":
    main()
