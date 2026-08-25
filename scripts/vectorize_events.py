"""Découpe le texte de chaque événement en chunks et génère leurs embeddings (Mistral).

Lit data/processed/events.json, découpe le champ text en chunks (text splitter LangChain),
génère un embedding par chunk via l'API Mistral (mistral-embed), et sauvegarde le résultat
dans data/vectors/events_vectors.json — "prêt à être indexé" au sens strict du brief.

Appelle l'API Mistral (mistral-embed, payant) — nécessite MISTRAL_API_KEY dans .env.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "events.json"
VECTORS_PATH = Path(__file__).resolve().parent.parent / "data" / "vectors" / "events_vectors.json"

# Taille cible d'un chunk (en caractères), dérivée de la distribution réelle des longueurs
# de texte observée sur data/processed/events.json (voir notebooks/03_chunking_exploration.ipynb) :
# CHUNK_SIZE = Q3(longueurs) — le 3e quartile (75e percentile), mesuré à 1034 caractères, arrondi à 1000.
# Choix délibéré : les 75% d'événements les plus courts (titre + description simple) restent
# intacts en un seul chunk, tandis que le quart le plus long (souvent avec une longdescription_fr
# substantielle) est justement celui qui bénéficie d'un découpage plus fin. Valeur fixe pour ce
# POC ; à recalculer sur Q3 si le volume/la nature des données changent significativement.
CHUNK_SIZE = 1000

# Chevauchement entre deux chunks consécutifs, pour ne pas couper une information pile à la
# frontière. Convention usuelle : CHUNK_OVERLAP = r x CHUNK_SIZE, avec r entre 0.10 et 0.20.
# Ici r = 0.10 (100 / 1000), cohérent avec cette convention.
CHUNK_OVERLAP = 100


def load_processed_events() -> list[dict]:
    """Charger les événements structurés produits par preprocess_events.py."""
    return json.loads(PROCESSED_PATH.read_text(encoding="utf-8"))


def split_into_chunks(events: list[dict]) -> list[dict]:
    """Découper le texte de chaque événement en chunks, chacun gardant les métadonnées de l'événement d'origine."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    for event in events:
        # Métadonnées identiques pour tous les chunks d'un même événement (dates, lieu...),
        # seul le texte diffère d'un chunk à l'autre.
        metadata = {key: value for key, value in event.items() if key != "text"}
        for i, chunk_text in enumerate(splitter.split_text(event["text"])):
            chunks.append({"chunk_id": f"{event['uid']}-{i}", "text": chunk_text, "metadata": metadata})
    return chunks


def main() -> None:
    """Point d'entrée : découper en chunks, vectoriser, et sauvegarder les vecteurs."""
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY manquant : renseignez-le dans .env (voir .env.example).")

    events = load_processed_events()
    chunks = split_into_chunks(events)
    print(f"{len(events)} événements découpés en {len(chunks)} chunks")

    embeddings_model = MistralAIEmbeddings(mistral_api_key=api_key)
    vectors = embeddings_model.embed_documents([chunk["text"] for chunk in chunks])

    records = [
        {"chunk_id": chunk["chunk_id"], "text": chunk["text"], "metadata": chunk["metadata"], "embedding": vector}
        for chunk, vector in zip(chunks, vectors)
    ]

    VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    VECTORS_PATH.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    print(f"{len(records)} vecteurs sauvegardés dans {VECTORS_PATH}")


if __name__ == "__main__":
    main()
