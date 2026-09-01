"""Construit l'index vectoriel FAISS à partir des vecteurs déjà calculés par vectorize_events.py.

Lit data/vectors/events_vectors.json (chunks + embeddings + métadonnées, déjà vectorisés à
l'Étape 2), construit l'index FAISS via FAISS.from_embeddings() — sans revectoriser — et le
sauvegarde dans data/index/.

Nécessite MISTRAL_API_KEY dans .env : pas pour revectoriser ici, mais parce que l'objet
MistralAIEmbeddings reste attaché à l'index pour vectoriser les futures questions au moment
de la recherche (même modèle qu'à l'indexation, indispensable pour comparer des vecteurs
compatibles entre eux).

Pour revenir à l'ancienne approche (vectoriser ET indexer en une seule étape, directement à
partir de data/processed/events.json, sans passer par vectorize_events.py/events_vectors.json) :

    from langchain_core.documents import Document
    processed_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "events.json"
    events = json.loads(processed_path.read_text(encoding="utf-8"))
    documents = [
        Document(page_content=e["text"], metadata={k: v for k, v in e.items() if k != "text"})
        for e in events
    ]
    vector_store = FAISS.from_documents(documents, embeddings)

... à la place de l'appel à FAISS.from_embeddings() ci-dessous. Utile si on retire le
découpage en chunks ou si les Étapes 2 (vectorisation) et 3 (indexation) doivent être
refusionnées en un seul script.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings

# Chemin du fichier produit par vectorize_events.py (chunks déjà vectorisés).
VECTORS_PATH = Path(__file__).resolve().parent.parent / "data" / "vectors" / "events_vectors.json"

# Dossier de destination de l'index FAISS (deux fichiers y seront écrits : index.faiss et index.pkl).
INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "index"


def load_vectors() -> list[dict]:
    """Charger les chunks déjà vectorisés par vectorize_events.py."""
    return json.loads(VECTORS_PATH.read_text(encoding="utf-8"))


def main() -> int:
    """Point d'entrée : construire l'index FAISS à partir des vecteurs déjà calculés.

    Renvoie le nombre de vecteurs indexés — utilisé par api/main.py (endpoint /rebuild)
    pour l'exposer dans le statut de reconstruction ; ignoré par l'appel en ligne de commande.
    """
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY manquant : renseignez-le dans .env (voir .env.example).")

    records = load_vectors()
    print(f"{len(records)} chunks déjà vectorisés chargés depuis {VECTORS_PATH}")

    # Client d'embeddings Mistral : nécessaire pour vectoriser les futures questions au moment
    # de la recherche, PAS pour revectoriser ces chunks (déjà fait par vectorize_events.py).
    embeddings = MistralAIEmbeddings(mistral_api_key=api_key)

    # (texte, vecteur) par chunk — FAISS.from_embeddings n'appelle jamais l'API Mistral,
    # contrairement à FAISS.from_documents qui, lui, vectorise à la volée (voir docstring).
    text_embeddings = [(record["text"], record["embedding"]) for record in records]
    metadatas = [record["metadata"] for record in records]
    ids = [record["chunk_id"] for record in records]

    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    # Vérification que tous les événements ont bien été indexés.
    indexed_count = vector_store.index.ntotal
    print(f"Vecteurs chargés : {len(records)} | vecteurs réellement indexés : {indexed_count}")
    if indexed_count != len(records):
        raise RuntimeError("Nombre de vecteurs indexés différent du nombre de vecteurs chargés — à investiguer.")

    # Création du dossier de destination s'il n'existe pas encore (premier lancement du script).
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    # Écriture de l'index sur disque : index.faiss (les vecteurs) + index.pkl (documents/métadonnées).
    vector_store.save_local(str(INDEX_PATH))

    print(f"Index FAISS sauvegardé dans {INDEX_PATH}")
    return indexed_count


if __name__ == "__main__":
    main()
