"""Diagnostic ponctuel : inspecte le contexte réellement transmis au LLM pour des questions où
`evaluate_rag.py` a montré un problème (hallucination sur contexte vide, ou faux négatif "aucun
événement" alors que des événements valides existent) — pour distinguer, à chaque étape du
pipeline (filtre FAISS -> résultats bruts -> après occurrence_in_period()), où l'information se
perd. Conservé volontairement (pas un script jetable) : preuve documentée du diagnostic ayant mené
aux fixes cités dans rag_chain.py/query_filters.py (voir leurs commentaires) — pas destiné à être
relancé en routine.
"""

import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_mistralai import MistralAIEmbeddings  # noqa: E402
from query_filters import build_faiss_filter, extract_filters, occurrence_in_period  # noqa: E402
from rag_chain import INDEX_PATH, RETRIEVED_CHUNKS, get_api_key, retrieve_context  # noqa: E402

QUESTIONS = [
    "Qu'est-ce qu'il y a à faire à Marseille dimanche prochain ?",
]


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = get_api_key()
    embeddings = MistralAIEmbeddings(mistral_api_key=api_key)
    vector_store = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
    total_vectors = vector_store.index.ntotal

    for question in QUESTIONS:
        print("=" * 100)
        print("QUESTION :", question)
        filters = extract_filters(question, api_key)
        today = date.today()
        print("FILTRES EXTRAITS :", filters)
        faiss_filter = build_faiss_filter(filters, today)
        print("FILTRE FAISS :", faiss_filter)

        # Étape 1 : ce que similarity_search() renvoie APRÈS le filtre FAISS mais AVANT le second
        # passage Python (occurrence_in_period) — pour savoir si AFRICA/Nouveau parcours sont
        # seulement mal classés (absents du top-k), ou déjà exclus par le filtre FAISS lui-même.
        docs = vector_store.similarity_search(
            question, k=RETRIEVED_CHUNKS, filter=faiss_filter or None, fetch_k=total_vectors
        )
        print(f"\nTOP {RETRIEVED_CHUNKS} APRÈS FILTRE FAISS (avant occurrence_in_period) :")
        for i, doc in enumerate(docs, start=1):
            print(f"  {i}. {doc.metadata.get('title')} (uid={doc.metadata.get('uid')})")

        # Étape 2 : ce qui survit au second passage (occurrence_in_period).
        kept = [doc for doc in docs if occurrence_in_period(doc.metadata, filters, today)]
        print(f"\nAPRÈS occurrence_in_period() ({len(kept)}/{len(docs)} conservés) :")
        for doc in kept:
            print(f"  - {doc.metadata.get('title')} (uid={doc.metadata.get('uid')})")

        # Étape 3 : le contexte final réellement transmis au LLM (identique à retrieve_context()).
        context = retrieve_context(question, vector_store, api_key, total_vectors)
        print(f"\nCONTEXTE FINAL ({len(context)} caractères) :")
        print(context if context else "(VIDE)")
        print()


if __name__ == "__main__":
    main()
