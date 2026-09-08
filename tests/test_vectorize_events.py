"""Tests unitaires de scripts/vectorize_events.py — MistralAIEmbeddings mocké (aucun appel API réel).

Même choix que tests/test_fetch_events.py (voir son docstring) : le comportement à vérifier ici
(découpage en chunks, assemblage chunk_id/texte/métadonnée/vecteur, écriture du résultat) est
mécanique et indépendant de la qualité des embeddings eux-mêmes — un mock reste donc représentatif,
contrairement à mocker une génération de réponse Mistral (rag_chain.py::retrieve_context, non
mocké, voir README.md).
"""

import json
import sys
from pathlib import Path

import pytest

# scripts/ n'est pas un package installé (pas de __init__.py) : on ajoute son chemin à sys.path
# pour pouvoir écrire "import vectorize_events" juste en dessous, comme si c'en était un.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import vectorize_events  # noqa: E402 (import après la modification de sys.path, volontaire)


# --- split_into_chunks : pure, aucun appel réseau -------------------------------------------


def test_split_into_chunks_builds_chunk_id_and_keeps_metadata_without_text():
    """chunk_id = "<uid>-<index>", et la métadonnée reprend tous les champs de l'événement SAUF
    "text" (déjà transféré dans le champ "text" du chunk lui-même, pas besoin de le dupliquer)."""
    # Un seul événement, texte court : aucun découpage attendu -> un seul chunk en sortie.
    events = [{"uid": "42", "title": "Festival de Jazz", "text": "Un texte court, un seul chunk."}]

    chunks = vectorize_events.split_into_chunks(events)

    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "42-0"  # uid de l'événement + index du chunk (le premier ici).
    assert chunks[0]["text"] == "Un texte court, un seul chunk."
    # "text" ne doit PAS réapparaître dans metadata (déjà présent comme champ "text" du chunk).
    assert chunks[0]["metadata"] == {"uid": "42", "title": "Festival de Jazz"}


def test_split_into_chunks_splits_long_text_into_several_ordered_chunks():
    """Un texte largement au-dessus de CHUNK_SIZE (1000 caractères) doit être découpé en
    plusieurs chunks, numérotés dans l'ordre à partir de 0."""
    long_text = "Une phrase représentative d'un événement culturel. " * 50  # bien > 1000 caractères
    events = [{"uid": "1", "text": long_text}]

    chunks = vectorize_events.split_into_chunks(events)

    assert len(chunks) > 1  # Le texte a bien été coupé en plusieurs morceaux.
    # Vérifie la numérotation exacte : "1-0", "1-1", "1-2"..., dans l'ordre, sans trou ni doublon.
    assert [c["chunk_id"] for c in chunks] == [f"1-{i}" for i in range(len(chunks))]


def test_split_into_chunks_handles_several_events_independently():
    """Deux événements courts -> un chunk chacun, avec des chunk_id distincts basés sur leur
    propre uid (pas de numérotation globale partagée entre événements)."""
    events = [
        {"uid": "1", "text": "Premier événement."},
        {"uid": "2", "text": "Second événement."},
    ]

    chunks = vectorize_events.split_into_chunks(events)

    # Chaque événement repart de l'index 0 : pas de compteur global partagé entre les deux.
    assert [c["chunk_id"] for c in chunks] == ["1-0", "2-0"]


# --- main : chargement, vectorisation (mockée), sauvegarde ----------------------------------


class FakeEmbeddings:
    """Remplace MistralAIEmbeddings : aucune clé API réelle ni appel réseau, un vecteur fixe
    par texte reçu."""

    def __init__(self, mistral_api_key=None):
        # Accepte le même argument que la vraie classe (mistral_api_key=...), sans rien en faire —
        # juste pour que vectorize_events.py puisse l'instancier de la même façon.
        self.mistral_api_key = mistral_api_key

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Un vecteur factice identique pour chaque texte reçu : suffisant pour vérifier l'assemblage
        # des résultats (main() ne teste pas la qualité des vecteurs, juste ce qu'il en fait).
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_main_vectorizes_processed_events_and_saves_records(tmp_path, monkeypatch):
    """Bout en bout (load -> split -> "vectorize" mocké -> save) : vérifie que le fichier de
    sortie contient bien chunk_id/text/metadata/embedding pour chaque chunk, sans appel Mistral réel."""
    # tmp_path : dossier temporaire fourni par pytest, supprimé après le test — on ne touche jamais
    # aux vrais data/processed/events.json ni data/vectors/events_vectors.json depuis un test.
    processed_path = tmp_path / "events.json"
    vectors_path = tmp_path / "events_vectors.json"
    # Fichier d'entrée factice, écrit à la main, comme si preprocess_events.py l'avait produit.
    processed_path.write_text(
        json.dumps([{"uid": "1", "title": "Festival", "text": "Un texte court."}]), encoding="utf-8"
    )

    # Redirige les chemins d'entrée/sortie du module vers les fichiers temporaires ci-dessus.
    monkeypatch.setattr(vectorize_events, "PROCESSED_PATH", processed_path)
    monkeypatch.setattr(vectorize_events, "VECTORS_PATH", vectors_path)
    # Remplace la vraie classe MistralAIEmbeddings par la fausse définie plus haut : main()
    # l'utilisera sans le savoir, sans jamais appeler l'API Mistral.
    monkeypatch.setattr(vectorize_events, "MistralAIEmbeddings", FakeEmbeddings)
    # Évite que load_dotenv() aille lire un vrai fichier .env pendant le test (sans effet ici).
    monkeypatch.setattr(vectorize_events, "load_dotenv", lambda: None)
    # main() vérifie la présence de MISTRAL_API_KEY avant de continuer : on en fournit une fausse,
    # jamais utilisée pour un vrai appel réseau grâce à FakeEmbeddings ci-dessus.
    monkeypatch.setenv("MISTRAL_API_KEY", "fake-key-for-tests")

    vectorize_events.main()

    # Si tout s'est bien enchaîné, le fichier de sortie contient un enregistrement complet par chunk.
    records = json.loads(vectors_path.read_text(encoding="utf-8"))
    assert len(records) == 1
    assert records[0]["chunk_id"] == "1-0"
    assert records[0]["text"] == "Un texte court."
    assert records[0]["metadata"] == {"uid": "1", "title": "Festival"}
    assert records[0]["embedding"] == [0.1, 0.2, 0.3]  # Le vecteur factice de FakeEmbeddings.


def test_main_raises_explicit_error_when_api_key_missing(tmp_path, monkeypatch):
    """Sans MISTRAL_API_KEY, main() doit échouer explicitement (RuntimeError avec un message
    clair), pas planter plus loin avec une erreur d'authentification Mistral confuse."""
    processed_path = tmp_path / "events.json"
    processed_path.write_text(json.dumps([{"uid": "1", "text": "Un texte."}]), encoding="utf-8")

    monkeypatch.setattr(vectorize_events, "PROCESSED_PATH", processed_path)
    monkeypatch.setattr(vectorize_events, "load_dotenv", lambda: None)
    # Supprime la variable d'environnement si elle existe (ex. un vrai .env chargé par ailleurs),
    # pour être sûr de tester le cas "clé absente" — raising=False : pas d'erreur si déjà absente.
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    # pytest.raises vérifie qu'une RuntimeError est bien levée, ET que son message contient
    # "MISTRAL_API_KEY" (match=...) — pas n'importe quelle erreur.
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
        vectorize_events.main()
