"""Tests fonctionnels de l'API (Étape 5) : routing, validation, gestion d'erreurs, protection
de /rebuild — via TestClient (fastapi.testclient), sans appel réel à Mistral ni au pipeline de
données.

lifespan() s'exécute réellement au démarrage de chaque TestClient (chargement de l'index FAISS
déjà présent sur disque + création des clients Mistral) : aucun appel réseau à ce stade, seule
la construction des objets. Les appels qui coûteraient réellement (chain.invoke(), les 4 étapes
du pipeline) sont mockés individuellement dans les tests qui en ont besoin, via monkeypatch.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# api/ n'est pas un package installé (pas de __init__.py) : ajout de la racine du projet au
# chemin d'import pour pouvoir faire "from api.main import app" (namespace package implicite,
# supporté nativement depuis Python 3.3, pas besoin de __init__.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api.main as api_main  # noqa: E402
from api.main import RebuildStatus, app  # noqa: E402


class FakeChain:
    """Remplace app.state.chain dans les tests /ask : pas de vrai appel Mistral."""

    def invoke(self, question: str) -> str:
        return f"Réponse factice pour : {question}"


class FailingChain:
    """Simule un échec du service Mistral pendant .invoke() (pour tester le 502)."""

    def invoke(self, question: str) -> str:
        raise RuntimeError("Panne simulée de l'API Mistral")


@pytest.fixture
def client():
    # "with" déclenche lifespan (startup) à l'entrée, puis son nettoyage à la sortie —
    # nécessaire pour que app.state.chain/rebuild_status existent avant chaque test.
    with TestClient(app) as test_client:
        yield test_client


# --- GET / et GET /health : routes simples, aucun mock nécessaire --------------------------


def test_root(client):
    """La racine répond et pointe vers la doc Swagger."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health(client):
    """Liveness check : répond "ok" indépendamment de l'état de la chaîne RAG."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- POST /ask ------------------------------------------------------------------------------


def test_ask_returns_answer(client, monkeypatch):
    """Chemin normal : la question atteint app.state.chain.invoke() et la réponse revient telle quelle."""
    monkeypatch.setattr(app.state, "chain", FakeChain())
    response = client.post("/ask", json={"question": "Quels concerts à Marseille ?"})
    assert response.status_code == 200
    assert "Marseille" in response.json()["answer"]


def test_ask_rejects_empty_question(client):
    """Question présente mais vide ("") -> rejetée par min_length=1 (Field), avant le corps de ask()."""
    # Aucun mock nécessaire : Pydantic rejette la requête avant même d'atteindre notre code.
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_rejects_missing_question_field(client):
    """Champ "question" absent du corps -> 422 aussi, mais un chemin de validation différent
    du cas précédent : ici Pydantic se plaint d'un champ REQUIS MANQUANT, pas d'une valeur
    invalide. Les deux sont à tester séparément ("mauvaise requête" du brief couvre les deux)."""
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_returns_503_when_chain_not_built(client, monkeypatch):
    """chain is None (échec de lifespan au démarrage) -> 503, pas un crash sur AttributeError."""
    monkeypatch.setattr(app.state, "chain", None)
    response = client.post("/ask", json={"question": "Une question valide ?"})
    assert response.status_code == 503


def test_ask_returns_502_on_upstream_failure(client, monkeypatch):
    """.invoke() lève une exception (ex. Mistral en panne) -> 502, message propre sans trace Python."""
    monkeypatch.setattr(app.state, "chain", FailingChain())
    response = client.post("/ask", json={"question": "Une question valide ?"})
    assert response.status_code == 502


# --- POST /rebuild : protection par clé + garde-fou "déjà en cours" -------------------------


def test_rebuild_rejects_missing_key(client, monkeypatch):
    """Aucun en-tête X-API-Key envoyé -> 401."""
    monkeypatch.setenv("X_API_KEY", "test-key")
    response = client.post("/rebuild")
    assert response.status_code == 401


def test_rebuild_rejects_wrong_key(client, monkeypatch):
    """En-tête envoyé mais valeur incorrecte -> 401 (même code que "absent", volontairement uniforme)."""
    monkeypatch.setenv("X_API_KEY", "test-key")
    response = client.post("/rebuild", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_rebuild_rejects_when_no_key_configured_server_side(client, monkeypatch):
    """X_API_KEY absent de l'environnement SERVEUR (pas juste de la requête) -> 401 quand même.

    Sécurité "fail-closed" : verify_admin_key() doit refuser par défaut si la clé attendue
    n'est même pas configurée, plutôt que d'accepter n'importe quelle valeur envoyée. Sans ce
    test, un bug qui ferait passer un "expected_key is None" comme "toujours autorisé" ne
    serait jamais détecté.
    """
    monkeypatch.delenv("X_API_KEY", raising=False)
    response = client.post("/rebuild", headers={"X-API-Key": "peu-importe-la-valeur"})
    assert response.status_code == 401


def test_rebuild_rejects_when_already_running(client, monkeypatch):
    """Un rebuild est déjà "running" -> 409, pas de second pipeline lancé en parallèle."""
    monkeypatch.setenv("X_API_KEY", "test-key")
    monkeypatch.setattr(app.state, "rebuild_status", RebuildStatus(state="running"))
    response = client.post("/rebuild", headers={"X-API-Key": "test-key"})
    assert response.status_code == 409


def test_rebuild_success_updates_status(client, monkeypatch):
    """Chemin de succès complet : 202 immédiat, puis statut "done" avec le compte d'événements."""
    monkeypatch.setenv("X_API_KEY", "test-key")
    monkeypatch.setattr(app.state, "rebuild_status", RebuildStatus(state="idle"))
    # Les 4 étapes du pipeline + le rechargement de la chaîne sont mockées : sans ça, ce test
    # déclencherait le vrai fetch/preprocess/vectorize (payant)/build_index à chaque lancement.
    monkeypatch.setattr(api_main.fetch_events, "main", lambda: None)
    monkeypatch.setattr(api_main.preprocess_events, "main", lambda: None)
    monkeypatch.setattr(api_main.vectorize_events, "main", lambda: None)
    monkeypatch.setattr(api_main.build_index, "main", lambda: 42)
    monkeypatch.setattr(api_main, "build_chain", lambda: FakeChain())

    response = client.post("/rebuild", headers={"X-API-Key": "test-key"})
    assert response.status_code == 202
    assert response.json() == {"status": "started"}

    # TestClient exécute les tâches de fond de façon SYNCHRONE (dans le même appel ASGI que
    # la requête, cf. starlette/responses.py) : le pipeline mocké est déjà terminé ici, pas
    # besoin d'attendre ni de faire du polling comme ce serait nécessaire avec un vrai serveur.
    status = client.get("/rebuild").json()
    assert status["state"] == "done"
    assert status["events_indexed"] == 42


def test_rebuild_failure_updates_status_with_error(client, monkeypatch):
    """Une étape du pipeline échoue -> statut "error" avec le message, pas de plantage silencieux.

    Couvre la branche "except" de run_rebuild_pipeline() : jusqu'ici aucun test ne la
    déclenchait, donc ce chemin n'était en réalité jamais vérifié.
    """
    monkeypatch.setenv("X_API_KEY", "test-key")
    monkeypatch.setattr(app.state, "rebuild_status", RebuildStatus(state="idle"))

    def failing_fetch() -> None:
        raise RuntimeError("Panne simulée pendant fetch_events")

    monkeypatch.setattr(api_main.fetch_events, "main", failing_fetch)
    # Les étapes suivantes ne devraient jamais être appelées (le pipeline s'arrête à la
    # première exception) — pas besoin de les mocker pour ce test.

    response = client.post("/rebuild", headers={"X-API-Key": "test-key"})
    assert response.status_code == 202  # la réponse HTTP reste 202 : l'échec arrive après, en tâche de fond

    status = client.get("/rebuild").json()
    assert status["state"] == "error"
    # detail doit rester générique (GET /rebuild n'est pas protégé) : le vrai message
    # ("Panne simulée...") part dans les logs serveur, pas dans la réponse HTTP publique.
    assert "Panne simulée" not in status["detail"]
    assert status["detail"]  # un message reste présent, juste pas le détail brut


def test_rebuild_status_get_is_public(client, monkeypatch):
    """GET /rebuild ne demande aucune clé API, contrairement à POST /rebuild."""
    monkeypatch.setattr(app.state, "rebuild_status", RebuildStatus(state="idle"))
    response = client.get("/rebuild")
    assert response.status_code == 200
    assert response.json()["state"] == "idle"
