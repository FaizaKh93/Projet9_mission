"""Tests unitaires de scripts/fetch_events.py — httpx.get() mocké (aucun appel réseau réel).

Contrairement à la politique appliquée ailleurs dans le projet (voir tests/test_rag_chain.py,
tests/test_api.py, README.md) qui exclut les fonctions faisant de vrais appels réseau/API :
ici, le comportement à vérifier (pagination, condition d'arrêt) est purement mécanique et ne
dépend d'aucune réponse d'un LLM à juger — un mock reste donc représentatif du comportement réel,
contrairement à mocker une génération Mistral dont la qualité ne se résume pas à sa structure.
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

# scripts/ n'est pas un package installé (pas de __init__.py) : on ajoute son chemin à sys.path
# pour pouvoir écrire "import fetch_events" juste en dessous, comme si c'en était un.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import fetch_events  # noqa: E402 (import après la modification de sys.path, volontaire)


class FakeResponse:
    """Réponse httpx factice : seules raise_for_status()/json() sont utilisées par fetch_events.py."""

    def __init__(self, payload: dict):
        # Le contenu JSON qu'on veut que .json() renvoie (simule le corps de la réponse HTTP).
        self._payload = payload

    def raise_for_status(self) -> None:
        # Ne fait rien : simule une réponse HTTP 200 (pas d'erreur à lever).
        pass

    def json(self) -> dict:
        return self._payload


# --- build_where_clause : filtre ODSQL calculé par rapport à aujourd'hui, aucun appel réseau ---


def test_build_where_clause_contains_department_and_recent_date():
    """Le filtre doit cibler les Bouches-du-Rhône et une date de début calculée à J-365, pas une
    date codée en dur — recalculée ici de la même façon pour la comparaison, sans dépendre d'un
    "aujourd'hui" figé (aucun paramètre injectable dans build_where_clause() pour le geler)."""
    # Fonction pure, sans appel réseau : appelée directement, sans aucun mock nécessaire.
    where_clause = fetch_events.build_where_clause()
    assert 'location_department="Bouches-du-Rhône"' in where_clause
    # Même calcul que dans le code testé (datetime.now() - 365 jours), pour comparer sans figer
    # artificiellement une date dans ce test.
    expected_date = (datetime.now(timezone.utc) - timedelta(days=fetch_events.WINDOW_DAYS)).strftime("%Y-%m-%d")
    assert f'firstdate_begin>="{expected_date}"' in where_clause


# --- fetch_page : retry automatique (tenacity) sur échec HTTP -------------------------------


def test_fetch_page_retries_after_transient_failure(monkeypatch):
    """Un premier appel qui échoue (erreur HTTP simulée) suivi d'un second qui réussit -> le
    retry de tenacity doit rattraper l'échec tout seul, sans faire remonter l'exception au
    premier essai. Vérifie que le retry se déclenche réellement, pas juste que le décorateur
    @retry est présent dans le code."""
    calls = []

    def flaky_get(url, params, timeout):
        calls.append(params)
        if len(calls) == 1:
            # Simule une vraie erreur HTTP (ex. panne serveur temporaire, 500) — c'est ce type
            # d'exception que response.raise_for_status() lèverait sur une vraie réponse en erreur.
            request = httpx.Request("GET", url)
            response = httpx.Response(500, request=request)
            raise httpx.HTTPStatusError("erreur simulée", request=request, response=response)
        return FakeResponse({"results": [{"uid": "1"}], "total_count": 1})

    monkeypatch.setattr(fetch_events.httpx, "get", flaky_get)

    payload = fetch_events.fetch_page("where=1", 0)

    assert payload == {"results": [{"uid": "1"}], "total_count": 1}
    assert len(calls) == 2  # 1er appel en échec + réessai réussi, pas plus


# --- fetch_all_events : pagination, httpx.get() mocké --------------------------------------


def test_fetch_all_events_paginates_until_total_count_reached(monkeypatch):
    """Deux pages à récupérer (total_count=3, 2 puis 1 résultat) -> les deux appels doivent
    avoir lieu, avec un offset qui avance du nombre de résultats réellement reçus, et le résultat
    final doit concaténer les deux pages dans l'ordre."""
    # Ce que l'API renverrait normalement : ici, deux pages fabriquées à la main, sans vraie requête.
    pages = [
        {"results": [{"uid": "1"}, {"uid": "2"}], "total_count": 3},
        {"results": [{"uid": "3"}], "total_count": 3},
    ]
    calls = []  # Garde une trace de chaque appel (et de ses params) pour vérifier la pagination après coup.

    def fake_get(url, params, timeout):
        # Remplace httpx.get() : renvoie la page suivante à chaque appel, dans l'ordre de "pages".
        calls.append(params)
        return FakeResponse(pages[len(calls) - 1])

    # monkeypatch.setattr remplace temporairement httpx.get par fake_get, le temps de ce test
    # seulement (remis automatiquement à l'original ensuite, même si le test échoue).
    monkeypatch.setattr(fetch_events.httpx, "get", fake_get)

    events = fetch_events.fetch_all_events()

    assert events == [{"uid": "1"}, {"uid": "2"}, {"uid": "3"}]  # Les 2 pages concaténées, dans l'ordre.
    assert len(calls) == 2  # Un appel par page, pas plus.
    assert calls[0]["offset"] == 0  # Premier appel : depuis le début.
    assert calls[1]["offset"] == 2  # Second appel : décalé du nombre de résultats déjà reçus (2).


def test_fetch_all_events_stops_on_empty_page_even_if_total_count_not_reached(monkeypatch):
    """Garde-fou contre une boucle infinie : une page vide doit arrêter la pagination, même si
    total_count annonce encore des résultats restants (ex. donnée source incohérente)."""

    def fake_get(url, params, timeout):
        # total_count=5 mais aucun résultat renvoyé : cas limite/incohérent à ne pas boucler dessus.
        return FakeResponse({"results": [], "total_count": 5})

    monkeypatch.setattr(fetch_events.httpx, "get", fake_get)

    events = fetch_events.fetch_all_events()

    assert events == []  # Doit s'arrêter proprement, pas boucler indéfiniment.


def test_fetch_all_events_single_page_stops_immediately(monkeypatch):
    """Un seul appel suffisant (total_count atteint dès la première page) -> pas de second appel."""
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        # total_count=1 et 1 résultat déjà reçu -> la boucle doit s'arrêter dès ce premier appel.
        return FakeResponse({"results": [{"uid": "1"}], "total_count": 1})

    monkeypatch.setattr(fetch_events.httpx, "get", fake_get)

    events = fetch_events.fetch_all_events()

    assert events == [{"uid": "1"}]
    assert len(calls) == 1  # Vérifie qu'il n'y a pas eu d'appel superflu.


# --- main : écrit le résultat de fetch_all_events() sur disque -----------------------------


def test_main_writes_fetched_events_to_output_path(tmp_path, monkeypatch):
    """main() doit créer le dossier de destination si besoin et y écrire exactement ce que
    renvoie fetch_all_events() — mocké directement ici, la pagination étant déjà testée plus haut."""
    # tmp_path : dossier temporaire fourni par pytest, supprimé automatiquement après le test —
    # on n'écrit jamais dans le vrai data/raw/events.json depuis un test.
    output_path = tmp_path / "raw" / "events.json"
    # Redirige la constante OUTPUT_PATH du module vers ce chemin temporaire.
    monkeypatch.setattr(fetch_events, "OUTPUT_PATH", output_path)
    # Ici on ne remocke pas httpx : on remplace directement fetch_all_events() par une fonction
    # qui renvoie un résultat tout prêt — la pagination elle-même est déjà couverte plus haut,
    # ce test-ci ne vérifie que ce que fait main() de ce résultat (écriture sur disque).
    monkeypatch.setattr(fetch_events, "fetch_all_events", lambda: [{"uid": "1"}, {"uid": "2"}])

    fetch_events.main()

    # Le dossier "raw/" n'existait pas dans tmp_path : s'il a été créé et contient le bon JSON,
    # c'est que main() a bien fait son travail (mkdir + écriture).
    saved_events = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved_events == [{"uid": "1"}, {"uid": "2"}]
