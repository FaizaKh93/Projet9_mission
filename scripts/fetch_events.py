"""Récupère les événements culturels des Bouches-du-Rhône (< 1 an) via le miroir
Opendatasoft des données Open Agenda, et sauvegarde le résultat brut en JSON.

Aucune clé API requise (dataset public). Filtre calculé par rapport à la date
du jour à chaque exécution, pour rester utilisable tel quel si ce script est
un jour déclenché automatiquement (cron / GitHub Actions) plutôt qu'à la main.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
# tenacity : lib de retry — ajoute une nouvelle tentative automatique en cas d'échec, sans avoir
# à écrire la boucle try/except/sleep à la main.
from tenacity import retry, stop_after_attempt, wait_exponential

# Dataset public "Événements Publics - OpenAgenda", miroir Opendatasoft du site openagenda.com.
API_URL = "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records"

# Zone géographique ciblée pour ce POC (modifiable si le volume de données est trop important).
LOCATION_DEPARTMENT = "Bouches-du-Rhône"

# Fenêtre de récence demandée par le brief : ne garder que les événements de moins d'un an.
WINDOW_DAYS = 365

# Nombre maximal de résultats que l'API accepte de renvoyer par appel (limite imposée par Opendatasoft).
PAGE_SIZE = 100

# Emplacement du fichier de sortie : data/raw/events.json, à la racine du projet.
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "events.json"


def build_where_clause() -> str:
    """Construire le filtre ODSQL (langage de requête Opendatasoft) combinant département et récence."""
    # Calcul de "aujourd'hui moins un an" à chaque exécution, pour que la fenêtre reste toujours à jour
    # (utile si ce script est relancé plus tard automatiquement, sans date codée en dur).
    one_year_ago = (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    return f'location_department="{LOCATION_DEPARTMENT}" AND firstdate_begin>="{one_year_ago}"'


@retry(
    # stop_after_attempt(3) : 3 tentatives au total (1 essai + 2 réessais), pas 3 réessais en plus
    # du premier essai — après ça, tenacity laisse remonter la dernière exception normalement.
    stop=stop_after_attempt(3),
    # wait_exponential : attend de plus en plus longtemps entre chaque tentative (1s puis 2s ici,
    # avec multiplier=1 -> 1 x 2^0, 1 x 2^1...) plutôt qu'un intervalle fixe — laisse le temps à
    # une panne temporaire du serveur de se résorber, sans le marteler de requêtes immédiates.
    wait=wait_exponential(multiplier=1),
)
def fetch_page(where_clause: str, offset: int) -> dict:
    """Récupérer UNE SEULE page de résultats, avec retry automatique sur échec HTTP.

    Isolée dans sa propre fonction (plutôt que l'appel direct dans fetch_all_events() d'avant)
    parce que @retry rejoue toute la fonction qu'il décore à chaque tentative — il doit donc
    entourer seulement l'appel réseau lui-même, pas toute la boucle de pagination (qui, elle, ne
    doit jamais être répétée depuis le début à cause d'une seule page en échec).
    """
    response = httpx.get(
        API_URL,
        params={"where": where_clause, "limit": PAGE_SIZE, "offset": offset},
        timeout=30,
    )
    # C'est cette exception (levée sur une erreur HTTP 4xx/5xx) que @retry détecte pour déclencher
    # une nouvelle tentative — par défaut, tenacity réessaie sur n'importe quelle exception.
    response.raise_for_status()
    return response.json()


def fetch_all_events() -> list[dict]:
    """Récupérer tous les événements correspondant au filtre, en parcourant les pages de résultats."""
    where_clause = build_where_clause()
    events: list[dict] = []
    offset = 0  # Index du premier résultat à récupérer sur la page courante.

    # Boucle de pagination : l'API ne renvoie que PAGE_SIZE résultats par appel,
    # donc répétition de l'appel en avançant offset jusqu'à couvoir la totalité des résultats.
    while True:
        payload = fetch_page(where_clause, offset)

        page_results = payload["results"]
        events.extend(page_results)

        total_count = payload["total_count"]  # Nombre total de résultats correspondant au filtre, renvoyé par l'API.
        offset += len(page_results)
        print(f"  {offset}/{total_count} événements récupérés")

        # Arrêt de la boucle une fois tous les résultats couverts, ou si l'API ne renvoie plus rien.
        if offset >= total_count or not page_results:
            break

    return events


def main() -> None:
    """Point d'entrée : récupérer les événements et les écrire dans data/raw/events.json."""
    events = fetch_all_events()

    # Création du dossier de destination s'il n'existe pas encore (premier lancement du script).
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{len(events)} événements sauvegardés dans {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
