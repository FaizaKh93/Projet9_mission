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


def fetch_all_events() -> list[dict]:
    """Récupérer tous les événements correspondant au filtre, en parcourant les pages de résultats."""
    where_clause = build_where_clause()
    events: list[dict] = []
    offset = 0  # Index du premier résultat à récupérer sur la page courante.

    # Boucle de pagination : l'API ne renvoie que PAGE_SIZE résultats par appel,
    # donc répétition de l'appel en avançant offset jusqu'à couvoir la totalité des résultats.
    while True:
        response = httpx.get(
            API_URL,
            params={"where": where_clause, "limit": PAGE_SIZE, "offset": offset},
            timeout=30,
        )
        response.raise_for_status()  # Interruption immédiate en cas d'erreur HTTP (ex. requête mal formée).
        payload = response.json()

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
