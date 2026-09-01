"""Nettoie et structure les événements bruts récupérés par fetch_events.py.

Lit data/raw/events.json, exclut les événements hors-sujet (non culturels) et les
événements incomplets ou mal géocodés (voir is_complete()), construit un texte unique
par événement (pour la vectorisation à venir), et sauvegarde le résultat structuré
dans data/processed/events.json.

Choix délibéré : garder le maximum de champs utiles à répondre aux questions des
utilisateurs (tarifs, âge, accessibilité, contact...), pas seulement titre/description.
Seuls sont écartés les champs constants/vides sur tout le jeu de données (aucune valeur
informative possible) et les coordonnées personnelles du contributeur (email, téléphone,
nom) qui n'ont pas leur place dans une donnée exposée publiquement par un chatbot.
"""

import json
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "events.json"
PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "events.json"

# Source à exclure : événements France Travail (forums emploi, recrutement) — hors-sujet
# pour un chatbot d'événements culturels, et ~49% du volume brut à eux seuls.
EXCLUDED_SOURCES = {"Mes événements France Travail"}

# Préfixe attendu pour location_postalcode : les codes postaux des Bouches-du-Rhône
# commencent tous par 13. Sert à détecter les événements mal géocodés (ex. un événement à
# Landerneau, code postal 29800, mais avec location_department indiquant à tort "Bouches-du-Rhône").
TARGET_POSTALCODE_PREFIX = "13"


def load_raw_events() -> list[dict]:
    """Charger les événements bruts téléchargés par fetch_events.py."""
    return json.loads(RAW_PATH.read_text(encoding="utf-8"))


def is_relevant(event: dict) -> bool:
    """Écarter les événements dont la source est hors-sujet (non culturelle)."""
    # Comparaison sur originagenda_title (nom de la source), pas sur le contenu de l'événement lui-même.
    return event.get("originagenda_title") not in EXCLUDED_SOURCES


def parse_labeled_field(raw_value: str | None) -> str | None:
    """Extraire le libellé français d'un champ à structure {"id": ..., "label": {"fr": ...}}.

    Plusieurs champs bruts (status, attendancemode) partagent ce même format, stocké
    sous forme de texte JSON imbriqué — d'où le second json.loads() pour l'interpréter.
    """
    if not raw_value:
        return None
    try:
        return json.loads(raw_value)["label"]["fr"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def parse_registration_link(raw_registration: str | None) -> str | None:
    """Extraire le premier lien d'inscription depuis le champ registration (liste JSON imbriquée)."""
    if not raw_registration:
        return None
    try:
        entries = json.loads(raw_registration)
        return entries[0]["value"] if entries else None
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return None


def parse_occurrences(raw_timings: str | None) -> list[dict]:
    """Extraire chaque occurrence individuelle d'un événement récurrent (champ timings, liste
    JSON de créneaux [{"begin": ..., "end": ...}, ...]).

    date_start/date_end (firstdate_begin/lastdate_end) ne donnent que la première et la
    dernière occurrence — insuffisant pour un événement récurrent (ex. un atelier chaque
    samedi d'avril à juin) dont il faut connaître la PROCHAINE occurrence, pas seulement la
    première. Stocké séparément pour être exploité au moment de la question (rag_chain.py),
    pas ici : "aujourd'hui" n'est pas connu au moment du preprocessing.
    """
    if not raw_timings:
        return []
    try:
        timings = json.loads(raw_timings)
    except (json.JSONDecodeError, TypeError):
        return []
    occurrences = []
    for timing in timings:
        begin = timing.get("begin")
        if begin:
            occurrences.append({"start": begin, "end": timing.get("end") or begin})
    return occurrences


def normalize_city(city: str | None) -> str | None:
    """Corriger la casse tout-majuscule (ex. MARSEILLE -> Marseille), sans toucher aux noms déjà bien formatés."""
    # Valeur vide ou absente : retour tel quel, rien à normaliser.
    if not city:
        return city
    city = city.strip()
    # Conversion en casse titre uniquement si tout-majuscule, pour ne pas abîmer
    # des noms déjà corrects comme "Aix-en-Provence" (qui échouerait le test isupper()).
    return city.title() if city.isupper() else city


def build_text(event: dict) -> str:
    """Construire le texte à vectoriser, enrichi de tout ce qui peut aider la recherche sémantique."""
    title = (event.get("title_fr") or "").strip()
    description = (event.get("description_fr") or "").strip()
    longdescription = (event.get("longdescription_fr") or "").strip()
    status = parse_labeled_field(event.get("status"))
    conditions = (event.get("conditions_fr") or "").strip()
    keywords = event.get("keywords_fr") or []
    accessibility_labels = event.get("accessibility_label_fr") or []
    location_name = (event.get("location_name") or "").strip()
    location_district = (event.get("location_district") or "").strip()

    parts = []
    # Mention du statut uniquement s'il sort du cas par défaut ("Programmé"), pour que le
    # texte vectorisé porte lui-même l'information — utile si le LLM doit répondre à
    # "cet événement est-il maintenu ?" sans dépendre d'un filtre structuré séparé.
    if status and status != "Programmé":
        parts.append(f"Statut de l'événement : {status}.")

    parts.extend([title, description])
    # Ajout de la version longue seulement si elle apporte du contenu en plus de la courte,
    # pour éviter de dupliquer deux fois le même texte dans les cas où les deux champs sont identiques.
    if longdescription and longdescription != description:
        parts.append(longdescription)

    # Tarifs/modalités intégrés au texte : aide la recherche sémantique sur des questions
    # comme "événements gratuits" sans dépendre d'un filtre structuré exact sur ce champ libre.
    if conditions:
        parts.append(f"Conditions/tarifs : {conditions}")

    if keywords:
        parts.append("Mots-clés : " + ", ".join(keywords))

    # Accessibilité intégrée au texte plutôt qu'en filtre structuré : seulement 5 valeurs
    # distinctes sur ~12,6% des événements (vérifié empiriquement), et ce sont des métadonnées
    # en LISTE — le filtre FAISS ($in) ne sait vérifier qu'une valeur scalaire parmi une liste
    # acceptée, pas l'inverse ("cette liste contient-elle X ?"). La recherche sémantique sur
    # une formulation libre ("accessible en fauteuil roulant") est plus simple à ce stade.
    if accessibility_labels:
        parts.append("Accessibilité : " + ", ".join(accessibility_labels))

    # Nom du lieu et arrondissement intégrés au texte (pas en filtre structuré, cf.
    # location_city) : ce sont des noms propres qu'une question peut citer directement
    # (ex. "à la médiathèque Louis Aragon", "dans le 8e arrondissement"), utiles à la
    # recherche sémantique, contrairement aux champs purement factuels (adresse, téléphone...)
    # qui n'apportent rien à la recherche et ne sont affichés qu'en métadonnée (format_docs).
    if location_name or location_district:
        lieu = " — ".join(part for part in [location_name, location_district] if part)
        parts.append(f"Lieu : {lieu}")

    # Filtrage des parties vides avant assemblage.
    return "\n\n".join(part for part in parts if part)


def structure_event(event: dict) -> dict:
    """Extraire les champs utiles au RAG, sous un format simple et stable.

    age_min/age_max valent respectivement 0 et 110 par défaut dans la donnée source
    quand aucune restriction n'est précisée (tout public) — valeur conservée telle quelle,
    ce n'est pas une donnée manquante.
    """
    # Renommage des champs bruts (suffixés _fr, préfixés location_) vers des noms courts et stables,
    # pour découpler le format de sortie de la structure changeante de l'API source.
    return {
        "uid": event.get("uid"),
        "title": event.get("title_fr"),
        "text": build_text(event),
        "status": parse_labeled_field(event.get("status")),
        "attendance_mode": parse_labeled_field(event.get("attendancemode")),
        "date_start": event.get("firstdate_begin"),
        "date_end": event.get("lastdate_end"),
        "occurrences": parse_occurrences(event.get("timings")),
        "conditions": (event.get("conditions_fr") or "").strip() or None,
        "age_min": event.get("age_min"),
        "age_max": event.get("age_max"),
        "accessibility_labels": event.get("accessibility_label_fr") or [],
        "keywords": event.get("keywords_fr") or [],
        "location_name": event.get("location_name"),
        "location_city": normalize_city(event.get("location_city")),
        "location_address": event.get("location_address"),
        "location_postalcode": event.get("location_postalcode"),
        "location_district": event.get("location_district"),
        "location_insee": event.get("location_insee"),
        "location_phone": event.get("location_phone"),
        "location_website": event.get("location_website"),
        "location_links": event.get("location_links"),
        "registration_link": parse_registration_link(event.get("registration")),
        "online_access_link": event.get("onlineaccesslink"),
        "url": event.get("canonicalurl"),
    }


def build_known_cities(structured_events: list[dict]) -> set[str]:
    """Construire l'ensemble des noms de ville déjà confirmés dans les Bouches-du-Rhône.

    Une ville est "confirmée" si au moins un événement du jeu de données l'associe à un
    code postal correct (préfixe 13). Sert de garde-fou contre les faux positifs de
    is_complete() : un code postal isolé mal saisi (ex. "10003" pour un événement à
    Marseille) ne doit pas suffire à exclure un événement dont la ville est par ailleurs
    bien identifiée ailleurs dans les mêmes données.
    """
    return {
        e["location_city"]
        for e in structured_events
        if e["location_city"] and e["location_postalcode"] and e["location_postalcode"].startswith(TARGET_POSTALCODE_PREFIX)
    }


def is_complete(structured_event: dict, known_cities: set[str]) -> bool:
    """Écarter les événements auxquels il manque une information jugée indispensable.

    Contrairement à is_relevant() (qui exclut une source entière avant transformation),
    ce filtre s'applique après structure_event() et vérifie le résultat final :
    - texte vide (titre ET description absents) : rien à vectoriser
    - date de début absente : impossible de dire si l'événement est récent/à venir
    - uid absent : pas d'identifiant fiable
    - code postal renseigné mais hors Bouches-du-Rhône (ne commence pas par 13) ET ville
      absente de known_cities : événement probablement mal géocodé. Si la ville est déjà
      confirmée ailleurs (ex. "Marseille"), on privilégie location_city — le code postal
      isolé est plus probablement une erreur de saisie qu'une preuve que l'événement est
      hors zone (cas réel rencontré : "10003" pour un événement à Marseille).
    - événement en présentiel (attendance_mode != "En ligne") sans aucune localisation
    - événement en ligne (attendance_mode == "En ligne") sans aucun lien pour y accéder,
      symétrique de la règle précédente : personne ne peut savoir comment y participer
    """
    if not structured_event["text"]:
        return False
    if not structured_event["date_start"]:
        return False
    if not structured_event["uid"]:
        return False

    postal_code = structured_event["location_postalcode"]
    city = structured_event["location_city"]
    if postal_code and not postal_code.startswith(TARGET_POSTALCODE_PREFIX) and city not in known_cities:
        return False

    is_in_person = structured_event["attendance_mode"] != "En ligne"
    has_location = bool(
        structured_event["location_name"]
        or structured_event["location_address"]
        or structured_event["location_city"]
    )
    if is_in_person and not has_location:
        return False

    is_online = structured_event["attendance_mode"] == "En ligne"
    has_access_link = bool(structured_event["online_access_link"] or structured_event["registration_link"])
    if is_online and not has_access_link:
        return False

    return True


def preprocess() -> list[dict]:
    """Filtrer, nettoyer et structurer l'ensemble des événements bruts."""
    raw_events = load_raw_events()
    # Exclusion des sources hors-sujet avant toute autre transformation.
    relevant_events = [e for e in raw_events if is_relevant(e)]
    structured_events = [structure_event(e) for e in relevant_events]
    # Villes déjà confirmées ailleurs dans le jeu de données, pour ne pas exclure à tort
    # un événement dont seul le code postal est mal saisi (voir is_complete()).
    known_cities = build_known_cities(structured_events)
    # Exclusion des événements incomplets, une fois structurés (voir is_complete()).
    complete_events = [e for e in structured_events if is_complete(e, known_cities)]

    print(
        f"{len(raw_events)} événements bruts, {len(relevant_events)} après filtrage de pertinence, "
        f"{len(complete_events)} après filtrage de complétude"
    )
    return complete_events


def main() -> None:
    """Point d'entrée : nettoyer les événements et écrire data/processed/events.json."""
    structured_events = preprocess()

    # Création du dossier de destination s'il n'existe pas encore (premier lancement du script).
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.write_text(
        json.dumps(structured_events, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"{len(structured_events)} événements sauvegardés dans {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
