"""Extrait un filtre structuré (date, ville, âge, mode) depuis la question de l'utilisateur.

Sépare la compréhension du langage naturel (confiée au LLM, via with_structured_output) du
calcul des dates (fait en Python, déterministe) : un LLM s'est montré peu fiable pour
calculer une vraie plage de dates 
"""

import json
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal

from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "events.json"

EXTRACTION_PROMPT = """Analyse cette question posée à un chatbot d'événements culturels \
et identifie les critères de filtrage qu'elle contient explicitement. Ne déduis rien qui \
n'est pas clairement exprimé — laisse un champ vide/None si la question ne le précise pas.

Question : {question}"""


# Un jour de semaine cité seul (ex. "mercredi", "ce samedi") ne correspondait auparavant à
# AUCUNE des catégories ci-dessous — extract_filters() retombait alors sur "aucune", et le LLM
# de génération devait calculer lui-même la date de ce jour, de façon non fiable (constaté
# empiriquement : juste une fois sur deux, sans garantie). D'où l'ajout des 7 jours nommément.
JOURS_SEMAINE = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]


class QueryFilters(BaseModel):
    """Critères de filtrage extraits d'une question en langage naturel."""

    period: Literal[
        "aujourd'hui", "ce_week_end", "cette_semaine", "semaine_prochaine", "ce_mois",
        "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
        "aucune",
    ] = Field(
        default="aucune",
        description=(
            "Période mentionnée dans la question. Utilise le nom du jour précis ('lundi' à "
            "'dimanche') si la question cite UN jour de la semaine en particulier (ex. "
            "'mercredi', 'ce samedi'), plutôt que 'ce_week_end' ou 'cette_semaine' qui couvrent "
            "plusieurs jours. 'ce_mois' pour 'ce mois-ci'/'ce mois'. 'aucune' si aucune période "
            "précise n'est demandée."
        ),
    )
    location_city: str | None = Field(
        default=None,
        description=(
            "Ville mentionnée dans la question (première lettre en majuscule, ex. 'Marseille'). "
            "None si aucune ville n'est mentionnée."
        ),
    )
    target_age: int | None = Field(
        default=None,
        description=(
            "Âge précis de la personne concernée, UNIQUEMENT si un âge ou une tranche d'âge exacte "
            "est donnée dans la question (ex. '8 ans', 'moins de 12 ans'). None si seulement un terme "
            "vague comme 'enfants' est utilisé sans âge précis."
        ),
    )
    attendance_mode: Literal["Sur place", "En ligne", "Mixte"] | None = Field(
        default=None,
        description=(
            "Mode de participation demandé, uniquement si explicitement mentionné "
            "(ex. 'en ligne', 'un webinaire', 'sur place'). None sinon."
        ),
    )


@lru_cache(maxsize=1)
def _known_cities() -> dict[str, str]:
    """Associer chaque ville en minuscules à sa casse canonique dans les données (ex.
    "aix-en-provence" -> "Aix-en-Provence"), pour normalize_location_city() ci-dessous. Chargé
    une seule fois (lru_cache) : la liste des villes ne change qu'après un nouveau
    preprocess_events.py, jamais entre deux questions.
    """
    events = json.loads(PROCESSED_PATH.read_text(encoding="utf-8"))
    return {city.lower(): city for e in events if (city := e.get("location_city"))}


def normalize_location_city(city: str | None) -> str | None:
    """Recaler la ville extraite par extract_filters() sur sa casse exacte dans les données.

    extract_filters() reprend souvent la casse telle qu'écrite dans la question (ex. l'utilisateur
    tape "Aix-En-Provence", le LLM extrait "Aix-En-Provence") — jamais passée par
    preprocess_events.py::normalize_city(), qui ne s'exécute qu'une fois côté données, pas ici.
    build_faiss_filter() compare ensuite avec $eq (chaînes strictement égales) : "Aix-En-Provence"
    != "Aix-en-Provence" (e minuscule en base) fait échouer le filtre à tort, alors que de vrais
    événements existent — constaté empiriquement (eval/diagnose_hallucination.py, contexte vide
    à tort pour une question sur Aix-en-Provence). Pas de simple .title() ici : Python capitalise
    après chaque tiret, "aix-en-provence".title() redonne "Aix-En-Provence" — exactement le bug.

    Ville non reconnue (hors zone couverte, ex. "Paris") : renvoyée telle quelle, le filtre FAISS
    ne retournera simplement aucun résultat — comportement correct, pas une erreur à corriger ici.
    """
    if not city:
        return city
    return _known_cities().get(city.lower(), city)


def extract_filters(question: str, api_key: str) -> QueryFilters:
    """Demander au LLM d'identifier les critères de filtrage présents dans la question.

    Utilise mistral-small-latest (tâche d'extraction simple) plutôt que mistral-large-latest
    (réservé à la génération de la réponse finale, dans rag_chain.py) — moins coûteux,
    suffisant pour une extraction structurée. temperature=0 : réponse la plus déterministe
    possible, pas de créativité recherchée sur une tâche d'extraction.
    """
    extractor = ChatMistralAI(mistral_api_key=api_key, model="mistral-small-latest", temperature=0)
    structured_extractor = extractor.with_structured_output(QueryFilters)
    return structured_extractor.invoke(EXTRACTION_PROMPT.format(question=question))


def period_to_date_range(period: str, today: date) -> tuple[str, str] | None:
    """Convertir une période nommée en plage de dates ISO, par calcul déterministe (pas par le LLM).

    "Cette semaine"/"ce week-end"/"ce mois" désignent toujours la période EN COURS (celle qui
    contient aujourd'hui), pas la suivante — qu'on soit lundi ou dimanche, "ce week-end" renvoie
    le même samedi/dimanche.

    Borne basse toujours >= today, jamais le début "théorique" de la période (ex. le lundi de la
    semaine courante) : un événement déjà terminé avant aujourd'hui n'est jamais une réponse
    valable, même s'il tombe dans la période nommée (ex. "cette semaine" un jeudi ne doit pas
    faire remonter un événement terminé lundi). Même principe que le filtre par défaut de
    build_faiss_filter() quand aucune période n'est précisée.
    """
    monday = today - timedelta(days=today.weekday())

    if period == "aujourd'hui":
        return today.isoformat(), today.isoformat()
    if period == "ce_week_end":
        saturday = monday + timedelta(days=5)
        sunday = monday + timedelta(days=6)
        return saturday.isoformat(), sunday.isoformat()
    if period == "cette_semaine":
        sunday = monday + timedelta(days=6)
        return today.isoformat(), sunday.isoformat()
    if period == "semaine_prochaine":
        next_monday = monday + timedelta(days=7)
        next_sunday = next_monday + timedelta(days=6)
        return next_monday.isoformat(), next_sunday.isoformat()
    if period == "ce_mois":
        if today.month == 12:
            first_of_next_month = date(today.year + 1, 1, 1)
        else:
            first_of_next_month = date(today.year, today.month + 1, 1)
        last_day_of_month = first_of_next_month - timedelta(days=1)
        return today.isoformat(), last_day_of_month.isoformat()
    if period in JOURS_SEMAINE:
        # Prochaine occurrence de ce jour, AUJOURD'HUI INCLUS si "today" tombe déjà sur ce
        # jour-là ("ce mercredi" dit un mercredi désigne ce jour-même, pas dans 7 jours).
        target_weekday = JOURS_SEMAINE.index(period)
        days_ahead = (target_weekday - today.weekday()) % 7
        target_date = today + timedelta(days=days_ahead)
        return target_date.isoformat(), target_date.isoformat()
    return None


def build_faiss_filter(filters: QueryFilters, today: date) -> dict:
    """Convertir les filtres extraits en dictionnaire de filtre FAISS ($gte/$lte/$eq combinés).

    Ne renvoie que les conditions réellement déduites de la question — un filtre vide {}
    laisse la recherche sémantique agir seule, sans restriction supplémentaire.

    date_start/date_end sont stockés en métadonnée sous forme de chaîne ISO complète (ex.
    "2026-08-29T19:30:00+00:00") : les comparer à une simple date ISO ("2026-08-29") fonctionne
    correctement en tant que chaînes de caractères (l'ordre alphabétique suit l'ordre
    chronologique pour ce format), à condition que toutes les dates du jeu de données utilisent
    le même format de fuseau horaire — c'est le cas ici (pipeline unique, source unique).

    Filtre sur un CHEVAUCHEMENT (date_end >= début période ET date_start <= fin période), pas
    seulement sur date_start : un événement déjà commencé avant la période mais encore en
    cours pendant celle-ci (ex. une exposition sur plusieurs semaines) doit rester inclus.
    date_end est toujours renseigné dans data/processed/events.json (vérifié empiriquement,
    0/4241 manquants), donc pas de risque de comparaison avec une valeur manquante ici.

    Si la question ne précise AUCUNE période, on filtre quand même sur date_end >= aujourd'hui :
    un événement déjà terminé n'est jamais une réponse valable, même à une question sans date
    explicite — ne pas laisser ce cas au LLM, qui s'est montré peu fiable pour comparer une
    date à aujourd'hui pendant la génération (ex. confondre un jour du mois similaire avec
    aujourd'hui, malgré current_date fourni dans le prompt).
    """
    conditions: dict = {}

    date_range = period_to_date_range(filters.period, today)
    if date_range:
        date_gte, date_lte = date_range
        conditions["date_end"] = {"$gte": date_gte}
        # "T23:59:59" sur la borne haute : inclut toute la journée de fin, pas seulement minuit.
        conditions["date_start"] = {"$lte": f"{date_lte}T23:59:59"}
    else:
        conditions["date_end"] = {"$gte": today.isoformat()}

    if filters.location_city:
        conditions["location_city"] = {"$eq": normalize_location_city(filters.location_city)}

    if filters.attendance_mode:
        conditions["attendance_mode"] = {"$eq": filters.attendance_mode}

    if filters.target_age is None:
        return conditions

    # age_min/age_max valent None pour la grande majorité des événements (4334/4502 mesurés sur
    # data/processed/events.json), pas 0/110 comme le documente à tort preprocess_events.py —
    # None doit se lire comme "tout public", pas comme absence de valeur à comparer : un simple
    # {"age_min": {"$lte": age}} plante avec un TypeError côté FAISS (comparaison None <= int,
    # langchain_community/vectorstores/faiss.py::filter_fn, aucun garde-fou intégré), constaté
    # empiriquement sur une vraie question posant un âge précis. D'où le $or explicite ci-dessous :
    # un événement passe le filtre s'il n'a pas de restriction d'âge, OU si l'âge demandé y entre.
    age = filters.target_age
    return {
        "$and": [
            conditions,
            {"$or": [{"age_min": {"$eq": None}}, {"age_min": {"$lte": age}}]},
            {"$or": [{"age_max": {"$eq": None}}, {"age_max": {"$gte": age}}]},
        ]
    }


def occurrence_in_period(meta: dict, filters: QueryFilters, today: date) -> bool:
    """Vérifier, pour un événement récurrent, qu'une occurrence réelle tombe dans la période
    demandée — pas seulement que la plage date_start/date_end s'y superpose.

    build_faiss_filter() ne compare que date_start/date_end (première/dernière occurrence) :
    un événement avec un grand écart entre deux occurrences (ex. avril puis octobre) passerait
    à tort ce filtre pour une question sur juin. Ici, on revérifie sur meta["occurrences"]
    (liste des créneaux individuels, construite par preprocess_events.py depuis le champ
    timings) si l'un d'eux tombe réellement dans la période. Retourne True sans vérifier
    davantage si l'événement n'a pas d'occurrences détaillées (cas normal, événement ponctuel,
    déjà correctement filtré par build_faiss_filter) ou si aucune période n'est demandée.
    """
    date_range = period_to_date_range(filters.period, today)
    if not date_range:
        return True

    occurrences = meta.get("occurrences") or []
    if not occurrences:
        return True

    period_start = date.fromisoformat(date_range[0])
    period_end = date.fromisoformat(date_range[1])
    return any(
        period_start <= datetime.fromisoformat(o["start"]).date() <= period_end
        for o in occurrences
        if o.get("start")
    )
