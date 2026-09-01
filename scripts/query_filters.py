"""Extrait un filtre structuré (date, ville, âge, mode) depuis la question de l'utilisateur.

Sépare la compréhension du langage naturel (confiée au LLM, via with_structured_output) du
calcul des dates (fait en Python, déterministe) : un LLM s'est montré peu fiable pour
calculer une vraie plage de dates 
"""

from datetime import date, datetime, timedelta
from typing import Literal

from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

EXTRACTION_PROMPT = """Analyse cette question posée à un chatbot d'événements culturels \
et identifie les critères de filtrage qu'elle contient explicitement. Ne déduis rien qui \
n'est pas clairement exprimé — laisse un champ vide/None si la question ne le précise pas.

Question : {question}"""


class QueryFilters(BaseModel):
    """Critères de filtrage extraits d'une question en langage naturel."""

    period: Literal["aujourd'hui", "ce_week_end", "cette_semaine", "semaine_prochaine", "aucune"] = Field(
        default="aucune",
        description="Période mentionnée dans la question. 'aucune' si aucune période précise n'est demandée.",
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

    "Cette semaine"/"ce week-end" désignent toujours la semaine EN COURS (celle qui contient
    aujourd'hui), pas la suivante — qu'on soit lundi ou dimanche, "ce week-end" renvoie le
    même samedi/dimanche.
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
        return monday.isoformat(), sunday.isoformat()
    if period == "semaine_prochaine":
        next_monday = monday + timedelta(days=7)
        next_sunday = next_monday + timedelta(days=6)
        return next_monday.isoformat(), next_sunday.isoformat()
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
        conditions["location_city"] = {"$eq": filters.location_city}

    if filters.target_age is not None:
        conditions["age_min"] = {"$lte": filters.target_age}
        conditions["age_max"] = {"$gte": filters.target_age}

    if filters.attendance_mode:
        conditions["attendance_mode"] = {"$eq": filters.attendance_mode}

    return conditions


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
