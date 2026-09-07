"""Tests unitaires de l'extraction de filtres (scripts/query_filters.py) : calcul de périodes,
construction du filtre FAISS, vérification des occurrences — toute la logique déterministe
(pas d'appel LLM). extract_filters() n'est PAS testée ici : elle appelle Mistral (payant),
hors périmètre d'un test unitaire gratuit.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from query_filters import (  # noqa: E402
    QueryFilters,
    build_faiss_filter,
    normalize_location_city,
    occurrence_in_period,
    period_to_date_range,
)

# Date de référence fixe pour tous les tests : un LUNDI vérifié par calcul (voir la
# conversation du projet) — permet des assertions déterministes sans dépendre de la date
# réelle du jour où les tests sont lancés.
MONDAY = date(2026, 8, 31)
SUNDAY_SAME_WEEK = date(2026, 9, 6)  # dimanche de la MÊME semaine que MONDAY, pas la suivante
THURSDAY = date(2026, 9, 3)  # jeudi vérifié par calcul, pour tester le "bouclage" sur un jour déjà passé dans la semaine


# --- period_to_date_range : conversion période nommée -> plage de dates ISO -----------------


def test_period_aujourdhui_returns_today_twice():
    """"aujourd'hui" -> la même date en borne basse et haute."""
    assert period_to_date_range("aujourd'hui", MONDAY) == ("2026-08-31", "2026-08-31")


def test_period_ce_week_end_from_monday():
    """"ce week-end" calculé depuis un lundi -> le samedi/dimanche de CETTE semaine-là."""
    assert period_to_date_range("ce_week_end", MONDAY) == ("2026-09-05", "2026-09-06")


def test_period_ce_week_end_from_sunday_stays_same_weekend():
    """Point non-évident du docstring de period_to_date_range : "ce week-end" désigne toujours
    la semaine EN COURS. Demandé un dimanche (le dernier jour de sa propre semaine), la fonction
    doit renvoyer LE MÊME week-end (déjà en cours), pas sauter au week-end suivant."""
    assert period_to_date_range("ce_week_end", SUNDAY_SAME_WEEK) == ("2026-09-05", "2026-09-06")


def test_period_cette_semaine_from_monday():
    """"cette semaine" demandé un lundi -> du lundi (= aujourd'hui) au dimanche de la semaine."""
    assert period_to_date_range("cette_semaine", MONDAY) == ("2026-08-31", "2026-09-06")


def test_period_cette_semaine_from_thursday_starts_today_not_monday():
    """Borne basse = aujourd'hui, jamais le lundi de la semaine si celui-ci est déjà passé — un
    événement terminé lundi ou mardi ne doit pas remonter pour une question posée jeudi. Avant
    ce fix, la borne basse était toujours le lundi de la semaine, même déjà passé."""
    assert period_to_date_range("cette_semaine", THURSDAY) == ("2026-09-03", "2026-09-06")


def test_period_semaine_prochaine_from_monday():
    """"semaine prochaine" -> la semaine complète suivante (lundi à dimanche)."""
    assert period_to_date_range("semaine_prochaine", MONDAY) == ("2026-09-07", "2026-09-13")


def test_period_ce_mois_from_mid_month():
    """"ce mois" -> du jour actuel au dernier jour du mois en cours (30 septembre)."""
    assert period_to_date_range("ce_mois", THURSDAY) == ("2026-09-03", "2026-09-30")


def test_period_ce_mois_on_last_day_of_month_returns_single_day():
    """Demandé le dernier jour du mois (31 août) -> aujourd'hui pour les deux bornes, pas de
    plage vide ni d'erreur de calcul."""
    assert period_to_date_range("ce_mois", MONDAY) == ("2026-08-31", "2026-08-31")


def test_period_ce_mois_handles_december_year_rollover():
    """Décembre -> le mois suivant est janvier de l'ANNÉE SUIVANTE, cas particulier du calcul
    de "dernier jour du mois" (pas de mois 13)."""
    assert period_to_date_range("ce_mois", date(2026, 12, 15)) == ("2026-12-15", "2026-12-31")


def test_period_aucune_returns_none():
    """"aucune" -> None (aucune contrainte de date à appliquer)."""
    assert period_to_date_range("aucune", MONDAY) is None


def test_period_unknown_value_returns_none():
    """Une valeur qui ne correspond à aucun cas géré -> None aussi (comportement par défaut sûr),
    pas une exception. Défensif : QueryFilters (Literal) devrait déjà empêcher ce cas, mais
    period_to_date_range() ne dépend pas de ce garde-fou pour rester correcte."""
    assert period_to_date_range("valeur-inconnue", MONDAY) is None


def test_period_weekday_today_is_that_day_returns_today():
    """"lundi" demandé un lundi -> renvoie AUJOURD'HUI (le jour cité peut désigner le jour même,
    pas nécessairement une occurrence future) — pas dans 7 jours."""
    assert period_to_date_range("lundi", MONDAY) == ("2026-08-31", "2026-08-31")


def test_period_weekday_later_this_week():
    """"mercredi" demandé un lundi -> le mercredi de cette même semaine (2 jours plus tard)."""
    assert period_to_date_range("mercredi", MONDAY) == ("2026-09-02", "2026-09-02")


def test_period_weekday_wraps_to_next_week_when_already_passed():
    """"lundi" demandé un jeudi -> le PROCHAIN lundi (4 jours plus tard), jamais un lundi déjà
    passé cette semaine-là — le calcul doit "boucler en avant", pas renvoyer une date passée.
    Cas central du bug corrigé : sans ce test, un jour "déjà passé" pourrait resurgir en négatif."""
    assert period_to_date_range("lundi", THURSDAY) == ("2026-09-07", "2026-09-07")


def test_period_weekday_returns_single_day_range():
    """Un jour de semaine précis renvoie toujours une plage d'UN SEUL jour (borne basse ==
    borne haute), contrairement à "cette_semaine"/"ce_week_end" (plusieurs jours)."""
    start, end = period_to_date_range("vendredi", MONDAY)
    assert start == end


# --- build_faiss_filter : filtres extraits -> dictionnaire de filtre FAISS -------------------


def test_build_faiss_filter_empty_filters_only_excludes_past_events():
    """Aucun critère renseigné (period="aucune" par défaut) -> seul le garde-fou "pas d'événement
    déjà terminé" (date_end >= aujourd'hui) s'applique."""
    result = build_faiss_filter(QueryFilters(), MONDAY)
    assert result == {"date_end": {"$gte": "2026-08-31"}}


def test_build_faiss_filter_with_period_uses_overlap_not_date_start_alone():
    """Une période précise -> chevauchement (date_end >= début période ET date_start <= fin
    période + 23:59:59), pas un simple filtre sur date_start (cf. le fix des événements
    déjà commencés mais encore en cours)."""
    result = build_faiss_filter(QueryFilters(period="ce_week_end"), MONDAY)
    assert result == {
        "date_end": {"$gte": "2026-09-05"},
        "date_start": {"$lte": "2026-09-06T23:59:59"},
    }


def test_build_faiss_filter_with_weekday_period():
    """Un jour de semaine précis fonctionne avec build_faiss_filter() exactement comme les
    autres périodes — réutilisation générique de period_to_date_range(), pas de cas spécial
    ajouté dans build_faiss_filter() elle-même."""
    result = build_faiss_filter(QueryFilters(period="mercredi"), MONDAY)
    assert result == {
        "date_end": {"$gte": "2026-09-02"},
        "date_start": {"$lte": "2026-09-02T23:59:59"},
    }


def test_build_faiss_filter_adds_location_city_condition():
    """Ville renseignée -> condition d'égalité exacte ajoutée au filtre."""
    result = build_faiss_filter(QueryFilters(location_city="Marseille"), MONDAY)
    assert result["location_city"] == {"$eq": "Marseille"}


def test_build_faiss_filter_omits_location_city_when_absent():
    """Ville non renseignée -> pas de clé "location_city" du tout (pas de contrainte forcée à None)."""
    result = build_faiss_filter(QueryFilters(), MONDAY)
    assert "location_city" not in result


def test_normalize_location_city_fixes_case_mismatch():
    """Non-régression : "Aix-En-Provence" (casse reprise de la question par le LLM) doit être
    recalé sur "Aix-en-Provence" (casse réelle en base) — sinon $eq échoue à tort et le contexte
    ressort vide alors que de vrais événements existent (constaté empiriquement,
    eval/diagnose_hallucination.py)."""
    assert normalize_location_city("Aix-En-Provence") == "Aix-en-Provence"


def test_normalize_location_city_is_case_insensitive_both_ways():
    """Fonctionne quelle que soit la casse fournie, pas seulement tout-minuscule ou le cas
    précis rencontré."""
    assert normalize_location_city("MARSEILLE") == "Marseille"
    assert normalize_location_city("marseille") == "Marseille"


def test_normalize_location_city_passes_through_unknown_city():
    """Ville hors zone couverte (ex. "Paris") -> renvoyée telle quelle, pas d'erreur. Le filtre
    FAISS ne retournera simplement aucun résultat — comportement correct, pas un cas à corriger."""
    assert normalize_location_city("Paris") == "Paris"


def test_normalize_location_city_handles_none():
    assert normalize_location_city(None) is None


def test_build_faiss_filter_normalizes_location_city_case():
    """Non-régression bout en bout : build_faiss_filter() applique bien normalize_location_city()
    avant de construire la condition $eq."""
    result = build_faiss_filter(QueryFilters(location_city="Aix-En-Provence"), MONDAY)
    assert result["location_city"] == {"$eq": "Aix-en-Provence"}


def test_build_faiss_filter_adds_age_conditions():
    """Âge renseigné -> filtre imbriqué ($and de 3 blocs : conditions de base + $or age_min +
    $or age_max), pas des clés age_min/age_max à plat (voir test de non-régression plus bas
    sur pourquoi : $or age_min/age_max None nécessaire, pas juste $lte/$gte)."""
    result = build_faiss_filter(QueryFilters(target_age=8), MONDAY)
    age_min_block, age_max_block = result["$and"][1], result["$and"][2]
    assert age_min_block == {"$or": [{"age_min": {"$eq": None}}, {"age_min": {"$lte": 8}}]}
    assert age_max_block == {"$or": [{"age_max": {"$eq": None}}, {"age_max": {"$gte": 8}}]}


def test_build_faiss_filter_age_zero_is_not_treated_as_absent():
    """target_age=0 doit quand même déclencher le filtre — un bug classique serait "if
    filters.target_age:" (0 est faux en Python), alors que le code utilise "is not None"."""
    result = build_faiss_filter(QueryFilters(target_age=0), MONDAY)
    assert result["$and"][1] == {"$or": [{"age_min": {"$eq": None}}, {"age_min": {"$lte": 0}}]}
    assert result["$and"][2] == {"$or": [{"age_max": {"$eq": None}}, {"age_max": {"$gte": 0}}]}


def test_build_faiss_filter_omits_age_conditions_when_absent():
    """Âge non renseigné -> pas de $and, filtre à plat comme avant (comportement inchangé)."""
    result = build_faiss_filter(QueryFilters(), MONDAY)
    assert "$and" not in result
    assert "age_min" not in result
    assert "age_max" not in result


def test_build_faiss_filter_age_none_metadata_does_not_crash_faiss():
    """Non-régression : une question avec un âge précis plantait avec un TypeError côté FAISS
    ('<=' non supporté entre None et int) dès qu'un document avait age_min/age_max=None — le cas
    de LOIN le plus fréquent du dataset (4334/4502 événements, voir preprocess_events.py).
    Constaté en conditions réelles (eval/evaluate_rag.py, question "activité pour un enfant de
    8 ans"). Reproduit ici le VRAI mécanisme de filtre FAISS (FAISS._create_filter_func), pas une
    réimplémentation approximative, pour que ce test casse si FAISS change son comportement.
    """
    from langchain_community.vectorstores import FAISS

    result = build_faiss_filter(QueryFilters(target_age=8), MONDAY)
    filter_fn = FAISS._create_filter_func(result)

    base_metadata = {"date_end": "2099-01-01", "date_start": "2000-01-01"}
    assert filter_fn({**base_metadata, "age_min": None, "age_max": None}) is True  # tout public
    assert filter_fn({**base_metadata, "age_min": 5, "age_max": 12}) is True  # 8 ans inclus
    assert filter_fn({**base_metadata, "age_min": 12, "age_max": 18}) is False  # 8 ans exclu


def test_build_faiss_filter_adds_attendance_mode_condition():
    """Mode de participation renseigné -> condition d'égalité exacte ajoutée au filtre."""
    result = build_faiss_filter(QueryFilters(attendance_mode="En ligne"), MONDAY)
    assert result["attendance_mode"] == {"$eq": "En ligne"}


def test_build_faiss_filter_omits_attendance_mode_when_absent():
    """Mode de participation non renseigné -> pas de clé "attendance_mode" du tout."""
    result = build_faiss_filter(QueryFilters(), MONDAY)
    assert "attendance_mode" not in result


def test_build_faiss_filter_combines_all_criteria_together():
    """Tous les critères renseignés en même temps -> date/ville/mode dans le bloc de base ($and[0]),
    plus les 2 blocs $or age_min/age_max ($and[1] et [2]) — imbrication décrite dans
    build_faiss_filter() dès qu'un target_age est présent."""
    filters = QueryFilters(
        period="cette_semaine", location_city="Marseille", target_age=10, attendance_mode="Sur place"
    )
    result = build_faiss_filter(filters, MONDAY)
    assert set(result["$and"][0].keys()) == {"date_end", "date_start", "location_city", "attendance_mode"}
    assert "age_min" in result["$and"][1]["$or"][1]
    assert "age_max" in result["$and"][2]["$or"][1]


# --- occurrence_in_period : vérification fine des occurrences individuelles -----------------


def test_occurrence_in_period_true_when_no_period_requested():
    """Aucune période demandée -> toujours True, peu importe le contenu de meta (rien à filtrer)."""
    assert occurrence_in_period({}, QueryFilters(), MONDAY) is True


def test_occurrence_in_period_true_when_occurrences_missing():
    """Événement sans occurrences détaillées (cas normal, événement ponctuel) -> True, déjà
    filtré correctement en amont par build_faiss_filter() sur date_start/date_end."""
    meta = {"date_start": "2026-09-05T10:00:00+00:00"}  # pas de clé "occurrences"
    assert occurrence_in_period(meta, QueryFilters(period="ce_week_end"), MONDAY) is True


def test_occurrence_in_period_true_when_an_occurrence_falls_inside():
    """Cas normal : une occurrence tombe réellement dans la période demandée -> True."""
    meta = {"occurrences": [{"start": "2026-09-05T10:00:00+02:00"}]}  # samedi, dans "ce week-end"
    assert occurrence_in_period(meta, QueryFilters(period="ce_week_end"), MONDAY) is True


def test_occurrence_in_period_false_when_no_occurrence_falls_inside():
    """Cas central du fix : un événement récurrent dont les occurrences existent mais AUCUNE
    ne tombe dans la période demandée (ex. avril puis octobre pour une question sur ce week-end)."""
    meta = {"occurrences": [{"start": "2026-04-21T10:00:00+02:00"}, {"start": "2026-10-03T10:00:00+02:00"}]}
    assert occurrence_in_period(meta, QueryFilters(period="ce_week_end"), MONDAY) is False


def test_occurrence_in_period_true_when_only_one_of_several_falls_inside():
    """any() : une seule occurrence dans la période suffit, les autres peuvent être hors période."""
    meta = {
        "occurrences": [
            {"start": "2026-04-21T10:00:00+02:00"},  # hors période
            {"start": "2026-09-06T10:00:00+02:00"},  # dans "ce week-end"
        ]
    }
    assert occurrence_in_period(meta, QueryFilters(period="ce_week_end"), MONDAY) is True


def test_occurrence_in_period_boundaries_are_inclusive():
    """Une occurrence pile sur la borne de début ou de fin de période doit compter comme incluse
    (comparaisons <=, pas <)."""
    on_start_boundary = {"occurrences": [{"start": "2026-09-05T00:00:00+00:00"}]}
    on_end_boundary = {"occurrences": [{"start": "2026-09-06T23:59:00+00:00"}]}
    assert occurrence_in_period(on_start_boundary, QueryFilters(period="ce_week_end"), MONDAY) is True
    assert occurrence_in_period(on_end_boundary, QueryFilters(period="ce_week_end"), MONDAY) is True


def test_occurrence_in_period_ignores_entries_without_start():
    """Une occurrence malformée (pas de clé "start") ne doit pas faire planter la vérification —
    juste être ignorée."""
    meta = {"occurrences": [{"end": "2026-09-05T12:00:00+02:00"}]}  # pas de "start"
    assert occurrence_in_period(meta, QueryFilters(period="ce_week_end"), MONDAY) is False
