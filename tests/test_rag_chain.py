"""Tests unitaires des fonctions pures de scripts/rag_chain.py : formatage de dates, sélection
de la prochaine occurrence, mise en forme du contexte transmis au LLM. build_chain(),
retrieve_context() et answer_question() ne sont PAS testées ici : elles appellent Mistral/FAISS
réellement, hors périmètre d'un test unitaire gratuit (couvertes indirectement par
notebooks/05_rag_chain_evaluation.ipynb et par api/main.py via lifespan()).
"""

import sys
from datetime import date
from pathlib import Path

import pytest
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from query_filters import QueryFilters  # noqa: E402
from rag_chain import build_period_note, format_date_fr, format_docs, format_event_date_fr, next_occurrence_date  # noqa: E402

# Dates de référence vérifiées par calcul dans la conversation du projet, pour des assertions
# déterministes sans dépendre du jour réel d'exécution des tests.
MONDAY = date(2026, 8, 31)
SATURDAY = date(2026, 9, 5)


# --- format_date_fr : jour de la semaine calculé en Python, jamais par le LLM ---------------


def test_format_date_fr_monday():
    """Date connue -> jour de semaine + mois en français, sans dépendre de la locale système."""
    assert format_date_fr(MONDAY) == "lundi 31 août 2026"


def test_format_date_fr_different_month_and_weekday():
    """Deuxième date (mois et jour de semaine différents) pour vérifier les deux tableaux
    JOURS_FR/MOIS_FR, pas seulement un seul indice de chacun."""
    assert format_date_fr(date(2026, 1, 1)) == "jeudi 1 janvier 2026"


# --- format_event_date_fr : date ISO d'événement -> français lisible, avec heure locale -----


def test_format_event_date_fr_none_input_returns_none():
    """Pas de date -> None, pas d'exception (contrairement à format_date_fr qui exige un objet date)."""
    assert format_event_date_fr(None) is None


def test_format_event_date_fr_empty_string_returns_none():
    """Chaîne vide -> None aussi, traitée comme "pas de date" (falsy)."""
    assert format_event_date_fr("") is None


def test_format_event_date_fr_formats_with_zero_padded_time():
    """Heure/minute à un seul chiffre -> doivent rester sur 2 caractères ("09h05", pas "9h5")."""
    assert format_event_date_fr("2026-09-05T09:05:00+02:00") == "samedi 5 septembre 2026 à 09h05"


def test_format_event_date_fr_formats_with_two_digit_time():
    """Heure/minute déjà sur 2 chiffres -> aucun padding superflu ajouté."""
    assert format_event_date_fr("2026-08-31T14:30:00+00:00") == "lundi 31 août 2026 à 14h30"


# --- next_occurrence_date : prochaine occurrence pertinente, pas systématiquement la 1ère ---


def test_next_occurrence_date_falls_back_to_date_start_when_no_occurrences():
    """Événement ponctuel (pas de champ "occurrences" ou liste vide) -> date_start telle quelle."""
    meta = {"date_start": "2026-09-05T10:00:00+00:00"}
    assert next_occurrence_date(meta, MONDAY) == "2026-09-05T10:00:00+00:00"
    assert next_occurrence_date({**meta, "occurrences": []}, MONDAY) == "2026-09-05T10:00:00+00:00"


def test_next_occurrence_date_falls_back_when_date_start_and_occurrences_both_missing():
    """Aucune des deux sources n'existe -> None, pas d'exception (KeyError évité par .get())."""
    assert next_occurrence_date({}, MONDAY) is None


def test_next_occurrence_date_picks_earliest_upcoming_not_list_order():
    """Plusieurs occurrences à venir, PAS dans l'ordre chronologique de la liste -> la fonction
    doit renvoyer la plus proche (min()), pas la première rencontrée dans la liste."""
    meta = {
        "occurrences": [
            {"start": "2026-10-03T10:00:00+02:00"},  # plus tardive, mais en premier dans la liste
            {"start": "2026-09-06T10:00:00+02:00"},  # la plus proche à venir
        ]
    }
    assert next_occurrence_date(meta, MONDAY) == "2026-09-06T10:00:00+02:00"


def test_next_occurrence_date_ignores_past_occurrences():
    """Une occurrence déjà passée par rapport à "today" ne doit jamais être choisie, même si
    elle est la seule présente dans la liste dans ce test — la future doit primer."""
    meta = {
        "occurrences": [
            {"start": "2026-01-01T10:00:00+00:00"},  # passée par rapport à MONDAY (31 août 2026)
            {"start": "2026-09-06T10:00:00+02:00"},  # à venir
        ]
    }
    assert next_occurrence_date(meta, MONDAY) == "2026-09-06T10:00:00+02:00"


def test_next_occurrence_date_falls_back_when_all_occurrences_are_past():
    """Toutes les occurrences sont passées -> repli sur date_start (comportement documenté,
    même si ce cas ne devrait normalement plus atteindre format_docs grâce au filtre de
    query_filters.py en amont)."""
    meta = {
        "date_start": "2026-01-01T10:00:00+00:00",
        "occurrences": [{"start": "2026-01-01T10:00:00+00:00"}],
    }
    assert next_occurrence_date(meta, MONDAY) == "2026-01-01T10:00:00+00:00"


def test_next_occurrence_date_ignores_entries_without_start():
    """Occurrence malformée (pas de clé "start") -> ignorée, pas d'exception."""
    meta = {
        "date_start": "2026-09-05T10:00:00+00:00",
        "occurrences": [{"end": "2026-09-06T12:00:00+02:00"}],  # pas de "start"
    }
    assert next_occurrence_date(meta, MONDAY) == "2026-09-05T10:00:00+00:00"


# --- build_period_note : résolution de la période en date(s) précise(s), donnée au LLM ------


def test_build_period_note_empty_when_no_period():
    """"aucune" -> aucune date à résoudre, pas de note ajoutée (chaîne vide)."""
    assert build_period_note(QueryFilters(period="aucune"), MONDAY) == ""


def test_build_period_note_single_day_period():
    """Période à un seul jour ("aujourd'hui" ou un jour de semaine précis) -> une seule date
    nommée, pas une plage "du ... au ..."."""
    note = build_period_note(QueryFilters(period="aujourd'hui"), MONDAY)
    assert note == "Précision : la période demandée dans la question correspond exactement à lundi 31 août 2026.\n\n"


def test_build_period_note_enumerates_every_day_of_a_range():
    """Point central du fix : pour une plage de plusieurs jours ("ce week-end"), chaque jour est
    listé nommément avec sa date ("aux dates suivantes : ..."), pas juste "du ... au ..." — pour
    que le LLM n'ait pas à déduire lui-même quel jour de la plage correspond au nom cité dans
    la question."""
    note = build_period_note(QueryFilters(period="ce_week_end"), MONDAY)
    assert "aux dates suivantes" in note
    assert "samedi 5 septembre 2026" in note
    assert "dimanche 6 septembre 2026" in note


def test_build_period_note_resolves_weekday_mentioned_alone():
    """Cas concret ayant motivé le fix : "mercredi" cité seul doit obtenir sa date précise."""
    note = build_period_note(QueryFilters(period="mercredi"), MONDAY)
    assert "mercredi 2 septembre 2026" in note


# --- format_docs : mise en forme du contexte transmis au LLM --------------------------------


def make_doc(**metadata) -> Document:
    """Construire un Document factice avec des métadonnées de base, personnalisables par test."""
    base = {
        "title": "Festival de Jazz",
        "date_start": "2026-09-05T20:00:00+02:00",
        "url": "https://openagenda.com/example",
    }
    base.update(metadata)
    return Document(page_content="Un concert en plein air.", metadata=base)


def test_format_docs_includes_title_date_and_link_always():
    """Titre, date et lien apparaissent toujours (pas de "if" les entourant dans format_docs)."""
    text = format_docs([make_doc()])
    assert "Titre : Festival de Jazz" in text
    # Pas de champ "occurrences" dans make_doc() -> repli sur date_start (5 septembre = samedi).
    assert "Date : samedi 5 septembre 2026 à 20h00" in text
    assert "Lien : https://openagenda.com/example" in text
    assert "Un concert en plein air." in text  # doc.page_content bien inclus


def test_format_docs_omits_optional_fields_when_absent():
    """Aucun champ optionnel renseigné -> aucune de leurs lignes n'apparaît (pas de "None" affiché)."""
    text = format_docs([make_doc()])
    for label in [
        "Lieu :", "Tarif :", "Établissement :", "Adresse :",
        "Téléphone :", "Site du lieu :", "Réseaux sociaux :", "Inscription :", "Accès en ligne :",
    ]:
        assert label not in text


@pytest.mark.parametrize(
    "field, label, value",
    [
        ("location_city", "Lieu :", "Marseille"),
        ("conditions", "Tarif :", "Gratuit"),
        ("location_phone", "Téléphone :", "0442443619"),
        ("location_website", "Site du lieu :", "https://example.com"),
        ("location_links", "Réseaux sociaux :", "https://facebook.com/example"),
        ("registration_link", "Inscription :", "https://example.com/inscription"),
        ("online_access_link", "Accès en ligne :", "https://example.com/live"),
    ],
)
def test_format_docs_includes_optional_field_when_present(field, label, value):
    """Chaque champ optionnel testé un par un : présent en métadonnée -> ligne correspondante
    affichée avec son libellé et sa valeur exacte."""
    text = format_docs([make_doc(**{field: value})])
    assert f"{label} {value}" in text


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"location_name": "Parc Longchamp", "location_district": "4e arrondissement"}, "Parc Longchamp — 4e arrondissement"),
        ({"location_name": "Parc Longchamp"}, "Parc Longchamp"),
        ({"location_district": "4e arrondissement"}, "4e arrondissement"),
    ],
)
def test_format_docs_etablissement_joins_name_and_district(overrides, expected):
    """"Établissement" combine nom + arrondissement séparés par un tiret cadratin, ou n'affiche
    que celui qui est renseigné si l'autre est absent."""
    text = format_docs([make_doc(**overrides)])
    assert f"Établissement : {expected}" in text


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"location_address": "12 rue de la Paix", "location_postalcode": "13001"}, "12 rue de la Paix, 13001"),
        ({"location_address": "12 rue de la Paix"}, "12 rue de la Paix"),
        ({"location_postalcode": "13001"}, "13001"),
    ],
)
def test_format_docs_adresse_joins_address_and_postalcode(overrides, expected):
    """"Adresse" combine adresse + code postal séparés par une virgule, ou n'affiche que celui
    qui est renseigné si l'autre est absent."""
    text = format_docs([make_doc(**overrides)])
    assert f"Adresse : {expected}" in text


def test_format_docs_separates_multiple_documents_with_marker():
    """Plusieurs documents sont séparés par "---" (délimiteur explicite pour le LLM), pas
    simplement concaténés."""
    text = format_docs([make_doc(title="Événement A"), make_doc(title="Événement B")])
    assert "\n\n---\n\n" in text
    assert "Événement A" in text
    assert "Événement B" in text


def test_format_docs_empty_list_returns_empty_string():
    """Cas limite : aucun document retrouvé -> chaîne vide, pas d'exception ni de séparateur orphelin."""
    assert format_docs([]) == ""
