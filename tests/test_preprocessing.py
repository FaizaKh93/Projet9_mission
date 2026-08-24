"""Tests unitaires du nettoyage/structuration des événements (scripts/preprocess_events.py).

Valide la logique de transformation sur des événements factices, sans dépendre de
data/raw/events.json (gitignoré, absent tant que fetch_events.py n'a pas tourné).
"""

import sys
from pathlib import Path

# Ajout de scripts/ au chemin d'import : ce n'est pas un package installé, juste un dossier de scripts.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

import preprocess_events  # noqa: E402 (import après la manipulation de sys.path, nécessaire ici)
from preprocess_events import (  # noqa: E402
    build_known_cities,
    build_text,
    is_complete,
    is_relevant,
    normalize_city,
    parse_labeled_field,
    parse_registration_link,
    structure_event,
)


def make_event(**overrides) -> dict:
    """Construire un événement brut minimal et réaliste, personnalisable pour chaque test."""
    event = {
        "uid": "12345",
        "title_fr": "Festival de Jazz",
        "description_fr": "Un concert en plein air.",
        "longdescription_fr": "",
        "originagenda_title": "Ville de Marseille",
        "status": '{"id": 1, "label": {"fr": "Programmé", "en": "Scheduled"}}',
        "attendancemode": '{"id": 1, "label": {"fr": "Sur place"}}',
        "conditions_fr": "Gratuit",
        "keywords_fr": ["jazz", "concert"],
        "age_min": 0,
        "age_max": 110,
        "accessibility_label_fr": [],
        "firstdate_begin": "2026-07-01T20:00:00+00:00",
        "lastdate_end": "2026-07-01T22:00:00+00:00",
        "location_name": "Parc Longchamp",
        "location_city": "Marseille",
        "location_address": "Boulevard Longchamp, Marseille",
        "location_postalcode": "13001",
        "location_district": None,
        "location_insee": "13055",
        "location_phone": None,
        "location_website": None,
        "location_links": None,
        "registration": None,
        "onlineaccesslink": None,
        "canonicalurl": "https://openagenda.com/example",
    }
    # Écrasement des valeurs par défaut par celles fournies au test (ex. make_event(location_city="MARSEILLE")).
    event.update(overrides)
    return event


# --- is_relevant : exclusion des événements hors-sujet (France Travail) ---


@pytest.mark.parametrize(
    "originagenda_title, expected",
    [
        pytest.param("Mes événements France Travail", False, id="excludes_france_travail"),
        pytest.param("Ville de Marseille", True, id="keeps_other_sources"),
    ],
)
def test_is_relevant(originagenda_title, expected):
    """Rejeter uniquement les événements dont la source est 'Mes événements France Travail'."""
    assert is_relevant(make_event(originagenda_title=originagenda_title)) is expected


# --- normalize_city : correction de la casse tout-majuscule ---


@pytest.mark.parametrize(
    "city, expected",
    [
        pytest.param("MARSEILLE", "Marseille", id="fixes_all_caps"),
        # "Aix-en-Provence" n'est pas tout-majuscule : ne doit pas être touché par .title(),
        # qui produirait "Aix-En-Provence" (majuscule incorrecte sur "En").
        pytest.param("Aix-en-Provence", "Aix-en-Provence", id="keeps_correct_mixed_case"),
        pytest.param(None, None, id="handles_none"),
        pytest.param("", "", id="handles_empty_string"),
    ],
)
def test_normalize_city(city, expected):
    """Corriger la casse tout-majuscule sans abîmer un nom déjà correct ni planter sur une valeur absente."""
    assert normalize_city(city) == expected


# --- parse_labeled_field : extraction du libellé français depuis un champ JSON imbriqué ---


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        pytest.param(
            '{"id": 6, "label": {"fr": "Annulé", "en": "Canceled"}}', "Annulé", id="extracts_french_label"
        ),
        pytest.param(None, None, id="handles_missing_value"),
        pytest.param("pas du json", None, id="handles_malformed_json"),
    ],
)
def test_parse_labeled_field(raw_value, expected):
    """Extraire le libellé français, ou renvoyer None proprement (absent / JSON invalide)."""
    assert parse_labeled_field(raw_value) == expected


# --- parse_registration_link : extraction du lien d'inscription depuis un champ JSON imbriqué ---


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        pytest.param(
            '[{"type": "link", "value": "https://example.com/inscription"}]',
            "https://example.com/inscription",
            id="extracts_first_link",
        ),
        pytest.param(None, None, id="handles_missing_value"),
        pytest.param("pas du json", None, id="handles_malformed_json"),
    ],
)
def test_parse_registration_link(raw_value, expected):
    """Extraire le premier lien d'inscription, ou renvoyer None proprement (absent / JSON invalide)."""
    assert parse_registration_link(raw_value) == expected


# --- build_text : construction du texte à vectoriser ---


def test_build_text_includes_title_and_description():
    """Inclure le titre et la description dans le texte assemblé."""
    event = make_event(title_fr="Festival de Jazz", description_fr="Un concert en plein air.")
    text = build_text(event)
    assert "Festival de Jazz" in text
    assert "Un concert en plein air." in text


def test_build_text_mentions_cancelled_status():
    """Mentionner explicitement le statut dans le texte quand l'événement est annulé."""
    event = make_event(status='{"id": 6, "label": {"fr": "Annulé"}}')
    text = build_text(event)
    assert "Annulé" in text


def test_build_text_omits_status_when_scheduled():
    """Omettre toute mention de statut quand l'événement suit son cours normal (Programmé)."""
    event = make_event()  # statut "Programmé" par défaut, ne doit pas être mentionné.
    text = build_text(event)
    assert "Statut" not in text


def test_build_text_includes_conditions_and_keywords():
    """Inclure les tarifs/conditions et les mots-clés dans le texte, pour enrichir la recherche sémantique."""
    event = make_event(conditions_fr="Gratuit", keywords_fr=["jazz", "plein air"])
    text = build_text(event)
    assert "Gratuit" in text
    assert "jazz" in text


def test_build_text_skips_duplicate_longdescription():
    """Ne pas répéter longdescription_fr si son contenu est identique à description_fr."""
    event = make_event(description_fr="Un concert en plein air.", longdescription_fr="Un concert en plein air.")
    text = build_text(event)
    assert text.count("Un concert en plein air.") == 1


def test_build_text_handles_all_fields_empty():
    """Renvoyer une chaîne vide, sans erreur, quand titre/description/longdescription sont tous vides."""
    event = make_event(title_fr="", description_fr="", longdescription_fr="", conditions_fr="", keywords_fr=[])
    assert build_text(event) == ""


# --- structure_event : mise en forme finale de l'événement ---


def test_structure_event_has_expected_keys():
    """Produire exactement les 24 champs attendus, ni plus ni moins."""
    structured = structure_event(make_event())
    expected_keys = {
        "uid", "title", "text", "status", "attendance_mode",
        "date_start", "date_end", "conditions", "age_min", "age_max",
        "accessibility_labels", "keywords", "location_name", "location_city",
        "location_address", "location_postalcode", "location_district",
        "location_insee", "location_phone", "location_website", "location_links",
        "registration_link", "online_access_link", "url",
    }
    assert set(structured.keys()) == expected_keys


def test_structure_event_normalizes_city():
    """Appliquer la normalisation de casse à location_city lors de la structuration complète."""
    structured = structure_event(make_event(location_city="MARSEILLE"))
    assert structured["location_city"] == "Marseille"


# --- build_known_cities : villes déjà confirmées ailleurs dans le jeu de données ---


def test_build_known_cities_keeps_only_cities_with_correct_postalcode():
    """Ne retenir que les villes associées à un code postal 13xxx quelque part dans les données."""
    structured_events = [
        structure_event(make_event(location_city="Marseille", location_postalcode="13001")),
        structure_event(make_event(location_city="Landerneau", location_postalcode="29800")),
    ]
    assert build_known_cities(structured_events) == {"Marseille"}


# --- is_complete : exclusion des événements incomplets ou mal géocodés ---


@pytest.mark.parametrize(
    "overrides, known_cities, expected",
    [
        pytest.param({}, set(), True, id="accepts_valid_event"),
        pytest.param(
            {
                "title_fr": "", "description_fr": "", "longdescription_fr": "",
                "conditions_fr": "", "keywords_fr": [],
            },
            set(),
            False,
            id="rejects_empty_text",
        ),
        pytest.param({"firstdate_begin": None}, set(), False, id="rejects_missing_date"),
        pytest.param({"uid": None}, set(), False, id="rejects_missing_uid"),
        # Cas réel rencontré : un événement à Landerneau (Finistère, code postal 29800),
        # avec location_department indiquant à tort "Bouches-du-Rhône", et une ville jamais
        # confirmée ailleurs dans les données (known_cities vide) : bien exclu.
        pytest.param(
            {"location_postalcode": "29800", "location_city": "Landerneau"},
            set(),
            False,
            id="rejects_postalcode_outside_target_department_when_city_unknown",
        ),
        # Cas réel rencontré : un événement à Marseille (ville bien connue par ailleurs dans
        # les données) mais avec un code postal aberrant ("10003", pas même un vrai code
        # marseillais) — on privilégie la ville, déjà confirmée, plutôt que ce code isolé.
        pytest.param(
            {"location_postalcode": "10003", "location_city": "Marseille"},
            {"Marseille"},
            True,
            id="accepts_bad_postalcode_when_city_already_known",
        ),
        pytest.param({"location_postalcode": None}, set(), True, id="accepts_missing_postalcode"),
        pytest.param(
            {"location_name": None, "location_address": None, "location_city": None},
            set(),
            False,
            id="rejects_in_person_event_without_any_location",
        ),
        pytest.param(
            {
                "attendancemode": '{"id": 2, "label": {"fr": "En ligne"}}',
                "onlineaccesslink": "https://example.com/live",
                "location_name": None, "location_address": None, "location_city": None,
            },
            set(),
            True,
            id="accepts_online_event_without_location",
        ),
        pytest.param(
            {
                "attendancemode": '{"id": 2, "label": {"fr": "En ligne"}}',
                "onlineaccesslink": None, "registration": None,
            },
            set(),
            False,
            id="rejects_online_event_without_access_link",
        ),
        pytest.param(
            {
                "attendancemode": '{"id": 2, "label": {"fr": "En ligne"}}',
                "onlineaccesslink": None,
                "registration": '[{"type": "link", "value": "https://example.com/inscription"}]',
            },
            set(),
            True,
            id="accepts_online_event_with_registration_link_only",
        ),
    ],
)
def test_is_complete(overrides, known_cities, expected):
    """Vérifier chaque règle primaire (texte, date, uid, code postal, localisation/lien d'accès)."""
    assert is_complete(structure_event(make_event(**overrides)), known_cities) is expected


# --- preprocess : comportement d'ensemble du pipeline de nettoyage ---


def test_preprocess_end_to_end(monkeypatch):
    """Vérifier le comportement d'ensemble : exclusion France Travail + normalisation, sur un petit jeu factice."""
    raw_events = [
        make_event(uid="1", originagenda_title="Ville de Marseille"),
        make_event(uid="2", originagenda_title="Mes événements France Travail"),
        make_event(uid="3", originagenda_title="Ville d'Istres", location_city="ISTRES"),
    ]
    # Remplacement de load_raw_events par une fonction renvoyant le jeu factice ci-dessus,
    # pour tester preprocess() sans dépendre d'un vrai fichier data/raw/events.json.
    monkeypatch.setattr(preprocess_events, "load_raw_events", lambda: raw_events)

    structured = preprocess_events.preprocess()

    # Événement 2 (France Travail) absent du résultat ; 1 et 3 conservés.
    assert {e["uid"] for e in structured} == {"1", "3"}
    istres_event = next(e for e in structured if e["uid"] == "3")
    assert istres_event["location_city"] == "Istres"
