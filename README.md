# Projet9 — POC Chatbot RAG pour Puls-Events

POC d'un chatbot capable de répondre à des questions sur des événements culturels à venir, à partir des données de l'[API Open Agenda](https://openagenda.com/), en s'appuyant sur un système RAG (Retrieval-Augmented Generation) combinant recherche vectorielle (FAISS) et génération de réponse en langage naturel (Mistral) orchestrés via LangChain, exposé par une API REST (FastAPI).

## Structure du projet

```
notebooks/   notebooks d'exploration et de vérification (ex. 00_check_environment.ipynb)
scripts/     scripts du pipeline de données (récupération, nettoyage, indexation)
```

Les autres dossiers (`api/`, `tests/`, `docs/`) seront ajoutés au fil des étapes suivantes, au moment où ils seront réellement utilisés.

## Reproduction de l'environnement

Prérequis : [uv](https://docs.astral.sh/uv/) installé, Python géré automatiquement par uv (version pinnée dans `.python-version`).

```bash
git clone <url-du-repo>
cd Projet9_mission
uv sync
```

`uv sync` recrée l'environnement virtuel `.venv/` et installe exactement les versions verrouillées dans `uv.lock`.

### Clé API Mistral

```bash
cp .env.example .env
```

Puis renseigner votre propre clé dans `.env` (jamais commité) :

```
MISTRAL_API_KEY=votre_clé
```

### Vérifier l'installation

```bash
uv run jupyter notebook notebooks/00_check_environment.ipynb
```

Toutes les cellules doivent s'exécuter sans erreur (faiss, langchain_community, langchain_huggingface, mistralai, fastapi).

## Pipeline de données

Pas de reconstruction automatique de l'index pour ce POC : les scripts se lancent manuellement, dans l'ordre, avant de démarrer l'API.

```bash
uv run python scripts/fetch_events.py
```

Récupère les événements culturels des Bouches-du-Rhône de moins d'un an (source : [miroir Opendatasoft des données Open Agenda](https://public.opendatasoft.com/explore/dataset/evenements-publics-openagenda/), aucune clé API requise) et les sauvegarde dans `data/raw/events.json`.

```bash
uv run python scripts/preprocess_events.py
```

Nettoie et structure les événements bruts : exclut les sources hors-sujet (forums emploi France Travail, ~49% du volume brut), normalise la casse des noms de ville, extrait les champs utiles (dates, lieu, tarifs, âge, accessibilité, contact...) et construit le texte à vectoriser. Écrit le résultat dans `data/processed/events.json`.

**Répartition des champs bruts (56) vers `data/processed/events.json` (24 champs structurés + texte vectorisé) :**

| Champ | Où | Pourquoi |
|---|---|---|
| `uid`, `url`, `date_start`, `date_end` | Structuré | Identifiants/dates exacts, fiables |
| `status` | Les deux | Structuré pour filtrage exact ; ajouté au texte seulement si ≠ "Programmé" (annulé/reprogrammé), pour que le LLM puisse répondre "cet événement est-il maintenu ?" |
| `attendance_mode` | Structuré | Sur place / En ligne / Mixte — filtrage exact |
| `location_city`, `location_address`, `location_postalcode`, `location_district`, `location_insee` | Structuré | Géolocalisation — fiabilité imparfaite documentée, sans effet sur leur place ici |
| `location_phone`, `location_website`, `location_links`, `registration_link`, `online_access_link` | Structuré | Contact/pratique — aucun intérêt à vectoriser, juste à afficher tel quel |
| `age_min`, `age_max`, `accessibility_labels` | Structuré | Filtrage potentiel ("pour enfants ?", accessibilité) — gardés malgré une faible fréquence de remplissage |
| `conditions` (tarifs) | Les deux | Structuré pour affichage exact, ET ajouté au texte pour la recherche sémantique ("événements gratuits") |
| `keywords` | Les deux | Structuré (liste), ET ajoutés au texte pour renforcer le matching sémantique |
| `title` | Les deux | Affichage en métadonnée, ET ajouté en tête du texte vectorisé pour la correspondance sur le nom de l'événement |
| `description_fr` / `longdescription_fr` | Vectorisé | Contenu libre, recherche sémantique |

Exclus, avec une vraie raison à chaque fois :
- `category`, `country_fr`, `location_countrycode` — constants/vides sur tout le dataset, aucune information
- `contributor_email`/`contactnumber`/`contactname`/`contactposition` — données personnelles du contributeur, pas de place dans une donnée exposée publiquement
- `slug`, `location_uid`, `originagenda_uid`, `image*` — identifiants/médias internes à la plateforme, sans valeur pour répondre à une question

D'autres scripts (vectorisation/indexation) seront ajoutés ici au fil de l'avancement — cette section sera complétée à chaque nouveau script.

## Statut

Étape 1 — configuration de l'environnement (terminée).
Étape 2 — pré-processing des données Open Agenda (en cours : récupération faite, nettoyage et vectorisation à venir).
