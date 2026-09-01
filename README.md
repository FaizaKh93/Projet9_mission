# Projet9 — POC Chatbot RAG pour Puls-Events

POC d'un chatbot capable de répondre à des questions sur des événements culturels à venir, à partir des données de l'[API Open Agenda](https://openagenda.com/), en s'appuyant sur un système RAG (Retrieval-Augmented Generation) combinant recherche vectorielle (FAISS) et génération de réponse en langage naturel (Mistral) orchestrés via LangChain, exposé par une API REST (FastAPI).

## Structure du projet

```
notebooks/   notebooks d'exploration et de vérification (ex. 00_check_environment.ipynb)
scripts/     scripts du pipeline de données (récupération, nettoyage, indexation)
api/         API REST FastAPI exposant le système RAG (Étape 5)
tests/       tests automatisés (pytest) : préprocessing et API
```

Le dossier `docs/` sera ajouté au fil des étapes suivantes, au moment où il sera réellement utilisé.

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

Pour utiliser l'API (voir plus bas), renseigner aussi `X_API_KEY` dans ce même `.env` — une valeur de VOTRE choix (pas fournie par Mistral), servant de mot de passe pour déclencher `POST /rebuild`. Sans elle, `POST /rebuild` répond systématiquement `401 Unauthorized` (voir `verify_admin_key()` dans `api/main.py`) :

```
X_API_KEY=votre_secret_au_choix
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

Nettoie et structure les événements bruts : exclut les sources hors-sujet (forums emploi France Travail, ~49% du volume brut) et les événements incomplets ou mal géocodés (texte vide, date/uid manquants, code postal hors Bouches-du-Rhône, présentiel sans aucune localisation, en ligne sans lien d'accès), normalise la casse des noms de ville, extrait les champs utiles (dates, lieu, tarifs, âge, accessibilité, contact...) et construit le texte à vectoriser. Écrit le résultat dans `data/processed/events.json`.

**Répartition des champs bruts (56) vers `data/processed/events.json` (25 champs structurés + texte vectorisé) :**

| Champ | Où | Pourquoi |
|---|---|---|
| `uid`, `url`, `date_start`, `date_end` | Structuré | Identifiants/dates exacts, fiables |
| `occurrences` | Structuré | Liste des créneaux individuels d'un événement récurrent (extraite du champ brut `timings`) — `date_start`/`date_end` ne donnent que la 1ère et la dernière occurrence, insuffisant pour savoir laquelle est encore à venir ou si une occurrence tombe réellement dans une période demandée (voir `next_occurrence_date()` et `occurrence_in_period()`, plus bas) |
| `status` | Les deux | Structuré pour filtrage exact ; ajouté au texte seulement si ≠ "Programmé" (annulé/reprogrammé), pour que le LLM puisse répondre "cet événement est-il maintenu ?" |
| `attendance_mode` | Structuré | Sur place / En ligne / Mixte — filtrage exact (recherche hybride, voir Étape 4) |
| `location_city` | Structuré | Filtrage exact (recherche hybride) + affiché |
| `location_name`, `location_district` | Les deux | Noms propres qu'une question peut citer directement (ex. "à la médiathèque Louis Aragon"), utiles à la recherche sémantique — ET affichés (bloc "Établissement") pour préciser le lieu au-delà de la seule ville |
| `location_address`, `location_postalcode`, `location_phone`, `location_website`, `location_links`, `registration_link`, `online_access_link` | Structuré | Contact/pratique — affichés tels quels dans le contexte transmis au LLM (`format_docs()`) quand renseignés, jamais vectorisés (aucune valeur de recherche sémantique) |
| `location_insee` | Structuré | Vérification interne de la fiabilité du géocodage au nettoyage (`build_known_cities()`) — jamais affiché ni vectorisé, aucun intérêt pour répondre à une question |
| `age_min`, `age_max` | Structuré | Filtrage exact ("pour un enfant de 8 ans ?") — gardés malgré une faible fréquence de remplissage |
| `accessibility_labels` | Vectorisé | Intégré au texte plutôt qu'en filtre structuré : c'est une métadonnée en LISTE, et le filtre FAISS (`$in`) ne sait vérifier qu'une valeur scalaire parmi une liste acceptée, pas l'inverse ("cette liste contient-elle X ?") — la recherche sémantique sur une formulation libre ("accessible en fauteuil roulant") est plus simple à ce stade |
| `conditions` (tarifs) | Les deux | Structuré pour affichage exact, ET ajouté au texte pour la recherche sémantique ("événements gratuits") |
| `keywords` | Les deux | Structuré (liste), ET ajoutés au texte pour renforcer le matching sémantique |
| `title` | Les deux | Affichage en métadonnée, ET ajouté en tête du texte vectorisé pour la correspondance sur le nom de l'événement |
| `description_fr` / `longdescription_fr` | Vectorisé | Contenu libre, recherche sémantique |

Exclus, avec une vraie raison à chaque fois :
- `category`, `country_fr`, `location_countrycode` — constants/vides sur tout le dataset, aucune information
- `contributor_email`/`contactnumber`/`contactname`/`contactposition` — données personnelles du contributeur, pas de place dans une donnée exposée publiquement
- `slug`, `location_uid`, `originagenda_uid`, `image*` — identifiants/médias internes à la plateforme, sans valeur pour répondre à une question

```bash
uv run python scripts/vectorize_events.py
```

Découpe le texte de chaque événement en chunks (`langchain_text_splitters`, les textes courts ne produisent le plus souvent qu'un seul chunk) et génère leurs embeddings via l'API Mistral (`mistral-embed`, payant — nécessite `MISTRAL_API_KEY` dans `.env`). Sauvegarde le résultat dans `data/vectors/events_vectors.json` — **clôture l'Étape 2** ("prêt à être indexé"), sans construire l'index FAISS lui-même (Étape 3). Coût estimé sur ce dataset : ~0,08 $ pour 4241 événements.

```bash
uv run python scripts/build_index.py
```

Étape 3 — construit l'index vectoriel FAISS à partir des vecteurs déjà calculés par `vectorize_events.py` (`FAISS.from_embeddings()`, aucun nouvel appel à Mistral), sauvegardé dans `data/index/` (`index.faiss` + `index.pkl`). Vérifie que le nombre de vecteurs indexés correspond au nombre de vecteurs chargés.

```bash
uv run jupyter notebook notebooks/04_search_evaluation.ipynb
```

Charge l'index et teste 5 questions représentatives (thème+lieu, filtre prix implicite, thème culturel différent, statut "Annulé", requête hors-sujet) pour vérifier la pertinence des résultats — la demande explicite du brief ("tests de recherche pour vérifier l'efficacité").

Étape 4 — `scripts/query_filters.py` + `scripts/rag_chain.py` assemblent la chaîne RAG complète (recherche + génération). La recherche est **hybride** : une recherche purement sémantique ne sait pas comparer une date à aujourd'hui, ce qui produisait de mauvaises réponses sur des questions temporelles ("ce week-end" renvoyait des événements passés ou le mauvais week-end). `query_filters.py` extrait de la question, via `mistral-small-latest` (`with_structured_output`), les critères explicitement exprimés — période, ville, âge, mode de participation — puis `rag_chain.py` convertit ces critères en filtre FAISS (`similarity_search(question, filter=..., fetch_k=<taille totale de l'index>)`), appliqué **en plus de** la recherche sémantique, pas à sa place :
- la période ("ce week-end", "cette semaine"...) est convertie en plage de dates par calcul Python déterministe (`timedelta`), jamais par le LLM — un LLM s'est montré peu fiable pour calculer une vraie plage de dates ;
- `fetch_k` est monté à la taille totale de l'index car le filtre FAISS s'applique *après* la recherche par similarité sur les `fetch_k` voisins les plus proches (pas un pré-filtre) — sans cela, le filtre pourrait n'avoir aucun événement valide à examiner ;
- un filtre vide laisse la recherche sémantique agir seule, sans restriction ;
- le filtre sur les dates compare `date_end`/`date_start` par CHEVAUCHEMENT (`date_end >= début période` ET `date_start <= fin période`), pas seulement `date_start` seul, pour garder un événement déjà commencé mais encore en cours pendant la période demandée ;
- si la question ne précise AUCUNE période, un filtre par défaut `date_end >= aujourd'hui` s'applique quand même — un événement déjà terminé n'est jamais une réponse valable, et ce tri ne peut pas être laissé au LLM au moment de la génération (constaté empiriquement : il compare mal une date à aujourd'hui même avec la date du jour fournie dans le prompt, ex. confondre un jour du mois similaire avec "aujourd'hui").

**Événements récurrents** : `date_start`/`date_end` ne représentent que la 1ère et la dernière occurrence d'un événement récurrent (ex. un atelier chaque samedi d'avril à juin) — insuffisant pour deux choses, corrigées séparément :
- **Affichage** : `next_occurrence_date()` (rag_chain.py) calcule, à partir de `occurrences`, la prochaine date pertinente à partir d'aujourd'hui, plutôt que de toujours montrer la toute première occurrence (potentiellement déjà passée) ;
- **Filtrage précis** : `occurrence_in_period()` (query_filters.py) revérifie, après la recherche FAISS, qu'une occurrence réelle tombe bien dans la période demandée — le filtre FAISS seul (chevauchement `date_start`/`date_end`) pourrait laisser passer à tort un événement dont les occurrences sont espacées (ex. avril puis octobre) pour une question sur juin. Limite assumée : ce second contrôle s'applique après que la recherche a déjà limité les résultats à `k` documents — un événement écarté à ce stade n'est pas remplacé par un autre candidat plus bas dans le classement.

`format_docs()` (rag_chain.py) met en forme le contexte transmis au LLM : titre, date (formatée en français avec jour de la semaine précalculé par `format_date_fr()` — jamais laissé au LLM, qui s'est trompé en le déduisant lui-même d'une date ISO brute), lieu, tarif, établissement, adresse, téléphone, site, réseaux sociaux, inscription, accès en ligne — chaque champ optionnel n'est ajouté que s'il est renseigné, pour éviter qu'un champ manquant (`None`) apparaisse littéralement dans le prompt et soit repris tel quel par le LLM.

**Limite connue, non traitée à ce stade** : la chaîne est *stateless* — `chain.invoke(question)` ne connaît que la question du tour actuel, aucun historique de conversation n'est conservé. Une question de suivi sans contexte explicite (ex. "je veux l'url de l'événement" après une première question) n'a structurellement aucun moyen d'être résolue correctement.

```bash
uv run jupyter notebook notebooks/05_rag_chain_evaluation.ipynb
```

Teste 6 scénarios représentatifs sur la chaîne complète (thème+lieu, contrainte de prix, question multi-contraintes, événement annulé signalé comme tel dans la réponse, question hors-sujet, question temporelle ambiguë) — nécessite `MISTRAL_API_KEY` (génération payante).

## API REST (Étape 5)

```bash
uv run uvicorn api.main:app --reload
```

Démarre l'API sur `http://127.0.0.1:8000`. Documentation Swagger interactive générée automatiquement sur `http://127.0.0.1:8000/docs`. La chaîne RAG est construite **une seule fois**, au démarrage du serveur (`lifespan`, voir `api/main.py`) — chaque appel à `/ask` réutilise cette même chaîne, sans recharger l'index FAISS ni recréer les clients Mistral à chaque requête.

| Route | Méthode | Description |
|---|---|---|
| `/` | GET | Informations générales, pointeur vers `/docs`. |
| `/health` | GET | Vérification de disponibilité du serveur, indépendante de l'état de la chaîne RAG. |
| `/ask` | POST | Corps `{"question": "..."}` → réponse générée. `422` si la question est vide/absente, `503` si la chaîne n'a pas pu être construite au démarrage, `502` en cas d'échec du service Mistral. |
| `/rebuild` | POST | **Nécessite l'en-tête HTTP `X-API-Key`**, avec la valeur de `X_API_KEY` définie dans `.env` — sans elle, `401 Unauthorized`. Relance le pipeline complet (`fetch_events` → `preprocess_events` → `vectorize_events` → `build_index`) en tâche de fond et recharge la chaîne RAG, sans redémarrer le serveur. Répond immédiatement `202 Accepted`, sans attendre la fin du pipeline (plusieurs minutes, appel payant à Mistral). `409 Conflict` si un rebuild est déjà en cours. |
| `/rebuild` | GET | Consulte l'état de la dernière reconstruction (`idle`/`running`/`done`/`error`) — pas de clé requise, lecture seule. |

### Tests

```bash
uv run pytest tests/ -v
```

`tests/test_preprocessing.py` teste la logique de `preprocess_events.py` (exclusion France Travail, normalisation de casse, extraction du statut, structuration des champs, extraction des occurrences) sur des événements factices — ne nécessite pas d'avoir lancé `fetch_events.py` au préalable.

`tests/test_api.py` teste l'API (`api/main.py`) via `TestClient` : routing, validation Pydantic, codes d'erreur (422/401/409/502/503), protection de `/rebuild`. Les appels coûteux (Mistral, pipeline complet) sont mockés — `lifespan()` s'exécute réellement au démarrage de chaque test (chargement de l'index FAISS déjà sur disque), mais aucun appel réseau payant n'a lieu.

`tests/test_query_filters.py` et `tests/test_rag_chain.py` testent la logique pure de `query_filters.py` (calcul de périodes, filtre FAISS, vérification des occurrences) et de `rag_chain.py` (formatage des dates, sélection de la prochaine occurrence, mise en forme du contexte) — aucune dépendance à Mistral ou FAISS, entièrement déterministes.

Fonctions volontairement non testées unitairement (appels réseau réels — Mistral et/ou disque) : `fetch_events.py`, `vectorize_events.py`, `build_index.py`, `preprocess_events.py::load_raw_events`, `query_filters.py::extract_filters`, `rag_chain.py::retrieve_context`/`answer_question`. Validées autrement, via `notebooks/04_search_evaluation.ipynb`/`05_rag_chain_evaluation.ipynb` et l'exécution réelle du pipeline complet.

Rapport de couverture (nécessite `pytest-cov`, dépendance de dev déjà installée) :

```bash
uv run pytest tests/ --cov=scripts --cov=api --cov-report=html
```

Génère `htmlcov/index.html` (gitignoré, régénérable). Couverture globale : 79% — répartie sans trou sur toute la logique testable unitairement (100% sur les fonctions pures citées ci-dessus), le reste correspondant aux appels réseau réels listés juste au-dessus.

## Statut

Étape 1 — configuration de l'environnement (terminée).
Étape 2 — pré-processing des données Open Agenda (terminée : récupération, nettoyage, tests unitaires, découpage en chunks, vectorisation — 4241 événements en 7169 chunks vectorisés dans `data/vectors/events_vectors.json`).
Étape 3 — base de données vectorielle FAISS (terminée : index construit via `FAISS.from_embeddings()` à partir des vecteurs déjà calculés, 7169/7169 vecteurs vérifiés, tests de recherche effectués dans `notebooks/04_search_evaluation.ipynb`).
Étape 4 — chaîne RAG (recherche + génération) : `scripts/rag_chain.py` + `scripts/query_filters.py` (recherche hybride filtre+sémantique, événements récurrents, affichage enrichi des métadonnées — voir plus haut). Terminée (mergée sur `main`).
Étape 5 — API REST (terminée) : `api/main.py` (FastAPI, endpoints `/`, `/health`, `/ask`, `/rebuild` — voir plus haut). Tests : 14 fonctionnels (`tests/test_api.py`) + tests unitaires sur la logique pure (`tests/test_query_filters.py`, `tests/test_rag_chain.py`, complétés dans `tests/test_preprocessing.py`) — couverture 79%, sans trou hors appels réseau réels (voir section Tests). Jeu de test annoté + évaluation automatisée (Ragas) volontairement laissés pour une étape ultérieure, non demandés explicitement par l'Étape 5.
