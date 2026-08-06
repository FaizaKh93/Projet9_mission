# Projet9 — POC Chatbot RAG pour Puls-Events

POC d'un chatbot capable de répondre à des questions sur des événements culturels à venir, à partir des données de l'[API Open Agenda](https://openagenda.com/), en s'appuyant sur un système RAG (Retrieval-Augmented Generation) combinant recherche vectorielle (FAISS) et génération de réponse en langage naturel (Mistral) orchestrés via LangChain, exposé par une API REST (FastAPI).

## Structure du projet

```
notebooks/   notebooks d'exploration et de vérification (ex. 00_check_environment.ipynb)
```

Les autres dossiers (`scripts/`, `api/`, `tests/`, `docs/`, `data/`) seront ajoutés au fil des étapes suivantes, au moment où ils seront réellement utilisés.

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

## Statut

Étape 1 — configuration de l'environnement (terminée). Les étapes suivantes (vectorisation, API, tests, rapport) sont en cours.
