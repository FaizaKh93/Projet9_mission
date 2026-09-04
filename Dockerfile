# Étape 6 : conteneurise l'API (api/main.py), pas le pipeline de données — build_index.py et
# les autres scripts du pipeline continuent de tourner hors Docker, comme avant (voir README).
# data/index/ (et le reste de data/) est monté en volume au lancement, jamais copié dans
# l'image : ni MISTRAL_API_KEY ni X_API_KEY n'interviennent à la construction de l'image
# (choix discuté et validé : Approche 1, index en volume plutôt qu'intégré au build).

FROM python:3.11-slim

# Copie le binaire officiel d'uv depuis son image dédiée (méthode recommandée par la doc uv
# pour Docker), plutôt que de l'installer via pip dans l'environnement du projet lui-même.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copiées seules d'abord : Docker met cette étape en cache tant que les dépendances ne
# changent pas, donc modifier le code (scripts/api) sans toucher pyproject.toml/uv.lock ne
# réinstalle pas tout à chaque reconstruction de l'image.
COPY pyproject.toml uv.lock ./
# --locked : échoue explicitement si uv.lock n'est pas synchronisé avec pyproject.toml, plutôt
# que de le régénérer silencieusement — image reproductible, pas de dépendances "au hasard".
# --no-dev : exclut pytest/jupyter/etc. (voir [dependency-groups] dans pyproject.toml),
# inutiles pour faire tourner l'API en production/démo.
RUN uv sync --locked --no-dev

# Code applicatif uniquement — ni notebooks/, ni tests/, ni data/ (voir .dockerignore) : le
# pipeline de données et les tests n'ont pas leur place dans l'image de l'API.
COPY scripts/ ./scripts/
COPY api/ ./api/

EXPOSE 8000

# --host 0.0.0.0 est indispensable dans un conteneur : 127.0.0.1 (valeur utilisée en local
# jusqu'ici) ne désignerait que le conteneur lui-même, injoignable depuis l'hôte.
# --no-sync : sans ça, "uv run" revérifie l'environnement à CHAQUE lancement du conteneur et,
# ne connaissant plus le "--no-dev" utilisé à la construction (ligne 23, une commande "uv"
# indépendante de celle-ci), réinstalle tout le groupe dev (jupyter, pytest...) au démarrage —
# constaté en testant : téléchargements inutiles, démarrage plus lent, dépendance réseau au
# lancement qu'on voulait justement éviter. "--no-sync" réutilise tel quel ce qui a déjà été
# installé (et vérifié "--locked") pendant le build, sans jamais re-synchroniser à l'exécution.
CMD ["uv", "run", "--no-sync", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
