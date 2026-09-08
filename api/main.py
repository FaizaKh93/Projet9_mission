"""API REST exposant le système RAG (Étape 5).

Ne réimplémente aucune logique métier : appelle build_chain() (scripts/rag_chain.py) une
seule fois au démarrage (voir lifespan ci-dessous), et .invoke() sur cette même chaîne mise
en cache à chaque question — jamais answer_question(), qui reconstruirait la chaîne (rechargement
de l'index FAISS, recréation des clients Mistral) à chaque appel.
"""

import os
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# Chargé ici (pas seulement via get_api_key()/build_chain() plus bas) pour que X_API_KEY soit
# disponible dès la définition des routes, sans dépendre de l'ordre d'exécution de lifespan().
load_dotenv()

# scripts/ n'est pas un package (pas de __init__.py, exécuté jusqu'ici via "uv run python
# scripts/xxx.py") — même approche que dans les notebooks pour réutiliser ses fonctions.
SCRIPTS_PATH = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_PATH))

# Imports placés après la modification de sys.path (ligne ci-dessus) : Python doit d'abord
# savoir où chercher ces modules avant de pouvoir les importer. "# noqa: E402" désactive
# l'avertissement du linter "import pas en haut de fichier", ici volontaire. On importe les
# modules eux-mêmes (pas juste leurs fonctions) pour appeler explicitement <module>.main()
# plus bas, ce qui rend sans ambiguïté quel script est déclenché à chaque étape du pipeline.
import build_index  # noqa: E402
import fetch_events  # noqa: E402
import preprocess_events  # noqa: E402
import vectorize_events  # noqa: E402
from rag_chain import build_chain  # noqa: E402


class AskRequest(BaseModel):
    """Corps de la requête POST /ask."""

    # "..." (Ellipsis) = champ obligatoire, pas de valeur par défaut. min_length=1 rejette
    # une question vide ("") automatiquement, avant même que la fonction ask() ne s'exécute
    # (réponse 422 générée par Pydantic/FastAPI, sans code à écrire pour ce cas).
    question: str = Field(
        ...,
        min_length=1,
        description="Question en langage naturel posée au chatbot Puls-Events.",
        examples=["Quels concerts de musique classique y a-t-il à Marseille ?"],
    )


class AskResponse(BaseModel):
    """Réponse renvoyée par POST /ask."""

    answer: str = Field(..., description="Réponse générée par le système RAG.")


class RebuildStatus(BaseModel):
    """État de la base vectorielle, renvoyé par GET /rebuild."""

    state: Literal["idle", "running", "done", "error"] = Field(
        ..., description="idle = jamais lancé, running = en cours, done = succès, error = échec."
    )
    started_at: str | None = Field(default=None, description="Horodatage ISO du dernier déclenchement.")
    finished_at: str | None = Field(default=None, description="Horodatage ISO de la fin (done/error uniquement).")
    events_indexed: int | None = Field(default=None, description="Nombre de vecteurs indexés (done uniquement).")
    detail: str | None = Field(default=None, description="Message d'erreur (error uniquement).")


class RebuildAck(BaseModel):
    """Réponse immédiate de POST /rebuild (avant que la tâche de fond ne se termine)."""

    status: Literal["started"]


class MetadataResponse(BaseModel):
    """État des données actuellement servies par /ask — pas l'état du pipeline (voir /rebuild)."""

    events_indexed: int | None = Field(
        default=None, description="Nombre d'événements actuellement chargés en mémoire, None si la chaîne n'a pas pu être construite."
    )
    geographic_coverage: str = Field(..., description="Zone géographique couverte par les données.")
    chain_available: bool = Field(..., description="False si /ask répondrait 503 (échec de build_chain()).")


# Header(default=None) : paramètre optionnel affiché directement dans les "Parameters" de
# l'opération /rebuild sur Swagger (champ texte simple, à remplir avant "Execute"), plutôt
# que via le bouton "Authorize" global (mécanisme fastapi.security.APIKeyHeader, plus discret
# mais moins visible pour ce cas d'usage). FastAPI convertit automatiquement le underscore de
# "x_api_key" en tiret pour l'en-tête HTTP réel ("X-Api-Key" — équivalent à "X-API-Key", les
# noms d'en-têtes HTTP étant insensibles à la casse).
def verify_admin_key(x_api_key: str | None = Header(default=None)) -> None:
    """Dépendance FastAPI protégeant POST /rebuild : compare l'en-tête X-API-Key à X_API_KEY (.env)."""
    expected_key = os.getenv("X_API_KEY")
    if not expected_key or x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="Clé API manquante ou incorrecte (en-tête X-API-Key).")


# Empêche deux POST /rebuild simultanés de passer tous les deux le contrôle "déjà en cours ?"
# avant que l'un des deux n'ait eu le temps de marquer l'état "running" (cf. trigger_rebuild).
_rebuild_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Construire la chaîne RAG une seule fois, au démarrage du serveur.

    Un échec ici n'empêche pas le serveur de démarrer : app.state.chain reste à None,
    et /ask répond alors 503 plutôt que de planter au lancement — /rebuild peut retenter
    une construction sans redémarrer le processus.
    """
    try:
        # app.state : emplacement fourni par FastAPI/Starlette pour stocker un objet partagé
        # entre toutes les requêtes, initialisé une fois ici plutôt que dans chaque route.
        # total_vectors stocké séparément (pas recalculé) : réutilisé par GET /metadata.
        app.state.chain, app.state.total_vectors = build_chain()
    except Exception as exc:
        # Sans ce print, l'échec est totalement invisible (ni logs, ni erreur) — impossible à
        # diagnostiquer depuis "docker logs" ou le terminal. Même raisonnement que pour
        # run_rebuild_pipeline() : visible côté serveur, jamais renvoyé au client (503 générique).
        print(f"[lifespan] échec de build_chain() au démarrage : {exc}")
        app.state.chain = None
        app.state.total_vectors = None
    app.state.rebuild_status = RebuildStatus(state="idle")
    # Tout ce qui est avant "yield" s'exécute au démarrage du serveur, une seule fois.
    # Tout ce qui serait après (rien ici) s'exécuterait à l'arrêt du serveur.
    yield


app = FastAPI(
    title="Puls-Events RAG API",
    description="API exposant le chatbot RAG de Puls-Events (recherche + génération sur des événements culturels).",
    # Relie la fonction lifespan ci-dessus au cycle de vie du serveur — sans ce paramètre,
    # elle ne serait jamais appelée.
    lifespan=lifespan,
)


@app.get("/", summary="Informations générales sur l'API")
def root() -> dict:
    """Point d'entrée racine : confirme que le serveur tourne et pointe vers la doc Swagger."""
    return {"name": app.title, "docs": "/docs"}


@app.get("/health", summary="Vérifier que le serveur répond")
def health() -> dict:
    """Vérification de disponibilité minimale (le serveur répond), sans dépendre de la chaîne
    RAG — un problème sur /ask (voir chain is None) ne doit pas faire croire que le SERVEUR
    lui-même est en panne à un outil de supervision qui interrogerait cette route."""
    return {"status": "ok"}


@app.get("/metadata", response_model=MetadataResponse, summary="Consulter l'état des données actuellement servies")
def metadata() -> MetadataResponse:
    """État des données actuellement chargées en mémoire — pas l'état du pipeline (voir /rebuild).

    events_indexed vient de app.state.total_vectors, mis à jour à chaque (re)construction de la
    chaîne (lifespan() au démarrage, run_rebuild_pipeline() après un /rebuild) — reflète donc
    toujours l'index réellement en mémoire, même si ce dernier a été (re)construit manuellement
    en local avant le démarrage du serveur, pas seulement via /rebuild (contrairement au compteur
    de GET /rebuild, qui ne se met à jour qu'après une reconstruction déclenchée par l'API).
    """
    return MetadataResponse(
        events_indexed=app.state.total_vectors,
        geographic_coverage=fetch_events.LOCATION_DEPARTMENT,
        chain_available=app.state.chain is not None,
    )


# response_model=AskResponse : documente à FastAPI la forme exacte de la réponse, utilisée
# pour générer le schéma Swagger (/docs) automatiquement, sans travail supplémentaire.
@app.post("/ask", response_model=AskResponse, summary="Poser une question au chatbot")
def ask(request: AskRequest) -> AskResponse:
    """Transmettre une question à la chaîne RAG déjà construite et renvoyer la réponse générée."""
    # Cas où lifespan() a échoué à construire la chaîne au démarrage (ex. clé API invalide) :
    # on prévient l'appelant plutôt que de planter sur un AttributeError (chain est None).
    if app.state.chain is None:
        raise HTTPException(
            status_code=503,
            detail="Le système RAG n'est pas disponible (échec au démarrage) — réessayez plus tard.",
        )
    try:
        # .invoke() réutilise la chaîne mise en cache — ne recharge pas l'index FAISS, ne
        # recrée pas les clients Mistral (contrairement à answer_question(), voir docstring
        # du module en tête de fichier).
        answer = app.state.chain.invoke(request.question)
    except Exception as exc:
        # Erreur venant d'un service tiers (Mistral) en amont de notre API : 502 ("Bad
        # Gateway") plutôt qu'un 500 générique, avec un message propre au lieu d'exposer la
        # trace Python brute à l'appelant. Le print() est nécessaire pour la garder visible
        # côté serveur : une HTTPException gérée (contrairement à une exception non attrapée)
        # n'est PAS journalisée automatiquement par uvicorn — sans cette ligne, la cause réelle
        # ne serait visible nulle part, ni dans la réponse HTTP (volontairement), ni dans les logs.
        print(f"[ask] échec de chain.invoke() : {exc}")
        raise HTTPException(
            status_code=502,
            detail="Échec de la génération de réponse (service Mistral indisponible ou en erreur).",
        ) from exc
    return AskResponse(answer=answer)


def run_rebuild_pipeline() -> None:
    """Exécutée en tâche de fond (pas au moment de la requête HTTP) : relance le pipeline
    complet dans l'ordre, puis recharge la chaîne RAG pour que /ask serve les données fraîches
    sans redémarrer le serveur.

    Fonction "def" classique (pas "async def") : nos scripts font des appels bloquants
    (httpx, Mistral) — FastAPI/Starlette exécute automatiquement une tâche de fond non-async
    dans un thread séparé, pour ne pas geler le reste de l'API (/ask notamment) pendant les
    quelques minutes que peut durer ce pipeline.
    """
    try:
        fetch_events.main()
        preprocess_events.main()
        vectorize_events.main()
        indexed_count = build_index.main()
        # Reconstruit la chaîne à partir du nouvel index sur disque — app.state.chain (utilisé
        # par /ask) pointe alors vers les données à jour, sans redémarrage du processus.
        # app.state.total_vectors mis à jour ici aussi (pas seulement dans lifespan()) : sans
        # ça, GET /metadata resterait figé sur le compte de démarrage après un /rebuild.
        app.state.chain, app.state.total_vectors = build_chain()
        app.state.rebuild_status = RebuildStatus(
            state="done",
            started_at=app.state.rebuild_status.started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            events_indexed=indexed_count,
        )
    except Exception as exc:
        # Échec à n'importe quelle étape du pipeline : on le trace dans le statut consultable
        # via GET /rebuild plutôt que de le laisser disparaître silencieusement (une tâche de
        # fond n'a personne à qui renvoyer une exception, contrairement à une route classique).
        # Le message complet (str(exc)) part dans les logs serveur, PAS dans la réponse HTTP :
        # GET /rebuild n'est volontairement pas protégé par X-API-Key (juste un statut en
        # lecture seule), donc detail ne doit jamais contenir un message d'erreur brut qui
        # pourrait, par accident, exposer un chemin local ou un détail d'erreur réseau.
        print(f"[rebuild] échec du pipeline : {exc}")
        app.state.rebuild_status = RebuildStatus(
            state="error",
            started_at=app.state.rebuild_status.started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            detail="Échec du pipeline de reconstruction — voir les logs du serveur pour le détail.",
        )


@app.post(
    "/rebuild",
    response_model=RebuildAck,
    status_code=202,
    summary="Reconstruire la base vectorielle (protégé par X-API-Key)",
    dependencies=[Depends(verify_admin_key)],
)
def trigger_rebuild(background_tasks: BackgroundTasks) -> RebuildAck:
    """Déclencher le pipeline complet en tâche de fond et répondre immédiatement (202)."""
    with _rebuild_lock:
        if app.state.rebuild_status.state == "running":
            raise HTTPException(status_code=409, detail="Une reconstruction est déjà en cours.")
        # Marqué "running" ICI, avant de programmer la tâche de fond — pas dans
        # run_rebuild_pipeline(), qui ne s'exécute qu'après la réponse HTTP : sans ça, un
        # second POST /rebuild presque simultané pourrait passer le contrôle ci-dessus avant
        # que le premier n'ait eu la main pour marquer l'état "running".
        app.state.rebuild_status = RebuildStatus(state="running", started_at=datetime.now(timezone.utc).isoformat())
    background_tasks.add_task(run_rebuild_pipeline)
    return RebuildAck(status="started")


@app.get("/rebuild", response_model=RebuildStatus, summary="Consulter l'état de la base vectorielle")
def get_rebuild_status() -> RebuildStatus:
    """Renvoyer l'état courant de la dernière reconstruction (ou "idle" si jamais lancée)."""
    return app.state.rebuild_status
