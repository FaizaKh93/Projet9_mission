"""Assemble la chaîne RAG complète : recherche FAISS + génération de réponse (Mistral).

Charge l'index construit par build_index.py, récupère les chunks les plus proches d'une
question, et les transmet à un LLM (mistral-large-latest) pour générer une réponse en
langage naturel, ancrée dans les événements réellement indexés.

Construit avec les primitives LCEL de langchain_core (pas via langchain_classic, dont les
chaînes pré-emballées comme create_stuff_documents_chain existent toujours mais sont
explicitement reléguées au paquet "classic" dans LangChain 1.x).

Nécessite MISTRAL_API_KEY dans .env (déjà utilisé pour les embeddings, réutilisé ici pour
la génération).
"""

import os
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from query_filters import build_faiss_filter, extract_filters, occurrence_in_period

INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "index"

# Nombre de chunks les plus proches transmis au LLM comme contexte pour chaque question.
RETRIEVED_CHUNKS = 5

# Noms français des jours/mois, pour ne pas dépendre du paramètre régional (locale) du
# système au moment de formater la date — évite un comportement différent d'une machine à l'autre.
JOURS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]

SYSTEM_PROMPT = """Tu es l'assistant de Puls-Events, une plateforme de recommandations \
culturelles. Nous sommes le {current_date}. Réponds à la question de l'utilisateur en te \
basant UNIQUEMENT sur les événements listés ci-dessous — n'invente aucun événement, aucune \
date, aucun lieu qui n'y figure pas.

Avant de répondre, pour CHAQUE événement listé : compare sa date à la date du jour \
ci-dessus. Si sa date est déjà passée par rapport à aujourd'hui, ne le mentionne PAS dans ta \
réponse, même s'il correspond au sujet de la question — un événement passé n'est jamais une \
réponse valable à une question sur un événement à venir. Si la question porte sur une \
période précise ("ce week-end", "bientôt", "la semaine prochaine"), ne garde que les \
événements dont la date tombe réellement dans cette période à partir d'aujourd'hui. \
Si, après ce tri, aucun événement du contexte ne correspond réellement à la question, \
dis-le clairement plutôt que de présenter des événements qui ne correspondent pas.

Réponds en français, de façon concise et bien formulée.

Événements disponibles :
{context}"""


def get_api_key() -> str:
    """Charger MISTRAL_API_KEY depuis .env, ou échouer explicitement si absente."""
    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY manquant : renseignez-le dans .env (voir .env.example).")
    return api_key


def format_date_fr(d: date) -> str:
    """Formater une date en français lisible (ex. "vendredi 28 août 2026"), sans dépendre de la locale."""
    return f"{JOURS_FR[d.weekday()]} {d.day} {MOIS_FR[d.month - 1]} {d.year}"


def format_event_date_fr(iso_datetime: str | None) -> str | None:
    """Formater une date d'événement (ISO, ex. "2026-09-05T17:00:00+00:00") en français lisible,
    jour de la semaine inclus — calculé ici plutôt que laissé au LLM. Même principe que pour
    current_date/format_date_fr : transmettre la date ISO brute forçait le LLM à déduire lui-même
    le jour de la semaine à chaque réponse, et il s'est trompé (ex. "vendredi 5 septembre 2026"
    au lieu de "samedi", scénario 3 vs scénario 6 de notebooks/05_rag_chain_evaluation.ipynb).
    """
    if not iso_datetime:
        return None
    dt = datetime.fromisoformat(iso_datetime)
    return f"{format_date_fr(dt.date())} à {dt.hour:02d}h{dt.minute:02d}"


def next_occurrence_date(meta: dict, today: date) -> str | None:
    """Renvoyer la date ISO de la prochaine occurrence pertinente (>= aujourd'hui) d'un événement.

    meta["date_start"] ne représente que la PREMIÈRE occurrence (firstdate_begin) — pour un
    événement récurrent (ex. un atelier chaque samedi d'avril à juin), l'afficher tel quel
    montrerait une date potentiellement déjà passée, alors que l'événement a encore des
    occurrences à venir. meta["occurrences"] (construit par preprocess_events.py à partir du
    champ brut timings) liste chaque créneau individuel ; on retombe sur date_start si la liste
    est vide (cas normal d'un événement ponctuel, sans occurrences détaillées).
    """
    occurrences = meta.get("occurrences") or []
    upcoming = [
        o["start"] for o in occurrences if o.get("start") and datetime.fromisoformat(o["start"]).date() >= today
    ]
    if upcoming:
        return min(upcoming)
    return meta.get("date_start")


def format_docs(docs: list[Document]) -> str:
    """Mettre en forme les chunks récupérés pour le prompt : métadonnées clés + texte.

    Ces champs ne sont présents qu'en métadonnée (pas dans le texte vectorisé, voir la
    répartition structuré/vectorisé documentée dans le README) — sans les rajouter ici
    explicitement, le LLM n'aurait aucun moyen de répondre à "quand"/"où"/"comment
    s'inscrire"/"comment accéder en ligne". Champs factuels ajoutés seulement s'ils sont
    renseignés, pour ne pas polluer le prompt de lignes vides sur des champs peu remplis.
    """
    blocks = []
    for doc in docs:
        meta = doc.metadata
        lines = [
            f"Titre : {meta.get('title')}",
            f"Date : {format_event_date_fr(next_occurrence_date(meta, date.today()))}",
        ]
        # Chaque champ ci-dessous n'est ajouté que s'il est renseigné : sans ce test, un champ
        # manquant (meta.get(...) -> None) afficherait littéralement "None" dans le prompt, que
        # le LLM pourrait reprendre tel quel dans sa réponse au lieu de simplement l'omettre.
        if meta.get("location_city"):
            lines.append(f"Lieu : {meta['location_city']}")
        if meta.get("conditions"):
            lines.append(f"Tarif : {meta['conditions']}")

        etablissement = " — ".join(p for p in [meta.get("location_name"), meta.get("location_district")] if p)
        if etablissement:
            lines.append(f"Établissement : {etablissement}")

        adresse = ", ".join(p for p in [meta.get("location_address"), meta.get("location_postalcode")] if p)
        if adresse:
            lines.append(f"Adresse : {adresse}")

        if meta.get("location_phone"):
            lines.append(f"Téléphone : {meta['location_phone']}")
        if meta.get("location_website"):
            lines.append(f"Site du lieu : {meta['location_website']}")
        if meta.get("location_links"):
            lines.append(f"Réseaux sociaux : {meta['location_links']}")
        if meta.get("registration_link"):
            lines.append(f"Inscription : {meta['registration_link']}")
        if meta.get("online_access_link"):
            lines.append(f"Accès en ligne : {meta['online_access_link']}")

        lines.append(f"Lien : {meta.get('url')}")
        blocks.append("\n".join(lines) + "\n\n" + doc.page_content)
    return "\n\n---\n\n".join(blocks)


def build_chain():
    """Construire la chaîne RAG complète (extraction de filtres + recherche hybride + prompt + LLM)."""
    api_key = get_api_key()

    # Étape 1 : préparer le modèle d'embeddings — le même qu'à l'indexation, pour que la
    # question et les vecteurs déjà stockés soient comparables.
    embeddings = MistralAIEmbeddings(mistral_api_key=api_key)

    # Étape 2 : charger sur disque l'index FAISS déjà construit par build_index.py.
    vector_store = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)

    # fetch_k = la totalité de l'index : le filtre (voir query_filters.py) doit pouvoir
    # examiner tous les événements, pas seulement les quelques plus proches sémantiquement.
    total_vectors = vector_store.index.ntotal

    def retrieve_context(question: str) -> str:
        """Extraire un filtre de la question, chercher avec ce filtre, mettre en forme le résultat.

        Remplace l'ancien "retriever = vector_store.as_retriever(...)" : ses search_kwargs
        étaient figés à la construction de la chaîne, alors que le filtre dépend maintenant
        de CHAQUE question — la recherche doit donc être appelée directement ici.
        """
        filters = extract_filters(question, api_key)
        today = date.today()
        faiss_filter = build_faiss_filter(filters, today)
        docs = vector_store.similarity_search(
            question, k=RETRIEVED_CHUNKS, filter=faiss_filter or None, fetch_k=total_vectors
        )
        # Second passage, en Python : build_faiss_filter() ne compare que date_start/date_end
        # (première/dernière occurrence), donc un événement récurrent avec un grand écart entre
        # deux occurrences pourrait passer le filtre FAISS à tort (cf. query_filters.py). On
        # revérifie ici sur les occurrences individuelles avant de transmettre au LLM.
        docs = [doc for doc in docs if occurrence_in_period(doc.metadata, filters, today)]
        return format_docs(docs)

    # Étape 3 : préparer le modèle de génération (différent du modèle d'embeddings ci-dessus).
    llm = ChatMistralAI(mistral_api_key=api_key, model="mistral-large-latest")

    # Étape 4 : construire le gabarit de prompt, avec {context} et {question} comme espaces réservés.
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    # Étape 5 : assembler la chaîne complète — question -> contexte (filtré + sémantique) +
    # question + date du jour -> prompt rempli -> réponse du LLM -> texte final. "current_date"
    # et "context" sont recalculés à CHAQUE question posée : ces fonctions ne s'exécutent
    # qu'au moment de chain.invoke(), jamais à la construction de la chaîne.
    return (
        {
            "context": retrieve_context,
            "question": RunnablePassthrough(),
            "current_date": lambda _: format_date_fr(date.today()),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def answer_question(question: str) -> str:
    """Poser une question à la chaîne RAG et renvoyer la réponse générée."""
    chain = build_chain()
    return chain.invoke(question)


if __name__ == "__main__":
    example_question = "Quels concerts de musique classique y a-t-il à Marseille ?"
    print(f"Question : {example_question}\n")
    print(answer_question(example_question))
