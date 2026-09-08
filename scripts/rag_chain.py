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
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from query_filters import QueryFilters, build_faiss_filter, extract_filters, occurrence_in_period, period_to_date_range

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

Si la liste "Événements disponibles" ci-dessous est ENTIÈREMENT VIDE (aucun événement listé, \
pas même un événement qui ne correspondrait pas), cela signifie qu'aucun événement de notre \
base ne correspond à la question — dis-le clairement. Tu n'as accès à AUCUNE information en \
dehors de cette liste : ne complète JAMAIS une réponse avec un événement, lieu, date ou \
détail tiré de tes propres connaissances générales, même s'il te semble exact ou plausible, \
même si tu es certain qu'il existe réellement — la liste ci-dessous est ta SEULE source \
d'information, sans aucune exception.

Réponds en français, de façon concise et bien formulée.

Événements disponibles :
{context}"""

# Réponse fixe pour un contexte vide — court-circuite le LLM plutôt que de compter sur lui pour
# suivre la consigne "ne complète JAMAIS..." du SYSTEM_PROMPT ci-dessus à 100% des appels.
# Constaté empiriquement (eval/eval_results.json, question "Quels événements y a-t-il à Paris ?") :
# contexte confirmé VRAIMENT vide (eval/diagnose_hallucination.py), et le LLM a quand même inventé
# 3 événements parisiens plausibles, malgré temperature=0 et une consigne explicite déjà en place.
# Une garantie déterministe en Python, pas une reformulation probabiliste de plus du prompt.
NO_RESULTS_MESSAGE = (
    "Je n'ai trouvé aucun événement correspondant à votre question dans notre base de données "
    "Puls-Events. N'hésitez pas à reformuler votre demande ou à essayer d'autres critères de recherche."
)


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


def next_occurrence_date(meta: dict, today: date, date_range: tuple[str, str] | None = None) -> str | None:
    """Renvoyer la date ISO de l'occurrence à afficher pour un événement.

    Si une période est demandée (date_range renseigné), priorité à une occurrence tombant DANS
    cette période plutôt qu'à la plus proche d'aujourd'hui — sinon un événement quasi quotidien
    affiche systématiquement sa date la plus proche (ex. le 8 ou 9 septembre), jamais celle
    demandée (ex. "dimanche prochain", le 13), même s'il a bien une occurrence ce jour-là.
    Constaté empiriquement (eval/diagnose_hallucination.py) : le LLM voyait alors des dates ne
    correspondant pas à la période demandée et concluait à tort qu'aucun événement ne convenait,
    alors qu'occurrence_in_period() (query_filters.py) avait déjà vérifié qu'une occurrence
    tombait bien dans cette période — seul l'AFFICHAGE ne le reflétait pas.

    Sans période demandée (date_range=None) ou aucune occurrence dans la période : repli sur la
    prochaine occurrence à partir d'aujourd'hui (comportement d'origine). meta["date_start"] ne
    représente que la PREMIÈRE occurrence (firstdate_begin) — pour un événement récurrent (ex.
    un atelier chaque samedi d'avril à juin), l'afficher tel quel montrerait une date
    potentiellement déjà passée, alors que l'événement a encore des occurrences à venir.
    meta["occurrences"] (construit par preprocess_events.py à partir du champ brut timings)
    liste chaque créneau individuel ; on retombe sur date_start si la liste est vide (cas normal
    d'un événement ponctuel, sans occurrences détaillées).
    """
    occurrences = meta.get("occurrences") or []

    if date_range:
        period_start = date.fromisoformat(date_range[0])
        period_end = date.fromisoformat(date_range[1])
        in_period = [
            o["start"]
            for o in occurrences
            if o.get("start") and period_start <= datetime.fromisoformat(o["start"]).date() <= period_end
        ]
        if in_period:
            return min(in_period)

    upcoming = [
        o["start"] for o in occurrences if o.get("start") and datetime.fromisoformat(o["start"]).date() >= today
    ]
    if upcoming:
        return min(upcoming)
    return meta.get("date_start")


def build_period_note(filters: QueryFilters, today: date) -> str:
    """Résoudre la période mentionnée dans la question ("ce samedi", "ce week-end"...) en
    date(s) précise(s), calculée en Python (period_to_date_range(), déjà utilisée pour le
    filtre de recherche) — à donner telle quelle au LLM de génération, jamais à lui faire
    recalculer lui-même : constaté empiriquement peu fiable (deux réponses à la même question
    "ce samedi" ont donné deux dates différentes, une correcte et une fausse).

    Un jour de semaine cité seul ("samedi", "mercredi"...) a sa propre catégorie dans
    QueryFilters.period (voir JOURS_SEMAINE, query_filters.py) et retombe donc toujours dans le
    cas start_date == end_date ci-dessous, une seule date — jamais dans une plage multi-jours.
    Les catégories multi-jours (ce_week_end, cette_semaine, ce_mois...) n'ont donc plus besoin
    d'énumérer chaque jour nommément (ancienne approche, jusqu'à 30 lignes pour ce_mois) : le LLM
    n'a plus qu'à comparer la date de CHAQUE événement (déjà donnée individuellement par
    format_docs()) aux deux bornes de la plage, une simple comparaison qu'il gère bien — un
    "du ... au ..." compact suffit.
    """
    date_range = period_to_date_range(filters.period, today)
    if not date_range:
        return ""
    start_date = date.fromisoformat(date_range[0])
    end_date = date.fromisoformat(date_range[1])
    if start_date == end_date:
        note = f"Précision : la période demandée dans la question correspond exactement à {format_date_fr(start_date)}."
    else:
        note = (
            "Précision : la période demandée dans la question va du "
            f"{format_date_fr(start_date)} au {format_date_fr(end_date)} inclus."
        )
    return note + "\n\n"


def format_docs(
    docs: list[Document], today: date | None = None, date_range: tuple[str, str] | None = None
) -> str:
    """Mettre en forme les chunks récupérés pour le prompt : métadonnées clés + texte.

    today/date_range transmis à next_occurrence_date() (voir son docstring) pour afficher
    l'occurrence pertinente à la période demandée, pas systématiquement la plus proche
    d'aujourd'hui. today optionnel (repli sur date.today()) pour ne pas casser un appel existant
    qui ne le fournirait pas.

    Ces champs ne sont présents qu'en métadonnée (pas dans le texte vectorisé, voir la
    répartition structuré/vectorisé documentée dans le README) — sans les rajouter ici
    explicitement, le LLM n'aurait aucun moyen de répondre à "quand"/"où"/"comment
    s'inscrire"/"comment accéder en ligne". Champs factuels ajoutés seulement s'ils sont
    renseignés, pour ne pas polluer le prompt de lignes vides sur des champs peu remplis.
    """
    if today is None:
        today = date.today()
    blocks = []
    for doc in docs:
        meta = doc.metadata
        lines = [
            f"Titre : {meta.get('title')}",
            f"Date : {format_event_date_fr(next_occurrence_date(meta, today, date_range))}",
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


def retrieve_context(question: str, vector_store: FAISS, api_key: str, total_vectors: int) -> str:
    """Extraire un filtre de la question, chercher avec ce filtre, mettre en forme le résultat.

    Fonction autonome (plus une fermeture interne à build_chain()) : réutilisée à l'identique par
    build_chain() (liée à un vector_store précis via une lambda) et par eval/evaluate_rag.py, qui a
    besoin du contexte ET de la réponse séparément pour Ragas — jamais juste la réponse finale comme
    le reste de la chaîne. Évite de dupliquer cette logique entre les deux usages.

    Remplace l'ancien "retriever = vector_store.as_retriever(...)" : ses search_kwargs étaient
    figés à la construction de la chaîne, alors que le filtre dépend maintenant de CHAQUE
    question — la recherche doit donc être appelée directement ici.
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
    # date_range recalculé ici (même appel que dans build_period_note()) pour que format_docs()
    # affiche, pour un événement récurrent, l'occurrence tombant DANS cette période plutôt que
    # systématiquement la plus proche d'aujourd'hui (voir next_occurrence_date()) — sinon un
    # événement quasi quotidien peut afficher une date différente de celle réellement demandée.
    date_range = period_to_date_range(filters.period, today)
    return build_period_note(filters, today) + format_docs(docs, today, date_range)


def build_chain() -> tuple:
    """Construire la chaîne RAG complète (extraction de filtres + recherche hybride + prompt + LLM).

    Renvoie (chain, total_vectors) — total_vectors (déjà calculé pour fetch_k, voir plus bas)
    exposé en plus de la chaîne pour que l'appelant puisse le réutiliser (ex. api/main.py::/metadata)
    sans recharger l'index FAISS une seconde fois juste pour compter les vecteurs.
    """
    api_key = get_api_key()

    # Étape 1 : préparer le modèle d'embeddings — le même qu'à l'indexation, pour que la
    # question et les vecteurs déjà stockés soient comparables.
    embeddings = MistralAIEmbeddings(mistral_api_key=api_key)

    # Étape 2 : charger sur disque l'index FAISS déjà construit par build_index.py.
    vector_store = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)

    # fetch_k = la totalité de l'index : le filtre (voir query_filters.py) doit pouvoir
    # examiner tous les événements, pas seulement les quelques plus proches sémantiquement.
    total_vectors = vector_store.index.ntotal

    # Étape 3 : préparer le modèle de génération (différent du modèle d'embeddings ci-dessus).
    # temperature=0 : réduit la variabilité d'une réponse à l'autre pour une même question
    # (comme pour extract_filters()), mais n'élimine pas complètement la variation possible
    # d'un appel à l'autre (calcul flottant non associatif, service réparti) — d'où
    # build_period_note() ci-dessus, qui traite la cause réelle plutôt que d'espérer un
    # comportement stable : le LLM n'a plus besoin de calculer la date lui-même.
    llm = ChatMistralAI(mistral_api_key=api_key, model="mistral-large-latest", temperature=0)

    # Étape 4 : construire le gabarit de prompt, avec {context} et {question} comme espaces réservés.
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    generation_chain = prompt | llm | StrOutputParser()

    def generate_or_refuse(inputs: dict) -> str:
        """Court-circuiter le LLM si le contexte est vide (voir NO_RESULTS_MESSAGE ci-dessus) —
        appelé une fois par question, avec le contexte déjà résolu par retrieve_context()."""
        if not inputs["context"]:
            return NO_RESULTS_MESSAGE
        return generation_chain.invoke(inputs)

    # Étape 5 : assembler la chaîne complète — question -> contexte (filtré + sémantique) +
    # question + date du jour -> réponse fixe (contexte vide) ou prompt rempli -> LLM -> texte
    # final. "current_date" et "context" sont recalculés à CHAQUE question posée : ces fonctions
    # ne s'exécutent qu'au moment de chain.invoke(), jamais à la construction de la chaîne.
    chain = (
        {
            "context": lambda question: retrieve_context(question, vector_store, api_key, total_vectors),
            "question": RunnablePassthrough(),
            "current_date": lambda _: format_date_fr(date.today()),
        }
        | RunnableLambda(generate_or_refuse)
    )
    return chain, total_vectors


def answer_question(question: str) -> str:
    """Poser une question à la chaîne RAG et renvoyer la réponse générée."""
    chain, _ = build_chain()
    return chain.invoke(question)


if __name__ == "__main__":
    example_question = "Quels concerts de musique classique y a-t-il à Marseille ?"
    print(f"Question : {example_question}\n")
    print(answer_question(example_question))
