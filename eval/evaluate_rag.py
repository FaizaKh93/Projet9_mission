"""Évalue la chaîne RAG sur les questions annotées (eval/qa_dataset_manual.json) avec Ragas.

Pour chaque question : récupère le contexte réellement utilisé (retrieve_context, scripts/
rag_chain.py) puis génère la réponse à partir de CE contexte précis — un seul appel à
extract_filters()/Mistral par question, jamais un second implicite via chain.invoke().

3 bugs ragas==0.4.3 x Mistral rencontrés et contournés ci-dessous, tous vérifiés empiriquement
(détail à chaque contournement) : import ChatVertexAI cassé, API "collections" incompatible avec
le client Mistral, combinaison de token_usage cassée dans langchain_mistralai au-delà d'une
génération combinée.

Métriques calculées : faithfulness, answer_relevancy, context_recall et answer_correctness (voir
commentaires ci-dessous). Les deux premières ignorent reference_answer (jugent la chaîne RAG dans
l'absolu) ; les deux dernières la comparent au ground truth du jeu de test.

Coût : jusqu'à N x (1 appel extract_filters + 1 génération + 4 appels métriques Ragas) en
mistral-large-latest/mistral-small-latest/mistral-embed, N = nombre de questions dans
qa_dataset_manual.json.
"""

import asyncio
import json
import os
import sys
import types
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# Bug 1/3 : import ChatVertexAI cassé dans ragas (issue #2753) — module factice, doit précéder
# tout import de ragas.
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _stub

# Permet d'importer rag_chain (scripts/) sans installer le projet comme package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings  # noqa: E402
from ragas.dataset_schema import SingleTurnSample  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
# Bug 2/3 : l'API "collections" (llm_factory) est incompatible avec Mistral — ancienne API
# ragas.metrics utilisée à la place.
from ragas.metrics import AnswerCorrectness, AnswerRelevancy, AnswerSimilarity, ContextRecall, Faithfulness  # noqa: E402

from rag_chain import INDEX_PATH, NO_RESULTS_MESSAGE, SYSTEM_PROMPT, format_date_fr, get_api_key, retrieve_context  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent / "qa_dataset_manual.json"
RESULTS_PATH = Path(__file__).resolve().parent / "eval_results.json"


async def evaluate() -> None:
    # Charge MISTRAL_API_KEY avant toute construction de client (embeddings, LLM, juges Ragas).
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = get_api_key()

    # Même index que la chaîne RAG réelle (data/index).
    embeddings = MistralAIEmbeddings(mistral_api_key=api_key)
    vector_store = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
    total_vectors = vector_store.index.ntotal  # fetch_k pour retrieve_context() (cf. rag_chain.py)

    # prompt | llm | parser à la main plutôt que build_chain() : évite un second retrieve_context()
    # interne (double appel Mistral, contexte incohérent avec celui mesuré par Ragas).
    llm = ChatMistralAI(mistral_api_key=api_key, model="mistral-large-latest", temperature=0)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])
    generation_chain = prompt | llm | StrOutputParser()

    # LLM/embeddings juges Ragas, distincts du LLM de génération ci-dessus.
    evaluator_llm = LangchainLLMWrapper(ChatMistralAI(mistral_api_key=api_key, model="mistral-large-latest"))
    evaluator_embeddings = LangchainEmbeddingsWrapper(MistralAIEmbeddings(mistral_api_key=api_key))
    faithfulness = Faithfulness(llm=evaluator_llm)

    # Bug 3/3 : strictness (nb de questions générées pour ce calcul) forcé à 1 au lieu de 3 — au-delà
    # d'une génération combinée, langchain_mistralai plante sur le token_usage (TypeError).
    answer_relevancy = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=1)

    context_recall = ContextRecall(llm=evaluator_llm)

    # answer_correctness a besoin d'un answer_similarity (embeddings), construit à part et passé
    # explicitement : sinon resterait None (normalement réglé par l'orchestrateur evaluate() de
    # ragas, qu'on n'utilise pas ici — single_turn_ascore() est appelé directement, voir plus bas).
    answer_similarity = AnswerSimilarity(embeddings=evaluator_embeddings)
    answer_correctness = AnswerCorrectness(llm=evaluator_llm, embeddings=evaluator_embeddings, answer_similarity=answer_similarity)

    # Écrites et vérifiées à la main contre data/processed/events.json (eval/qa_dataset_manual.json).
    qa_pairs = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    results = []
    for i, pair in enumerate(qa_pairs, start=1):
        question = pair["question"]
        print(f"[{i}/{len(qa_pairs)}] {question}")

        # Contexte réutilisé tel quel pour la génération et pour Ragas ci-dessous. Même
        # court-circuit que build_chain() (rag_chain.py) sur contexte vide, pour que ce script
        # évalue exactement le comportement réel de l'API, pas une variante plus permissive.
        context = retrieve_context(question, vector_store, api_key, total_vectors)
        if context:
            answer = generation_chain.invoke({
                "context": context,
                "question": question,
                "current_date": format_date_fr(date.today()),
            })
        else:
            answer = NO_RESULTS_MESSAGE

        # Un contexte par événement pour Ragas, pas un bloc fusionné (format_docs() les sépare par "\n\n---\n\n").
        retrieved_contexts = [block for block in context.split("\n\n---\n\n") if block.strip()] or [""]

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=retrieved_contexts,
            reference=pair["reference_answer"],
        )
        # Métrique 1 - Faithfulness : la réponse invente-t-elle des faits absents du contexte récupéré (hallucination) ?
        faithfulness_score = await faithfulness.single_turn_ascore(sample)

        # Métrique 2 - Answer relevancy : la réponse répond-elle vraiment à la question posée (pas hors-sujet, pas incomplète) ?
        relevancy_score = await answer_relevancy.single_turn_ascore(sample)

        # Métrique 3 - Context recall : le contexte récupéré contient-il tout ce qu'il faut pour bien répondre ?
        recall_score = await context_recall.single_turn_ascore(sample)

        # Métrique 4 - Answer correctness : la réponse générée correspond-elle à la réponse de référence (ground truth) ?
        correctness_score = await answer_correctness.single_turn_ascore(sample)

        # Réponses conservées pour la relecture humaine, pas seulement les scores.
        results.append({
            "question": question,
            "reference_answer": pair["reference_answer"],
            "generated_answer": answer,
            "source": pair["source"],
            "dimension": pair.get("dimension"),
            "faithfulness": faithfulness_score,
            "answer_relevancy": relevancy_score,
            "context_recall": recall_score,
            "answer_correctness": correctness_score,
        })
        print(
            f"    faithfulness={faithfulness_score:.2f}  answer_relevancy={relevancy_score:.2f}  "
            f"context_recall={recall_score:.2f}  answer_correctness={correctness_score:.2f}"
        )

        # Sauvegarde incrémentale : réécrit le fichier à chaque question, pas seulement à la fin.
        # Un plantage (API, réseau...) ne fait alors perdre que la question en cours, pas tout le
        # travail déjà payé en appels Mistral — constaté à nos dépens sur un crash à la question 20/20.
        RESULTS_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # Moyennes globales, en plus du détail par question déjà affiché ci-dessus.
    for metric in ["faithfulness", "answer_relevancy", "context_recall", "answer_correctness"]:
        avg = sum(r[metric] for r in results) / len(results)
        print(f"Moyenne {metric} : {avg:.2f}")
    print(f"Résultats détaillés sauvegardés dans {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(evaluate())
