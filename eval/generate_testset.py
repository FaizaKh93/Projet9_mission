"""NON UTILISÉ pour construire eval/qa_dataset_manual.json (voir plus bas pourquoi) — conservé et
documenté à titre d'alternative explorée, au cas où la génération automatique redevienne
pertinente (ex. jeu de test beaucoup plus grand, où la vérification manuelle de chaque paire
deviendrait trop coûteuse en temps).

Génère des paires question/réponse de référence via la génération de jeu de test synthétique de
Ragas, à partir de vrais événements du dataset (voir eval/qa_dataset_generated.json pour un
exemple de sortie, 14 paires générées lors de l'essai).

Pourquoi abandonné : sur les 14 paires generées lors de l'essai, 10 se sont révélées porter sur
des événements déjà passés au moment de l'évaluation (`generate_with_langchain_docs()` pioche
sans tenir compte de la fraîcheur des dates, voir pick_diverse_sample() ci-dessous) — un chatbot
qui respecte sa propre consigne anti-hallucination (Étape 6) ne PEUT PAS répondre correctement à
ces questions par construction, indépendamment de sa qualité réelle. Un deuxième défaut, plus
insidieux, s'est aussi révélé : `data/processed/events.json` contient de nombreux doublons
(même activité reconduite à plusieurs dates différentes) sans qu'aucun lien automatique
n'existe vers l'événement source réellement utilisé par Ragas (voir plus bas) — un simple
matching par mots-clés a confondu plusieurs de ces doublons, avec des paires de référence
incorrectes comme résultat, détecté seulement en comparant les réponses réellement générées par
le RAG aux données réelles. Plutôt que de blinder ce pipeline (filtrer sur la fraîcheur au
moment de la génération, fiabiliser le lien vers l'événement source...), le jeu de test est
désormais entièrement écrit et vérifié à la main (eval/qa_dataset_manual.json, seul fichier
consommé par evaluate_rag.py) : plus lent par paire, mais chaque paire est garantie correcte au
moment de la vérification, sans dépendre de la fraîcheur du graphe de connaissances de Ragas ni
d'un matching automatique fragile.

Deux bugs de ragas==0.4.3 avec Mistral, vérifiés empiriquement, contournés ci-dessous :
(1) import inconditionnel de ChatVertexAI cassé (issue #2753, non corrigé) ; (2) l'API
"collections" (llm_factory) suppose une méthode client.messages.create() propre à Anthropic —
sans objet ici puisque generate_with_langchain_docs() utilise LangchainLLMWrapper directement,
pas concerné par ce second bug.

Limite connue : l'adaptation en français (adapt_prompts("french", ...)) ne fonctionne pas de
façon fiable sur tous les synthétiseurs (constaté : 2/3 adaptés correctement, le troisième —
multi_hop_specific — parfois encore en anglais). Ne pose plus de problème ici : on se limite au
seul synthétiseur single-hop (voir plus bas), qui s'est toujours adapté correctement.

Uniquement des questions single-hop (un document = une question), pas multi-hop : les questions
multi-hop nécessitent des documents THÉMATIQUEMENT LIÉS pour former des "clusters" dans le
graphe de connaissances de ragas — or l'échantillon ci-dessous maximise volontairement la
diversité géographique (peu de documents liés entre eux), ce qui fait échouer la génération
multi-hop ("No clusters found in the knowledge graph", constaté empiriquement). Choix assumé,
pas juste un contournement : les questions multi-hop générées lors du test isolé sonnaient de
toute façon artificielles (combinaison forcée d'événements sans rapport).

source_event_uids ajoutés à la main dans qa_dataset_generated.json après génération, pas par ce
script : le SingleTurnSample renvoyé par ragas (ragas/testset/synthesizers/single_hop/base.py)
ne contient que reference_contexts (texte source brut), jamais les métadonnées uid/title du
Document passées en entrée — aucun lien automatique possible vers l'événement source. Un essai de
matching automatique par mots-clés capitalisés (question/réponse vs texte des 18 événements
candidats) a été tenté puis abandonné : fiable sur 10/14 cas seulement, ambigu ou franchement faux
sur les 4 autres (ex. confondu deux randonnées cyclotouristes différentes dans des villes
voisines). Résolution manuelle finale par recoupement titre + ville + date sur data/processed/
events.json.

Coût : plusieurs appels à mistral-large-latest + mistral-embed. À lancer soi-même.
"""

import asyncio
import json
import os
import random
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

# Bug #1 (voir docstring) : doit s'exécuter AVANT tout import de ragas.
_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _stub

from langchain_core.documents import Document  # noqa: E402
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.testset import TestsetGenerator  # noqa: E402
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer  # noqa: E402

PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "events.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "qa_dataset_generated.json"

# Nombre de documents source : plus que les 6 du test isolé, pour une vraie diversité
# thématique/géographique, sans aller jusqu'au corpus complet (coût et temps de construction
# du graphe de connaissances trop élevés pour ce POC).
SAMPLE_SIZE = 18
TESTSET_SIZE = 14
RANDOM_SEED = 42  # reproductible : même échantillon à chaque exécution


def pick_diverse_sample(events: list[dict]) -> list[dict]:
    """Choisir des événements avec assez de texte, répartis sur des villes différentes.

    Marseille garantie dans l'échantillon (plusieurs entrées) : la ville la plus représentée
    dans le dataset, et celle utilisée dans tous nos scénarios manuels — un tirage aléatoire
    pur sur 106 villes distinctes pourrait sinon la manquer entièrement (constaté).
    """
    candidates = [e for e in events if len(e.get("text") or "") > 250]
    by_city: dict[str, list[dict]] = {}
    for e in candidates:
        by_city.setdefault(e.get("location_city") or "inconnu", []).append(e)

    rng = random.Random(RANDOM_SEED)

    sample = []
    if "Marseille" in by_city:
        sample.extend(rng.sample(by_city["Marseille"], k=min(3, len(by_city["Marseille"]))))

    other_cities = [c for c in by_city if c != "Marseille"]
    rng.shuffle(other_cities)
    for city in other_cities:
        if len(sample) >= SAMPLE_SIZE:
            break
        sample.append(rng.choice(by_city[city]))
    return sample


async def adapt_distribution_to_french(distribution, llm) -> None:
    for synthesizer, _weight in distribution:
        adapted_prompts = await synthesizer.adapt_prompts("french", llm=llm)
        synthesizer.set_prompts(**adapted_prompts)


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY manquant dans .env")

    events = json.loads(PROCESSED_PATH.read_text(encoding="utf-8"))
    sample = pick_diverse_sample(events)
    docs = [
        Document(page_content=e["text"], metadata={"uid": e["uid"], "title": e["title"]}) for e in sample
    ]
    print(f"{len(docs)} documents source (villes : {sorted({e.get('location_city') for e in sample})})")

    generator_llm = LangchainLLMWrapper(ChatMistralAI(mistral_api_key=api_key, model="mistral-large-latest"))
    generator_embeddings = LangchainEmbeddingsWrapper(MistralAIEmbeddings(mistral_api_key=api_key))
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

    # Un seul synthétiseur (single-hop), poids 1.0 — voir docstring pour pourquoi les
    # synthétiseurs multi-hop sont volontairement exclus.
    distribution = [(SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0)]
    asyncio.run(adapt_distribution_to_french(distribution, generator_llm))

    dataset = generator.generate_with_langchain_docs(docs, testset_size=TESTSET_SIZE, query_distribution=distribution)
    df = dataset.to_pandas()

    pairs = [
        {
            "question": row["user_input"],
            "reference_answer": row["reference"],
            "source": "ragas_generated",
            "synthesizer": row["synthesizer_name"],
        }
        for _, row in df.iterrows()
    ]
    OUTPUT_PATH.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(pairs)} paires sauvegardées dans {OUTPUT_PATH}")
    print("Relire chaque paire avant usage : langue (français attendu), exactitude de la réponse de référence.")


if __name__ == "__main__":
    main()
