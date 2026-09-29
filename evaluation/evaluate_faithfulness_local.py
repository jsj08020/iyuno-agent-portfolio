import json
import re
import sys
from pathlib import Path

import numpy as np


# =========================================================
# Path
# =========================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parents[1]
)

APP_DIR = (
    ROOT_DIR
    / "app"
)

EVALUATION_DIR = (
    ROOT_DIR
    / "evaluation"
)

sys.path.insert(
    0,
    str(APP_DIR)
)


from retriever import model


METRICS_PATH = (
    EVALUATION_DIR
    / "metrics.json"
)


# =========================================================
# Sentence Split
# =========================================================

def split_sentences(text):

    if not text:
        return []

    # 출처 부분은 평가에서 제외
    text = re.split(
        r"\n#+\s*출처|\n출처\s*:?",
        text,
        maxsplit=1,
        flags=re.IGNORECASE
    )[0]

    sentences = re.split(
        r"(?<=[.!?。])\s+|\n+",
        text
    )

    cleaned = []

    for sentence in sentences:

        sentence = (
            sentence.strip()
        )

        # 너무 짧은 문장은 평가 제외
        if len(sentence) < 15:
            continue

        cleaned.append(
            sentence
        )

    return cleaned


# =========================================================
# Similarity
# =========================================================

def cosine_similarity(
    a,
    b
):

    return float(
        np.dot(
            a,
            b
        )
    )


# =========================================================
# Faithfulness Proxy
# =========================================================

def calculate_faithfulness(
    answer,
    contexts,
    threshold=0.70
):

    sentences = split_sentences(
        answer
    )


    if not sentences:

        return {

            "score": 0.0,

            "supported_sentences": 0,

            "total_sentences": 0,

            "details": []
        }


    if not contexts:

        return {

            "score": 0.0,

            "supported_sentences": 0,

            "total_sentences": (
                len(sentences)
            ),

            "details": []
        }


    # -----------------------------------------------------
    # E5 prefix
    # -----------------------------------------------------

    sentence_inputs = [

        "query: " + sentence

        for sentence in sentences
    ]


    context_inputs = [

        "passage: " + context

        for context in contexts
    ]


    # -----------------------------------------------------
    # Embedding
    # -----------------------------------------------------

    sentence_embeddings = (
        model.encode(
            sentence_inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    )


    context_embeddings = (
        model.encode(
            context_inputs,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
    )


    supported_count = 0

    details = []


    # =====================================================
    # Sentence-by-Sentence
    # =====================================================

    for index, sentence_embedding in enumerate(
        sentence_embeddings
    ):

        similarities = [

            cosine_similarity(
                sentence_embedding,
                context_embedding
            )

            for context_embedding
            in context_embeddings
        ]


        best_score = max(
            similarities
        )


        best_context_index = int(
            np.argmax(
                similarities
            )
        )


        supported = (
            best_score
            >= threshold
        )


        if supported:

            supported_count += 1


        details.append(
            {

                "sentence": (
                    sentences[index]
                ),

                "best_similarity": (
                    round(
                        best_score,
                        4
                    )
                ),

                "supported": (
                    supported
                ),

                "best_context_index": (
                    best_context_index
                )
            }
        )


    score = (
        supported_count
        / len(sentences)
    )


    return {

        "score": round(
            score,
            4
        ),

        "supported_sentences": (
            supported_count
        ),

        "total_sentences": (
            len(sentences)
        ),

        "details": details
    }


# =========================================================
# Main
# =========================================================

def main():

    if not METRICS_PATH.exists():

        print(
            "metrics.json이 없습니다."
        )

        return


    with open(
        METRICS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)


    results = data.get(
        "results",
        []
    )


    evaluated_results = []

    scores = []


    print()
    print(
        "================================"
    )

    print(
        " Local Faithfulness Evaluation"
    )

    print(
        "================================"
    )

    print()


    # =====================================================
    # Evaluation
    # =====================================================

    for item in results:

        # RAG 질문만 평가
        if item.get(
            "type"
        ) != "rag":

            continue


        answer = item.get(
            "answer"
        )


        contexts = item.get(
            "retrieved_contexts",
            []
        )


        # retrieval-only 결과는
        # answer가 없기 때문에 스킵
        if not answer:

            continue


        if not contexts:

            continue


        faithfulness = (
            calculate_faithfulness(
                answer=answer,
                contexts=contexts,
                threshold=0.70
            )
        )


        score = (
            faithfulness[
                "score"
            ]
        )


        scores.append(
            score
        )


        result_item = {

            "id": item.get(
                "id"
            ),

            "question": item.get(
                "question"
            ),

            "faithfulness_proxy": (
                score
            ),

            "supported_sentences": (
                faithfulness[
                    "supported_sentences"
                ]
            ),

            "total_sentences": (
                faithfulness[
                    "total_sentences"
                ]
            ),

            "details": (
                faithfulness[
                    "details"
                ]
            )
        }


        evaluated_results.append(
            result_item
        )


        print(
            f"Q{item.get('id')} "
            f"Faithfulness Proxy: "
            f"{score * 100:.1f}% "
            f"("
            f"{faithfulness['supported_sentences']}/"
            f"{faithfulness['total_sentences']}"
            f")"
        )


    # =====================================================
    # No Answers
    # =====================================================

    if not scores:

        print()
        print(
            "평가 가능한 AI 답변이 없습니다."
        )

        print()
        print(
            "현재 metrics.json이 "
            "Retrieval-only 평가 결과라면 정상입니다."
        )

        print()
        print(
            "Gemini quota가 복구된 뒤:"
        )

        print()

        print(
            "python evaluation\\evaluate.py "
            "--full --limit 5"
        )

        print()

        print(
            "그 다음:"
        )

        print()

        print(
            "python evaluation\\"
            "evaluate_faithfulness_local.py"
        )

        return


    # =====================================================
    # Summary
    # =====================================================

    average_score = (
        sum(scores)
        / len(scores)
    )


    total_supported = sum(

        item[
            "supported_sentences"
        ]

        for item
        in evaluated_results
    )


    total_sentences = sum(

        item[
            "total_sentences"
        ]

        for item
        in evaluated_results
    )


    overall_sentence_support = (

        total_supported
        / total_sentences

        if total_sentences > 0

        else 0.0
    )


    faithfulness_data = {

        "method": (
            "Embedding-based "
            "Faithfulness Proxy"
        ),

        "model": (
            "intfloat/"
            "multilingual-e5-base"
        ),

        "threshold": 0.70,

        "description": (
            "Each answer sentence is "
            "compared against the retrieved "
            "RAG contexts using normalized "
            "E5 embedding cosine similarity."
        ),

        "questions_evaluated": (
            len(scores)
        ),

        "average_faithfulness_proxy": (
            round(
                average_score,
                4
            )
        ),

        "overall_sentence_support_rate": (
            round(
                overall_sentence_support,
                4
            )
        ),

        "supported_sentences": (
            total_supported
        ),

        "total_sentences": (
            total_sentences
        ),

        "results": (
            evaluated_results
        )
    }


    # =====================================================
    # metrics.json에 저장
    # =====================================================

    data[
        "local_faithfulness_evaluation"
    ] = faithfulness_data


    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


    # =====================================================
    # Print
    # =====================================================

    print()
    print(
        "================================"
    )

    print(
        " Faithfulness Evaluation Result"
    )

    print(
        "================================"
    )


    print(
        f"평가 질문 수: "
        f"{len(scores)}"
    )


    print(
        f"평균 Faithfulness Proxy: "
        f"{average_score * 100:.1f}%"
    )


    print(
        "전체 문장 Support Rate: "
        f"{overall_sentence_support * 100:.1f}%"
    )


    print(
        f"지원 문장: "
        f"{total_supported}/"
        f"{total_sentences}"
    )


    print()

    print(
        f"결과 저장: "
        f"{METRICS_PATH}"
    )


if __name__ == "__main__":
    main()