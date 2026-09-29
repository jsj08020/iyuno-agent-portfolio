import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
EVALUATION_DIR = ROOT_DIR / "evaluation"

sys.path.insert(0, str(APP_DIR))


from retriever import (
    load_documents,
    create_chunks,
    create_vector_store,
    search,
    format_source,
)


METRICS_PATH = EVALUATION_DIR / "metrics.json"
QUESTIONS_PATH = EVALUATION_DIR / "questions.json"


# -----------------------------
# 평가 결과 불러오기
# -----------------------------

with open(
    METRICS_PATH,
    "r",
    encoding="utf-8"
) as f:
    metrics = json.load(f)


with open(
    QUESTIONS_PATH,
    "r",
    encoding="utf-8"
) as f:
    questions = json.load(f)


question_map = {
    question["id"]: question
    for question in questions
}


# -----------------------------
# MISS 목록 추출
# -----------------------------

misses = []

for item in metrics["results"]:

    hit = item.get(
        "recall_at_3_hit"
    )

    if hit is False:
        misses.append(item)


print()
print("=" * 80)
print("Retrieval MISS 상세 분석")
print("=" * 80)

print(
    f"\nMISS 질문 수: {len(misses)}"
)


# -----------------------------
# Vector Store 재생성
# -----------------------------

print("\n문서와 Vector Store를 불러오는 중...")

documents = load_documents()

chunks = create_chunks(
    documents
)

index = create_vector_store(
    chunks
)

print("준비 완료")


# -----------------------------
# MISS 상세 출력
# -----------------------------

for miss in misses:

    question_id = miss["id"]

    question_info = question_map[
        question_id
    ]

    query = question_info[
        "question"
    ]

    expected_terms = question_info.get(
        "expected_terms",
        []
    )

    results = search(
        query=query,
        index=index,
        chunks=chunks,
        top_k=3
    )

    print()
    print("=" * 80)

    print(
        f"Q{question_id}"
    )

    print(
        f"질문: {query}"
    )

    print(
        "Expected Terms:"
    )

    for term in expected_terms:
        print(
            f"  - {term}"
        )

    print()

    for rank, result in enumerate(
        results,
        start=1
    ):

        source = format_source(
            result
        )

        print(
            f"[TOP {rank}] "
            f"{source}"
        )

        print(
            f"Score: "
            f"{result['score']:.4f}"
        )

        print(
            "검색 내용:"
        )

        text = result["text"]

        print(
            text[:800]
        )

        print(
            "-" * 80
        )