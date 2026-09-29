import argparse
import json
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types
from dotenv import load_dotenv
import os


# =========================================================
# Path 설정
# =========================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
EVALUATION_DIR = ROOT_DIR / "evaluation"

sys.path.insert(
    0,
    str(APP_DIR)
)


from retriever import (
    load_documents,
    create_chunks,
    create_vector_store,
    format_source,
)

from agent import ask_agent


QUESTIONS_PATH = (
    EVALUATION_DIR
    / "questions.json"
)

METRICS_PATH = (
    EVALUATION_DIR
    / "metrics.json"
)


# =========================================================
# Gemini 설정
# =========================================================

load_dotenv(
    ROOT_DIR / ".env"
)

API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY가 .env에 없습니다."
    )


client = genai.Client(
    api_key=API_KEY
)


# =========================================================
# 데이터 로딩
# =========================================================

def load_questions():
    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def load_existing_metrics():

    if not METRICS_PATH.exists():
        return {}

    with open(
        METRICS_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


# =========================================================
# Context 생성
# =========================================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        source = format_source(
            result
        )

        context_parts.append(
            f"[근거 {i}]\n"
            f"출처: {source}\n"
            f"{result['text']}"
        )

    return "\n\n".join(
        context_parts
    )


# =========================================================
# Faithfulness Judge
# =========================================================

def judge_faithfulness(
    question,
    answer,
    retrieved_results
):

    context = build_context(
        retrieved_results
    )

    prompt = f"""
너는 RAG 시스템의 Faithfulness를 평가하는 심사자다.

Faithfulness는 AI 답변의 주장들이
제공된 검색 근거에 의해 얼마나 뒷받침되는지를 의미한다.

아래 질문, AI 답변, 검색 근거를 비교하라.

평가 규칙:

- 검색 근거에서 직접 확인 가능한 주장: faithful
- 검색 근거에 없는 사실을 추가한 경우: unfaithful
- 일반 상식이라도 검색 근거에 없다면 근거 없는 주장으로 본다.
- 문장 표현이 달라도 의미가 근거와 일치하면 인정한다.
- 답변의 주요 주장 대부분이 근거에 있으면 높은 점수를 준다.

0.0 ~ 1.0 사이의 점수를 부여하라.

1.0:
모든 주요 주장이 근거에 의해 뒷받침됨

0.75:
대부분 근거가 있으나 작은 추가 설명이 있음

0.5:
근거가 있는 내용과 없는 내용이 섞여 있음

0.25:
대부분 근거가 부족함

0.0:
근거와 무관하거나 사실상 환각에 가까움

반드시 아래 JSON 형식으로만 출력하라.

{{
    "score": 0.0,
    "reason": "평가 이유"
}}


[질문]

{question}


[AI 답변]

{answer}


[검색 근거]

{context}
"""

    max_retries = 3

    for attempt in range(
        max_retries
    ):

        try:

            response = (
                client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type=(
                            "application/json"
                        )
                    )
                )
            )

            result = json.loads(
                response.text
            )

            score = float(
                result.get(
                    "score",
                    0
                )
            )

            # 범위 강제
            score = max(
                0.0,
                min(
                    1.0,
                    score
                )
            )

            return {
                "score": score,
                "reason": result.get(
                    "reason",
                    ""
                )
            }

        except Exception as e:

            message = str(e)

            temporary_error = (
                "503" in message
                or
                "429" in message
                or
                "RESOURCE_EXHAUSTED"
                in message
                or
                "UNAVAILABLE"
                in message
                or
                "high demand"
                in message.lower()
            )

            if (
                temporary_error
                and
                attempt < max_retries - 1
            ):

                wait_time = (
                    3 * (
                        attempt + 1
                    )
                )

                print(
                    f"  Judge API 일시 오류. "
                    f"{wait_time}초 후 재시도..."
                )

                time.sleep(
                    wait_time
                )

                continue

            raise


# =========================================================
# Evaluation
# =========================================================

def run_evaluation(
    limit=None,
    delay=2.0
):

    questions = load_questions()

    # RAG 질문만 Faithfulness 평가
    rag_questions = [
        q
        for q in questions
        if q.get("type") == "rag"
    ]

    if limit is not None:
        rag_questions = (
            rag_questions[:limit]
        )

    print()
    print(
        "================================"
    )

    print(
        "   Faithfulness Evaluation"
    )

    print(
        "================================"
    )

    print(
        f"평가 질문 수: "
        f"{len(rag_questions)}"
    )

    print()

    # -----------------------------
    # RAG 준비
    # -----------------------------

    print(
        "문서 및 Vector Store 준비 중..."
    )

    documents = load_documents()

    chunks = create_chunks(
        documents
    )

    index = create_vector_store(
        chunks
    )

    print(
        f"페이지: {len(documents)}"
    )

    print(
        f"Chunk: {len(chunks)}"
    )

    print(
        "준비 완료"
    )

    print()

    results = []

    scores = []

    agent_latencies = []

    error_count = 0

    # -----------------------------
    # 질문별 평가
    # -----------------------------

    for i, question in enumerate(
        rag_questions,
        start=1
    ):

        query = question[
            "question"
        ]

        print(
            f"[{i}/"
            f"{len(rag_questions)}] "
            f"Q{question['id']}: "
            f"{query}"
        )

        item = {
            "id": question["id"],
            "question": query
        }

        try:

            # -------------------------
            # Agent 답변
            # -------------------------

            start = (
                time.perf_counter()
            )

            (
                answer,
                retrieved_results,
                tool_logs
            ) = ask_agent(
                query=query,
                index=index,
                chunks=chunks,
                top_k=3
            )

            end = (
                time.perf_counter()
            )

            latency = (
                end - start
            )

            agent_latencies.append(
                latency
            )

            item[
                "agent_latency_sec"
            ] = round(
                latency,
                3
            )

            item["answer"] = answer

            item[
                "retrieved_sources"
            ] = [
                format_source(
                    result
                )
                for result
                in retrieved_results
            ]

            # -------------------------
            # Faithfulness Judge
            # -------------------------

            judge_result = (
                judge_faithfulness(
                    question=query,
                    answer=answer,
                    retrieved_results=(
                        retrieved_results
                    )
                )
            )

            score = (
                judge_result["score"]
            )

            scores.append(
                score
            )

            item[
                "faithfulness_score"
            ] = score

            item[
                "faithfulness_reason"
            ] = judge_result[
                "reason"
            ]

            results.append(
                item
            )

            print(
                f"  Faithfulness: "
                f"{score:.2f}"
            )

            print(
                f"  Agent Latency: "
                f"{latency:.2f}s"
            )

        except Exception as e:

            error_count += 1

            item["error"] = str(e)

            results.append(
                item
            )

            print(
                f"  ERROR: {e}"
            )

        # API 과부하 방지
        if (
            i
            < len(rag_questions)
        ):

            time.sleep(
                delay
            )

        print()

    # =====================================================
    # Summary
    # =====================================================

    average_score = None

    if scores:
        average_score = (
            sum(scores)
            / len(scores)
        )

    average_latency = None

    if agent_latencies:
        average_latency = (
            sum(agent_latencies)
            / len(agent_latencies)
        )

    summary = {
        "questions_evaluated": (
            len(rag_questions)
        ),

        "successful_evaluations": (
            len(scores)
        ),

        "average_faithfulness": (
            round(
                average_score,
                4
            )
            if average_score
            is not None
            else None
        ),

        "average_agent_latency_sec": (
            round(
                average_latency,
                3
            )
            if average_latency
            is not None
            else None
        ),

        "errors": error_count,

        "judge_method": (
            "Gemini LLM-as-a-Judge"
        )
    }


    # =====================================================
    # 기존 metrics.json에 추가
    # =====================================================

    metrics = (
        load_existing_metrics()
    )

    metrics[
        "faithfulness_evaluation"
    ] = {
        "summary": summary,
        "results": results
    }


    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2
        )


    # =====================================================
    # 결과 출력
    # =====================================================

    print()
    print(
        "================================"
    )

    print(
        "      Faithfulness Result"
    )

    print(
        "================================"
    )

    print(
        f"평가 질문: "
        f"{summary['questions_evaluated']}"
    )

    if average_score is not None:

        print(
            f"평균 Faithfulness: "
            f"{average_score * 100:.1f}%"
        )

    if average_latency is not None:

        print(
            f"평균 Agent Latency: "
            f"{average_latency:.2f} sec"
        )

    print(
        f"오류 수: "
        f"{error_count}"
    )

    print()

    print(
        f"결과 저장: "
        f"{METRICS_PATH}"
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=2.0
    )

    args = parser.parse_args()

    run_evaluation(
        limit=args.limit,
        delay=args.delay
    )


if __name__ == "__main__":
    main()