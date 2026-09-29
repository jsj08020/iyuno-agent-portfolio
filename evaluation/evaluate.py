import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path


# =========================================================
# Path
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
    search,
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
# Load
# =========================================================

def load_questions():

    with open(
        QUESTIONS_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# =========================================================
# Text Normalization
# =========================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text).lower()

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    # 하이픈/대시 → 공백
    text = re.sub(
        r"[-‐-‒–—]",
        " ",
        text
    )

    # 특수문자 정리
    text = re.sub(
        r"[^a-z0-9가-힣\s]",
        " ",
        text
    )

    # 중복 공백 제거
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# Retrieval Evaluation
# =========================================================

def check_retrieval_hit(
    question,
    results
):

    expected_terms = question.get(
        "expected_terms",
        []
    )

    if not expected_terms:
        return None, []

    combined_text = "\n".join(
        result["text"]
        for result in results
    )

    combined_text = normalize_text(
        combined_text
    )

    matched_terms = []

    for term in expected_terms:

        normalized_term = normalize_text(
            term
        )

        if (
            normalized_term
            in combined_text
        ):

            matched_terms.append(
                term
            )

    hit = (
        len(matched_terms) > 0
    )

    return (
        hit,
        matched_terms
    )


# =========================================================
# Tool Evaluation
# =========================================================

def check_tool_call(
    question,
    tool_logs
):

    expected_tool = question.get(
        "expected_tool"
    )

    if not expected_tool:
        return None

    used_tools = [
        log.get("tool")
        for log in tool_logs
    ]

    return (
        expected_tool
        in used_tools
    )


# =========================================================
# Main Evaluation
# =========================================================

def evaluate(
    questions,
    top_k=3,
    full=False,
    limit=None
):

    print()
    print(
        "================================"
    )
    print(
        "      Evaluation Start"
    )
    print(
        "================================"
    )

    print()
    print(
        "문서 로딩 중..."
    )

    documents = load_documents()

    if not documents:

        raise RuntimeError(
            "data/documents에 "
            "평가할 문서가 없습니다."
        )

    chunks = create_chunks(
        documents
    )

    print(
        f"문서 페이지 수: "
        f"{len(documents)}"
    )

    print(
        f"Chunk 수: "
        f"{len(chunks)}"
    )

    print()
    print(
        "Vector Store 생성 중..."
    )

    index = create_vector_store(
        chunks
    )

    print(
        "Vector Store 준비 완료"
    )


    # -----------------------------------------------------
    # Question 제한
    # -----------------------------------------------------

    if limit is not None:

        questions = (
            questions[:limit]
        )


    total_count = len(
        questions
    )


    print()
    print(
        f"평가 질문 수: "
        f"{total_count}"
    )

    print(
        "실행 모드: "
        + (
            "FULL AGENT"
            if full
            else "RETRIEVAL ONLY"
        )
    )

    print()


    # -----------------------------------------------------
    # Metric 변수
    # -----------------------------------------------------

    evaluation_results = []

    rag_total = 0
    rag_hits = 0

    tool_total = 0
    tool_success = 0

    retrieval_latencies = []

    agent_latencies = []

    error_count = 0


    # =====================================================
    # Question Loop
    # =====================================================

    for current, question in enumerate(
        questions,
        start=1
    ):

        question_id = question[
            "id"
        ]

        question_type = question[
            "type"
        ]

        query = question[
            "question"
        ]

        print(
            f"[{current}/{total_count}] "
            f"Q{question_id}"
        )

        print(
            f"  {query}"
        )


        item = {
            "id": question_id,
            "type": question_type,
            "question": query
        }


        try:

            # =================================================
            # Retrieval
            # =================================================

            retrieval_start = (
                time.perf_counter()
            )

            results = search(
                query=query,
                index=index,
                chunks=chunks,
                top_k=top_k
            )

            retrieval_end = (
                time.perf_counter()
            )


            retrieval_latency_ms = (
                (
                    retrieval_end
                    - retrieval_start
                )
                * 1000
            )


            retrieval_latencies.append(
                retrieval_latency_ms
            )


            item[
                "retrieval_latency_ms"
            ] = round(
                retrieval_latency_ms,
                2
            )


            # 출처 저장
            item[
                "retrieved_sources"
            ] = [

                {
                    "source": (
                        format_source(
                            result
                        )
                    ),

                    "score": round(
                        result["score"],
                        4
                    )
                }

                for result in results
            ]


            # ★ Faithfulness용 실제 context 저장
            item[
                "retrieved_contexts"
            ] = [

                result["text"]

                for result in results
            ]


            # =================================================
            # RAG Recall
            # =================================================

            if question_type == "rag":

                rag_total += 1

                (
                    retrieval_hit,
                    matched_terms
                ) = check_retrieval_hit(
                    question,
                    results
                )


                item[
                    f"recall_at_{top_k}_hit"
                ] = retrieval_hit


                item[
                    "matched_terms"
                ] = matched_terms


                if retrieval_hit:

                    rag_hits += 1


                print(
                    "  Retrieval: "
                    + (
                        "HIT"
                        if retrieval_hit
                        else "MISS"
                    )
                )


            # =================================================
            # Full Agent Evaluation
            # =================================================

            if full:

                agent_start = (
                    time.perf_counter()
                )


                (
                    answer,
                    agent_results,
                    tool_logs
                ) = ask_agent(
                    query=query,
                    index=index,
                    chunks=chunks,
                    top_k=top_k
                )


                agent_end = (
                    time.perf_counter()
                )


                agent_latency = (
                    agent_end
                    - agent_start
                )


                agent_latencies.append(
                    agent_latency
                )


                item[
                    "agent_latency_sec"
                ] = round(
                    agent_latency,
                    3
                )


                item[
                    "answer"
                ] = answer


                item[
                    "tool_logs"
                ] = tool_logs


                print(
                    f"  Agent Latency: "
                    f"{agent_latency:.2f}s"
                )


                # =============================================
                # Tool
                # =============================================

                if question_type == "tool":

                    tool_total += 1


                    success = (
                        check_tool_call(
                            question,
                            tool_logs
                        )
                    )


                    item[
                        "tool_call_success"
                    ] = success


                    if success:

                        tool_success += 1


                    print(
                        "  Tool Calling: "
                        + (
                            "SUCCESS"
                            if success
                            else "FAIL"
                        )
                    )


            elif question_type == "tool":

                item[
                    "tool_call_success"
                ] = None


                item[
                    "note"
                ] = (
                    "Tool 평가는 "
                    "--full 모드에서 실행"
                )


            evaluation_results.append(
                item
            )


        except Exception as e:

            error_count += 1

            item[
                "error"
            ] = str(e)

            evaluation_results.append(
                item
            )

            print(
                f"  ERROR: {e}"
            )


        print()


    # =====================================================
    # Summary
    # =====================================================

    recall_at_k = None

    if rag_total > 0:

        recall_at_k = (
            rag_hits
            / rag_total
        )


    average_retrieval_latency = None

    if retrieval_latencies:

        average_retrieval_latency = (
            sum(
                retrieval_latencies
            )
            / len(
                retrieval_latencies
            )
        )


    average_agent_latency = None

    if agent_latencies:

        average_agent_latency = (
            sum(
                agent_latencies
            )
            / len(
                agent_latencies
            )
        )


    tool_success_rate = None

    if tool_total > 0:

        tool_success_rate = (
            tool_success
            / tool_total
        )


    summary = {

        "timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "mode": (
            "full"
            if full
            else "retrieval_only"
        ),

        "top_k": top_k,

        "total_questions": (
            total_count
        ),

        "rag_questions": (
            rag_total
        ),

        "rag_hits": (
            rag_hits
        ),

        f"recall_at_{top_k}": (
            round(
                recall_at_k,
                4
            )
            if recall_at_k
            is not None
            else None
        ),

        "average_retrieval_latency_ms": (
            round(
                average_retrieval_latency,
                2
            )
            if average_retrieval_latency
            is not None
            else None
        ),

        "agent_questions_evaluated": (
            len(
                agent_latencies
            )
        ),

        "average_agent_latency_sec": (
            round(
                average_agent_latency,
                3
            )
            if average_agent_latency
            is not None
            else None
        ),

        "tool_questions_evaluated": (
            tool_total
        ),

        "tool_call_success_count": (
            tool_success
        ),

        "tool_call_success_rate": (
            round(
                tool_success_rate,
                4
            )
            if tool_success_rate
            is not None
            else None
        ),

        "errors": error_count
    }


    # =====================================================
    # 기존 Local Faithfulness 보존
    # =====================================================

    old_data = {}

    if METRICS_PATH.exists():

        try:

            with open(
                METRICS_PATH,
                "r",
                encoding="utf-8"
            ) as f:

                old_data = json.load(
                    f
                )

        except Exception:

            old_data = {}


    output = {

        "summary": summary,

        "results": (
            evaluation_results
        )
    }


    # 이전 faithfulness 결과 있으면 보존
    if (
        "local_faithfulness_evaluation"
        in old_data
    ):

        output[
            "local_faithfulness_evaluation"
        ] = old_data[
            "local_faithfulness_evaluation"
        ]


    # =====================================================
    # Save
    # =====================================================

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )


    return summary


# =========================================================
# CLI
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Iyuno RAG Agent Evaluation"
        )
    )


    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Gemini Agent + "
            "Tool Calling까지 실행"
        )
    )


    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "앞에서부터 N개만 평가"
        )
    )


    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Retriever Top-K"
    )


    args = parser.parse_args()


    questions = load_questions()


    summary = evaluate(
        questions=questions,
        top_k=args.top_k,
        full=args.full,
        limit=args.limit
    )


    # =====================================================
    # Result Print
    # =====================================================

    print()
    print(
        "================================"
    )

    print(
        "       Evaluation Result"
    )

    print(
        "================================"
    )


    print(
        f"총 질문 수: "
        f"{summary['total_questions']}"
    )


    recall_key = (
        f"recall_at_"
        f"{summary['top_k']}"
    )


    recall = summary[
        recall_key
    ]


    if recall is not None:

        print(
            f"Recall@"
            f"{summary['top_k']}: "
            f"{recall * 100:.1f}%"
        )


    retrieval_latency = summary[
        "average_retrieval_latency_ms"
    ]


    if (
        retrieval_latency
        is not None
    ):

        print(
            "평균 Retrieval Latency: "
            f"{retrieval_latency:.2f} ms"
        )


    agent_latency = summary[
        "average_agent_latency_sec"
    ]


    if (
        agent_latency
        is not None
    ):

        print(
            "평균 Agent Latency: "
            f"{agent_latency:.2f} sec"
        )


    tool_rate = summary[
        "tool_call_success_rate"
    ]


    if tool_rate is not None:

        print(
            "Tool Calling 성공률: "
            f"{tool_rate * 100:.1f}%"
        )


    print(
        f"오류 수: "
        f"{summary['errors']}"
    )


    print()

    print(
        "결과 저장:"
    )

    print(
        METRICS_PATH
    )


if __name__ == "__main__":
    main()