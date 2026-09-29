import os
import time

from dotenv import load_dotenv
from google import genai

from retriever import (
    load_documents,
    create_chunks,
    create_vector_store,
    search,
    format_source,
)


load_dotenv()

API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY가 "
        ".env 파일에 설정되어 있지 않습니다."
    )


client = genai.Client(
    api_key=API_KEY
)


def build_context(results):
    """
    검색 결과를 Gemini에게 전달할
    context 문자열로 변환합니다.
    """

    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        source_text = format_source(
            result
        )

        context_parts.append(
            f"[검색 문서 {i}]\n"
            f"출처: {source_text}\n"
            f"내용:\n"
            f"{result['text']}"
        )

    return "\n\n".join(
        context_parts
    )


def ask_agent(
    query,
    index,
    chunks,
    top_k=3
):

    # 1. 관련 문서 검색
    results = search(
        query=query,
        index=index,
        chunks=chunks,
        top_k=top_k
    )

    if len(results) == 0:
        return (
            "관련 문서를 찾지 못했습니다.",
            results
        )

    # 2. 검색 문서들을 context로 구성
    context = build_context(
        results
    )

    # 3. Gemini prompt
    prompt = f"""
너는 공개 기술 문서를 기반으로 질문에 답변하는
RAG AI Agent다.

반드시 아래 규칙을 지켜라.

1. [검색된 문서]에 제공된 내용만 근거로 답변한다.

2. 검색 문서에 없는 사실을 임의로 추측하거나
   만들어내지 않는다.

3. 문서만으로 질문에 답변할 수 없다면
   "제공된 문서에서는 확인할 수 없습니다."
   라고 명확하게 말한다.

4. 답변은 한국어로 작성한다.

5. 전문용어는 필요한 경우 쉽게 설명한다.

6. 답변에서 중요한 주장에는 가능한 경우
   관련 문서 출처를 표시한다.

7. 답변 마지막에는 반드시
   "출처" 항목을 만들어 사용한
   파일명과 페이지 번호를 표시한다.

8. 제공되지 않은 페이지 번호를
   임의로 만들어내지 않는다.


[검색된 문서]

{context}


[사용자 질문]

{query}
"""

    max_retries = 3

    for attempt in range(
        max_retries
    ):

        try:

            response = (
                client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
            )

            return (
                response.text,
                results
            )

        except Exception as e:

            error_message = str(e)

            temporary_error = (
                "503" in error_message
                or
                "UNAVAILABLE"
                in error_message
                or
                "high demand"
                in error_message.lower()
            )

            if (
                temporary_error
                and
                attempt < max_retries - 1
            ):

                wait_time = (
                    2 ** attempt
                )

                print(
                    "Gemini 서버 "
                    "일시 오류 발생. "
                    f"{wait_time}초 후 "
                    "재시도합니다. "
                    f"({attempt + 1}/"
                    f"{max_retries})"
                )

                time.sleep(
                    wait_time
                )

                continue

            raise


if __name__ == "__main__":

    print(
        "문서를 불러오는 중..."
    )

    documents = load_documents()

    if len(documents) == 0:

        print(
            "data/documents 폴더에 "
            "문서가 없습니다."
        )

        exit()

    chunks = create_chunks(
        documents
    )

    print(
        f"읽은 문서 페이지 수: "
        f"{len(documents)}"
    )

    print(
        f"생성된 chunk 수: "
        f"{len(chunks)}"
    )

    index = create_vector_store(
        chunks
    )

    print(
        f"FAISS 저장 chunk 수: "
        f"{index.ntotal}"
    )

    print(
        "RAG Agent 준비 완료"
    )

    while True:

        query = input(
            "\n질문 입력 "
            "(종료: exit): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        try:

            answer, results = (
                ask_agent(
                    query=query,
                    index=index,
                    chunks=chunks,
                    top_k=3
                )
            )

            print(
                "\n===== AI 답변 ====="
            )

            print(
                answer
            )

            print(
                "\n===== 검색된 출처 ====="
            )

            for result in results:

                source_text = (
                    format_source(
                        result
                    )
                )

                print(
                    f"- {source_text} "
                    f"| chunk "
                    f"{result['chunk_id']} "
                    f"| score "
                    f"{result['score']:.4f}"
                )

        except Exception as e:

            error_message = str(e)

            if (
                "503" in error_message
                or
                "UNAVAILABLE"
                in error_message
                or
                "high demand"
                in error_message.lower()
            ):

                print(
                    "\n현재 Gemini 모델 "
                    "사용량이 많습니다. "
                    "잠시 후 다시 "
                    "질문해주세요."
                )

            else:

                print(
                    f"\n오류 발생: {e}"
                )