import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from retriever import (
    load_documents,
    create_chunks,
    create_vector_store,
    search,
    format_source,
)

from tools import (
    calculate_expression,
    lookup_cve,
    reset_tool_log,
    get_tool_log,
)


load_dotenv()

API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY가 "
        ".env에 설정되어 있지 않습니다."
    )


client = genai.Client(
    api_key=API_KEY
)


def build_context(results):

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

    # 이전 tool log 초기화
    reset_tool_log()

    # RAG 검색
    results = search(
        query=query,
        index=index,
        chunks=chunks,
        top_k=top_k
    )

    context = build_context(
        results
    )

    prompt = f"""
너는 사이버보안 기술 문서를 기반으로
질문에 답변하는 RAG AI Agent다.

너에게는 다음 도구도 제공된다.

1. calculate_expression
   - 정확한 수학 계산이 필요한 경우 사용한다.

2. lookup_cve
   - 사용자가 CVE ID에 대해 질문할 경우
     NIST NVD API를 조회하는 데 사용한다.

규칙:

1. 일반적인 보안 개념 질문은
   아래 검색된 문서를 우선 근거로 사용한다.

2. CVE 번호가 포함된 질문은
   가능하면 lookup_cve 도구를 사용한다.

3. 계산이 필요한 경우
   직접 암산하지 말고 calculate_expression을 사용한다.

4. 문서에 없는 내용을
   임의로 만들어내지 않는다.

5. tool 결과를 사용했다면
   어떤 도구의 정보를 사용했는지
   답변에 명확하게 표시한다.

6. 문서를 사용했다면
   답변 마지막에 파일명과 페이지 번호를 표시한다.

7. NVD API 결과를 사용했다면
   출처를 "NIST NVD API"라고 표시한다.

8. 답변은 한국어로 작성한다.


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
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[
                            calculate_expression,
                            lookup_cve
                        ],
                        automatic_function_calling=(
                            types.AutomaticFunctionCallingConfig(
                                maximum_remote_calls=3
                            )
                        )
                    )
                )
            )

            tool_logs = get_tool_log()

            return (
                response.text,
                results,
                tool_logs
            )

        except Exception as e:

            error_message = str(e)

            temporary_error = (
                "503" in error_message
                or "UNAVAILABLE"
                in error_message
                or "high demand"
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
                    f"Gemini 서버 일시 오류. "
                    f"{wait_time}초 후 재시도."
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

    chunks = create_chunks(
        documents
    )

    index = create_vector_store(
        chunks
    )

    print(
        "RAG + Tool Calling Agent 준비 완료"
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

            (
                answer,
                results,
                tool_logs
            ) = ask_agent(
                query=query,
                index=index,
                chunks=chunks
            )

            print(
                "\n===== AI 답변 ====="
            )

            print(answer)

            print(
                "\n===== Tool Calls ====="
            )

            if tool_logs:
                for log in tool_logs:
                    print(log)

            else:
                print(
                    "사용된 Tool 없음"
                )

        except Exception as e:

            print(
                f"오류 발생: {e}"
            )