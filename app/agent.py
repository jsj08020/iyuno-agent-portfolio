import os
from dotenv import load_dotenv
from google import genai

from retriever import (
    load_documents,
    create_chunks,
    create_vector_store,
    search,
)

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)


def build_context(results):
    context_parts = []

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"[문서 {i}]\n"
            f"출처: {result['source']}\n"
            f"내용:\n{result['text']}"
        )

    return "\n\n".join(context_parts)


def ask_agent(query, index, chunks):
    results = search(
        query=query,
        index=index,
        chunks=chunks,
        top_k=3
    )

    context = build_context(results)

    prompt = f"""
너는 공개 기술 문서를 기반으로 답변하는 AI Agent다.

아래 검색된 문서 내용만 근거로 사용해서 질문에 답변해라.
문서에 없는 내용은 추측하지 말고
"제공된 문서에서는 확인할 수 없습니다."라고 답변해라.

반드시 답변 마지막에 출처 파일명을 표시해라.

[검색된 문서]

{context}

[사용자 질문]

{query}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text, results


if __name__ == "__main__":
    if not API_KEY:
        print("GEMINI_API_KEY가 설정되지 않았습니다.")
        exit()

    documents = load_documents()
    chunks = create_chunks(documents)
    index = create_vector_store(chunks)

    print("RAG Agent 준비 완료")

    while True:
        query = input("\n질문 입력 (종료: exit): ")

        if query.lower() == "exit":
            break

        answer, results = ask_agent(
            query=query,
            index=index,
            chunks=chunks
        )

        print("\n===== AI 답변 =====")
        print(answer)

        print("\n===== 검색된 출처 =====")

        for result in results:
            print(
                f"- {result['source']} "
                f"(chunk {result['chunk_id']}, "
                f"score {result['score']:.4f})"
            )