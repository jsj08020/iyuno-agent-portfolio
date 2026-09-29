import streamlit as st

from retriever import (
    load_documents,
    create_chunks,
    create_vector_store,
)

from agent import ask_agent


st.set_page_config(
    page_title="Iyuno AI Agent Portfolio",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def initialize_rag():
    documents = load_documents()
    chunks = create_chunks(documents)
    index = create_vector_store(chunks)

    return index, chunks


st.title("🤖 Iyuno AI Agent Portfolio")

st.write(
    "공개 기술 문서를 검색하고, "
    "검색된 근거를 기반으로 답변하는 RAG AI Agent입니다."
)


try:
    index, chunks = initialize_rag()

    st.success(
        f"RAG 시스템 준비 완료 / 총 {len(chunks)}개 chunk"
    )

except Exception as e:
    st.error(f"RAG 초기화 실패: {e}")
    st.stop()


query = st.text_input(
    "질문을 입력하세요",
    placeholder="예: IDS와 IPS의 차이가 뭐야?"
)


if st.button("질문하기"):

    if not query.strip():
        st.warning("질문을 입력해주세요.")

    else:

        with st.spinner("문서를 검색하고 답변을 생성하는 중..."):

            try:

                answer, results = ask_agent(
                    query=query,
                    index=index,
                    chunks=chunks
                )

                st.subheader("AI 답변")

                st.write(answer)

                st.subheader("검색된 출처")

                for i, result in enumerate(results, start=1):

                    with st.expander(
                        f"{i}. {result['source']} "
                        f"(score: {result['score']:.4f})"
                    ):

                        st.write(
                            f"Chunk ID: {result['chunk_id']}"
                        )

                        st.write(result["text"])

            except Exception as e:

                st.error(f"답변 생성 중 오류 발생: {e}")