import streamlit as st

from retriever import (
    load_documents,
    create_chunks,
    create_vector_store,
    format_source,
)

from agent import (
    ask_agent
)


st.set_page_config(
    page_title=(
        "Iyuno AI Agent Portfolio"
    ),
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def initialize_rag():

    documents = load_documents()

    if len(documents) == 0:

        raise RuntimeError(
            "data/documents 폴더에 "
            "문서가 없습니다."
        )

    chunks = create_chunks(
        documents
    )

    if len(chunks) == 0:

        raise RuntimeError(
            "문서에서 chunk를 "
            "생성하지 못했습니다."
        )

    index = create_vector_store(
        chunks
    )

    return (
        documents,
        chunks,
        index
    )


st.title(
    "🤖 Iyuno AI Agent Portfolio"
)

st.write(
    "공개 기술 문서를 검색하고, "
    "검색된 근거를 기반으로 답변하는 "
    "RAG AI Agent입니다."
)


try:

    (
        documents,
        chunks,
        index
    ) = initialize_rag()

    unique_files = set(
        document["source"]
        for document in documents
    )

    st.success(
        f"RAG 시스템 준비 완료 / "
        f"파일 {len(unique_files)}개 / "
        f"문서 페이지 {len(documents)}개 / "
        f"총 {len(chunks)}개 chunk"
    )

except Exception as e:

    st.error(
        f"RAG 초기화 실패: {e}"
    )

    st.stop()


st.divider()


query = st.text_input(
    "질문을 입력하세요",
    placeholder=(
        "예: 제로 트러스트 "
        "아키텍처란 무엇인가?"
    )
)


ask_button = st.button(
    "질문하기",
    type="primary"
)


if ask_button:

    if not query.strip():

        st.warning(
            "질문을 입력해주세요."
        )

    else:

        with st.spinner(
            "관련 문서를 검색하고 "
            "답변을 생성하는 중..."
        ):

            try:

                answer, results = (
                    ask_agent(
                        query=query,
                        index=index,
                        chunks=chunks,
                        top_k=3
                    )
                )

                st.subheader(
                    "🤖 AI 답변"
                )

                st.markdown(
                    answer
                )

                st.divider()

                st.subheader(
                    "📚 검색된 근거"
                )

                st.caption(
                    "아래 문서 조각들이 "
                    "AI 답변 생성에 "
                    "사용되었습니다."
                )

                for i, result in enumerate(
                    results,
                    start=1
                ):

                    source_text = (
                        format_source(
                            result
                        )
                    )

                    title = (
                        f"{i}. "
                        f"{source_text} "
                        f"| 유사도 "
                        f"{result['score']:.4f}"
                    )

                    with st.expander(
                        title
                    ):

                        col1, col2 = (
                            st.columns(2)
                        )

                        with col1:

                            st.write(
                                "**출처**"
                            )

                            st.write(
                                source_text
                            )

                        with col2:

                            st.write(
                                "**Chunk ID**"
                            )

                            st.write(
                                result[
                                    "chunk_id"
                                ]
                            )

                        st.write(
                            "**검색된 원문**"
                        )

                        st.write(
                            result["text"]
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

                    st.warning(
                        "현재 AI 모델 "
                        "사용량이 많아 "
                        "응답 생성이 "
                        "지연되고 있습니다. "
                        "잠시 후 다시 "
                        "시도해주세요."
                    )

                else:

                    st.error(
                        "답변 생성 중 "
                        f"오류 발생: {e}"
                    )