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
    page_title="Iyuno AI Agent Portfolio",
    page_icon="🤖",
    layout="wide"
)


@st.cache_resource
def initialize_rag():

    documents = load_documents()

    if not documents:
        raise RuntimeError(
            "data/documents 폴더에 "
            "문서가 없습니다."
        )

    chunks = create_chunks(
        documents
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
    "RAG 검색과 Tool Calling을 결합한 "
    "사이버보안 AI Agent입니다."
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
        f"시스템 준비 완료 / "
        f"파일 {len(unique_files)}개 / "
        f"페이지 {len(documents)}개 / "
        f"chunk {len(chunks)}개"
    )

except Exception as e:

    st.error(
        f"초기화 실패: {e}"
    )

    st.stop()


st.info(
    "사용 가능한 Tool: "
    "🧮 Calculator / "
    "🔐 NIST NVD CVE Lookup"
)


query = st.text_input(
    "질문을 입력하세요",
    placeholder=(
        "예: CVE-2021-44228의 "
        "CVSS 점수를 알려줘"
    )
)


if st.button(
    "질문하기",
    type="primary"
):

    if not query.strip():

        st.warning(
            "질문을 입력해주세요."
        )

    else:

        with st.spinner(
            "문서 검색 및 Tool 실행 중..."
        ):

            try:

                (
                    answer,
                    results,
                    tool_logs
                ) = ask_agent(
                    query=query,
                    index=index,
                    chunks=chunks,
                    top_k=3
                )

                st.subheader(
                    "🤖 AI 답변"
                )

                st.markdown(
                    answer
                )

                st.divider()

                st.subheader(
                    "🛠️ Tool Calling"
                )

                if tool_logs:

                    for log in tool_logs:

                        tool_name = log[
                            "tool"
                        ]

                        if (
                            tool_name
                            == "lookup_cve"
                        ):
                            icon = "🔐"

                        else:
                            icon = "🧮"

                        with st.expander(
                            f"{icon} {tool_name}",
                            expanded=True
                        ):

                            st.json(
                                log
                            )

                else:

                    st.caption(
                        "이번 질문에서는 "
                        "외부 Tool이 필요하지 않았습니다."
                    )

                st.divider()

                st.subheader(
                    "📚 RAG 검색 근거"
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

                    with st.expander(
                        f"{i}. "
                        f"{source_text} "
                        f"| score "
                        f"{result['score']:.4f}"
                    ):

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
                ):

                    st.warning(
                        "Gemini 서버가 "
                        "혼잡합니다. "
                        "잠시 후 다시 "
                        "시도해주세요."
                    )

                else:

                    st.error(
                        f"오류 발생: {e}"
                    )