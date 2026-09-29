from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DOCUMENT_DIR = Path("data/documents")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(MODEL_NAME)


def load_txt(file_path: Path):
    """
    TXT 파일을 읽습니다.
    TXT는 페이지 개념이 없으므로 page=None으로 저장합니다.
    """

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    return [
        {
            "source": file_path.name,
            "page": None,
            "text": text
        }
    ]


def load_pdf(file_path: Path):
    """
    PDF를 페이지 단위로 읽습니다.

    각 페이지마다:
    {
        source: 파일명,
        page: 실제 페이지 번호,
        text: 페이지 내용
    }
    형태로 반환합니다.
    """

    reader = PdfReader(file_path)

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        text = page.extract_text()

        if not text:
            continue

        text = text.strip()

        if not text:
            continue

        documents.append(
            {
                "source": file_path.name,
                "page": page_number,
                "text": text
            }
        )

    return documents


def load_documents():
    """
    data/documents 폴더의 PDF와 TXT를 읽습니다.
    """

    documents = []

    if not DOCUMENT_DIR.exists():
        DOCUMENT_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

    for file_path in DOCUMENT_DIR.iterdir():

        if file_path.name.startswith("."):
            continue

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".txt":
                loaded = load_txt(file_path)

            elif suffix == ".pdf":
                loaded = load_pdf(file_path)

            else:
                continue

            documents.extend(loaded)

        except Exception as e:
            print(
                f"[경고] {file_path.name} 읽기 실패: {e}"
            )

    return documents


def chunk_text(
    text: str,
    chunk_size=800,
    overlap=150
):
    """
    긴 텍스트를 일정 크기의 chunk로 나눕니다.

    chunk_size:
        chunk 최대 문자 수

    overlap:
        이전 chunk와 겹치는 문자 수
    """

    if not text:
        return []

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        next_start = end - overlap

        if next_start <= start:
            break

        start = next_start

    return chunks


def create_chunks(documents):
    """
    페이지별 문서를 검색 가능한 chunk로 변환합니다.
    """

    chunks = []

    global_chunk_id = 0

    for document in documents:

        text_chunks = chunk_text(
            document["text"]
        )

        for page_chunk_id, chunk in enumerate(
            text_chunks
        ):

            chunks.append(
                {
                    "source": document["source"],
                    "page": document["page"],
                    "chunk_id": global_chunk_id,
                    "page_chunk_id": page_chunk_id,
                    "text": chunk
                }
            )

            global_chunk_id += 1

    return chunks


def create_vector_store(chunks):
    """
    모든 chunk를 embedding하여
    FAISS vector index를 생성합니다.
    """

    if len(chunks) == 0:
        raise ValueError(
            "Embedding할 chunk가 없습니다."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    # cosine similarity와 유사하게 사용
    # embedding을 normalize했기 때문에
    # Inner Product를 사용
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


def search(
    query,
    index,
    chunks,
    top_k=3
):
    """
    질문을 embedding하여
    가장 관련성이 높은 chunk를 검색합니다.
    """

    if not query.strip():
        return []

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    search_count = min(
        top_k,
        len(chunks)
    )

    scores, indices = index.search(
        query_embedding,
        search_count
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        result = chunks[idx].copy()

        result["score"] = float(score)

        results.append(result)

    return results


def format_source(result):
    """
    출처 표시를 통일합니다.

    PDF:
    filename.pdf / p.10

    TXT:
    filename.txt
    """

    source = result["source"]
    page = result.get("page")

    if page is not None:
        return f"{source} / p.{page}"

    return source


if __name__ == "__main__":

    print("문서를 읽는 중...")

    documents = load_documents()

    print(
        f"읽은 문서 페이지 수: "
        f"{len(documents)}"
    )

    if len(documents) == 0:

        print(
            "data/documents 폴더에 "
            "PDF 또는 TXT 문서를 넣어주세요."
        )

        exit()

    chunks = create_chunks(
        documents
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
        "\n검색 시스템 준비 완료"
    )

    while True:

        query = input(
            "\n질문 입력 (종료: exit): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        results = search(
            query=query,
            index=index,
            chunks=chunks,
            top_k=3
        )

        print(
            "\n===== 검색 결과 ====="
        )

        for number, result in enumerate(
            results,
            start=1
        ):

            source_text = format_source(
                result
            )

            print(
                f"\n[{number}] "
                f"{source_text} "
                f"| chunk "
                f"{result['chunk_id']} "
                f"| score "
                f"{result['score']:.4f}"
            )

            print(
                result["text"][:800]
            )