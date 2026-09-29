from pathlib import Path
import re

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =========================================================
# 설정
# =========================================================

DOCUMENT_DIR = Path("data/documents")

# 한국어 질문 ↔ 영어 문서 검색 성능 개선
MODEL_NAME = "intfloat/multilingual-e5-base"

# 기존보다 조금 더 작은 chunk 사용
CHUNK_SIZE = 600
CHUNK_OVERLAP = 120

model = SentenceTransformer(MODEL_NAME)


# =========================================================
# Text Cleaning
# =========================================================

def clean_text(text: str) -> str:
    """
    PDF에서 추출된 텍스트를 정리합니다.

    - 반복 공백 제거
    - NIST 반복 헤더 제거
    - DOI 안내 문구 제거
    - 빈 줄 정리
    """

    if not text:
        return ""

    # 줄바꿈 통일
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # NIST 반복 헤더 제거
    text = re.sub(
        r"NIST SP 800-207\s+ZERO TRUST ARCHITECTURE",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # 반복 DOI 문구 제거
    text = re.sub(
        r"This publication is available free of charge from:\s*"
        r"https://doi\.org/10\.6028/NIST\.SP\.800-207",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # URL만 있는 줄 제거
    text = re.sub(
        r"(?m)^\s*https?://\S+\s*$",
        " ",
        text
    )

    # 탭 제거
    text = text.replace("\t", " ")

    # 여러 공백 정리
    text = re.sub(
        r"[ ]+",
        " ",
        text
    )

    # 너무 많은 빈 줄 정리
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# =========================================================
# Document Loading
# =========================================================

def load_txt(file_path: Path):
    """
    TXT 파일 로딩.
    TXT는 페이지 번호가 없으므로 page=None.
    """

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    text = clean_text(text)

    if not text:
        return []

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

    각 페이지는:
    {
        source,
        page,
        text
    }
    형태로 저장됩니다.
    """

    reader = PdfReader(file_path)

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        try:
            text = page.extract_text()

        except Exception:
            continue

        if not text:
            continue

        text = clean_text(text)

        # 텍스트가 너무 짧은 페이지는 제외
        if len(text) < 80:
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
    data/documents 안의 PDF/TXT를 모두 읽습니다.
    """

    documents = []

    DOCUMENT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    for file_path in DOCUMENT_DIR.iterdir():

        # .gitkeep 등 숨김 파일 제외
        if file_path.name.startswith("."):
            continue

        suffix = file_path.suffix.lower()

        try:

            if suffix == ".pdf":
                loaded = load_pdf(
                    file_path
                )

            elif suffix == ".txt":
                loaded = load_txt(
                    file_path
                )

            else:
                continue

            documents.extend(
                loaded
            )

        except Exception as e:

            print(
                f"[경고] "
                f"{file_path.name} "
                f"읽기 실패: {e}"
            )

    return documents


# =========================================================
# Chunking
# =========================================================

def chunk_text(
    text: str,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    긴 텍스트를 일정 크기로 분할합니다.

    가능하면 문장 경계 근처에서 끊도록 합니다.
    """

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        target_end = min(
            start + chunk_size,
            text_length
        )

        end = target_end

        # 마지막 chunk가 아닐 경우
        # 문장 끝을 찾아 조금 더 자연스럽게 자름
        if target_end < text_length:

            search_start = max(
                start + 300,
                target_end - 150
            )

            candidate = text[
                search_start:
                target_end
            ]

            sentence_endings = [
                candidate.rfind(". "),
                candidate.rfind("? "),
                candidate.rfind("! "),
                candidate.rfind("\n")
            ]

            best_ending = max(
                sentence_endings
            )

            if best_ending != -1:
                end = (
                    search_start
                    + best_ending
                    + 1
                )

        chunk = text[
            start:end
        ].strip()

        # 의미 없는 너무 짧은 chunk 제거
        if len(chunk) >= 100:
            chunks.append(
                chunk
            )

        if end >= text_length:
            break

        next_start = (
            end - overlap
        )

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def create_chunks(documents):
    """
    페이지별 문서를 검색용 chunk로 변환.
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
                    "source": document[
                        "source"
                    ],
                    "page": document[
                        "page"
                    ],
                    "chunk_id": global_chunk_id,
                    "page_chunk_id": page_chunk_id,
                    "text": chunk
                }
            )

            global_chunk_id += 1

    return chunks


# =========================================================
# Embedding / FAISS
# =========================================================

def create_vector_store(chunks):
    """
    E5 모델을 이용해 chunk embedding을 만들고
    FAISS index를 생성합니다.

    E5 계열 모델은 문서 앞에
    'passage:' prefix를 붙이는 방식이 권장됩니다.
    """

    if not chunks:
        raise ValueError(
            "Embedding할 chunk가 없습니다."
        )

    texts = [
        "passage: " + chunk["text"]
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

    # normalize된 embedding이므로
    # Inner Product ≈ Cosine Similarity
    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# =========================================================
# Search
# =========================================================

def search(
    query,
    index,
    chunks,
    top_k=3
):
    """
    질문을 E5 query embedding으로 변환 후
    관련 chunk를 검색합니다.
    """

    if not query or not query.strip():
        return []

    if not chunks:
        return []

    # E5 검색 query prefix
    query_text = (
        "query: "
        + query.strip()
    )

    query_embedding = model.encode(
        [query_text],
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

        if idx < 0:
            continue

        result = chunks[
            int(idx)
        ].copy()

        result["score"] = float(
            score
        )

        results.append(
            result
        )

    return results


# =========================================================
# Citation
# =========================================================

def format_source(result):
    """
    검색 결과 출처 표시.

    PDF:
        filename.pdf / p.12

    TXT:
        filename.txt
    """

    source = result.get(
        "source",
        "unknown"
    )

    page = result.get(
        "page"
    )

    if page is not None:

        return (
            f"{source} / p.{page}"
        )

    return source


# =========================================================
# CLI 테스트
# =========================================================

if __name__ == "__main__":

    print()
    print(
        "문서를 불러오는 중..."
    )

    documents = load_documents()

    unique_files = set(
        document["source"]
        for document in documents
    )

    print(
        f"파일 수: "
        f"{len(unique_files)}"
    )

    print(
        f"문서 페이지 수: "
        f"{len(documents)}"
    )

    if not documents:

        print(
            "data/documents 폴더에 "
            "PDF 또는 TXT 문서를 넣어주세요."
        )

        raise SystemExit

    chunks = create_chunks(
        documents
    )

    print(
        f"생성된 chunk 수: "
        f"{len(chunks)}"
    )

    print()
    print(
        "Embedding 및 "
        "FAISS index 생성 중..."
    )

    index = create_vector_store(
        chunks
    )

    print()
    print(
        f"FAISS 저장 chunk 수: "
        f"{index.ntotal}"
    )

    print(
        "검색 시스템 준비 완료"
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

        results = search(
            query=query,
            index=index,
            chunks=chunks,
            top_k=3
        )

        print()
        print(
            "===== 검색 결과 ====="
        )

        for rank, result in enumerate(
            results,
            start=1
        ):

            source = format_source(
                result
            )

            print()
            print(
                f"[TOP {rank}] "
                f"{source}"
            )

            print(
                f"Score: "
                f"{result['score']:.4f}"
            )

            print()

            print(
                result["text"][:1000]
            )

            print(
                "-" * 80
            )