from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DOCUMENT_DIR = Path("data/documents")

# 한국어/영어 문서를 둘 다 어느 정도 처리하기 위해 다국어 모델 사용
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

model = SentenceTransformer(MODEL_NAME)


def load_txt(file_path: Path) -> str:
    """TXT 파일을 읽어서 문자열로 반환합니다."""
    return file_path.read_text(encoding="utf-8")


def load_pdf(file_path: Path) -> str:
    """PDF 파일의 텍스트를 추출합니다."""
    reader = PdfReader(file_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n".join(pages)


def load_documents():
    """
    data/documents 폴더의 txt, pdf 파일을 읽습니다.

    반환 예시:
    [
        {
            "source": "sample.txt",
            "text": "문서 내용..."
        }
    ]
    """

    documents = []

    for file_path in DOCUMENT_DIR.iterdir():

        if file_path.suffix.lower() == ".txt":
            text = load_txt(file_path)

        elif file_path.suffix.lower() == ".pdf":
            text = load_pdf(file_path)

        else:
            continue

        documents.append(
            {
                "source": file_path.name,
                "text": text
            }
        )

    return documents


def chunk_text(text: str, chunk_size=500, overlap=100):
    """
    긴 문서를 일정 길이로 나눕니다.

    chunk_size = 각 chunk 크기
    overlap = 앞 chunk와 겹치는 문자 수
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_chunks(documents):
    """문서 전체를 chunk 단위로 변환합니다."""

    chunks = []

    for document in documents:

        text_chunks = chunk_text(document["text"])

        for index, chunk in enumerate(text_chunks):

            chunks.append(
                {
                    "source": document["source"],
                    "chunk_id": index,
                    "text": chunk
                }
            )

    return chunks


def create_vector_store(chunks):
    """Chunk들을 embedding한 뒤 FAISS index를 생성합니다."""

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


def search(query, index, chunks, top_k=3):
    """질문과 가장 유사한 chunk를 검색합니다."""

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(query_embedding, dtype="float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        result = chunks[idx].copy()
        result["score"] = float(score)

        results.append(result)

    return results


if __name__ == "__main__":

    documents = load_documents()

    print(f"읽은 문서 수: {len(documents)}")

    if len(documents) == 0:
        print("data/documents 폴더에 문서를 넣어주세요.")
        exit()

    chunks = create_chunks(documents)

    print(f"생성된 chunk 수: {len(chunks)}")

    index = create_vector_store(chunks)

    print(f"FAISS 저장 문서 수: {index.ntotal}")

    while True:

        query = input("\n질문 입력 (종료: exit): ")

        if query.lower() == "exit":
            break

        results = search(
            query=query,
            index=index,
            chunks=chunks,
            top_k=3
        )

        print("\n===== 검색 결과 =====")

        for number, result in enumerate(results, start=1):

            print(
                f"\n[{number}] "
                f"출처: {result['source']} "
                f"| chunk: {result['chunk_id']} "
                f"| score: {result['score']:.4f}"
            )

            print(result["text"][:500])