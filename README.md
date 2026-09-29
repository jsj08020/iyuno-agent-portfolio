
# Iyuno AI Agent Portfolio

![Python Tests](https://github.com/jsj08020/iyuno-agent-portfolio/actions/workflows/test.yml/badge.svg)

## 프로젝트 개요

본 프로젝트는 Iyuno의 AI Agent Engineer 채용공고에서 요구하는
LLM Agent, RAG, Tool Calling, API Integration 및 Evaluation 역량을
직접 구현하여 증명하기 위한 포트폴리오 프로젝트입니다.

사용자가 공개 보안 및 기술 문서에 대해 질문하면,
관련 문서를 검색하고 검색된 근거를 기반으로 답변하는
AI Agent를 구현하는 것을 목표로 합니다.

---

## Target Job Posting

- Company: Iyuno
- Position: AI Agent Engineer
- Location: Seoul
- Requisition: JR101122

### 주요 요구 기술

- LLM 기반 AI Agent 시스템 설계 및 개발
- RAG 기반 검색 및 응답 시스템
- Tool Calling
- API 및 Database Integration
- Multi-step Workflow
- Evaluation / Feedback Loop
- Latency / Cost / Reliability 개선

---

## 프로젝트 목표

다음 기능을 구현합니다.

1. 공개 기술 문서 수집
2. Document Chunking
3. Embedding 생성
4. Vector Search
5. RAG 기반 질의응답
6. 답변 출처 Citation 제공
7. Tool Calling
8. Evaluation Dataset 구축
9. Accuracy / Latency 평가
10. FastAPI 또는 Streamlit 기반 Demo 제공

---

## Architecture

User Question

↓

Agent / Router

↓

RAG Retriever + Tools

↓

LLM

↓

Answer + Citation

↓

Evaluation / Feedback

---

## Tech Stack

- Python
- FastAPI
- Streamlit
- FAISS
- Sentence Transformers
- Gemini API
- Pytest

---

## Project Structure

```text
app/
    main.py
    agent.py
    retriever.py
    tools.py

data/
    documents/

evaluation/
    questions.json
    metrics.json

tests/
    test_basic.py

    ## Evaluation

총 30개의 평가 질문을 구성하여 Retriever 성능을 측정했습니다.

### Retrieval Performance

| Retriever | Recall@3 | Average Latency |
|---|---:|---:|
| Multilingual MiniLM | 66.7% | 15.74 ms |
| Multilingual E5 Base | 83.3% | 34.07 ms |

초기 Multilingual MiniLM 기반 Retriever에서 한국어 질문과 영어 기술 문서 간 검색 성능 저하가 확인되었습니다.

오류 분석 후 Retriever를 `intfloat/multilingual-e5-base`로 변경하고, E5의 `query:` / `passage:` prefix 및 chunking 방식을 개선했습니다.

그 결과 Recall@3가 **66.7%에서 83.3%로 향상**되었습니다.

검색 지연시간은 15.74 ms에서 34.07 ms로 증가했지만, 실시간 질의응답에 사용 가능한 수준을 유지했습니다.

> Recall@3는 평가 질문의 expected term이 Top-3 검색 chunk에서 발견되는지를 기준으로 측정한 retrieval proxy metric입니다.

### Evaluation Files

- `evaluation/questions.json` - 30개 평가 질문
- `evaluation/metrics.json` - 평가 결과
- `evaluation/graphs/recall_comparison.png`
- `evaluation/graphs/latency_comparison.png`

### Limitations

현재 공개 NIST 보안 문서를 중심으로 데이터셋을 구성했습니다.

Gemini Free Tier의 일일 API 호출 제한으로 인해 전체 질문에 대한 LLM-as-a-Judge Faithfulness 평가는 수행하지 못했습니다.

향후 문서 데이터셋 확대, Hybrid Search 및 별도 평가 모델을 이용하여 평가 범위를 확장할 예정입니다.