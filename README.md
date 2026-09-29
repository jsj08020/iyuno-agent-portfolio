# Iyuno AI Agent Portfolio

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