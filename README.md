# RetrIQ — Retrieval Intelligence, Simplified.

A Multimodal Corrective RAG System for Enterprise Document Intelligence

## Live Demo
- Frontend: https://retriq-frontend.vercel.app
- Backend: https://compliq-backend-production-6a0d.up.railway.app

## Tech Stack
- LangChain + FAISS + Voyage AI voyage-code-3
- Claude claude-opus-4-6 (Anthropic)
- Tavily Web Search API
- FastAPI + Python 3.11
- React + Vite + Tailwind CSS
- Deployed on Vercel + Railway

## Features
- Multi-format ingestion: PDF, Word, Excel, PowerPoint, CSV, Images
- Claude Vision OCR for scanned documents
- Corrective RAG with Tavily web fallback
- Persistent vector storage via Railway Volume
- Source transparency: document vs web

## RAGAS Evaluation Results
| Metric | Score |
|--------|-------|
| Faithfulness | 0.9227 |
| Answer Relevancy | 0.7042 |
| Context Precision | 0.9000 |

## Dataset
Synthetic fleet dataset for Lone Star Freight LLC (69 documents)
- fleet_docs/pdfs/ — 56 PDF documents
- fleet_docs/texts/ — 13 text documents

## Research Paper
"RetrIQ: A Multimodal Corrective RAG System for Enterprise Document Intelligence"
- Submitted to arXiv and IEEE Access (pending)

## Setup
```bash
pip install -r backend/requirements.txt
cd backend && uvicorn main:app --reload
```