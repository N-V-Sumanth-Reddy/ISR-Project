"""ragsearch: a private, citation-grounded RAG search engine over local documents.

Everything runs on the local machine: BM25 + dense retrieval, rank fusion,
optional cross-encoder re-ranking, and grounded generation through a locally
hosted LLM (Ollama or any llama.cpp / OpenAI-compatible server on localhost).
"""
import os

# torch and faiss both ship an OpenMP runtime on macOS; without this the
# process can abort with "OMP: Error #15" when both are imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "1.0.0"
