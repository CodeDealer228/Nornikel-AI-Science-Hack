"""Query dispatcher: turns routing decisions into execution + context + final answer.

The RAG client is not implemented in this package — see
``agent.rag_factory`` for the plug-in slot. The default fallback
``StubRAGClient`` returns empty results with a marker note.
"""

from .dispatcher import Dispatcher, DispatchResult
from .deep_research import DeepResearchAgent, DeepResearchResult, ResearchIteration
from .rag_client import (
    NumericFilter,
    RAGClient,
    RAGDocument,
    RAGResult,
    StubRAGClient,
)
from .rag_factory import build_rag_client, register_rag_backend
from .synthesizer import AnswerSynthesizer, SynthesisResult, attach_synthesis

__all__ = [
    "AnswerSynthesizer",
    "DeepResearchAgent",
    "DeepResearchResult",
    "Dispatcher",
    "DispatchResult",
    "NumericFilter",
    "RAGClient",
    "RAGDocument",
    "RAGResult",
    "ResearchIteration",
    "StubRAGClient",
    "SynthesisResult",
    "attach_synthesis",
    "build_rag_client",
    "register_rag_backend",
]
