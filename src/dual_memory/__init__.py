"""dual-memory: working memory plus long-term SQLite memory for AI agents."""

from .answer import RuleAnswerer, exact_match, parse_question
from .embedders import Embedder, HashingEmbedder, OnnxEmbedder, default_embedder
from .extract import RuleExtractor, render_fact
from .llm import (
    LLM,
    LLMAnswerer,
    LLMError,
    LLMExtractor,
    LLMJudge,
    LLMSummarizer,
    OpenAICompatibleLLM,
    StubLLM,
)
from .longterm import LongTermConfig, LongTermMemory, RetrievalWeights
from .memory import DualMemory
from .store import EmbedderMismatchError, SQLiteStore
from .summarize import RuleSummarizer
from .types import Fact, MemoryItem, Provenance, RememberResult, ScoredMemory, ScoreParts, Turn
from .working import WorkingMemory

__version__ = "0.1.0"

__all__ = [
    "LLM",
    "DualMemory",
    "Embedder",
    "EmbedderMismatchError",
    "Fact",
    "HashingEmbedder",
    "LLMAnswerer",
    "LLMError",
    "LLMExtractor",
    "LLMJudge",
    "LLMSummarizer",
    "LongTermConfig",
    "LongTermMemory",
    "MemoryItem",
    "OnnxEmbedder",
    "OpenAICompatibleLLM",
    "Provenance",
    "RememberResult",
    "RetrievalWeights",
    "RuleAnswerer",
    "RuleExtractor",
    "RuleSummarizer",
    "SQLiteStore",
    "ScoreParts",
    "ScoredMemory",
    "StubLLM",
    "Turn",
    "WorkingMemory",
    "default_embedder",
    "exact_match",
    "parse_question",
    "render_fact",
    "__version__",
]
