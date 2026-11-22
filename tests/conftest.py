from __future__ import annotations

import pytest

from dual_memory import HashingEmbedder, LongTermMemory, SQLiteStore

DAY = 86400.0
T0 = 1_736_000_000.0  # a fixed start time, early January 2025


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=128)


@pytest.fixture
def store(embedder: HashingEmbedder):
    s = SQLiteStore(":memory:", embedder_name=embedder.name, dim=embedder.dim)
    yield s
    s.close()


@pytest.fixture
def ltm(store: SQLiteStore, embedder: HashingEmbedder) -> LongTermMemory:
    return LongTermMemory(store, embedder)
