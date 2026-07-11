from customer_agent.config import Settings


def test_collection_name_derived_from_index_config():
    s = Settings(chunk_size_tokens=512, chunk_overlap_tokens=64,
                 embedding_model="text-embedding-3-small")
    assert s.collection_name == "KB_chunk512o64_te3small"


def test_collection_name_changes_with_config():
    a = Settings(chunk_size_tokens=512, chunk_overlap_tokens=64)
    b = Settings(chunk_size_tokens=256, chunk_overlap_tokens=64)
    c = Settings(chunk_size_tokens=512, chunk_overlap_tokens=64,
                 embedding_model="text-embedding-3-large")
    assert len({a.collection_name, b.collection_name, c.collection_name}) == 3


def test_env_var_override(monkeypatch):
    monkeypatch.setenv("K_RETRIEVE", "7")
    monkeypatch.setenv("AGENT_MODEL", "gpt-test")
    s = Settings()
    assert s.k_retrieve == 7
    assert s.agent_model == "gpt-test"
