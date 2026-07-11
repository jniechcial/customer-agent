import tiktoken

from customer_agent.indexing.chunking import TokenChunker

ENC = tiktoken.get_encoding("cl100k_base")


def make_article(contents: str, article_id: str = "art1") -> dict:
    return {
        "id": article_id,
        "url": "https://support.wix.com/x",
        "title": "My Title",
        "article_type": "article",
        "contents": contents,
    }


def test_short_article_single_chunk():
    chunks = TokenChunker(chunk_size=512, overlap=64).chunk_article(make_article("short body"))
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].article_id == "art1"


def test_title_is_prefixed_into_chunk_text():
    chunks = TokenChunker(chunk_size=512, overlap=64).chunk_article(make_article("short body"))
    assert chunks[0].text.startswith("My Title")
    assert "short body" in chunks[0].text
    assert chunks[0].title == "My Title"


def test_long_article_splits_with_overlap():
    words = " ".join(f"word{i}" for i in range(1000))
    chunker = TokenChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_article(make_article(words))
    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Consecutive chunks share the overlap: end of chunk N == start of chunk N+1.
    first = ENC.encode(chunks[0].text, disallowed_special=())
    second = ENC.encode(chunks[1].text, disallowed_special=())
    assert first[-20:] == second[:20]
    # Every chunk except possibly the last is exactly chunk_size tokens.
    for c in chunks[:-1]:
        assert len(ENC.encode(c.text, disallowed_special=())) == 100


def test_no_content_lost_between_chunks():
    words = " ".join(f"word{i}" for i in range(500))
    chunker = TokenChunker(chunk_size=100, overlap=20)
    chunks = chunker.chunk_article(make_article(words))
    body_tokens = ENC.encode(f"My Title\n\n{words}", disallowed_special=())
    # Reassemble by stripping the overlap from every chunk after the first.
    reassembled = list(ENC.encode(chunks[0].text, disallowed_special=()))
    for c in chunks[1:]:
        reassembled.extend(ENC.encode(c.text, disallowed_special=())[20:])
    assert reassembled == body_tokens


def test_empty_contents_still_yields_title_chunk():
    chunks = TokenChunker(chunk_size=512, overlap=64).chunk_article(make_article(""))
    assert len(chunks) == 1
    assert "My Title" in chunks[0].text


def test_all_chunks_carry_article_metadata():
    words = " ".join(f"word{i}" for i in range(300))
    chunks = TokenChunker(chunk_size=100, overlap=10).chunk_article(make_article(words, "abc"))
    assert all(c.article_id == "abc" for c in chunks)
    assert all(c.url == "https://support.wix.com/x" for c in chunks)
    assert all(c.article_type == "article" for c in chunks)
