from rag_src.pre_embedding_enricher.metadata_injector import MetadataInjector


def test_metadata_injector_with_full_metadata():
    docs = ["Document A", "Document B"]
    metadata = {0: "Title: A\nAuthor: Alice", 1: "Title: B\nAuthor: Bob"}
    enricher = MetadataInjector(metadata)
    enriched = enricher.enrich(docs)

    # Content correctness
    assert enriched[0] == "Title: A\nAuthor: Alice\n\nDocument A"
    assert enriched[1] == "Title: B\nAuthor: Bob\n\nDocument B"
    # Type checks
    assert isinstance(enriched, list)
    assert all(isinstance(doc, str) for doc in enriched)


def test_metadata_injector_with_partial_metadata():
    docs = ["Doc X", "Doc Y", "Doc Z"]
    metadata = {1: "Author: Charlie"}
    enricher = MetadataInjector(metadata)
    enriched = enricher.enrich(docs)

    assert enriched[0] == "Doc X"  # no metadata for index 0
    assert enriched[1].startswith("Author: Charlie")
    assert "Doc Y" in enriched[1]
    assert enriched[2] == "Doc Z"  # no metadata for index 2


def test_metadata_injector_with_no_metadata():
    docs = ["Just text"]
    metadata = {}
    enricher = MetadataInjector(metadata)
    enriched = enricher.enrich(docs)

    assert enriched == docs  # unchanged


def test_metadata_injector_empty_input():
    enricher = MetadataInjector({})
    enriched = enricher.enrich([])

    assert enriched == []
