from src import embedding


def test_create_embeddings():
    texts = ["car", "pet", "carpet"]
    embeddings = embedding.create_embeddings(texts)

    assert isinstance(embeddings, dict)
