import pandas as pd
from pandas import DataFrame
from sklearn.cluster import KMeans
from typing import List
from langchain.embeddings.base import Embeddings
from langchain.base_language import BaseLanguageModel


def cluster_and_summarize(
    texts: List[str], n_summaries: int, embeddings: Embeddings, llm: BaseLanguageModel, prompt: str, n_clusters: int = 5
) -> DataFrame:
    """Method to recursively cluster and summarize documents.

    Args:
        texts (List[str]): List of documents to be clustered and summarized
        n_summaries (int): Number of summaries to be generated
        embeddings (Embeddings): Langchain embeddings object
        llm (BaseLanguageModel): Langchain language model object
        prompt (str): Summary prompt
        n_clusters (int, optional): Number of clusters to be generated. Defaults to 5.
    """
    embedded_docs = embeddings.embed_documents(texts)
    d = {i: [texts[i], embedding_vec] for i, embedding_vec in enumerate(embedded_docs)}

    df = pd.DataFrame(d).T.rename(columns={1: "embeddings", 0: "text"})

    decrement = (n_clusters - 2) / (n_summaries - 1) if n_summaries > 1 else 0

    prev_clusters = n_clusters
    for i in range(n_summaries):
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model = model.fit(df["embeddings"].tolist())
        df[f"cluster_{i+1}"] = model.labels_

        cluster_summaries = {}
        for j in df[f"cluster_{i+1}"].unique():
            cluster_contents = ",".join(df[df[f"cluster_{i+1}"] == j]["text"].tolist())
            cluster_summary = llm.invoke(prompt.format(paragraph=cluster_contents))
            cluster_summaries[j] = cluster_summary.content

        df[f"cluster_{i+1}_summary"] = df[f"cluster_{i+1}"].map(cluster_summaries)

        cluster_summaries_embedded = {}
        for summary in df[f"cluster_{i+1}_summary"].unique():
            embedded_summary = embeddings.embed_documents([summary])
            cluster_summaries_embedded[summary] = embedded_summary[0]

        df[f"cluster_{i+1}_summary_embeddings"] = df[f"cluster_{i+1}_summary"].map(cluster_summaries_embedded)

        n_clusters = max(2, round(n_clusters - decrement))
        if n_clusters == prev_clusters:
            break
        prev_clusters = n_clusters

    return df
