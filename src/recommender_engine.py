import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.embedding_model import create_or_load_embeddings


class AIDestinationRecommender:
    def __init__(self, df):
        self.df = df.drop_duplicates(subset=["Site Name"]).reset_index(drop=True).copy()
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = create_or_load_embeddings(self.df)

    def recommend(self, user_query, top_n=10):
        if not str(user_query).strip():
            user_query = "travel destinations"

        user_embedding = self.model.encode([user_query])
        similarities = cosine_similarity(user_embedding, self.embeddings)[0]

        ranked = self.df.copy()
        ranked["score"] = similarities
        ranked = ranked.sort_values(by="score", ascending=False)
        ranked = ranked.drop_duplicates(subset=["Site Name"])

        return ranked.head(top_n)
