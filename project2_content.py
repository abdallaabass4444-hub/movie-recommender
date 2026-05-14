import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


class ContentBasedFilter:
    def __init__(self, movies_df):
        self.movies_df = movies_df.reset_index(drop=True)
        self.tfidf_matrix = None
        self.cosine_sim   = None
        self.movie_index_map = None
        self.scaler = MinMaxScaler()
        self._build_feature_matrix()

    def _build_feature_matrix(self):
        corpus = self.movies_df['genres_str'].fillna('').tolist()
        self.tfidf = TfidfVectorizer(
            token_pattern=r'(?u)\b\w+\b', stop_words=None, ngram_range=(1, 2)
        )
        self.tfidf_matrix = self.tfidf.fit_transform(corpus)
        self.cosine_sim   = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        self.movie_index_map = pd.Series(
            self.movies_df.index, index=self.movies_df['movie_id']
        ).to_dict()
        print(f'Built TF-IDF matrix: {self.tfidf_matrix.shape}')

    def get_similar_movies(self, movie_id, top_n=10):
        if movie_id not in self.movie_index_map:
            return pd.DataFrame()
        idx = self.movie_index_map[movie_id]
        scores = sorted(
            enumerate(self.cosine_sim[idx]), key=lambda x: x[1], reverse=True
        )[1:top_n + 1]
        indices = [i[0] for i in scores]
        sims    = [i[1] for i in scores]
        result  = self.movies_df.iloc[indices][['movie_id', 'title', 'genres_str']].copy()
        result['similarity_score'] = sims
        return result

    def get_recommendations_for_user(self, user_rated_movies, top_n=10):
        if not user_rated_movies:
            return pd.DataFrame()
        all_scores = {}
        for movie_id in user_rated_movies:
            for _, row in self.get_similar_movies(movie_id, top_n=20).iterrows():
                mid = row['movie_id']
                if mid not in user_rated_movies:
                    all_scores.setdefault(mid, []).append(row['similarity_score'])
        if not all_scores:
            return pd.DataFrame()
        rec_df = pd.DataFrame(
            [{'movie_id': mid, 'content_score': max(s)} for mid, s in all_scores.items()]
        )
        rec_df = rec_df.sort_values('content_score', ascending=False).head(top_n)
        rec_df = rec_df.merge(
            self.movies_df[['movie_id', 'title', 'genres_str']], on='movie_id', how='left'
        )
        rec_df['content_score_normalized'] = self.scaler.fit_transform(rec_df[['content_score']])
        return rec_df

    def predict_score(self, user_id, movie_id, user_rated_movies):
        if not user_rated_movies or movie_id not in self.movie_index_map:
            return 3.0
        target_idx = self.movie_index_map[movie_id]
        sims = [
            self.cosine_sim[target_idx][self.movie_index_map[rm]]
            for rm in user_rated_movies if rm in self.movie_index_map
        ]
        if not sims:
            return 3.0
        avg_sim = np.mean(sims)
        return float(np.clip(1 + 4 * (avg_sim + 0.2) / 1.2, 1, 5))


def build_cb_model(movies_df):
    """Convenience builder — returns a trained ContentBasedFilter."""
    model = ContentBasedFilter(movies_df)
    return model
