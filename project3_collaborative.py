import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error


class CollaborativeFilter:
    def __init__(self, algo='svd', n_factors=20):
        self.algo_name     = algo
        self.n_factors     = n_factors
        self.is_trained    = False
        self.user_index    = {}
        self.item_index    = {}
        self.matrix        = None
        self.reconstructed = None

    def train(self, ratings_df, user_col='user_id', item_col='item_id', rating_col='rating'):
        print(f'Training {self.algo_name.upper()} model...')
        users = ratings_df[user_col].unique()
        items = ratings_df[item_col].unique()
        self.user_index  = {u: i for i, u in enumerate(users)}
        self.item_index  = {it: i for i, it in enumerate(items)}
        self.global_mean = ratings_df[rating_col].mean()

        matrix = np.zeros((len(users), len(items)))
        for _, row in ratings_df.iterrows():
            u  = self.user_index.get(row[user_col])
            it = self.item_index.get(row[item_col])
            if u is not None and it is not None:
                matrix[u][it] = row[rating_col] - self.global_mean

        self.matrix = matrix
        svd = TruncatedSVD(n_components=self.n_factors, random_state=42)
        reduced = svd.fit_transform(matrix)
        self.reconstructed = np.dot(reduced, svd.components_)
        self.is_trained = True
        print('Training completed!')

    def predict(self, user_id, item_id):
        u  = self.user_index.get(user_id)
        it = self.item_index.get(item_id)
        if u is None or it is None:
            return self.global_mean
        raw = self.reconstructed[u][it] + self.global_mean
        return float(np.clip(raw, 1, 5))

    def predict_batch(self, user_item_pairs):
        return [self.predict(u, i) for u, i in user_item_pairs]

    def get_recommendations_for_user(self, user_id, all_movies_df, n_recommendations=10,
                                     exclude_rated=True, ratings_df=None):
        if not self.is_trained:
            raise ValueError('Model not trained yet.')
        all_movie_ids = all_movies_df['movie_id'].unique()
        rated_movies  = set()
        if exclude_rated and ratings_df is not None:
            rated_movies = set(ratings_df[ratings_df['user_id'] == user_id]['item_id'].values)

        predictions = [
            {
                'movie_id': mid,
                'predicted_rating': self.predict(user_id, mid),
                'collab_score': self.predict(user_id, mid) / 5.0,
            }
            for mid in all_movie_ids if mid not in rated_movies
        ]
        if not predictions:
            return pd.DataFrame()
        df = (
            pd.DataFrame(predictions)
            .sort_values('predicted_rating', ascending=False)
            .head(n_recommendations)
        )
        return df.merge(
            all_movies_df[['movie_id', 'title', 'genres_str']], on='movie_id', how='left'
        )

    def evaluate(self, test_df, user_col='user_id', item_col='item_id', rating_col='rating'):
        if not self.is_trained:
            raise ValueError('Model not trained yet.')
        actuals, preds = [], []
        for _, row in test_df.iterrows():
            actuals.append(row[rating_col])
            preds.append(self.predict(row[user_col], row[item_col]))
        rmse = np.sqrt(mean_squared_error(actuals, preds))
        mae  = np.mean(np.abs(np.array(actuals) - np.array(preds)))
        return {'RMSE': rmse, 'MAE': mae}

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
        self.__dict__.update(loaded.__dict__)
        self.is_trained = True


def build_cf_model(train_df, n_factors=50):
    """Convenience builder — returns a trained CollaborativeFilter."""
    model = CollaborativeFilter(algo='svd', n_factors=n_factors)
    model.train(train_df)
    return model
