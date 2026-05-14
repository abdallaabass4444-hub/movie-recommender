import os
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = r'C:\Users\DELL\Downloads\files\files\data'


def load_movielens_100k(data_path=DATA_PATH):
    """Load MovieLens dataset from CSV files (ratings and movies only)"""
    movies_path  = os.path.join(data_path, 'movies.csv')
    ratings_path = os.path.join(data_path, 'ratings.csv')

    print('Loading dataset from CSV files...')
    movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")

    return ratings, movies


def preprocess_movies(movies):
    """Clean and enrich the movies DataFrame"""
    if 'movieId' in movies.columns:
        movies = movies.rename(columns={'movieId': 'movie_id'})

    genre_columns = ['unknown','Action','Adventure','Animation','Childrens',
                     'Comedy','Crime','Documentary','Drama','Fantasy',
                     'Film_Noir','Horror','Musical','Mystery','Romance',
                     'Sci_Fi','Thriller','War','Western']
    existing_genres = [c for c in genre_columns if c in movies.columns]

    def get_genre_string(row):
        genres = [g.replace('_', ' ') for g in existing_genres if row[g] == 1]
        return ' '.join(genres) if genres else 'unknown'

    if existing_genres:
        movies['genres_str'] = movies.apply(get_genre_string, axis=1)
    elif 'genres' in movies.columns:
        movies['genres_str'] = movies['genres'].str.replace('|', ' ', regex=False)
    else:
        movies['genres_str'] = 'unknown'

    if 'title' in movies.columns:
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').fillna(0).astype(int)
        movies['clean_title'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()

    return movies


def prepare_data(ratings, movies, test_size=0.2, random_state=42):
    """Merge and split data into train/test sets"""
    if 'movieId' in movies.columns:  movies  = movies.rename(columns={'movieId': 'movie_id'})
    if 'userId'  in ratings.columns: ratings = ratings.rename(columns={'userId':  'user_id'})
    if 'movieId' in ratings.columns: ratings = ratings.rename(columns={'movieId': 'item_id'})

    df = ratings.merge(
        movies[['movie_id', 'title', 'genres_str']],
        left_on='item_id', right_on='movie_id', how='left'
    )
    df = df.dropna(subset=['rating', 'user_id', 'item_id'])

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df['user_id']
    )
    print(f'Training set: {len(train_df):,} rows  |  Test set: {len(test_df):,} rows')
    return train_df, test_df, df


def load_all(data_path=DATA_PATH):
    """Full pipeline: load → preprocess → split. Returns (ratings, movies, train, test, full_df, ratings_ui, movies_ui)"""
    ratings, movies = load_movielens_100k(data_path)
    movies = preprocess_movies(movies)
    train, test, full_df = prepare_data(ratings, movies)

    ratings_ui = ratings.rename(columns={'userId': 'user_id', 'movieId': 'item_id'}) \
        if 'userId' in ratings.columns else ratings.copy()
    movies_ui = movies.rename(columns={'movie_id': 'item_id', 'genres_str': 'genres'}).copy()

    print('Data ready!')
    return ratings, movies, train, test, full_df, ratings_ui, movies_ui
