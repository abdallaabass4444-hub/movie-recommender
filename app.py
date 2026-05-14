import streamlit as st
import pandas as pd

from project1_data import load_all, DATA_PATH
from project2_content import build_cb_model
from project3_collaborative import build_cf_model
from project4_hybrid import (
    cb_scores, cb_recommend, cf_predict, cf_recommend,
    hybrid_predict_rating, hybrid_recommend, get_similar_movies,
)
from project5_evaluation import RecommenderEvaluator

ALL_GENRES = [
    'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci Fi',
    'Thriller', 'War', 'Western',
]

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='🎬 Movie Recommendation System',
    page_icon='🎬',
    layout='wide',
)

# ── Load models (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner='Loading data & training models…')
def load_models():
    ratings, movies, train, test, full_df, ratings_ui, movies_ui = load_all(DATA_PATH)
    cb_model = build_cb_model(movies)
    cf_model = build_cf_model(train, n_factors=50)
    return ratings, movies, train, test, full_df, ratings_ui, movies_ui, cb_model, cf_model


try:
    ratings, movies, train, test, full_df, ratings_ui, movies_ui, cb_model, cf_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    load_error = str(e)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title('🎬 Hybrid Movie Recommendation System')
st.caption('Content-Based · Collaborative (SVD) · Hybrid · Evaluation')

if not models_loaded:
    st.error(f'❌ Could not load data: {load_error}')
    st.info(
        'Make sure `movies.csv` and `ratings.csv` are inside a `data/` folder '
        'in the same directory as `app.py`.'
    )
    st.stop()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    '👤 By User', '🔍 Similar Movies', '🎭 Browse by Genre', '📊 Evaluation'
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Recommendations for a User
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader('Get Recommendations for a User')
    col_left, col_right = st.columns([1, 2])

    with col_left:
        uid = st.selectbox('User ID', sorted(train['user_id'].unique()), key='uid')
        model_choice = st.radio(
            'Model',
            ['Hybrid', 'Collaborative (SVD)', 'Content-Based'],
            key='model_choice',
        )
        top_n = st.slider('Top N', 5, 20, 10, key='topn_user')
        cb_w  = st.slider('CB Weight (Hybrid only)', 0.0, 1.0, 0.3, 0.05, key='cb_w')
        run_btn = st.button('Get Recommendations', type='primary', key='run_user')

    with col_right:
        if run_btn:
            cf_w = 1.0 - cb_w

            # User history
            history = (
                train[train['user_id'] == uid]
                .sort_values('rating', ascending=False)
                .head(5)[['title', 'genres_str', 'rating']]
                .rename(columns={'genres_str': 'genres'})
            )
            st.markdown(f'**User {uid} — Top Rated Movies**')
            st.dataframe(history, use_container_width=True, hide_index=True)

            st.markdown(f'**{model_choice} Recommendations**')
            with st.spinner('Generating…'):
                if model_choice == 'Hybrid':
                    recs = hybrid_recommend(
                        uid, train, cb_model, cf_model, movies,
                        cb_weight=cb_w, cf_weight=cf_w, top_n=top_n,
                    )
                    cols = [c for c in ['title', 'genres', 'cb_score', 'cf_score', 'hybrid_score'] if c in recs.columns]
                elif model_choice == 'Collaborative (SVD)':
                    recs = cf_recommend(uid, cf_model, movies, train, top_n=top_n)
                    cols = [c for c in ['title', 'genres', 'cf_score'] if c in recs.columns]
                else:
                    recs = cb_recommend(uid, train, cb_model, top_n=top_n)
                    cols = [c for c in ['title', 'genres', 'cb_score'] if c in recs.columns]

            if recs.empty:
                st.warning('No recommendations found for this user.')
            else:
                st.dataframe(recs[cols].round(4), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Similar Movies
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader('Find Similar Movies')
    col_left, col_right = st.columns([1, 2])

    with col_left:
        movie_title = st.selectbox('Movie', sorted(movies['title'].tolist()), key='movie_title')
        top_n_sim   = st.slider('Top N', 5, 20, 10, key='topn_sim')
        sim_btn = st.button('Find Similar', type='primary', key='sim_btn')

    with col_right:
        if sim_btn:
            item_id = movies.loc[movies['title'] == movie_title, 'movie_id'].values[0]
            with st.spinner('Searching…'):
                similar = get_similar_movies(item_id, cb_model, top_n=top_n_sim)
            st.markdown(f'**Movies similar to: {movie_title}**')
            if similar.empty:
                st.warning('No similar movies found.')
            else:
                cols = [c for c in ['title', 'genres', 'cb_score'] if c in similar.columns]
                st.dataframe(similar[cols].round(4), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Browse by Genre / Keyword
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader('Browse Movies by Genre / Keyword')
    col_left, col_right = st.columns([1, 2])

    with col_left:
        genre_choice = st.selectbox('Genre', ['All'] + ALL_GENRES, key='genre')
        keyword      = st.text_input('Title keyword', placeholder='e.g. Batman', key='keyword')
        top_n_browse = st.slider('Top N', 5, 30, 10, key='topn_browse')
        browse_btn   = st.button('Search', type='primary', key='browse_btn')

    with col_right:
        if browse_btn:
            filtered = movies_ui.copy()
            if genre_choice and genre_choice != 'All':
                filtered = filtered[filtered['genres'].str.contains(genre_choice, case=False, na=False)]
            if keyword.strip():
                filtered = filtered[filtered['title'].str.contains(keyword.strip(), case=False, na=False)]

            avg = (
                ratings_ui.groupby('item_id')['rating']
                .agg(avg_rating='mean', n_ratings='count')
                .reset_index()
            )
            filtered = filtered.merge(avg, on='item_id', how='left')
            filtered = filtered.sort_values('avg_rating', ascending=False).head(top_n_browse)

            st.markdown(f'**Found {len(filtered)} movies**')
            if len(filtered) > 0:
                st.dataframe(
                    filtered[['title', 'genres', 'avg_rating', 'n_ratings']].round(2),
                    use_container_width=True, hide_index=True,
                )
            else:
                st.warning('No movies found! Try a different genre or keyword.')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Evaluation
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader('Model Evaluation')
    n_users = st.slider('Number of eval users (more = slower)', 10, 200, 50, key='n_eval')
    eval_btn = st.button('Run Evaluation', type='primary', key='eval_btn')

    if eval_btn:
        with st.spinner('Evaluating all models… this may take a minute.'):
            evaluator = RecommenderEvaluator(k=10, relevance_threshold=4.0, n_eval_users=n_users)
            evaluator.evaluate_all(
                [
                    (
                        'Content-Based',
                        lambda u, i: cb_scores(u, [i], train, cb_model).get(i, 0.0) * 5,
                        lambda uid, top_n=10: cb_recommend(uid, train, cb_model, top_n=top_n),
                    ),
                    (
                        'Collaborative (SVD)',
                        lambda u, i: cf_predict(u, i, cf_model),
                        lambda uid, top_n=10: cf_recommend(uid, cf_model, movies, train, top_n=top_n),
                    ),
                    (
                        'Hybrid',
                        lambda u, i: hybrid_predict_rating(u, i, train, cb_model, cf_model),
                        lambda uid, top_n=10: hybrid_recommend(uid, train, cb_model, cf_model, movies, top_n=top_n),
                    ),
                ],
                test_df=test,
            )
        results_df = evaluator.get_results_df()
        st.markdown('### Results')
        st.dataframe(results_df, use_container_width=True)

        # Bar charts
        st.markdown('### RMSE Comparison')
        st.bar_chart(results_df['RMSE'])
        st.markdown('### MAE Comparison')
        st.bar_chart(results_df['MAE'])
