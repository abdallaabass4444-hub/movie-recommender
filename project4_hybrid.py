import numpy as np

CB_WEIGHT = 0.3
CF_WEIGHT = 0.7


def cb_scores(user_id, item_ids, train_df, cb_model):
    user_rated = train_df[train_df['user_id'] == user_id]['item_id'].tolist()
    return {iid: cb_model.predict_score(user_id, iid, user_rated) / 5.0 for iid in item_ids}


def cb_recommend(user_id, train_df, cb_model, top_n=10):
    user_rated = train_df[train_df['user_id'] == user_id]['item_id'].tolist()
    recs = cb_model.get_recommendations_for_user(user_rated, top_n=top_n)
    if recs.empty:
        return recs
    return recs.rename(columns={
        'movie_id': 'item_id',
        'content_score': 'cb_score',
        'genres_str': 'genres',
    })


def cf_predict(user_id, item_id, cf_model):
    return cf_model.predict(user_id, item_id)


def cf_recommend(user_id, cf_model, movies_df, train_df, top_n=10):
    recs = cf_model.get_recommendations_for_user(
        user_id, movies_df, n_recommendations=top_n, ratings_df=train_df
    )
    if recs.empty:
        return recs
    return recs.rename(columns={
        'movie_id': 'item_id',
        'predicted_rating': 'cf_score',
        'genres_str': 'genres',
    })


def hybrid_predict_rating(user_id, item_id, train_df, cb_model, cf_model,
                           cb_weight=CB_WEIGHT, cf_weight=CF_WEIGHT):
    cb_s = cb_scores(user_id, [item_id], train_df, cb_model).get(item_id, 0.0)
    cf_s = cf_predict(user_id, item_id, cf_model)
    return float(np.clip(cb_weight * (cb_s * 5) + cf_weight * cf_s, 0.5, 5.0))


def hybrid_recommend(user_id, train_df, cb_model, cf_model, movies_df,
                     cb_weight=CB_WEIGHT, cf_weight=CF_WEIGHT, top_n=10):
    recs = cf_recommend(user_id, cf_model, movies_df, train_df, top_n=top_n * 2)
    if recs.empty:
        return recs
    recs['hybrid_score'] = recs['item_id'].apply(
        lambda iid: hybrid_predict_rating(user_id, iid, train_df, cb_model, cf_model, cb_weight, cf_weight)
    )
    recs['cb_score'] = recs['item_id'].apply(
        lambda iid: cb_scores(user_id, [iid], train_df, cb_model).get(iid, 0.0)
    )
    return recs.sort_values('hybrid_score', ascending=False).head(top_n)


def get_similar_movies(item_id, cb_model, top_n=10):
    recs = cb_model.get_similar_movies(item_id, top_n=top_n)
    if recs.empty:
        return recs
    return recs.rename(columns={
        'movie_id': 'item_id',
        'similarity_score': 'cb_score',
        'genres_str': 'genres',
    })
