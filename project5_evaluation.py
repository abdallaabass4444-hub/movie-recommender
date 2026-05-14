import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RecommenderEvaluator:
    def __init__(self, k=10, relevance_threshold=4.0, n_eval_users=200):
        self.k = k
        self.relevance_threshold = relevance_threshold
        self.n_eval_users = n_eval_users
        self.results = {}

    def calculate_rmse(self, y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    def calculate_mae(self, y_true, y_pred):
        return float(mean_absolute_error(y_true, y_pred))

    def calculate_precision_recall_f1(self, recommended, relevant):
        recommended = set(list(recommended)[:self.k])
        relevant    = set(relevant)
        if not relevant:
            return 0.0, 0.0, 0.0
        hits = len(recommended & relevant)
        p = hits / self.k
        r = hits / len(relevant)
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f

    def evaluate_model(self, model_name, predict_fn, recommend_fn, test_df):
        print(f'\nEvaluating: {model_name}')
        preds  = np.array([predict_fn(r.user_id, r.item_id) for r in test_df.itertuples(index=False)])
        actual = test_df['rating'].values.astype(float)
        rmse_val = self.calculate_rmse(actual, preds)
        mae_val  = self.calculate_mae(actual, preds)
        print(f'  RMSE : {rmse_val:.4f}  |  MAE : {mae_val:.4f}')

        eval_users = np.random.choice(
            test_df['user_id'].unique(),
            min(self.n_eval_users, test_df['user_id'].nunique()),
            replace=False,
        )
        precisions, recalls, f1s = [], [], []
        for uid in eval_users:
            user_test = test_df[test_df['user_id'] == uid]
            relevant  = set(user_test[user_test['rating'] >= self.relevance_threshold]['item_id'])
            if not relevant:
                continue
            recs = recommend_fn(uid, top_n=self.k)
            if recs is None or recs.empty:
                continue
            p, r, f = self.calculate_precision_recall_f1(recs['item_id'].values, relevant)
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        print(f'  Precision@{self.k} : {np.mean(precisions):.4f}')
        print(f'  Recall@{self.k}    : {np.mean(recalls):.4f}')
        print(f'  F1@{self.k}        : {np.mean(f1s):.4f}')

        self.results[model_name] = {
            'RMSE': round(rmse_val, 4),
            'MAE':  round(mae_val, 4),
            f'Precision@{self.k}': round(np.mean(precisions), 4),
            f'Recall@{self.k}':    round(np.mean(recalls), 4),
            f'F1@{self.k}':        round(np.mean(f1s), 4),
        }

    def evaluate_all(self, models, test_df):
        for model_name, predict_fn, recommend_fn in models:
            self.evaluate_model(model_name, predict_fn, recommend_fn, test_df)
        self.print_results()

    def print_results(self):
        if not self.results:
            print('No results yet.')
            return
        print('\n' + '=' * 55)
        print('Model Comparison')
        print('=' * 55)
        print(pd.DataFrame(self.results).T.to_string())

    def get_results_df(self):
        return pd.DataFrame(self.results).T
