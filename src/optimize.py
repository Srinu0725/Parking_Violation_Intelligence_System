import optuna
import xgboost as xgb

from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

from src.model import FEATURE_COLS, TARGET


def objective(trial, ml_df):

    ml_df = ml_df.sort_values("date").reset_index(drop=True)

    X = ml_df[FEATURE_COLS].fillna(0)
    y = ml_df[TARGET]

    tscv = TimeSeriesSplit(n_splits=3)

    scores = []

    for train_idx, test_idx in tscv.split(X):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = xgb.XGBClassifier(
            n_estimators=trial.suggest_int(
                "n_estimators",
                100,
                800
            ),
            max_depth=trial.suggest_int(
                "max_depth",
                3,
                10
            ),
            learning_rate=trial.suggest_float(
                "learning_rate",
                0.01,
                0.3,
                log=True
            ),
            subsample=trial.suggest_float(
                "subsample",
                0.6,
                1.0
            ),
            colsample_bytree=trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0
            ),
            min_child_weight=trial.suggest_int(
                "min_child_weight",
                1,
                20
            ),
            gamma=trial.suggest_float(
                "gamma",
                0,
                5
            ),
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
            verbosity=0,
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        scores.append(
            f1_score(y_test, preds)
        )

    return sum(scores) / len(scores)


def tune_xgb(ml_df, n_trials=50):

    study = optuna.create_study(
        direction="maximize"
    )

    study.optimize(
        lambda trial: objective(
            trial,
            ml_df
        ),
        n_trials=n_trials
    )

    print("\nBest F1:")
    print(study.best_value)

    print("\nBest Params:")
    print(study.best_params)

    return study.best_params