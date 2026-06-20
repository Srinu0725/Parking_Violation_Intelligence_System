from src.preprocess import full_pipeline
from src.clustering import assign_h3_cells
from src.model import build_ml_dataset
from src.optimize import tune_xgb

_, approved, exp = full_pipeline(
    "data/parking_violations.csv"
)

df_h3 = assign_h3_cells(
    exp,
    resolution=8
)

ml_df = build_ml_dataset(
    df_h3
)

best_params = tune_xgb(
    ml_df,
    n_trials=50
)

print(best_params)