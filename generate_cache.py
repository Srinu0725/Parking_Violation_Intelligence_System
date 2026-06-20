from src.preprocess import full_pipeline
from src.clustering import (
    assign_h3_cells,
    run_hdbscan,
    build_cluster_profiles,
    get_repeat_offenders
)

print("Loading data...")
_, approved, exp = full_pipeline("data/parking_violations.parquet")

print("Running H3...")
df_h3 = assign_h3_cells(exp, resolution=8)

print("Running HDBSCAN...")
df_clust = run_hdbscan(df_h3, min_cluster_size=40)

print("Building profiles...")
profiles = build_cluster_profiles(df_clust)

print("Finding repeat offenders...")
repeats = get_repeat_offenders(df_clust, min_violations=3)

print("Saving cache files...")
df_h3.to_parquet("data/df_h3.parquet")
df_clust.to_parquet("data/df_clust.parquet")
profiles.to_parquet("data/profiles.parquet")
repeats.to_parquet("data/repeats.parquet")

print("Done.")