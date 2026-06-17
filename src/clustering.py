import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    from sklearn.cluster import DBSCAN

try:
    import h3
    HAS_H3 = True
except ImportError:
    HAS_H3 = False

from src.preprocess import compute_congestion_impact_score


def assign_h3_cells(df: pd.DataFrame, resolution: int = 8) -> pd.DataFrame:
    print("Columns:", df.columns.tolist())
    """Assign H3 hex cell to each violation record."""
    if not HAS_H3:
        raise ImportError("Install h3: pip install h3")
    df = df.copy()
#     df['hex8'] = df.apply(
#         lambda r: h3.latlng_to_cell(
#     r['latitude'],
#     r['longitude'],
#     resolution
# )
    # )
    print(type(df))
    print(df.head())
    def test_row(r):
        # print("INDEX:", r.index.tolist()[:10])
        # print("NAME:", r.name)
        return "test"

    df['hex8'] = df.apply(test_row, axis=1)

    return df


def run_hdbscan(df: pd.DataFrame, min_cluster_size: int = 40) -> pd.DataFrame:
    """
    Run HDBSCAN on violation lat/lon to detect spatial hotspot clusters.
    Uses Haversine distance (great-circle) for geographic accuracy.
    Returns df with 'cluster' column (-1 = noise/outlier).
    """
    df = df.copy()
    coords = np.radians(df[['latitude', 'longitude']].values)

    if HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=10,
            metric='haversine',
            cluster_selection_epsilon=0.0005,  # ~50m
            core_dist_n_jobs=-1,
        )
        df['cluster'] = clusterer.fit_predict(coords)
        df['cluster_prob'] = clusterer.probabilities_
    else:
        # Fallback to DBSCAN if hdbscan not installed
        # eps=0.0005 radians ≈ 50m on earth surface
        clusterer = DBSCAN(eps=0.0005, min_samples=10, metric='haversine', n_jobs=-1)
        df['cluster'] = clusterer.fit_predict(coords)
        df['cluster_prob'] = (df['cluster'] >= 0).astype(float)

    n_clusters = df[df['cluster'] >= 0]['cluster'].nunique()
    noise_pct = (df['cluster'] == -1).mean() * 100
    print(f"  Clusters found: {n_clusters}")
    print(f"  Noise points: {noise_pct:.1f}%")
    return df


def build_cluster_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a rich profile for each cluster including:
    - Centroid lat/lon
    - Dominant violation type, vehicle type
    - Peak hour ratio
    - Congestion impact components
    - Enforcement response lag
    - Repeat offender count
    """
    clustered = df[df['cluster'] >= 0].copy()

    profiles = clustered.groupby('cluster').agg(
        violation_count=('id', 'count'),
        centroid_lat=('latitude', 'mean'),
        centroid_lon=('longitude', 'mean'),
        lat_std=('latitude', 'std'),
        lon_std=('longitude', 'std'),
        top_violation=('violation_list', lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'),
        top_vehicle=('vehicle_type', lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'),
        top_junction=('junction_name', lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown'),
        top_station=('police_station', lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown'),
        peak_ratio=('is_peak', 'mean'),
        weekend_ratio=('is_weekend', 'mean'),
        avg_vehicle_weight=('vehicle_weight', 'mean'),
        avg_severity=('violation_severity', 'mean'),
        avg_response_lag=('response_lag_hrs', 'mean'),
        scooter_ratio=('vehicle_type', lambda x: (x == 'SCOOTER').mean()),
        car_ratio=('vehicle_type', lambda x: (x == 'CAR').mean()),
        heavy_ratio=('vehicle_type', lambda x: x.isin([
            'TANKER','HGV','LORRY/GOODS VEHICLE','BUS (BMTC/KSRTC)',
            'PRIVATE BUS','MINI LORRY','LGV'
        ]).mean()),
        unique_vehicles=('vehicle_number', 'nunique'),
        unique_devices=('device_id', 'nunique'),
    ).reset_index()

    # Radius estimate from std dev (rough spread in meters, ~111km per degree)
    profiles['radius_m'] = (
        np.sqrt(profiles['lat_std']**2 + profiles['lon_std']**2) * 111000
    ).fillna(0).round(0)

    # Congestion Impact Score per cluster
    profiles['avg_response_lag'] = profiles['avg_response_lag'].fillna(24)
    profiles['cis_score'] = (
        profiles['violation_count']
        * profiles['avg_vehicle_weight']
        * (1 + profiles['peak_ratio'])
        * profiles['avg_severity']
        * (1 + np.log1p(profiles['avg_response_lag']) / 10)
    )

    # Normalise to 0–100
    cis_min, cis_max = profiles['cis_score'].min(), profiles['cis_score'].max()
    profiles['cis_score'] = (
        (profiles['cis_score'] - cis_min) / (cis_max - cis_min) * 100
    ).round(2)

    # Recommended patrol shift based on peak_ratio
    def recommend_shift(row):
        if row['peak_ratio'] > 0.6:
            return 'Morning (7–10am) + Evening (5–9pm)'
        elif row['weekend_ratio'] > 0.4:
            return 'Weekend all-day'
        else:
            return 'Off-peak hours (10am–5pm)'

    profiles['recommended_shift'] = profiles.apply(recommend_shift, axis=1)
    profiles['priority_rank'] = profiles['cis_score'].rank(ascending=False, method='min').astype(int)

    return profiles.sort_values('cis_score', ascending=False).reset_index(drop=True)


def get_repeat_offenders(df: pd.DataFrame, min_violations: int = 3) -> pd.DataFrame:
    """Find vehicles with repeated violations."""
    veh_col = 'updated_vehicle_number'
    valid = df[df[veh_col].notna()].copy()

    repeats = (
        valid.groupby(veh_col)
        .agg(
            total_violations=('id', 'count'),
            unique_locations=('junction_name', 'nunique'),
            top_violation=('violation_list', lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'),
            vehicle_type=('updated_vehicle_type', lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'),
            first_seen=('created_ist', 'min'),
            last_seen=('created_ist', 'max'),
        )
        .reset_index()
        .rename(columns={veh_col: 'vehicle_number'})
    )

    repeats = repeats[repeats['total_violations'] >= min_violations]
    repeats = repeats.sort_values('total_violations', ascending=False).reset_index(drop=True)
    print(f"  Repeat offenders (≥{min_violations} violations): {len(repeats):,}")
    return repeats
