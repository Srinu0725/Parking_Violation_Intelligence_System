import pandas as pd
import numpy as np
import ast
import warnings
warnings.filterwarnings('ignore')

# ─── Vehicle weight map for Congestion Impact Score ───────────────────────────
VEHICLE_WEIGHTS = {
    'TANKER': 4.0,
    'HGV': 3.5,
    'LORRY/GOODS VEHICLE': 3.5,
    'BUS (BMTC/KSRTC)': 3.0,
    'PRIVATE BUS': 3.0,
    'TOURIST BUS': 3.0,
    'FACTORY BUS': 3.0,
    'SCHOOL VEHICLE': 3.0,
    'MINI LORRY': 2.5,
    'TRACTOR': 2.5,
    'LGV': 2.0,
    'TEMPO': 2.0,
    'VAN': 2.0,
    'MAXI-CAB': 1.8,
    'JEEP': 1.5,
    'CAR': 1.5,
    'GOODS AUTO': 1.2,
    'PASSENGER AUTO': 1.2,
    'MOTOR CYCLE': 1.0,
    'SCOOTER': 1.0,
    'MOPED': 1.0,
    'OTHERS': 1.2,
}

# ─── Violation severity weights ───────────────────────────────────────────────
VIOLATION_SEVERITY = {
    'PARKING NEAR ROAD CROSSING': 3.0,
    'PARKING IN A MAIN ROAD': 2.5,
    'PARKING ON FOOTPATH': 2.0,
    'WRONG PARKING': 1.5,
    'NO PARKING': 1.5,
    'DEFECTIVE NUMBER PLATE': 0.5,
}

# Peak hours: morning 7-10am, evening 5-9pm
PEAK_HOURS = list(range(7, 10)) + list(range(17, 21))


def load_data(filepath: str) -> pd.DataFrame:

    print(f"Loading data from {filepath}...")

    if filepath.endswith(".parquet"):
        df = pd.read_parquet(filepath)

    else:
        df = pd.read_csv(filepath, low_memory=False)

    print(f"  Raw shape: {df.shape}")

    # Drop fully null columns — confirmed from inspection
    drop_cols = ['description', 'closed_datetime', 'action_taken_timestamp']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Parse datetimes
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], utc=True, errors='coerce')
    df['modified_datetime'] = pd.to_datetime(df['modified_datetime'], utc=True, errors='coerce')
    df['validation_timestamp'] = pd.to_datetime(df['validation_timestamp'], utc=True, errors='coerce')

    # Convert to IST (UTC+5:30)
    df['created_ist'] = df['created_datetime'].dt.tz_convert('Asia/Kolkata')
    print("\nTimestamp Check")
    print(
        df[
            ['created_datetime', 'created_ist']
        ].head(10)
)

    # Drop rows with invalid lat/lon (Bengaluru bounds)
    df = df[
        df['latitude'].between(12.7, 13.2) &
        df['longitude'].between(77.3, 77.9)
    ]

    print(f"  After geo filter: {df.shape}")
    print("\nHour Distribution")
    print(
        df['created_ist']
        .dt.hour
        .value_counts()
        .sort_index()
    )
    return df


def filter_approved(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only approved records for primary analysis."""
    approved = df[df['validation_status'] == 'approved'].copy()
    print(f"  Approved records: {len(approved):,} ({len(approved)/len(df)*100:.1f}%)")
    return approved


def parse_violations(df: pd.DataFrame) -> pd.DataFrame:
    """Parse JSON violation_type arrays and explode to one row per violation."""
    def safe_parse(x):
        try:
            return ast.literal_eval(x)
        except Exception:
            return ['UNKNOWN']

    df = df.copy()
    df['violation_list'] = df['violation_type'].apply(safe_parse)
    df_exp = df.explode('violation_list').copy()
    df_exp['violation_list'] = df_exp['violation_list'].str.strip()
    print(f"  Exploded violations: {len(df_exp):,} rows")
    return df_exp


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day-of-week, peak flag, and month features."""
    df = df.copy()
    dt = df['created_ist']
    df['hour'] = dt.dt.hour
    df['dow'] = dt.dt.dayofweek          # 0=Monday, 6=Sunday
    df['dow_name'] = dt.dt.day_name()
    df['month'] = dt.dt.month
    df['month_name'] = dt.dt.month_name()
    df['is_weekend'] = df['dow'].isin([5, 6]).astype(int)
    df['is_peak'] = df['hour'].isin(PEAK_HOURS).astype(int)
    df['date'] = dt.dt.date
    return df


def add_vehicle_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Map vehicle type to congestion weight."""
    df = df.copy()
    df['vehicle_weight'] = df['vehicle_type'].map(VEHICLE_WEIGHTS).fillna(1.2)
    return df


def add_violation_severity(df: pd.DataFrame, violation_col: str = 'violation_list') -> pd.DataFrame:
    """Map violation type to severity score."""
    df = df.copy()
    df['violation_severity'] = df[violation_col].map(VIOLATION_SEVERITY).fillna(1.0)
    return df


def compute_response_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Compute enforcement response lag in hours (modified - created)."""
    df = df.copy()
    df['response_lag_hrs'] = (
        df['modified_datetime'] - df['created_datetime']
    ).dt.total_seconds() / 3600
    # Cap at 168 hours (1 week) — outliers beyond that are data issues
    df['response_lag_hrs'] = df['response_lag_hrs'].clip(0, 168)
    return df


def compute_congestion_impact_score(hex_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Congestion Impact Score per H3 hex cell.

    Formula:
        CIS = violation_count
              × avg_vehicle_weight
              × (1 + peak_ratio)
              × avg_violation_severity
              × (1 + log1p(avg_response_lag_hrs) / 10)

    Higher score = higher enforcement priority.
    """
    agg = hex_df.groupby('hex8').agg(
        violation_count=('id', 'count'),
        avg_vehicle_weight=('vehicle_weight', 'mean'),
        peak_ratio=('is_peak', 'mean'),
        avg_severity=('violation_severity', 'mean'),
        avg_response_lag=('response_lag_hrs', 'mean'),
        lat=('latitude', 'mean'),
        lon=('longitude', 'mean'),
        top_junction=('junction_name', lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'),
        top_police_station=('police_station', lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'),
        top_vehicle=('vehicle_type', lambda x: x.mode()[0] if len(x) > 0 else 'Unknown'),
        weekend_ratio=('is_weekend', 'mean'),
    ).reset_index()

    agg['avg_response_lag'] = agg['avg_response_lag'].fillna(24)

    agg['cis'] = (
        agg['violation_count']
        * agg['avg_vehicle_weight']
        * (1 + agg['peak_ratio'])
        * agg['avg_severity']
        * (1 + np.log1p(agg['avg_response_lag']) / 10)
    )

    # Normalise to 0–100 for readability
    agg['cis_score'] = (
        (agg['cis'] - agg['cis'].min()) /
        (agg['cis'].max() - agg['cis'].min()) * 100
    ).round(2)

    agg['priority_rank'] = agg['cis_score'].rank(ascending=False, method='min').astype(int)

    return agg.sort_values('cis_score', ascending=False).reset_index(drop=True)


def full_pipeline(filepath: str):
    """Run the complete preprocessing pipeline. Returns (raw_df, approved_df, hex_agg_df)."""
    df = load_data(filepath)
    approved = filter_approved(df)
    approved = add_temporal_features(approved)
    approved = add_vehicle_weight(approved)
    approved = compute_response_lag(approved)

    # Exploded version for violation-level analysis
    exp = parse_violations(approved)
    exp = add_violation_severity(exp)

    print(f"\nPipeline complete.")
    print(f"  Approved records: {len(approved):,}")
    print(f"  Exploded violation rows: {len(exp):,}")
    return df, approved, exp
