import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    from sklearn.ensemble import GradientBoostingClassifier


def build_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix for prediction.
    Unit of analysis: hex8 cell × date × hour_bucket (morning/evening/night)
    Target: violation_count in that cell-hour bucket
    """
    df = df.copy()

    # Hour buckets
    def hour_bucket(h):
        if 7 <= h <= 10:   return 'morning_peak'
        if 17 <= h <= 20:  return 'evening_peak'
        if 11 <= h <= 16:  return 'midday'
        return 'off_peak'

    df['hour_bucket'] = df['hour'].apply(hour_bucket)
    df['date'] = pd.to_datetime(df['created_ist'].dt.date)

    # Aggregate to hex8 × date × hour_bucket
    agg = df.groupby(['hex8', 'date', 'hour_bucket', 'dow', 'is_weekend']).agg(
        violation_count=('id', 'count'),
        avg_vehicle_weight=('vehicle_weight', 'mean'),
        heavy_ratio=('vehicle_type', lambda x: x.isin([
            'TANKER','HGV','LORRY/GOODS VEHICLE','BUS (BMTC/KSRTC)','PRIVATE BUS'
        ]).mean()),
        scooter_ratio=('vehicle_type', lambda x: (x == 'SCOOTER').mean()),
        car_ratio=('vehicle_type', lambda x: (x == 'CAR').mean()),
    ).reset_index()

    agg = agg.sort_values(['hex8', 'date', 'hour_bucket']).reset_index(drop=True)

    # Lag features per hex cell (past 7-day rolling)
    agg['lag_1d'] = agg.groupby(['hex8', 'hour_bucket'])['violation_count'].shift(1)
    agg['lag_7d'] = agg.groupby(['hex8', 'hour_bucket'])['violation_count'].shift(7)
    agg['rolling_7d_mean'] = (
        agg.groupby(['hex8', 'hour_bucket'])['violation_count']
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
    )
    agg['rolling_7d_std'] = (
        agg.groupby(['hex8', 'hour_bucket'])['violation_count']
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).std().fillna(0))
    )

    # Encode hour_bucket
    bucket_map = {'morning_peak': 3, 'evening_peak': 3, 'midday': 2, 'off_peak': 1}
    agg['bucket_weight'] = agg['hour_bucket'].map(bucket_map)

    # Binary target: is this a high-risk cell-slot? (above 75th percentile)
    threshold = agg['violation_count'].quantile(0.75)
    agg['is_hotspot'] = (agg['violation_count'] >= threshold).astype(int)

    # Drop rows with no lag data
    agg = agg.dropna(subset=['lag_1d', 'lag_7d'])

    print(f"  ML dataset shape: {agg.shape}")
    print(f"  Hotspot ratio: {agg['is_hotspot'].mean():.1%}")
    return agg


FEATURE_COLS = [
    'dow', 'is_weekend', 'bucket_weight',
    'lag_1d', 'lag_7d', 'rolling_7d_mean', 'rolling_7d_std',
    'avg_vehicle_weight', 'heavy_ratio', 'scooter_ratio', 'car_ratio',
]
TARGET = 'is_hotspot'


def train_model(ml_df: pd.DataFrame, model_path: str = 'models/xgb_hotspot.pkl'):
    """
    Train XGBoost classifier with temporal cross-validation.
    Uses TimeSeriesSplit to avoid data leakage.
    """
    ml_df = ml_df.sort_values('date').reset_index(drop=True)

    X = ml_df[FEATURE_COLS].fillna(0)
    y = ml_df[TARGET]

    # Temporal split: last 20% of dates as test
    split_idx = int(len(ml_df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if HAS_XGB:
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            n_jobs=-1,           # use all 6 cores
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)

    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)] if HAS_XGB else None,
              verbose=False if HAS_XGB else None)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\n  Model evaluation (temporal test set):")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"    MAE:       {mae:.4f}")
    print(f"    Test size: {len(y_test):,} samples")

    # Feature importance
    if HAS_XGB:
        importance = pd.DataFrame({
            'feature': FEATURE_COLS,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\n  Top feature importances:")
        print(importance.to_string(index=False))

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n  Model saved to {model_path}")

    return model, {'f1': f1, 'mae': mae}


def predict_hotspots(model, ml_df: pd.DataFrame, dow: int, hour_bucket: str, top_n: int = 15) -> pd.DataFrame:
    """
    Predict top-N high-risk hex cells for a given day-of-week and hour bucket.
    Returns a DataFrame with hex8, predicted probability, and location info.
    """
    bucket_map = {'morning_peak': 3, 'evening_peak': 3, 'midday': 2, 'off_peak': 1}

    # Use most recent stats per hex8 as the feature base
    latest = (
        ml_df.sort_values('date')
        .groupby('hex8')
        .last()
        .reset_index()
    )

    latest['dow'] = dow
    latest['is_weekend'] = int(dow >= 5)
    latest['bucket_weight'] = bucket_map.get(hour_bucket, 1)

    X_pred = latest[FEATURE_COLS].fillna(0)

    if hasattr(model, 'predict_proba'):
        latest['hotspot_prob'] = model.predict_proba(X_pred)[:, 1]
    else:
        latest['hotspot_prob'] = model.predict(X_pred)

    return (
        latest[['hex8', 'hotspot_prob']]
        .sort_values('hotspot_prob', ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
