# Bengaluru Parking Violation Intelligence System
### Theme 1 — AI-Driven Parking Intelligence | BTP × Flipkart Hackathon 2026

---

## Setup (do this once)

```bash
# 1. Create environment
conda create -n parking python=3.10 -y
conda activate parking

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your dataset
mkdir data
# Copy the HackerEarth CSV into data/ and rename it:
mv your_dataset.csv data/parking_violations.csv
```

---

## Run the dashboard

```bash
cd parking_intel
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Project structure

```
parking_intel/
├── app.py                  ← Streamlit dashboard (run this)
├── requirements.txt
├── data/
│   └── parking_violations.csv   ← place dataset here
├── src/
│   ├── preprocess.py       ← data loading, feature engineering, CIS score
│   ├── clustering.py       ← H3 binning, HDBSCAN, cluster profiles
│   ├── model.py            ← XGBoost hotspot predictor
│   └── maps.py             ← Folium heatmap generation
├── models/
│   └── xgb_hotspot.pkl     ← saved model (auto-generated on first run)
└── outputs/
    ├── heatmap.html        ← interactive map (auto-generated)
    └── time_heatmap.html   ← time-animated map (auto-generated)
```

---

## Key innovations

### 1. Congestion Impact Score (CIS)
Novel metric derived entirely from the dataset:
```
CIS = violations × avg_vehicle_weight × (1 + peak_ratio) × avg_severity × (1 + log(response_lag)/10)
```
- Vehicle weight: TANKER=4.0, BUS=3.0, CAR=1.5, SCOOTER=1.0
- Peak ratio: fraction of violations during 7–10am and 5–9pm
- Response lag: hours between created_datetime and modified_datetime
- Normalised to 0–100 for readability

### 2. HDBSCAN spatial clustering
- Haversine distance metric (geographically accurate)
- Finds variable-density clusters without needing to specify number of clusters
- Each cluster = one enforcement zone with full profile

### 3. XGBoost predictive model
- Temporal train/test split (no data leakage)
- Predicts next-period violation probability per H3 hex cell
- Features: lag-1d, lag-7d, rolling 7-day mean, hour bucket, vehicle mix

### 4. Dataset-only (no external data)
All analysis uses only the provided HackerEarth dataset. No external APIs or enrichment.

---

## Deploy to Streamlit Cloud (free, 5 minutes)

1. Push this repo to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo, set main file = `app.py`
4. Add your CSV as a GitHub LFS file or use Streamlit secrets for path
5. Deploy → share the public URL in your submission

---

## Hardware requirements
- 6-core CPU: XGBoost uses all cores (`n_jobs=-1`)
- 16GB RAM: 298k rows uses ~800MB peak — well within limits
- No GPU needed
