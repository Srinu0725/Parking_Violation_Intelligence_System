import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import full_pipeline, VEHICLE_WEIGHTS
from src.clustering import assign_h3_cells, run_hdbscan, build_cluster_profiles, get_repeat_offenders
from src.model import (
    build_ml_dataset,
    predict_hotspots,
    generate_patrol_recommendations,
    FEATURE_COLS
)
from src.maps import make_heatmap, make_time_heatmap
import streamlit.components.v1 as components

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bengaluru Parking Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card{background:#f8f9fa;border-radius:10px;padding:16px;text-align:center;border:1px solid #e0e0e0}
.metric-val{font-size:28px;font-weight:700;color:#d73027}
.metric-lbl{font-size:13px;color:#555;margin-top:4px}
.section-head{font-size:18px;font-weight:600;margin:20px 0 10px;border-left:4px solid #d73027;padding-left:10px}
</style>
""", unsafe_allow_html=True)

# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and processing dataset...")
def load_all(filepath):
    df_raw, approved, exp = full_pipeline(filepath)
    return df_raw, approved, exp

@st.cache_data(show_spinner="Running spatial clustering...")
def run_clustering(_exp_df):
    df_h3 = assign_h3_cells(_exp_df, resolution=8)
    

    df_clust = run_hdbscan(
        df_h3,
        min_cluster_size=40
    )
    profiles = build_cluster_profiles(df_clust)
    repeats = get_repeat_offenders(df_clust, min_violations=3)
    return df_h3, df_clust, profiles, repeats
@st.cache_data(show_spinner=False)
def get_heatmap_html(fdf, profiles):

    map_path = "outputs/heatmap.html"

    make_heatmap(
        fdf,
        profiles,
        map_path
    )

    with open(
        map_path,
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()

@st.cache_data(show_spinner=False)
def get_time_heatmap_html(fdf):

    path = "outputs/time_heatmap.html"

    make_time_heatmap(
        fdf,
        path
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()

def get_filtered_results(
    violation,
    vehicle,
    station,
    peak_only,
    _df_h3,
    _df_clust,
    _profiles,
    _repeats
):

    fdf = _df_h3

    mask = pd.Series(True, index=_df_h3.index)

    if violation != "All":
        mask &= (_df_h3["violation_list"] == violation)

    if vehicle != "All":
        mask &= (_df_h3["vehicle_type"] == vehicle)

    if station != "All":
        mask &= (_df_h3["police_station"] == station)

    if peak_only:
        mask &= (_df_h3["is_peak"] == 1)

    fdf = _df_h3.loc[mask]

    if len(fdf) == 0:
        return (
            fdf,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame()
        )

    if len(fdf) < 20:
        return (
            fdf,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame()
        )
 
    # keep only cluster rows that belong to filtered records
    filtered_clusters = _df_clust.loc[
        _df_clust.index.intersection(fdf.index)
    ]

    filtered_profiles = _profiles.copy()
    filtered_repeats = _repeats.copy()

    if vehicle != "All":
        filtered_profiles = filtered_profiles[
            filtered_profiles["top_vehicle"] == vehicle
        ]

        filtered_repeats = filtered_repeats[
            filtered_repeats["vehicle_type"] == vehicle
        ]

    if station != "All":
        filtered_profiles = filtered_profiles[
            filtered_profiles["top_station"] == station
        ]

    return (
        fdf,
        filtered_clusters,
        filtered_profiles,
        filtered_repeats
    )
@st.cache_data(show_spinner=False)
def load_cached_files():

    df_h3 = pd.read_parquet("data/df_h3.parquet")
    df_clust = pd.read_parquet("data/df_clust.parquet")
    profiles = pd.read_parquet("data/profiles.parquet")
    repeats = pd.read_parquet("data/repeats.parquet")

    return df_h3, df_clust, profiles, repeats

@st.cache_data(show_spinner=False)
def build_hex_locations(df):

    return (
        df.groupby("hex8")
        .agg(
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
            top_station=(
                "police_station",
                lambda x: x.mode().iloc[0]
                if not x.mode().empty
                else "Unknown"
            )
        )
        .reset_index()
    )

@st.cache_data(show_spinner=False)
def load_ml_dataset(df):
    return build_ml_dataset(df)

@st.cache_resource(show_spinner="Training predictive model...")
def load_model(_ml_df):

    from src.model import train_model

    model_path = "models/xgb_hotspot.pkl"

    if os.path.exists(model_path):

        saved = joblib.load(model_path)

        if isinstance(saved, dict):
            return saved["model"], saved["metrics"]

        return saved, None

    model, metrics = train_model(_ml_df, model_path)

    return model, metrics


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🚦 Parking Intelligence")
st.sidebar.markdown("---")


DATA_PATH = st.sidebar.text_input(
    "Dataset path",
    value="data/parking_violations.parquet"
)



if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at `{DATA_PATH}`. Update the path in the sidebar.")
    st.info("Expected columns: id, latitude, longitude, vehicle_type, violation_type, created_datetime, etc.")
    st.stop()



df_h3, df_clust, profiles, repeats = load_cached_files()
HEX_LOCS = build_hex_locations(df_h3)
for col in [
    "vehicle_type",
    "violation_list",
    "police_station"
]:
    if col in df_h3.columns:
        df_h3[col] = df_h3[col].astype("category")

exp = df_h3
approved = df_h3
df_raw = df_h3

print("CACHE FILES LOADED")
print(df_h3.shape)
print(df_clust.shape)
print(profiles.shape)
print(repeats.shape)
print(
    "df_h3 MB:",
    round(
        df_h3.memory_usage(deep=True).sum()
        / 1024**2,
        2
    )
)

print(
    "df_clust MB:",
    round(
        df_clust.memory_usage(deep=True).sum()
        / 1024**2,
        2
    )
)

print(
    "profiles MB:",
    round(
        profiles.memory_usage(deep=True).sum()
        / 1024**2,
        2
    )
)

print(
    "repeats MB:",
    round(
        repeats.memory_usage(deep=True).sum()
        / 1024**2,
        2
    )
)

print("STEP 5")

# keep original index alignment
df_h3 = df_h3.sort_index()
df_clust = df_clust.sort_index()
# Sidebar filters
st.sidebar.markdown("### Filters")
violation_types = ['All'] + sorted(exp['violation_list'].dropna().unique().tolist())
sel_violation = st.sidebar.selectbox("Violation type", violation_types)

# vehicle_types = ['All'] + sorted(approved['vehicle_type'].dropna().unique().tolist())
if sel_violation != "All":

    valid_vehicles = (
        exp[
            exp["violation_list"] == sel_violation
        ]["vehicle_type"]
        .dropna()
        .unique()
        .tolist()
    )

else:

    valid_vehicles = (
        approved["vehicle_type"]
        .dropna()
        .unique()
        .tolist()
    )

vehicle_types = ["All"] + sorted(valid_vehicles)

sel_vehicle = st.sidebar.selectbox(
    "Vehicle type",
    vehicle_types
)
stations = ['All'] + sorted(approved['police_station'].dropna().unique().tolist())
sel_station = st.sidebar.selectbox("Police station", stations)

peak_only = st.sidebar.checkbox("Peak hours only (7–10am, 5–9pm)", value=False)

# Apply filters
print("BEFORE FILTERING")

(
    fdf,
    filtered_clusters,
    filtered_profiles,
    filtered_repeats
) = get_filtered_results(
    sel_violation,
    sel_vehicle,
    sel_station,
    peak_only,
    df_h3,
    df_clust,
    profiles,
    repeats
)
print("After FILTERING")

if len(fdf) == 0:
    st.warning(
        "No records match the selected filters."
    )
    st.stop()



# ── Navigation ────────────────────────────────────────────────────────────────
page = st.sidebar.radio(
    "View",
    ["Overview", "Heatmap", "Priority Zones", "Time Patterns",
     "Predictions", "Repeat Offenders"],
)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Bengaluru Parking Violation Intelligence")
    st.caption("AI-driven hotspot detection and enforcement prioritisation")
    st.markdown("---")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    total_v = len(fdf)
    n_zones = len(filtered_profiles)
    top_zone = filtered_profiles.iloc[0]['top_station'] if len(filtered_profiles) > 0 else 'N/A'
    avg_lag = fdf['response_lag_hrs'].mean()
    repeat_count = len(filtered_repeats)

    for col, val, lbl in zip(
        [c1, c2, c3, c4, c5],
        [f"{total_v:,}", f"{n_zones}", top_zone, f"{avg_lag:.1f} hrs", f"{repeat_count:,}"],
        ["Approved violations", "Hotspot zones", "Highest impact area", "Avg response lag", "Repeat offenders"]
    ):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-val">{val}</div>
          <div class="metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-head">Top 10 violation types</div>', unsafe_allow_html=True)
        # vt_counts = exp['violation_list'].value_counts().head(10).reset_index()
        vt_counts = fdf['violation_list'].value_counts().head(10).reset_index()
        vt_counts.columns = ['Violation', 'Count']
        fig = px.bar(vt_counts, x='Count', y='Violation', orientation='h',
                     color='Count', color_continuous_scale='Reds')
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          yaxis={'categoryorder': 'total ascending'}, height=350)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.markdown('<div class="section-head">Violations by vehicle type</div>', unsafe_allow_html=True)
        vc = fdf['vehicle_type'].value_counts().head(10).reset_index()
        vc.columns = ['Vehicle', 'Count']
        fig2 = px.bar(vc, x='Vehicle', y='Count', color='Count',
                      color_continuous_scale='Blues')
        fig2.update_layout(showlegend=False, coloraxis_showscale=False,
                           xaxis_tickangle=-30, height=350)
        st.plotly_chart(fig2, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-head">Monthly trend</div>', unsafe_allow_html=True)
        monthly = fdf.groupby('month_name')['id'].count().reset_index()
        monthly.columns = ['Month', 'Count']
        month_order = ['November','December','January','February','March','April']
        monthly['Month'] = pd.Categorical(monthly['Month'], categories=month_order, ordered=True)
        monthly = monthly.sort_values('Month')
        fig3 = px.line(monthly, x='Month', y='Count', markers=True,
                       color_discrete_sequence=['#d73027'])
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, width="stretch")

    with col4:
        st.markdown('<div class="section-head">Top 10 police stations by violations</div>', unsafe_allow_html=True)
        ps = fdf['police_station'].value_counts().head(10).reset_index()
        ps.columns = ['Station', 'Count']
        fig4 = px.bar(ps, x='Count', y='Station', orientation='h',
                      color='Count', color_continuous_scale='Oranges')
        fig4.update_layout(showlegend=False, coloraxis_showscale=False,
                           yaxis={'categoryorder': 'total ascending'}, height=320)
        st.plotly_chart(fig4, width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — HEATMAP
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Heatmap":
    st.title("Violation Heatmap")
    st.caption("Intensity weighted by vehicle size. Red numbered markers = top 10 enforcement priorities.")

    map_path = "outputs/heatmap.html"
  
    if filtered_profiles.empty:
        st.warning(
            "No hotspot clusters found for selected filters."
        )
        st.stop()

    map_html = get_heatmap_html(
        fdf,
        filtered_profiles
)

    # components.html(map_html, height=600, scrolling=False)

    st.components.v1.html(
        map_html,
        height=600,
        scrolling=False
    )
    st.markdown("---")

    st.subheader("Violation density by hour of day")

    st.info(
        """
        Temporal hotspot animation based on recorded parking violations.
        The timeline reflects when violations were captured by enforcement
        personnel and helps identify recurring hotspot periods for patrol
        deployment and enforcement planning.
        """
    )

    time_map_path = "outputs/time_heatmap.html"

    t_html = get_time_heatmap_html(
    fdf
)
    components.html(t_html, height=500, scrolling=False)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PRIORITY ZONES
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Priority Zones":
    st.title("Enforcement Priority Zones")
    st.caption("Ranked by Congestion Impact Score (CIS) — accounts for violation density, vehicle size, peak hours, and response lag.")

    st.info("""
    **Congestion Impact Score formula:**
    `CIS = violations × avg_vehicle_weight × (1 + peak_ratio) × avg_severity × (1 + log(response_lag)/10)`
    Normalised to 0–100. Higher = deploy enforcement here first.
    """)

    top_n = st.slider("Show top N zones", 5, 50, 20)
    if filtered_profiles.empty:
        st.warning(
            "No hotspot clusters found for selected filters."
        )
        st.stop()
    display = filtered_profiles.head(top_n)[[
        'priority_rank', 'top_station', 'top_junction', 'violation_count',
        'cis_score', 'top_violation', 'top_vehicle', 'peak_ratio',
        'heavy_ratio', 'avg_response_lag', 'recommended_shift'
    ]].copy()

    display.columns = [
        'Rank', 'Police Station', 'Junction', 'Violations',
        'CIS Score', 'Top Violation', 'Top Vehicle', 'Peak Hour %',
        'Heavy Vehicle %', 'Avg Response Lag (hrs)', 'Recommended Patrol Shift'
    ]
    display['Peak Hour %'] = (display['Peak Hour %'] * 100).round(1)
    display['Heavy Vehicle %'] = (display['Heavy Vehicle %'] * 100).round(1)
    display['Avg Response Lag (hrs)'] = display['Avg Response Lag (hrs)'].round(1)

    st.dataframe(
        display.style.background_gradient(subset=['CIS Score'], cmap='Reds')
                     .format({'CIS Score': '{:.1f}', 'Peak Hour %': '{:.1f}%',
                              'Heavy Vehicle %': '{:.1f}%'}),
        width="stretch", height=500
    )

    st.download_button(
        "⬇️ Download priority zones CSV",
        display.to_csv(index=False),
        "priority_zones.csv",
        "text/csv"
    )

    # CIS score bar chart
    fig = px.bar(display.head(15), x='CIS Score', y='Police Station',
                 orientation='h', color='CIS Score',
                 color_continuous_scale='Reds', title='Top 15 zones by CIS score')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=450)
    st.plotly_chart(fig, width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TIME PATTERNS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Time Patterns":
    st.title("Violation Time Patterns")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Violations by hour of day")
        hourly = fdf.groupby('hour')['id'].count().reset_index()
        hourly.columns = ['Hour', 'Count']
        fig = px.bar(hourly, x='Hour', y='Count', color='Count',
                     color_continuous_scale='Reds',
                     labels={'Hour': 'Hour of day (0=midnight)'})
        fig.add_vrect(x0=6.5, x1=10.5, fillcolor='orange', opacity=0.1, annotation_text='Morning peak')
        fig.add_vrect(x0=16.5, x1=21.5, fillcolor='red', opacity=0.1, annotation_text='Evening peak')
        fig.update_layout(height=350, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig,width="stretch")

    with col2:
        st.subheader("Violations by day of week")
        dow_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
        daily = fdf.groupby('dow')['id'].count().reset_index()
        daily['Day'] = daily['dow'].map(dow_map)
        fig2 = px.bar(daily, x='Day', y='id', color='id',
                      color_continuous_scale='Blues',
                      category_orders={'Day': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']})
        fig2.update_layout(height=350, showlegend=False, coloraxis_showscale=False,
                           yaxis_title='Violations')
        st.plotly_chart(fig2, width="stretch")

    st.subheader("Hour × Day heatmap (violation density)")
    pivot = fdf.groupby(['dow', 'hour'])['id'].count().reset_index()
    pivot['Day'] = pivot['dow'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
    pivot_table = pivot.pivot(index='Day', columns='hour', values='id').fillna(0)
    fig3 = px.imshow(
        pivot_table,
        labels=dict(x="Hour of day", y="Day", color="Violations"),
        color_continuous_scale='Reds',
        aspect='auto',
        title='When and which days violations peak'
    )
    fig3.update_layout(height=320)
    st.plotly_chart(fig3, width="stretch")

    st.subheader("Vehicle type breakdown by hour")
    veh_hour = fdf.groupby(['hour', 'vehicle_type'])['id'].count().reset_index()
    top_vehs = fdf['vehicle_type'].value_counts().head(5).index.tolist()
    veh_hour = veh_hour[veh_hour['vehicle_type'].isin(top_vehs)]
    fig4 = px.line(veh_hour, x='hour', y='id', color='vehicle_type', markers=True,
                   labels={'hour': 'Hour', 'id': 'Violations', 'vehicle_type': 'Vehicle'})
    fig4.update_layout(height=350)
    st.plotly_chart(fig4, width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — PREDICTIONS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Predictions":
    st.title("Predictive Hotspot Forecasting")
    st.caption("XGBoost model predicts which zones will have high violations for a given time slot.")
    st.info(
    """
    Forecasting Note

    The predictive hotspot model is trained on the complete historical
    Bengaluru parking violation dataset to capture long-term spatial and
    temporal patterns.

    Sidebar filters affect exploratory analytics, heatmaps, hotspot zones,
    and offender analysis, but hotspot forecasting uses the full historical
    dataset for more reliable city-wide predictions.
    """
)

    ml_df = load_ml_dataset(df_h3)
    model, metrics = load_model(ml_df)

    # if metrics:
    #     c1, c2 = st.columns(2)
    #     c1.metric("F1 Score", f"{metrics['f1']:.3f}")
    #     c2.metric("MAE", f"{metrics['mae']:.3f}")
    if metrics and "precision" in metrics:

        c1, c2, c3, c4, c5 = st.columns(5)

        c1.metric("F1 Score", f"{metrics['f1']:.3f}")
        c2.metric("MAE", f"{metrics['mae']:.3f}")
        c3.metric("Precision", f"{metrics['precision']:.3f}")
        c4.metric("Recall", f"{metrics['recall']:.3f}")
        c5.metric("ROC-AUC", f"{metrics['auc']:.3f}")

    if metrics and "confusion_matrix" in metrics:

        st.markdown("### Confusion Matrix")

        cm = np.array(metrics["confusion_matrix"])

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Normal", "Actual Hotspot"],
            columns=["Pred Normal", "Pred Hotspot"]
        )

        st.dataframe(cm_df, width="stretch")

        if metrics and "feature_importance" in metrics:

            st.markdown("---")
            st.subheader("Feature Importance")

            imp_df = pd.DataFrame(
                metrics["feature_importance"]
            )

            fig = px.bar(
                imp_df,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Reds",
                title="Top Factors Driving Hotspot Predictions"
            )

            fig.update_layout(
                height=500,
                yaxis={"categoryorder": "total ascending"},
                showlegend=False
            )

            st.plotly_chart(
                fig,
                width="stretch"
            )

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        dow_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        sel_day = st.selectbox("Select day", dow_names)
        sel_dow = dow_names.index(sel_day)
    with col2:
        bucket_options = {
            'Morning peak (7–10am)': 'morning_peak',
            'Midday (10am–5pm)': 'midday',
            'Evening peak (5–9pm)': 'evening_peak',
            'Off-peak / night': 'off_peak',
        }
        sel_bucket_label = st.selectbox("Select time slot", list(bucket_options.keys()))
        sel_bucket = bucket_options[sel_bucket_label]

    if st.button("🔮 Predict hotspots"):
        preds = predict_hotspots(
                model,
                ml_df,
                sel_dow,
                sel_bucket,
                top_n=50
            )

        preds = generate_patrol_recommendations(preds)

     

        preds = preds.merge(
                HEX_LOCS,
                on="hex8",
                how="left"
            )

   
        display_preds = preds.head(20).copy()

        display_preds = display_preds.dropna(
            subset=["lat", "lon"]
        )
        
        st.session_state["preds"] = preds
        st.session_state["display_preds"] = display_preds
        st.session_state["sel_bucket_label"] = sel_bucket_label
        
    if "preds" in st.session_state:

        preds = st.session_state["preds"]
        display_preds = st.session_state["display_preds"]
        sel_bucket_label = st.session_state["sel_bucket_label"]

        if display_preds.empty:
            st.warning("No hotspot locations available.")
            st.stop()

        st.success(
            f"Forecast generated for {sel_day} • {sel_bucket_label}"
        )

        import folium
        import streamlit.components.v1 as components

        m = folium.Map(
            location=[
                display_preds["lat"].mean(),
                display_preds["lon"].mean()
            ],
            zoom_start=12,
            tiles="CartoDB positron"
        )

        for _, row in display_preds.iterrows():

            if row["priority"] == "Critical":
                marker_color = "red"

            elif row["priority"] == "High":
                marker_color = "orange"

            else:
                marker_color = "green"

            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=int(row["hotspot_prob"] * 25) + 6,
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.8,
                tooltip=(
                    f"{row['top_station']}<br>"
                    f"Risk: {row['hotspot_prob']:.1%}<br>"
                    f"Priority: {row['priority']}"
                ),
            ).add_to(m)

        components.html(
            m._repr_html_(),
            height=600
        )
        st.markdown("""
            ### Risk Legend

            🔴 Critical Zone (>99%)

            🟠 High Priority Zone (98–99%)

            🟢 Monitoring Zone (<98%)
            """)
        
        priority_counts_raw = preds["priority"].value_counts()

        critical = priority_counts_raw.get("Critical", 0)
        high = priority_counts_raw.get("High", 0)
        medium = priority_counts_raw.get("Medium", 0)

        tow_trucks = critical * 2 + high
        officers = critical * 4 + high * 2
        patrol_units = max(1, critical + high)
        monitor_zones = medium

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Tow Trucks Required",
            tow_trucks
        )

        c2.metric(
            "Officers Required",
            officers
        )

        c3.metric(
            "Patrol Units",
            patrol_units
        )

        c4.metric(
            "Monitoring Zones",
            monitor_zones
        )

        priority_counts = pd.DataFrame({
            "Priority": ["Critical", "High", "Medium"],
            "Count": [critical, high, medium]
        })

        # fig_priority = px.pie(
        #     priority_counts,
        #     names="Priority",
        #     values="Count",
        #     title="Enforcement Allocation",
        #     color="Priority",
        #     color_discrete_map={
        #         "Critical": "red",
        #         "High": "orange",
        #         "Medium": "green"
        #     }
        # )

        # st.plotly_chart(
        #     fig_priority,
        #     width="stretch"
        # )
        fig_priority = px.bar(
            priority_counts,
            x="Count",
            y="Priority",
            orientation="h",
            color="Priority",
            color_discrete_map={
                "Critical": "red",
                "High": "orange",
                "Medium": "green"
            },
            title="AI-Driven Enforcement Prioritization Across High-Risk Parking Zones"
        )

        fig_priority.update_layout(
            height=350,
            xaxis_title="Number of Predicted Zones",
            yaxis_title="Priority Level",
            yaxis={
                "categoryorder": "array",
                "categoryarray": ["Critical", "High", "Medium"]
            }
        )

        st.plotly_chart(
            fig_priority,
            width="stretch"
        )

        avg_risk = (
            display_preds["hotspot_prob"]
            .mean() * 100
        )
        # top_zone = display_preds.iloc[0]["top_station"]
        if filtered_profiles.empty:
            top_zone = "N/A"
        else:
            top_zone = (
                filtered_profiles
                .sort_values("cis_score", ascending=False)
                .iloc[0]["top_station"]
            )

        st.info(
            f"""
        Forecast Summary

        • Critical Zones: {critical}
        • High Priority Zones: {high}
        • Monitoring Zones: {medium}

        • Highest Risk Area: {top_zone}
        • Average Risk Score: {avg_risk:.1f}%
           """
        )
        
        # Table
        preds["Risk %"] = (
            preds["hotspot_prob"] * 100
        ).round(1).astype(str) + "%"


        display_preds = (
        display_preds
            .sort_values(
                "hotspot_prob",
                ascending=False
            )
            .copy()
        )

        display_preds["Risk %"] = (
            display_preds["hotspot_prob"] * 100
        ).round(1).astype(str) + "%"
     

        st.subheader("Patrol Recommendations")

        st.dataframe(
            display_preds[
                [
                    "hex8",
                    "top_station",
                    "Risk %",
                    "priority",
                    "action"
                ]
            ],
            width="stretch"
        )
        st.markdown("---")
        st.subheader("AI Resource Recommendation")

        recommended_officers = (
            critical * 4
            + high * 2
            + medium * 1
        )

        recommended_tows = (
            critical * 2
            + high * 1
        )

        recommended_patrols = max(
            1,
            int(np.ceil(recommended_officers / 4))
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Recommended Officers",
            recommended_officers
        )

        c2.metric(
            "Recommended Patrol Vehicles",
            recommended_patrols
        )

        c3.metric(
            "Recommended Tow Trucks",
            recommended_tows
        )
        st.markdown("---")
        st.subheader("Available Resources")
        c1, c2, c3 = st.columns(3)

        available_officers = c1.number_input(
            "Traffic Officers Available",
            min_value=0,
            value=min(recommended_officers, 50)
        )

        available_patrols = c2.number_input(
            "Patrol Vehicles Available",
            min_value=0,
            value=min(recommended_patrols, 10)
        )

        available_tows = c3.number_input(
            "Tow Trucks Available",
            min_value=0,
            value=min(recommended_tows, 5)
        )
        if st.button("🚔 Generate Deployment Plan"):
            weights = {
            "Critical": 5,
            "High": 3,
            "Medium": 1
        }

            deployment = display_preds.copy()
            deployment = deployment.sort_values(
            "hotspot_prob",
            ascending=False
            )

            deployment["Deployment Rank"] = range(
                1,
                len(deployment) + 1
            )

            deployment["weight"] = deployment["priority"].map(weights)
            total_weight = deployment["weight"].sum()

            deployment["allocated_officers"] = (
                deployment["weight"]
                / total_weight
                * available_officers
            ).round().astype(int)

            deployment["allocated_tows"] = (
            deployment["weight"]
            / total_weight
            * available_tows
            ).round().astype(int)

            deployment["allocated_patrols"] = (
                deployment["weight"]
                / total_weight
                * available_patrols
            ).round().astype(int)

            diff = (
                available_officers
                - deployment["allocated_officers"].sum()
            )

            deployment.loc[
                deployment["allocated_officers"].idxmax(),
                "allocated_officers"
            ] += diff

            diff = (
                available_tows
                - deployment["allocated_tows"].sum()
            )



            deployment.loc[
                deployment["allocated_tows"].idxmax(),
                "allocated_tows"
            ] += diff 

            diff = (
                    available_patrols
                    - deployment["allocated_patrols"].sum()
                        )

            deployment.loc[
                deployment["allocated_patrols"].idxmax(),
                "allocated_patrols"
            ] += diff

            deployment["deployment_window"] = sel_bucket_label



            coverage_officers = (
            available_officers
            / max(1, recommended_officers)
            * 100
            )

            coverage_patrols = (
                available_patrols
                / max(1, recommended_patrols)
                * 100
            )

            coverage_tows = (
                available_tows
                / max(1, recommended_tows)
                * 100
            )


            c1,c2,c3 = st.columns(3)

            c1.metric(
                "Officer Coverage",
                f"{coverage_officers:.1f}%"
            )

            c2.metric(
                "Patrol Coverage",
                f"{coverage_patrols:.1f}%"
            )

            c3.metric(
                "Tow Coverage",
                f"{coverage_tows:.1f}%"
            )

            overall_coverage = min(
                coverage_officers,
                coverage_patrols,
                coverage_tows
            )

            if overall_coverage >= 90:
                st.success(
                    f"Resources Sufficient ({overall_coverage:.1f}% coverage)"
                )

            elif overall_coverage >= 70:
                st.warning(
                    f"Resources Partially Sufficient ({overall_coverage:.1f}% coverage)"
                )

            else:
                st.error(
                    f"Resources Insufficient ({overall_coverage:.1f}% coverage)"
                )


            officer_shortage = max(
                                0,
                                recommended_officers - available_officers
                            )
            
            patrol_shortage = max(
                                0,
                                recommended_patrols - available_patrols
                            )
            
            tow_shortage = max(
                                0,
                                recommended_tows - available_tows
                            )
            
            if officer_shortage > 0:
                                            st.warning(
                                                f"Officer shortage: {officer_shortage}"
                                            )
                        
            if patrol_shortage > 0:
                                            st.warning(
                                                f"Patrol shortage: {patrol_shortage}"
                                            )
                        
            if tow_shortage > 0:
                                            st.warning(
                                                f"Tow shortage: {tow_shortage}"
                                            )
            
            deployment["Expected Action"] = deployment["action"]    
            
            deployment["impact_score"] = (
                            deployment["allocated_officers"] * 0.4
                                + deployment["allocated_tows"] * 0.4
                                + deployment["allocated_patrols"] * 0.2
                            )
            
            estimated_reduction = min(
                                60,
                                deployment["impact_score"].sum() * 1.5
                            )
            
            avg_risk_before = (
                                display_preds["hotspot_prob"]
                                .mean() * 100
                            )
            
            avg_risk_after = max(
                                0,
                                avg_risk_before - estimated_reduction
                            )
            
            st.subheader(
                                "Risk Comparison"
                            )
            
            c1,c2 = st.columns(2)
            
            c1.metric(
                                "Average Risk Before Deployment",
                                f"{avg_risk_before:.1f}%"
                            )
            
            c2.metric(
                                "Expected Risk After Deployment",
                                f"{avg_risk_after:.1f}%"
                            )
            
                            
            
            st.subheader("Optimized Deployment Plan")
            
            st.dataframe(
                                deployment[[
                                    "Deployment Rank",
                                    "top_station",
                                    "priority",
                                    "Risk %",
                                    "Expected Action",
                                    "allocated_officers",
                                    "allocated_patrols",
                                    "allocated_tows",
                                    "deployment_window"
                                ]],
                                width="stretch"
                            )
            
                            
            
            st.download_button(
                                "⬇️ Download Deployment Plan",
                                deployment.to_csv(index=False),
                                "deployment_plan.csv",
                                "text/csv"
                            )
            
            st.success(
                                    f"""
                                Deployment Summary
            
                                Zones Covered: {len(deployment)}
            
                                Officers Assigned: {deployment['allocated_officers'].sum()}
            
                                Patrol Vehicles Assigned: {deployment['allocated_patrols'].sum()}
            
                                Tow Trucks Assigned: {deployment['allocated_tows'].sum()}
            
                                Expected Congestion Reduction: {estimated_reduction:.1f}%
            
                                Highest Priority Zone: {deployment.iloc[0]['top_station']}
                                """
                            )
            
            st.subheader(
                                "Top 5 Immediate Action Zones"
                            )
            
            st.dataframe(
                                deployment.head(5)[[
                                    "Deployment Rank",
                                    "top_station",
                                    "priority",
                                    "Risk %",
                                    "allocated_officers",
                                    "allocated_patrols",
                                    "allocated_tows"
                                ]]
                            )
            

        export_cols = [
            "hex8",
            "top_station",
            "hotspot_prob",
            "priority",
            "action",
            "lat",
            "lon"
        ]

        st.download_button(
            "⬇️ Download Predicted Hotspots",
            preds[export_cols].to_csv(index=False),
            "predicted_hotspots.csv",
            "text/csv"
        )

        st.subheader("Predicted Hotspot Locations")

        st.dataframe(
            display_preds[
                [
                    "hex8",
                    "top_station",
                    "lat",
                    "lon"
                ]
            ],
            width="stretch"
        )


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — REPEAT OFFENDERS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Repeat Offenders":
    st.title("Repeat Offender Analysis")
    st.caption("Vehicles with 3+ violations. These represent habitual violators who require targeted action.")
    if filtered_repeats.empty:
            st.warning(
                    "No repeat offenders found for selected filters."
                )
            st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total repeat offenders", f"{len(filtered_repeats):,}")
    c2.metric("Max violations (single vehicle)", f"{filtered_repeats['total_violations'].max()}")
    c3.metric("Avg violations per repeat offender", f"{filtered_repeats['total_violations'].mean():.1f}")

    st.dataframe(filtered_repeats.head(50), width="stretch", height=400)
    st.download_button(
        "⬇️ Download repeat offenders CSV",
        filtered_repeats.to_csv(index=False),
        "repeat_offenders.csv",
        "text/csv"
    )

    fig = px.histogram(filtered_repeats, x='total_violations', nbins=30,
                       title='Distribution of violations per repeat offender',
                       color_discrete_sequence=['#d73027'])
    fig.update_layout(height=300)
    st.plotly_chart(fig, width="stretch")

    top_veh = filtered_repeats['vehicle_type'].value_counts().head(8).reset_index()
    top_veh.columns = ['Vehicle type', 'Count']
    fig2 = px.pie(top_veh, names='Vehicle type', values='Count',
                  title='Repeat offenders by vehicle type',
                  color_discrete_sequence=px.colors.sequential.Reds_r)
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, width="stretch")
