import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.preprocess import full_pipeline, VEHICLE_WEIGHTS
from src.clustering import assign_h3_cells, run_hdbscan, build_cluster_profiles, get_repeat_offenders
from src.model import build_ml_dataset, predict_hotspots, FEATURE_COLS
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
    df_clust = run_hdbscan(df_h3, min_cluster_size=40)
    profiles = build_cluster_profiles(df_clust)
    repeats = get_repeat_offenders(df_clust, min_violations=3)
    return df_h3, df_clust, profiles, repeats

@st.cache_resource(show_spinner="Training predictive model...")
def load_model(_ml_df):
    from model import train_model
    model_path = 'models/xgb_hotspot.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path), None
    model, metrics = train_model(_ml_df, model_path)
    return model, metrics

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🚦 Parking Intelligence")
st.sidebar.markdown("---")

DATA_PATH = st.sidebar.text_input(
    "Dataset CSV path",
    value="data/parking_violations.csv",
    help="Path to the HackerEarth dataset CSV"
)

if not os.path.exists(DATA_PATH):
    st.error(f"Dataset not found at `{DATA_PATH}`. Update the path in the sidebar.")
    st.info("Expected columns: id, latitude, longitude, vehicle_type, violation_type, created_datetime, etc.")
    st.stop()

df_raw, approved, exp = load_all(DATA_PATH)
df_h3, df_clust, profiles, repeats = run_clustering(exp)

# Sidebar filters
st.sidebar.markdown("### Filters")
violation_types = ['All'] + sorted(exp['violation_list'].dropna().unique().tolist())
sel_violation = st.sidebar.selectbox("Violation type", violation_types)

vehicle_types = ['All'] + sorted(approved['vehicle_type'].dropna().unique().tolist())
sel_vehicle = st.sidebar.selectbox("Vehicle type", vehicle_types)

stations = ['All'] + sorted(approved['police_station'].dropna().unique().tolist())
sel_station = st.sidebar.selectbox("Police station", stations)

peak_only = st.sidebar.checkbox("Peak hours only (7–10am, 5–9pm)", value=False)

# Apply filters
fdf = exp.copy()
if sel_violation != 'All':
    fdf = fdf[fdf['violation_list'] == sel_violation]
if sel_vehicle != 'All':
    fdf = fdf[fdf['vehicle_type'] == sel_vehicle]
if sel_station != 'All':
    fdf = fdf[fdf['police_station'] == sel_station]
if peak_only:
    fdf = fdf[fdf['is_peak'] == 1]

# ── Navigation ────────────────────────────────────────────────────────────────
page = st.sidebar.radio(
    "View",
    ["📊 Overview", "🗺️ Heatmap", "🎯 Priority Zones", "⏰ Time Patterns",
     "🔮 Predictions", "🔁 Repeat Offenders"],
)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("Bengaluru Parking Violation Intelligence")
    st.caption("AI-driven hotspot detection and enforcement prioritisation")
    st.markdown("---")

    # KPI cards
    c1, c2, c3, c4, c5 = st.columns(5)
    total_v = len(approved)
    n_zones = len(profiles)
    top_zone = profiles.iloc[0]['top_station'] if len(profiles) > 0 else 'N/A'
    avg_lag = approved['response_lag_hrs'].mean()
    repeat_count = len(repeats)

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
        vt_counts = exp['violation_list'].value_counts().head(10).reset_index()
        vt_counts.columns = ['Violation', 'Count']
        fig = px.bar(vt_counts, x='Count', y='Violation', orientation='h',
                     color='Count', color_continuous_scale='Reds')
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          yaxis={'categoryorder': 'total ascending'}, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-head">Violations by vehicle type</div>', unsafe_allow_html=True)
        vc = approved['vehicle_type'].value_counts().head(10).reset_index()
        vc.columns = ['Vehicle', 'Count']
        fig2 = px.bar(vc, x='Vehicle', y='Count', color='Count',
                      color_continuous_scale='Blues')
        fig2.update_layout(showlegend=False, coloraxis_showscale=False,
                           xaxis_tickangle=-30, height=350)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-head">Monthly trend</div>', unsafe_allow_html=True)
        monthly = approved.groupby('month_name')['id'].count().reset_index()
        monthly.columns = ['Month', 'Count']
        month_order = ['November','December','January','February','March','April']
        monthly['Month'] = pd.Categorical(monthly['Month'], categories=month_order, ordered=True)
        monthly = monthly.sort_values('Month')
        fig3 = px.line(monthly, x='Month', y='Count', markers=True,
                       color_discrete_sequence=['#d73027'])
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown('<div class="section-head">Top 10 police stations by violations</div>', unsafe_allow_html=True)
        ps = approved['police_station'].value_counts().head(10).reset_index()
        ps.columns = ['Station', 'Count']
        fig4 = px.bar(ps, x='Count', y='Station', orientation='h',
                      color='Count', color_continuous_scale='Oranges')
        fig4.update_layout(showlegend=False, coloraxis_showscale=False,
                           yaxis={'categoryorder': 'total ascending'}, height=320)
        st.plotly_chart(fig4, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — HEATMAP
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Heatmap":
    st.title("Violation Heatmap")
    st.caption("Intensity weighted by vehicle size. Red numbered markers = top 10 enforcement priorities.")

    map_path = "outputs/heatmap.html"
    if not os.path.exists(map_path):
        with st.spinner("Generating map..."):
            os.makedirs("outputs", exist_ok=True)
            make_heatmap(approved, profiles, map_path)

    with open(map_path, 'r', encoding='utf-8') as f:
        map_html = f.read()
    components.html(map_html, height=600, scrolling=False)

    time_map_path = "outputs/time_heatmap.html"
    st.markdown("---")
    st.subheader("Violation density by hour of day")
    if not os.path.exists(time_map_path):
        with st.spinner("Generating time animation..."):
            make_time_heatmap(approved, time_map_path)
    with open(time_map_path, 'r', encoding='utf-8') as f:
        t_html = f.read()
    components.html(t_html, height=500, scrolling=False)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PRIORITY ZONES
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Priority Zones":
    st.title("Enforcement Priority Zones")
    st.caption("Ranked by Congestion Impact Score (CIS) — accounts for violation density, vehicle size, peak hours, and response lag.")

    st.info("""
    **Congestion Impact Score formula:**
    `CIS = violations × avg_vehicle_weight × (1 + peak_ratio) × avg_severity × (1 + log(response_lag)/10)`
    Normalised to 0–100. Higher = deploy enforcement here first.
    """)

    top_n = st.slider("Show top N zones", 5, 50, 20)
    display = profiles.head(top_n)[[
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
        use_container_width=True, height=500
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
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TIME PATTERNS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "⏰ Time Patterns":
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
elif page == "🔮 Predictions":
    st.title("Predictive Hotspot Forecasting")
    st.caption("XGBoost model predicts which zones will have high violations for a given time slot.")

    ml_df = build_ml_dataset(df_h3)
    model, metrics = load_model(ml_df)

    if metrics:
        c1, c2 = st.columns(2)
        c1.metric("F1 Score", f"{metrics['f1']:.3f}")
        c2.metric("MAE", f"{metrics['mae']:.3f}")

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
        preds = predict_hotspots(model, ml_df, sel_dow, sel_bucket, top_n=15)

        # Merge with hex centroid locations
        hex_locs = df_h3.groupby('hex8').agg(
            lat=('latitude', 'mean'), lon=('longitude', 'mean'),
            top_station=('police_station', lambda x: x.mode()[0])
        ).reset_index()
        preds = preds.merge(hex_locs, on='hex8', how='left')

        st.success(f"Top 15 predicted hotspot zones for {sel_day} {sel_bucket_label}")

        # Map
        import folium, streamlit.components.v1 as components
        m = folium.Map(location=[preds['lat'].mean(), preds['lon'].mean()],
                       zoom_start=12, tiles='CartoDB positron')
        for i, row in preds.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=int(row['hotspot_prob'] * 25) + 6,
                color='#d73027', fill=True, fill_color='#d73027',
                fill_opacity=float(row['hotspot_prob']),
                tooltip=f"Risk: {row['hotspot_prob']:.0%} | {row['top_station']}",
            ).add_to(m)
        components.html(m._repr_html_(), height=450)

        # Table
        preds['Risk %'] = (preds['hotspot_prob'] * 100).round(1).astype(str) + '%'
        st.dataframe(preds[['hex8', 'Risk %', 'top_station', 'lat', 'lon']].head(15),
                     use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — REPEAT OFFENDERS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔁 Repeat Offenders":
    st.title("Repeat Offender Analysis")
    st.caption("Vehicles with 3+ violations. These represent habitual violators who require targeted action.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total repeat offenders", f"{len(repeats):,}")
    c2.metric("Max violations (single vehicle)", f"{repeats['total_violations'].max()}")
    c3.metric("Avg violations per repeat offender", f"{repeats['total_violations'].mean():.1f}")

    st.dataframe(repeats.head(50), use_container_width=True, height=400)
    st.download_button(
        "⬇️ Download repeat offenders CSV",
        repeats.to_csv(index=False),
        "repeat_offenders.csv",
        "text/csv"
    )

    fig = px.histogram(repeats, x='total_violations', nbins=30,
                       title='Distribution of violations per repeat offender',
                       color_discrete_sequence=['#d73027'])
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    top_veh = repeats['vehicle_type'].value_counts().head(8).reset_index()
    top_veh.columns = ['Vehicle type', 'Count']
    fig2 = px.pie(top_veh, names='Vehicle type', values='Count',
                  title='Repeat offenders by vehicle type',
                  color_discrete_sequence=px.colors.sequential.Reds_r)
    fig2.update_layout(height=350)
    st.plotly_chart(fig2, use_container_width=True)
