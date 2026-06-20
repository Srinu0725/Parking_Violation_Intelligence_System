import pandas as pd
import numpy as np
import folium
from folium.plugins import HeatMap, MarkerCluster, HeatMapWithTime
import json


def make_heatmap(approved_df: pd.DataFrame, cluster_profiles: pd.DataFrame,
                 output_path: str = 'outputs/heatmap.html') -> folium.Map:
    """
    Build an interactive Folium map with:
    - HeatMap layer weighted by CIS score proxy (vehicle_weight × is_peak)
    - Cluster centroid markers with popup details
    - Layer control to toggle on/off
    """
    center_lat = approved_df['latitude'].mean()
    center_lon = approved_df['longitude'].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron',
    )

    # ── HeatMap layer ─────────────────────────────────────────────────────────
    heat_data = approved_df[['latitude', 'longitude', 'vehicle_weight']].dropna()
    heat_fg = folium.FeatureGroup(name='Violation heatmap (intensity = vehicle size)', show=True)
    HeatMap(
        data=heat_data.values.tolist(),
        radius=12,
        blur=15,
        max_zoom=13,
        min_opacity=0.3,
        gradient={0.2: '#4575b4', 0.4: '#74add1', 0.6: '#fee090', 0.8: '#f46d43', 1.0: '#d73027'},
    ).add_to(heat_fg)
    heat_fg.add_to(m)

    # ── Peak-hour only heatmap ────────────────────────────────────────────────
    peak_df = approved_df[approved_df['is_peak'] == 1]
    peak_fg = folium.FeatureGroup(name='Peak-hour violations only', show=False)
    HeatMap(
        data=peak_df[['latitude', 'longitude']].values.tolist(),
        radius=10,
        blur=12,
        gradient={0.2: '#ffffb2', 0.5: '#fd8d3c', 1.0: '#bd0026'},
    ).add_to(peak_fg)
    peak_fg.add_to(m)

    # ── Cluster markers with popup ────────────────────────────────────────────
    cluster_fg = folium.FeatureGroup(name='Hotspot zones (click for details)', show=True)

    for _, row in cluster_profiles.iterrows():
        rank = int(row['priority_rank'])
        score = float(row['cis_score'])

        # Color by priority rank
        if rank <= 5:
            color = '#d73027'
            icon_color = 'red'
        elif rank <= 15:
            color = '#f46d43'
            icon_color = 'orange'
        elif rank <= 30:
            color = '#fdae61'
            icon_color = 'beige'
        else:
            color = '#74add1'
            icon_color = 'blue'

        popup_html = f"""
        <div style="font-family:Arial,sans-serif;font-size:13px;width:260px">
          <b style="color:{color};font-size:15px">Zone #{rank} — CIS: {score:.1f}/100</b><hr>
          <table style="width:100%;border-collapse:collapse">
            <tr><td><b>Violations</b></td><td>{int(row['violation_count']):,}</td></tr>
            <tr><td><b>Top violation</b></td><td>{row['top_violation']}</td></tr>
            <tr><td><b>Top vehicle</b></td><td>{row['top_vehicle']}</td></tr>
            <tr><td><b>Junction</b></td><td>{row['top_junction']}</td></tr>
            <tr><td><b>Police station</b></td><td>{row['top_station']}</td></tr>
            <tr><td><b>Peak hour %</b></td><td>{row['peak_ratio']*100:.0f}%</td></tr>
            <tr><td><b>Heavy vehicle %</b></td><td>{row['heavy_ratio']*100:.0f}%</td></tr>
            <tr><td><b>Avg response lag</b></td><td>{row['avg_response_lag']:.1f} hrs</td></tr>
            <tr><td><b>Recommended shift</b></td><td>{row['recommended_shift']}</td></tr>
          </table>
        </div>
        """

        folium.CircleMarker(
            location=[row['centroid_lat'], row['centroid_lon']],
            radius=max(8, min(30, score / 5)),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.6,
            weight=2,
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"Zone #{rank} | CIS: {score:.1f} | {row['top_station']}",
        ).add_to(cluster_fg)

    cluster_fg.add_to(m)

    # ── Top 10 priority zones with numbered markers ───────────────────────────
    top10_fg = folium.FeatureGroup(name='Top 10 enforcement priorities', show=True)
    top10 = cluster_profiles.head(10)
    for i, row in top10.iterrows():
        folium.Marker(
            location=[row['centroid_lat'], row['centroid_lon']],
            icon=folium.DivIcon(
                html=f'<div style="background:#d73027;color:white;border-radius:50%;'
                     f'width:28px;height:28px;display:flex;align-items:center;'
                     f'justify-content:center;font-weight:bold;font-size:13px;'
                     f'border:2px solid white;box-shadow:0 2px 4px rgba(0,0,0,0.4)">'
                     f'{i+1}</div>',
                icon_size=(28, 28),
                icon_anchor=(14, 14),
            ),
            tooltip=f"#{i+1} Priority | CIS: {row['cis_score']:.1f} | {row['top_station']}",
        ).add_to(top10_fg)
    top10_fg.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Title overlay
    title_html = """
    <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:9999;
         background:white;padding:10px 20px;border-radius:8px;
         box-shadow:0 2px 8px rgba(0,0,0,0.2);font-family:Arial,sans-serif">
      <b style="font-size:15px">Bengaluru Parking Violation Intelligence</b><br>
      <span style="font-size:12px;color:#666">AI-driven enforcement priority map</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    legend_html = """
    <div style="
    position: fixed;
    bottom: 50px;
    right: 50px;
    z-index: 9999;
    background: white;
    padding: 12px;
    border-radius: 8px;
    box-shadow: 0 0 10px rgba(0,0,0,0.3);
    font-size: 13px;
    width: 220px;
    ">

    <b>Map Legend</b>
    <hr style="margin:5px 0">

    <div>
    <span style="display:inline-block;width:15px;height:15px;
    background:#d73027;border-radius:50%;"></span>
    Top Priority Zone
    </div>

    <div style="margin-top:5px;">
    <span style="display:inline-block;width:15px;height:15px;
    background:#f46d43;border-radius:50%;"></span>
    High Priority Zone
    </div>

    <div style="margin-top:5px;">
    <span style="display:inline-block;width:15px;height:15px;
    background:#74add1;border-radius:50%;"></span>
    Hotspot Cluster
    </div>

    <div style="margin-top:5px;">
    🔢 Numbered Markers = Top 10 Enforcement Zones
    </div>

    <div style="margin-top:8px;">
    🔥 Heatmap = Violation Density
    </div>

    </div>
    """

    m.get_root().html.add_child(
        folium.Element(legend_html)
    )

    m.save(output_path)
    print(f"  Map saved to {output_path}")
    return m


def make_time_heatmap(approved_df: pd.DataFrame,
                      output_path: str = 'outputs/time_heatmap.html') -> folium.Map:
    """
    Build a HeatMapWithTime showing how violations shift across 24 hours.
    """
    center_lat = approved_df['latitude'].mean()
    center_lon = approved_df['longitude'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12,
                   tiles='CartoDB positron')

    hour_data = []
    time_labels = []
    for hour in range(24):
        h_df = approved_df[approved_df['hour'] == hour]
        hour_data.append(h_df[['latitude', 'longitude']].values.tolist())
        time_labels.append(f"{hour:02d}:00")

    # HeatMapWithTime(
    #     data=hour_data,
    #     index=time_labels,
    #     radius=12,
    #     blur=15,
    #     min_opacity=0.6,
    #     gradient={0.2: '#4575b4', 0.5: '#fee090', 1.0: '#d73027'},
    #     auto_play=True,
    #     display_index=True,
    # ).add_to(m)
    print(
        "Total points:",
        sum(len(x) for x in hour_data)
        )
    HeatMapWithTime(
        hour_data,
        index=time_labels,
        radius=25,
        auto_play=False,
        max_opacity=0.8,
    ).add_to(m)
    print("\nChecking hourly frames")

    for hour in range(24):
        print(
            hour,
            len(hour_data[hour])
        )

    m.save(output_path)
    print(f"  Time heatmap saved to {output_path}")
    return m
