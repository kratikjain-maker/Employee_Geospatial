# ============================================================================
# 🗺️ STREAMLIT EMPLOYEE GEOSPATIAL ANALYTICS DASHBOARD
# Run: streamlit run app.py
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN, KMeans
import json
from datetime import datetime

# ============================================================================
# PAGE CONFIG (Must be first Streamlit command)
# ============================================================================
st.set_page_config(
    page_title="Employee Geo Analytics",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e88e5 0%, #1565c0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }
</style>
""",
    unsafe_allow_html=True
)

# ============================================================================
# CONSTANTS
# ============================================================================
OFFICE_LAT, OFFICE_LON = 26.871785965027584, 75.77585852466206

# ============================================================================
# HELPERS
# ============================================================================


def compute_summary_stats(df: pd.DataFrame) -> dict:
    """Precompute all frequently used distance stats to avoid redundancy."""
    if df.empty:
        return {
            "total": 0,
            "avg_dist": 0.0,
            "max_dist": 0.0,
            "within_5km": 0,
            "within_10km": 0,
            "beyond_15km": 0,
            "within_10_pct": 0.0,
        }

    distances = df["Distance_km"]
    total = len(df)

    within_5km = (distances <= 5).sum()
    within_10km = (distances <= 10).sum()
    beyond_15km = (distances > 15).sum()

    return {
        "total": total,
        "avg_dist": float(distances.mean()),
        "max_dist": float(distances.max()),
        "within_5km": int(within_5km),
        "within_10km": int(within_10km),
        "beyond_15km": int(beyond_15km),
        "within_10_pct": float(within_10km / total * 100) if total > 0 else 0.0,
    }


def get_cluster_count(df: pd.DataFrame) -> int:
    """Count DBSCAN clusters, ignoring noise (-1)."""
    if df.empty or "Cluster" not in df.columns:
        return 0
    labels = set(df["Cluster"].values)
    return len(labels) - (1 if -1 in labels else 0)


# ============================================================================
# DATA LOADING & PROCESSING
# ============================================================================

@st.cache_data
def load_and_process_data(uploaded_file):
    """Load and process employee data with caching."""
    df = pd.read_csv(uploaded_file)

    def parse_coordinates(coord_str):
        try:
            parts = str(coord_str).split(",")
            lat, lon = float(parts[0].strip()), float(parts[1].strip())
            if 26.5 <= lat <= 27.2 and 75.5 <= lon <= 76.0:
                return lat, lon
            return None, None
        except Exception:
            return None, None

    df["Lat"], df["Lon"] = zip(*df["Coordinates"].apply(parse_coordinates))
    df_valid = df.dropna(subset=["Lat", "Lon"]).copy()

    # If no valid coordinates, return early with empty processed columns
    if df_valid.empty:
        return df_valid.assign(
            Distance_km=pd.Series(dtype=float),
            Direction=pd.Series(dtype=str),
            Cluster=pd.Series(dtype=int),
            Zone=pd.Series(dtype=int),
        )

    # Calculate distances
    df_valid["Distance_km"] = df_valid.apply(
        lambda row: geodesic((OFFICE_LAT, OFFICE_LON), (row["Lat"], row["Lon"])).km,
        axis=1,
    )

    # Directional analysis
    def get_direction(lat, lon):
        dlat = lat - OFFICE_LAT
        dlon = lon - OFFICE_LON
        angle = np.degrees(np.arctan2(dlon, dlat))

        if -22.5 <= angle < 22.5:
            return "N"
        elif 22.5 <= angle < 67.5:
            return "NE"
        elif 67.5 <= angle < 112.5:
            return "E"
        elif 112.5 <= angle < 157.5:
            return "SE"
        elif angle >= 157.5 or angle < -157.5:
            return "S"
        elif -157.5 <= angle < -112.5:
            return "SW"
        elif -112.5 <= angle < -67.5:
            return "W"
        else:
            return "NW"

    df_valid["Direction"] = df_valid.apply(
        lambda r: get_direction(r["Lat"], r["Lon"]), axis=1
    )

    # Clustering
    coords = df_valid[["Lat", "Lon"]].values

    # DBSCAN clusters
    dbscan = DBSCAN(eps=0.01, min_samples=5)
    dbscan_labels = dbscan.fit_predict(coords)
    df_valid["Cluster"] = dbscan_labels

    # KMeans zones (avoid n_clusters > n_samples or 0)
    n_samples = len(df_valid)
    n_clusters = min(8, n_samples) if n_samples > 0 else 0
    if n_clusters > 0:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        df_valid["Zone"] = kmeans.fit_predict(coords)
    else:
        df_valid["Zone"] = -1

    return df_valid


@st.cache_data
def get_region_stats(df_valid):
    """Calculate region statistics."""
    if df_valid.empty:
        return pd.DataFrame(
            columns=["Region", "Count", "Lat", "Lon", "Avg_Dist", "Depts"]
        )

    region_stats = (
        df_valid.groupby("Region")
        .agg(
            {
                "Name": "count",
                "Lat": "mean",
                "Lon": "mean",
                "Distance_km": "mean",
                "Department": lambda x: x.value_counts().to_dict(),
            }
        )
        .reset_index()
    )
    region_stats.columns = ["Region", "Count", "Lat", "Lon", "Avg_Dist", "Depts"]
    return region_stats.sort_values("Count", ascending=False)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown(
        '<h1 class="main-header">🗺️ Employee Geospatial Analytics Dashboard</h1>',
        unsafe_allow_html=True,
    )
    st.markdown("**AI-Powered Location Intelligence for Workforce Optimization**")
    st.divider()

    # Sidebar: file upload
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/000000/google-maps-new.png", width=80
        )
        st.title("📂 Data Upload")

        uploaded_file = st.file_uploader(
            "Upload Employee CSV",
            type=["csv"],
            help="Upload your Region_based-Master_Data.csv file",
        )

        if uploaded_file is None:
            st.info("👆 Upload your CSV file to begin")
            st.stop()
        else:
            st.success("✅ File loaded successfully!")

    # Load data
    df_valid = load_and_process_data(uploaded_file)
    region_stats = get_region_stats(df_valid)

    # Sidebar filters (now that we have data)
    with st.sidebar:
        st.divider()
        st.subheader("🎛️ Filters")

        max_distance_slider = int(df_valid["Distance_km"].max()) + 1 if not df_valid.empty else 1
        max_distance = st.slider(
            "Max Distance (km)",
            min_value=0,
            max_value=max_distance_slider,
            value=max_distance_slider,
            help="Filter employees by distance from office",
        )

        departments = ["All"] + sorted(df_valid["Department"].dropna().unique().tolist())
        selected_dept = st.selectbox("Department", departments)

        grades = ["All"] + sorted(df_valid["Grade"].dropna().unique().tolist())
        selected_grade = st.selectbox("Grade", grades)

        directions = ["All"] + sorted(df_valid["Direction"].dropna().unique().tolist())
        selected_direction = st.selectbox("Direction", directions)

    # Apply filters
    filtered_df = df_valid[df_valid["Distance_km"] <= max_distance].copy()
    if selected_dept != "All":
        filtered_df = filtered_df[filtered_df["Department"] == selected_dept]
    if selected_grade != "All":
        filtered_df = filtered_df[filtered_df["Grade"] == selected_grade]
    if selected_direction != "All":
        filtered_df = filtered_df[filtered_df["Direction"] == selected_direction]

    # If no data after filters, stop early
    if filtered_df.empty:
        st.warning("No employees match the selected filters.")
        st.stop()

    # Precompute stats once, reuse everywhere
    summary = compute_summary_stats(filtered_df)
    cluster_count = get_cluster_count(filtered_df)

    # =========================================================================
    # KPI METRICS
    # =========================================================================
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("👥 Total Employees", f"{summary['total']:,}")

    with col2:
        st.metric("📍 Unique Regions", f"{filtered_df['Region'].nunique()}")

    with col3:
        st.metric("📏 Avg Distance", f"{summary['avg_dist']:.2f} km")

    with col4:
        st.metric(
            "Within 10km",
            f"{summary['within_10km']:,}",
            f"{summary['within_10_pct']:.1f}%",
        )

    with col5:
        st.metric("🎯 Clusters Found", f"{cluster_count}")

    st.divider()

    # =========================================================================
    # MAIN TABS
    # =========================================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "🗺️ Interactive Map",
            "📊 Analytics Dashboard",
            "🎯 3D Visualization",
            "📈 Insights & Reports",
            "💾 Export Data",
        ]
    )

    # =========================================================================
    # TAB 1: INTERACTIVE MAP
    # =========================================================================
    with tab1:
        st.subheader("🌍 Employee Distribution Map")

        col1_map, col2_map = st.columns([3, 1])

        with col2_map:
            st.markdown("### Map Controls")
            map_style = st.selectbox(
                "Map Style",
                ["open-street-map", "carto-positron", "carto-darkmatter", "stamen-terrain"],
                index=0,
            )

            show_office = st.checkbox("Show Office Location", value=True)
            point_size = st.slider("Point Size", 5, 20, 10)

        with col1_map:
            fig_map = px.scatter_mapbox(
                filtered_df,
                lat="Lat",
                lon="Lon",
                color="Distance_km",
                size="Distance_km",
                hover_name="Name",
                hover_data={
                    "Department": True,
                    "Grade": True,
                    "Region": True,
                    "Distance_km": ":.2f",
                    "Direction": True,
                    "Lat": False,
                    "Lon": False,
                },
                color_continuous_scale="Turbo",
                size_max=point_size,
                zoom=11,
                height=600,
                mapbox_style=map_style,
            )

            if show_office:
                fig_map.add_trace(
                    go.Scattermapbox(
                        lat=[OFFICE_LAT],
                        lon=[OFFICE_LON],
                        mode="markers+text",
                        marker=dict(size=20, color="red", symbol="star"),
                        text=["🏢 Office"],
                        textposition="top center",
                        name="Office",
                        hovertemplate="<b>Main Office</b><br>Jaipur HQ<extra></extra>",
                    )
                )

            fig_map.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                mapbox_center={"lat": OFFICE_LAT, "lon": OFFICE_LON},
                showlegend=False,
            )

            st.plotly_chart(fig_map, use_container_width=True)

        st.subheader("📍 Regional Distribution")
        region_stats_filtered = region_stats[
            region_stats["Region"].isin(filtered_df["Region"].unique())
        ]

        if not region_stats_filtered.empty:
            fig_regions = px.scatter_mapbox(
                region_stats_filtered.head(20),
                lat="Lat",
                lon="Lon",
                size="Count",
                color="Avg_Dist",
                hover_name="Region",
                hover_data={"Count": True, "Avg_Dist": ":.2f", "Lat": False, "Lon": False},
                color_continuous_scale="Reds",
                size_max=40,
                zoom=11,
                height=500,
                mapbox_style=map_style,
                title="Top 20 Regions by Employee Count",
            )

            fig_regions.update_layout(
                mapbox_center={"lat": OFFICE_LAT, "lon": OFFICE_LON}
            )
            st.plotly_chart(fig_regions, use_container_width=True)
        else:
            st.info("No regional statistics available for the current filters.")

    # =========================================================================
    # TAB 2: ANALYTICS DASHBOARD
    # =========================================================================
    with tab2:
        st.subheader("📊 Comprehensive Analytics")

        fig_dash = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "📊 Distance Distribution",
                "🧭 Directional Breakdown",
                "🏢 Top 10 Departments",
                "⭐ Grade Distribution",
                "📈 Cumulative Distance",
                "🎯 Top 15 Regions",
            ),
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "bar"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15,
        )

        # 1. Distance histogram
        dist_bins = pd.cut(
            filtered_df["Distance_km"], bins=[0, 5, 10, 15, 20, 100], include_lowest=True
        )
        dist_counts = dist_bins.value_counts().sort_index()
        fig_dash.add_trace(
            go.Bar(
                x=[str(x) for x in dist_counts.index],
                y=dist_counts.values,
                marker_color="#1e88e5",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # 2. Directional pie
        dir_counts = filtered_df["Direction"].value_counts()
        fig_dash.add_trace(
            go.Pie(
                labels=dir_counts.index,
                values=dir_counts.values,
                hole=0.4,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        # 3. Department bar
        dept_counts = filtered_df["Department"].value_counts().head(10)
        fig_dash.add_trace(
            go.Bar(
                y=dept_counts.index,
                x=dept_counts.values,
                orientation="h",
                marker_color="#43a047",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        # 4. Grade pie
        grade_counts = filtered_df["Grade"].value_counts()
        fig_dash.add_trace(
            go.Pie(
                labels=grade_counts.index,
                values=grade_counts.values,
                hole=0.3,
                showlegend=False,
            ),
            row=2,
            col=2,
        )

        # 5. Cumulative distance
        sorted_dist = np.sort(filtered_df["Distance_km"].values)
        cumulative = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist) * 100
        fig_dash.add_trace(
            go.Scatter(
                x=sorted_dist,
                y=cumulative,
                mode="lines",
                line=dict(color="#e53935", width=3),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        # 6. Top regions
        top_regions = filtered_df["Region"].value_counts().head(15)
        fig_dash.add_trace(
            go.Bar(
                y=top_regions.index,
                x=top_regions.values,
                orientation="h",
                marker_color="#fb8c00",
                showlegend=False,
            ),
            row=3,
            col=2,
        )

        fig_dash.update_layout(height=1200, showlegend=False)
        fig_dash.update_xaxes(title_text="Distance Range (km)", row=1, col=1)
        fig_dash.update_xaxes(title_text="Employee Count", row=2, col=1)
        fig_dash.update_xaxes(title_text="Distance (km)", row=3, col=1)
        fig_dash.update_yaxes(title_text="Cumulative %", row=3, col=1)
        fig_dash.update_xaxes(title_text="Employee Count", row=3, col=2)

        st.plotly_chart(fig_dash, use_container_width=True)

    # =========================================================================
    # TAB 3: 3D VISUALIZATION
    # =========================================================================
    with tab3:
        st.subheader("🎯 3D Hexagon Density Map")
        st.markdown(
            "*GPU-accelerated visualization showing employee concentration as building heights*"
        )

        hex_layer = pdk.Layer(
            "HexagonLayer",
            data=filtered_df[["Lon", "Lat"]].rename(columns={"Lon": "lng", "Lat": "lat"}),
            get_position=["lng", "lat"],
            auto_highlight=True,
            elevation_scale=50,
            elevation_range=[0, 1000],
            extruded=True,
            coverage=0.95,
            radius=500,
            pickable=True,
        )

        office_layer = pdk.Layer(
            "ScatterplotLayer",
            data=pd.DataFrame({"lon": [OFFICE_LON], "lat": [OFFICE_LAT], "size": [500]}),
            get_position=["lon", "lat"],
            get_radius="size",
            get_fill_color=[255, 0, 0, 200],
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=OFFICE_LAT,
            longitude=OFFICE_LON,
            zoom=11.5,
            pitch=45,
            bearing=0,
        )

        deck = pdk.Deck(
            layers=[hex_layer, office_layer],
            initial_view_state=view_state,
            tooltip={"text": "🏢 Density: {elevationValue}"},
            map_style="mapbox://styles/mapbox/dark-v10",  # set MAPBOX_API_KEY in Streamlit Cloud
        )

        st.pydeck_chart(deck)
        st.info(
            "💡 **Tip:** Use mouse to rotate and tilt the 3D view. Taller hexagons indicate higher employee density."
        )

    # =========================================================================
    # TAB 4: INSIGHTS & REPORTS
    # =========================================================================
    with tab4:
        st.subheader("📈 Business Insights & Recommendations")

        col1_ins, col2_ins = st.columns(2)

        with col1_ins:
            st.markdown("### 🎯 Key Findings")

            within_5km = summary["within_5km"]
            within_10km = summary["within_10km"]
            beyond_15km = summary["beyond_15km"]
            total = summary["total"]

            st.markdown(
                f"""
1. **Distance Analysis:**
   - Average commute: **{summary['avg_dist']:.2f} km**
   - Maximum distance: **{summary['max_dist']:.2f} km**
   - Within 5km: **{within_5km}** ({within_5km/total*100:.1f}%)
   - Within 10km: **{within_10km}** ({within_10km/total*100:.1f}%)
   - Beyond 15km: **{beyond_15km}** ({beyond_15km/total*100:.1f}%)

2. **Geographic Spread:**
   - Employees concentrated in **{filtered_df['Direction'].value_counts().index[0]}** direction
   - Top region: **{filtered_df['Region'].value_counts().index[0]}** ({filtered_df['Region'].value_counts().values[0]} employees)

3. **Natural Clusters:**
   - Identified **{cluster_count}** distinct employee clusters
   - Potential shuttle pickup zones identified
"""
            )

        with col2_ins:
            st.markdown("### 💡 Recommendations")

            recommendations = []

            if beyond_15km > 0:
                recommendations.append(
                    f"🚌 **Shuttle Service**: {beyond_15km} employees ({beyond_15km/total*100:.1f}%) live beyond 15km. "
                    f"Consider implementing shuttle routes for these areas."
                )

            region_counts = filtered_df["Region"].value_counts()
            if not region_counts.empty:
                top_region_name = region_counts.index[0]
                top_region_count = region_counts.values[0]
                if top_region_count > 30:
                    recommendations.append(
                        f"🏢 **Satellite Office**: {top_region_name} has {top_region_count} employees. "
                        f"Evaluate feasibility of a satellite office or co-working space partnership."
                    )

            remote_count = (filtered_df["Distance_km"] > 20).sum()
            if remote_count > 10:
                recommendations.append(
                    f"🏠 **Remote Work Policy**: {remote_count} employees live 20km+ away. "
                    f"Consider flexible remote work options to improve work-life balance."
                )

            if cluster_count >= 3:
                recommendations.append(
                    f"🎯 **Housing Partnerships**: {cluster_count} natural clusters identified. "
                    f"Partner with apartment complexes in these zones for employee housing benefits."
                )

            main_direction = filtered_df["Direction"].value_counts().index[0]
            recommendations.append(
                f"📍 **Recruitment Focus**: Most employees are in the {main_direction} direction. "
                f"Focus recruitment efforts and job postings in these neighborhoods."
            )

            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")

        st.divider()

        col1_tbl, col2_tbl = st.columns(2)

        with col1_tbl:
            st.markdown("### 📊 Top 10 Regions")
            top_regions_df = (
                filtered_df["Region"].value_counts().head(10).reset_index()
            )
            top_regions_df.columns = ["Region", "Employee Count"]
            st.dataframe(top_regions_df, use_container_width=True, hide_index=True)

        with col2_tbl:
            st.markdown("### 🧭 Directional Distribution")
            dir_df = filtered_df["Direction"].value_counts().reset_index()
            dir_df.columns = ["Direction", "Employee Count"]
            dir_df["Percentage"] = (
                dir_df["Employee Count"] / total * 100
            ).round(1)
            st.dataframe(dir_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 5: EXPORT DATA
    # =========================================================================
    with tab5:
        st.subheader("💾 Export Analytics & Reports")

        col1_exp, col2_exp, col3_exp = st.columns(3)

        with col1_exp:
            st.markdown("### 📋 Enhanced Employee Data")
            csv_employees = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Employee CSV",
                data=csv_employees,
                file_name=f"employees_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
            st.caption(
                f"Includes {len(filtered_df)} employees with distance, direction, and cluster data"
            )

        with col2_exp:
            st.markdown("### 🌍 Regional Statistics")
            region_export = region_stats[
                region_stats["Region"].isin(filtered_df["Region"].unique())
            ].copy()
            region_export = region_export[["Region", "Count", "Lat", "Lon", "Avg_Dist"]]
            csv_regions = region_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Region CSV",
                data=csv_regions,
                file_name=f"region_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
            st.caption("Regional aggregations and statistics")

        with col3_exp:
            st.markdown("### 📊 JSON Report")
            report = {
                "timestamp": datetime.now().isoformat(),
                "filters_applied": {
                    "max_distance": max_distance,
                    "department": selected_dept,
                    "grade": selected_grade,
                    "direction": selected_direction,
                },
                "summary": {
                    "total_employees": summary["total"],
                    "unique_regions": int(filtered_df["Region"].nunique()),
                    "avg_distance_km": summary["avg_dist"],
                    "max_distance_km": summary["max_dist"],
                    "within_10km": summary["within_10km"],
                    "within_10km_pct": summary["within_10_pct"],
                },
                "top_regions": region_export.head(10).to_dict("records")
                if not region_export.empty
                else [],
                "directional_distribution": filtered_df["Direction"]
                .value_counts()
                .to_dict(),
            }
            json_report = json.dumps(report, indent=2).encode("utf-8")
            st.download_button(
                label="📥 Download JSON Report",
                data=json_report,
                file_name=f"analytics_report_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
            )
            st.caption("Complete analytics in JSON format")

        st.divider()
        st.markdown("### 📸 Export Visualizations")
        st.info(
            "💡 **Tip:** You can download any chart as PNG by hovering over it and clicking the camera icon in the top-right corner."
        )


# ============================================================================
# RUN APP
# ============================================================================
if __name__ == "__main__":
    main()
