import streamlit as st
import pandas as pd
import base64
import requests
import os
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import MinMaxScaler
import io

API_Url = os.getenv("BACKEND_URL", "http://localhost:8000/predict")

def get_base64(bin_file):
    with open(bin_file,'rb') as f:
        data=f.read()
    return base64.b64encode(data).decode()

bg_file='assets/bg.png'
bg_base64=get_base64(bg_file)


st.markdown(f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{bg_base64}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    background-position: center;
}}

section.main > div {{
    background-color: rgba(255,255,255,0);
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
    right: 2rem;
}}
.stFileUploader {{
        border: 2px solid #000000;
        border-radius: 8px;
        padding: 0.6em;
        background-color: #E2EFDA;
}}
.stFileUploader label p {{
        color: #000000 !important;
        font-weight: bold;
}}
.stFileUploader section {{
        background-color: #E2EFDA !important;
}}
.stButton > button {{
        border: 2px solid #262730 !important;
        color: #FFFFF0 !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        background-color: #006400 !important;  
}}
.stButton > button:hover {{
        border-color: #000000 !important;
        color: #000000 !important;             
        background-color: #E2EFDA !important;  
        transform: scale(1.1);
}}
.stMetric label p {{
        font-weight: bold !important;
        font-size: 1.2rem !important;
}}  
div[data-testid="stTextInput"]:nth-of-type(1) label p,
div[data-testid="stTextInput"]:nth-of-type(2) label p {{
    font-size: 1.5rem !important;
    font-weight: bold !important;
}}
</style>
""", unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.markdown("<h1 style='text-align: center; color: #002D04; font-weight: bold;'>Login</h1>", unsafe_allow_html=True)
    email = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if email == "abc@gmail.com" and password == "12345@123":
            st.session_state.logged_in = True
            st.success("Login successful! Redirecting...")
            st.rerun()
        else:
            st.error("Invalid credentials")
def hydrology_engine():

    st.markdown("<h1 style='text-align:center;font-weight:bold;font-style:italic;color:#002D04;'>Urban Flooding & Hydrology Engine</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-style:italic; font-size:27px; font-weight:bold; color:black;'>""AI-based Hydrology Engine for Pre-Monsoon Readiness</p>",unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])
    if uploaded_file:
        st.session_state["csv_file"] = uploaded_file
        st.success("CSV uploaded successfully")

    def generate_ward_rankings(final_map):
        final_rankings = final_map[['WardName','readiness_score']].drop_duplicates().sort_values(by='readiness_score')
        q_steps = [round(i/7,2) for i in range(1,7)]
        quantiles = final_rankings['readiness_score'].quantile(q_steps).to_dict()
        def get_7_tier_priority(score):
            if score <= quantiles[q_steps[0]]: return "🔴 TIER 1: EMERGENCY"
            if score <= quantiles[q_steps[1]]: return "🔴 TIER 2: SEVERE"
            if score <= quantiles[q_steps[2]]: return "🟠 TIER 3: CRITICAL"
            if score <= quantiles[q_steps[3]]: return "🟠 TIER 4: VULNERABLE"
            if score <= quantiles[q_steps[4]]: return "🟡 TIER 5: MONITORING"
            if score <= quantiles[q_steps[5]]: return "🟢 TIER 6: PREPARED"
            return "🟢 TIER 7: OPTIMAL"
        final_rankings['Status'] = final_rankings['readiness_score'].apply(get_7_tier_priority)
        return final_rankings

    @st.cache_data
    def load_wards():
        return gpd.read_file("assets/delhi_wards.kml")
    wards = load_wards()

    if st.button("Run Flood Risk Analysis"):

        if "csv_file" not in st.session_state:
            st.warning("Please upload a CSV first")
        else:
            with st.spinner("Running hydrology model..."):
                files = {"file": st.session_state["csv_file"]}
                response = requests.post(API_Url, files=files)
                if response.status_code != 200:
                    st.error("API request failed")
                else:
                    csv_data = io.StringIO(response.content.decode("utf-8"))
                    st.success("Prediction completed")
                    df = pd.read_csv(csv_data)
                    st.info(f"Dataset loaded with {len(df)} rows")
                    c4 = df[df["cluster"] == 4]
                    hotspots_4 = c4.sample(n=min(4000, len(c4)), weights="inundation_weight", random_state=42)
                    c2 = df[df["cluster"] == 2]
                    hotspots_2 = c2.sample(n=min(3500, len(c2)), weights=(1.1 - c2["elev_norm"]), random_state=42)
                    c1 = df[df["cluster"] == 1]
                    hotspots_1 = c1[c1["rain_norm"] >= c1["rain_norm"].quantile(0.90)].sample(n=min(850, len(c1)), random_state=42)
                    c3 = df[df["cluster"] == 3]
                    c3_filtered = c3[(c3["rain_norm"] >= c3["rain_norm"].quantile(0.80)) &(c3["elev_norm"] <= c3["elev_norm"].quantile(0.40))]
                    hotspots_3 = c3_filtered.sample(n=min(500, len(c3)), random_state=42)
                    hotspots = pd.concat([hotspots_4,hotspots_2,hotspots_1,hotspots_3])
                    st.success(f"{len(hotspots)} Flood Micro-Hotspots Identified")
                    st.subheader("Flood Micro-Hotspot Map")
                    fig, ax = plt.subplots(figsize=(14,14))
                    ax.scatter(
                        hotspots["lon"],
                        hotspots["lat"],
                        s=20,
                        c="red",
                        alpha=0.3,
                        edgecolors="none"
                    )
                    ctx.add_basemap(
                        ax,
                        crs="EPSG:4326",
                        source=ctx.providers.OpenStreetMap.Mapnik,
                        zoom=12
                    )
                    ax.set_title("Delhi Flood Risk: OpenStreetMap Visualization (Hydrology Engine)",fontsize=16)
                    ax.axis("off")
                    st.pyplot(fig)
                    st.subheader("Ward-Level Flood Readiness Analysis")
                    geometry = [Point(xy) for xy in zip(hotspots["lon"], hotspots["lat"])]
                    gdf_hotspots = gpd.GeoDataFrame(hotspots,geometry=geometry,crs="EPSG:4326")


                    joined = gpd.sjoin(gdf_hotspots,wards,how="inner",predicate="intersects")
                    exposure = joined.groupby("WardName").size().rename("hotspot_count")   
                    severity = joined.groupby("WardName")["inundation_weight"].quantile(0.85).rename("peak_severity")
                    wards_metrics = wards.copy()
                    wards_metrics["area"] = wards_metrics.geometry.area
                    wards_metrics = wards_metrics.merge(exposure, on="WardName", how="left")
                    wards_metrics = wards_metrics.merge(severity, on="WardName", how="left").fillna(0)
                    wards_metrics["hotspot_density"] = wards_metrics["hotspot_count"] / wards_metrics["area"].replace(0,1)

                    
                    mscaler = MinMaxScaler()
                    cols = ["hotspot_count", "hotspot_density", "peak_severity"]
                    wards_metrics[[f"{c}_n" for c in cols]] = mscaler.fit_transform(wards_metrics[cols])
                    
                    wards_metrics["base_readiness"] = 1 - (
                        0.15 * wards_metrics["hotspot_count_n"] +
                        0.50 * wards_metrics["hotspot_density_n"] +
                        0.35 * wards_metrics["peak_severity_n"]
                    )

                    river_wards = [
                        'CIVIL LINES','SONIA VIHAR','YAMUNA VIHAR','JAMA MASJID',
                        'DARYAGANJ','NEW ASHOK NAGAR','MAYUR VIHAR PHASE-I','BADARPUR'
                    ]
                    wards_metrics["readiness_score"] = wards_metrics.apply(
                        lambda x: x["base_readiness"] - (0.30 + 0.10 * x["peak_severity_n"])
                        if x["WardName"] in river_wards else x["base_readiness"],
                        axis=1
                    )
                    wards_metrics["readiness_score"] = wards_metrics["readiness_score"].clip(0.05, 1.0)

                
                    final_map = wards.merge(wards_metrics[['WardName','readiness_score']],on="WardName",how="left")
                    final_map["readiness_score"] = final_map["readiness_score"].fillna(1)
                    fig2, ax2 = plt.subplots(figsize=(12,12))
                    final_map.plot(
                        column="readiness_score",
                        cmap="RdYlGn",
                        legend=True,
                        scheme="Quantiles",
                        k=7,
                        edgecolor="black",
                        linewidth=0.3,
                        ax=ax2,
                        legend_kwds={
                            'loc': 'upper left',
                            'title': "Readiness Score\n(Green=Ready, Red=Critical)"
                        }
                    )
                    ax2.set_title(
                        "Delhi Ward-Level Pre-Monsoon Readiness Map",
                        fontsize=18
                    )
                    ax2.axis("off")
                    st.pyplot(fig2)
                    st.subheader("Ward Readiness Rankings")

                    final_map = final_map[final_map["WardName"].notna()]
                    final_map = final_map[final_map["WardName"] != ""]

                    rankings = generate_ward_rankings(final_map)
                    st.metric("Total Wards Analysed", len(rankings))
                    st.metric("Critical Wards (Tier 1-2)", len(rankings[rankings["Status"].str.contains("TIER 1|TIER 2")]))
                    st.dataframe(
                        rankings.style.format({"readiness_score":"{:.4f}"}),
                        use_container_width=True
                    )         
if not st.session_state.logged_in:
    login()
else:
    hydrology_engine()
