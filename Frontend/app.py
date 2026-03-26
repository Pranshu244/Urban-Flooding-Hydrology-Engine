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
from datetime import date

API_Url = os.getenv("BACKEND_URL", "http://backend:8000/predict")

st.set_page_config(
    page_title="Urban Flooding & Hydrology Engine",
    page_icon="🌊",
)

#Helpers
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def days_to_monsoon():
    today = date.today()
    monsoon = date(today.year, 6, 15)
    if today > monsoon:
        monsoon = date(today.year + 1, 6, 15)
    return (monsoon - today).days

bg_file = 'assets/bg.png'
bg_base64 = get_base64(bg_file)

#CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=DM+Sans:wght@400;500&display=swap');

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

/* ── Navbar ── */
.navbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(0,60,10,0.88);
    border-radius: 10px;
    padding: 10px 24px;
    margin-bottom: 20px;
    border: 1px solid rgba(100,200,100,0.20);
}}
.navbar-brand {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #90EE90;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    text-align: center;
    line-height: 1.2;
}}
.navbar-links {{
    display: flex;
    gap: 28px;
    align-items: center;
}}
.navbar-links a {{
    color: rgba(200,255,200,0.80);
    text-decoration: none;
    font-size: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    transition: color 0.2s;
}}
.navbar-links a:hover {{
    color: #90EE90;
}}
.navbar-badge {{
    background: #006400;
    color: #90EE90;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid rgba(100,200,100,0.35);
    letter-spacing: 0.05em;
}}

/* ── About section ── */
.about-section {{
    background: rgba(226,239,218,0.88);
    border: 1px solid rgba(0,100,0,0.20);
    border-radius: 10px;
    padding: 22px 28px;
    margin-bottom: 16px;
}}
.about-section h3 {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #002D04;
    margin-bottom: 10px;
    letter-spacing: 0.04em;
}}
.about-section p {{
    font-size: 0.88rem;
    color: #003300;
    line-height: 1.7;
    margin-bottom: 10px;
}}
.about-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 14px;
}}
.about-card {{
    background: #006400;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}}
.about-card-icon {{
    font-size: 1.4rem;
    margin-bottom: 4px;
}}
.about-card-title {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: #90EE90;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.about-card-desc {{
    font-size: 0.75rem;
    color: #b8e8b8;
    margin-top: 2px;
    line-height: 1.4;
}}
.hover-card {{
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: pointer;
}}
.hover-card:hover {{
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0,80,0,0.30);
}}

/* ── Contact section ── */
.contact-section {{
    background: rgba(226,239,218,0.88);
    border: 1px solid rgba(0,100,0,0.20);
    border-radius: 10px;
    padding: 18px 28px;
    margin-bottom: 16px;
}}
.contact-section h3 {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #002D04;
    margin-bottom: 12px;
    letter-spacing: 0.04em;
}}
.contact-grid {{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
}}
.contact-card {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    background: rgba(0,80,0,0.08);
    border: 1px solid rgba(0,100,0,0.15);
    border-radius: 8px;
    padding: 10px 14px;
}}
.contact-icon {{
    font-size: 1.2rem;
    margin-top: 2px;
}}
.contact-label {{
    font-size: 0.72rem;
    color: #4a7a4a;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}}
.contact-value {{
    font-size: 0.84rem;
    color: #002D04;
    font-weight: 500;
    margin-top: 1px;
}}

/* ── File uploader ── */
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

/* ── Buttons ── */
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

/* ── Metric labels ── */
.stMetric label p {{
    font-weight: bold !important;
    font-size: 1.2rem !important;
}}

div[data-testid="stTextInput"]:nth-of-type(1) label p,
div[data-testid="stTextInput"]:nth-of-type(2) label p {{
    font-size: 1.5rem !important;
    font-weight: bold !important;
}}
div[data-testid="stTextInput"] div[data-baseweb="base-input"] {{
    background-color: #E2EFDA !important;
    border: 1px solid #006400 !important;
}}
div[data-testid="stTextInput"] input {{
    background-color: #E2EFDA !important;
    color: #002D04 !important;
}}
div[data-testid="stTextInput"] div[data-baseweb="base-input"] button {{
    background-color: #E2EFDA !important;
    border: none !important;
    color: #006400 !important;
}}
div[data-testid="stTextInput"] div[data-baseweb="base-input"] button:hover {{
    background-color: #c8e6c0 !important;
}}

/* ── Monsoon countdown banner ── */
.countdown-banner {{
    background: #006400;
    border: 2px solid #004d00;
    border-radius: 10px;
    padding: 14px 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
}}
.countdown-days {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    color: #90EE90;
    line-height: 1;
    display: inline-block;
    margin-right: 14px;
    vertical-align: middle;
}}
.countdown-text {{
    display: inline-block;
    vertical-align: middle;
}}
.countdown-text strong {{
    display: block;
    color: #ffffff;
    font-size: 0.95rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
.countdown-text span {{
    font-size: 0.78rem;
    color: #b8e8b8;
}}
.countdown-right {{
    font-size: 0.82rem;
    color: #90EE90;
    display: flex;
    align-items: center;
    gap: 6px;
}}
.pulse-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #90EE90;
    display: inline-block;
    animation: pulse 1.8s ease-in-out infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.3; }}
}}

/* ── Stat cards ── */
.stat-row {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 16px 0;
}}
.stat-card {{
    background: #006400;
    border: 2px solid #004d00;
    border-radius: 10px;
    padding: 16px 18px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    cursor: default;
}}
.stat-card:hover {{
    transform: scale(1.05);
    box-shadow: 0 6px 20px rgba(0,80,0,0.30);
}}
.stat-value {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #90EE90;
    line-height: 1;
}}
.stat-label {{
    font-size: 0.72rem;
    color: #d4f5d4;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 4px;
}}

/* ── Section header ── */
.section-hdr {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #002D04;
    border-left: 4px solid #006400;
    padding: 6px 10px;
    margin: 24px 0 12px;
    background: rgba(226,239,218,0.80);
    border-radius: 0 6px 6px 0;
}}

/* ── Ward table ── */
.ward-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #ccddcc;
}}
.ward-table thead tr {{
    background: #006400;
}}
.ward-table thead th {{
    padding: 10px 14px;
    text-align: left;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.80rem;
    font-weight: 600;
    color: #d4f5d4;
    text-transform: uppercase;
    letter-spacing: 0.07em;
}}
.ward-table tbody tr {{
    background: rgba(226,239,218,0.88);
    border-bottom: 1px solid rgba(0,80,0,0.10);
}}
.ward-table tbody tr:nth-child(even) {{
    background: rgba(210,230,200,0.88);
}}
.ward-table td {{
    padding: 8px 14px;
    color: #002D04;
}}

/* ── Score bar ── */
.score-bar-wrap {{
    display: flex;
    align-items: center;
    gap: 8px;
}}
.score-bar-bg {{
    flex: 1;
    height: 6px;
    background: rgba(0,80,0,0.15);
    border-radius: 4px;
    overflow: hidden;
}}
.score-bar-fill {{
    height: 100%;
    border-radius: 4px;
}}
.score-num {{
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    min-width: 38px;
    color: #002D04;
}}

/* ── Tier badges ── */
.tier-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.74rem;
    font-weight: 600;
    white-space: nowrap;
}}
.t1 {{ background: #FF4444; color: #fff; }}
.t2 {{ background: #FF7730; color: #fff; }}
.t3 {{ background: #FFB830; color: #002D04; }}
.t4 {{ background: #E8D020; color: #002D04; }}
.t5 {{ background: #90C030; color: #002D04; }}
.t6 {{ background: #30A060; color: #fff; }}
.t7 {{ background: #006400; color: #90EE90; }}

/* ── Action pill ── */
.action-pill {{
    font-size: 0.77rem;
    color: #003300;
    background: rgba(0,100,0,0.10);
    border: 1px solid rgba(0,100,0,0.25);
    border-radius: 12px;
    padding: 2px 8px;
}}

/* ── Intervention bullet ── */
.intervention-list {{
    list-style: none;
    padding: 0;
    margin: 0;
}}
.intervention-list li {{
    font-size: 0.80rem;
    color: #002D04;
    padding: 2px 0;
    line-height: 1.5;
}}
.intervention-list li::before {{
    content: "▸ ";
    color: #006400;
    font-weight: 700;
}}

/* ── Methodology note ── */
.method-note {{
    background: rgba(226,239,218,0.85);
    border-left: 3px solid #006400;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    font-size: 0.78rem;
    color: #003300;
    margin-top: 12px;
    line-height: 1.6;
}}
/* ── Footer ── */
.footer {{
    background: rgba(0,60,10,0.88);
    border-radius: 10px;
    padding: 16px 24px;
    margin-top: 32px;
    text-align: center;
    border: 1px solid rgba(100,200,100,0.15);
}}
.footer p {{
    font-size: 0.78rem;
    color: rgba(200,255,200,0.60);
    margin: 2px 0;
}}
.footer strong {{
    color: #90EE90;
}}
</style>
""", unsafe_allow_html=True)

#Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

#Tier config
TIER_CONFIG = {
    "TIER 1": ("t1", "#FF4444", "Emergency pump deployment + drain clearance"),
    "TIER 2": ("t2", "#FF7730", "Priority drain inspection + sandbag preposition"),
    "TIER 3": ("t3", "#FFB830", "Scheduled drain clearance within 72 hrs"),
    "TIER 4": ("t4", "#E8D020", "Vulnerability mapping + community alert"),
    "TIER 5": ("t5", "#90C030", "Routine monitoring, no immediate action"),
    "TIER 6": ("t6", "#30A060", "Infrastructure check before monsoon"),
    "TIER 7": ("t7", "#006400", "Maintain current readiness status"),
}

#Intervention framework
INTERVENTIONS = {
    "TIER 1": {
        "label": "TIER 1 — Emergency",
        "badge": "t1",
        "points": [
            "Deploy mobile dewatering pumps immediately at identified micro-hotspot locations",
            "Emergency desiltation of blocked storm water drains before first monsoon rainfall",
            "Pre-position sandbags and flood barriers at ward entry points and low-lying roads",
            "Activate NDRF / SDRF coordination for rapid response teams in the ward",
            "Issue early warning SMS alerts to all residents in flood-plain sub-areas",
            "Identify and map evacuation routes; set up temporary relief camps on high ground",
        ]
    },
    "TIER 2": {
        "label": "TIER 2 — Severe",
        "badge": "t2",
        "points": [
            "Priority inspection of all storm water drain outfalls within ward boundary",
            "Remove encroachments and debris blocking drain flow before monsoon onset",
            "Pre-position sandbags at historically waterlogged road intersections",
            "Repair or replace non-functional pump stations and sump pits",
            "Community awareness drive on waterlogging hotspots and self-evacuation protocol",
            "Coordinate with MCD for drain gradient correction in identified adverse-slope segments",
        ]
    },
    "TIER 3": {
        "label": "TIER 3 — Critical",
        "badge": "t3",
        "points": [
            "Schedule full drain clearance and desilting within 72 hours of this report",
            "Inspect and clear all road-side drain inlets and gullies of garbage and sediment",
            "Check and repair manholes — identify punctured or broken sewer-storm drain connections",
            "Implement rain water harvesting structures in public buildings to reduce runoff load",
            "Survey low-lying areas for potential water body rejuvenation as flood retention zones",
            "Coordinate with Delhi Jal Board for SPS (Sewage Pumping Station) capacity check",
        ]
    },
    "TIER 4": {
        "label": "TIER 4 — Vulnerable",
        "badge": "t4",
        "points": [
            "Conduct ward-level vulnerability mapping identifying sub-areas prone to waterlogging",
            "Issue community alert notices with waterlogging hotspot locations and contact numbers",
            "Inspect condition of existing drain infrastructure — record capacity vs current load",
            "Plan medium-term drain augmentation based on population density and built-up area growth",
            "Promote installation of roof-top rain water harvesting in residential colonies",
            "Engage Resident Welfare Associations (RWAs) for local drain maintenance awareness",
        ]
    },
    "TIER 5": {
        "label": "TIER 5 — Monitoring",
        "badge": "t5",
        "points": [
            "Maintain routine monsoon monitoring schedule — weekly drain inspection from June",
            "Ensure all dewatering pumps are serviced and operational before June 1st",
            "Check adequacy of green cover and permeable surfaces to reduce runoff coefficient",
            "Identify parks and open spaces that can serve as temporary storm water recharge zones",
            "Maintain updated ward-level drainage map for future GIS integration",
            "No emergency deployment required — focus on preventive maintenance only",
        ]
    },
    "TIER 6": {
        "label": "TIER 6 — Prepared",
        "badge": "t6",
        "points": [
            "Conduct annual infrastructure check of storm drains and outfall points before monsoon",
            "Assess potential for Low Impact Development (LID) — porous pavements, bio-swales",
            "Document current drainage network condition for long-term master plan updates",
            "Encourage rain water harvesting adoption in new construction approvals",
            "Review and update ward-level emergency contact directory for flood events",
            "No immediate structural intervention required — focus on long-term planning",
        ]
    },
    "TIER 7": {
        "label": "TIER 7 — Optimal",
        "badge": "t7",
        "points": [
            "Maintain current infrastructure and drainage network in good condition",
            "Continue existing rain water harvesting and green infrastructure programmes",
            "Serve as model ward for best practices in urban flood resilience",
            "Provide drainage capacity data to support neighbouring high-risk ward planning",
            "No intervention required — periodic review recommended post-monsoon season",
            "Document practices as case study for replication in lower-tier wards",
        ]
    },
}

#Ward ranking logic
def generate_ward_rankings(final_map):
    final_rankings = final_map[['WardName', 'readiness_score']].drop_duplicates().sort_values(by='readiness_score')
    q_steps = [round(i / 7, 2) for i in range(1, 7)]
    quantiles = final_rankings['readiness_score'].quantile(q_steps).to_dict()

    def get_tier(score):
        if score <= quantiles[q_steps[0]]: return "TIER 1"
        if score <= quantiles[q_steps[1]]: return "TIER 2"
        if score <= quantiles[q_steps[2]]: return "TIER 3"
        if score <= quantiles[q_steps[3]]: return "TIER 4"
        if score <= quantiles[q_steps[4]]: return "TIER 5"
        if score <= quantiles[q_steps[5]]: return "TIER 6"
        return "TIER 7"

    final_rankings['Tier'] = final_rankings['readiness_score'].apply(get_tier)
    return final_rankings

#Cached ward loader
@st.cache_data
def load_wards():
    return gpd.read_file("assets/delhi_wards.kml")

#LOGIN
def login():
    st.markdown(
        "<h1 style='text-align:center;color:#002D04;font-weight:bold;'>Login</h1>",
        unsafe_allow_html=True)
    email = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if email == "abc@gmail.com" and password == "12345@123":
            st.session_state.logged_in = True
            st.success("Login successful! Redirecting...")
            st.rerun()
        else:
            st.error("Invalid credentials")

#MAIN DASHBOARD
def hydrology_engine():

    st.markdown("""
    <div class="navbar"><div class="navbar-brand">🌊 Urban Flooding & Hydrology <br> Engine</div>
        <div class="navbar-links">
            <a href="#about">About</a>
            <a href="#system">System</a>
            <a href="#info">info</a>
            <span class="navbar-badge">NeuroCodex</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<h1 style='text-align:center;font-weight:bold;font-style:italic;color:#002D04;'>"
        "Urban Flooding & Hydrology Engine</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center;font-style:italic;font-size:22px;font-weight:bold;color:black;'>"
        "AI-based Hydrology Engine for Pre-Monsoon Readiness</p>", unsafe_allow_html=True)

    dtm = days_to_monsoon()
    st.markdown(f"""
    <div class="countdown-banner" id="system">
        <div>
            <span class="countdown-days">{dtm}</span>
            <div class="countdown-text">
                <strong>Days to Monsoon Season</strong>
                <span>Pre-monsoon readiness window active &nbsp;·&nbsp; June 15 onset</span>
            </div>
        </div>
        <div class="countdown-right">
            <span class="pulse-dot"></span>&nbsp; System Active &nbsp;·&nbsp; Engine Ready
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("""
    <div class="about-section" id="about">
    <h3>🏙️ About This System</h3>

    <p>
    The <strong>Urban Flooding & Hydrology Engine</strong> is a GIS-integrated AI system designed
    for urban environments to identify flood-prone areas before the monsoon season.
    It processes large-scale geospatial data including terrain elevation, rainfall intensity,
    and drainage infrastructure to detect localized flood vulnerabilities.
    </p>

    <div class="about-grid">

    <div class="about-card hover-card">
    <div class="about-card-icon">🧠</div>
    <div class="about-card-title">AI Modeling</div>
    <div class="about-card-desc">
    Uses Gaussian Mixture Model (GMM) for clustering flood-risk zones and generating readiness scores.
    </div>
    </div>

    <div class="about-card hover-card">
    <div class="about-card-icon">🌍</div>
    <div class="about-card-title">Scalable System</div>
    <div class="about-card-desc">
    Adaptable architecture deployable across cities with geospatial and environmental datasets.
    </div>
    </div>

    <div class="about-card hover-card">
    <div class="about-card-icon">⚙️</div>
    <div class="about-card-title">Decision Support</div>
    <div class="about-card-desc">
    Enables proactive planning, resource allocation, and flood impact mitigation.
    </div>
    </div>

    </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])
    if uploaded_file:
        st.session_state["csv_file"] = uploaded_file
        st.success("CSV uploaded successfully")

    if st.button("Run Flood Risk Analysis"):
        if "csv_file" not in st.session_state:
            st.warning("Please upload a CSV first")
            return

        with st.status("⚙️  Analysis in progress…", expanded=True) as status:

            st.write("🤖  Running AI clustering model…")
            files = {"file": st.session_state["csv_file"]}
            response = requests.post(API_Url, files=files)
            if response.status_code != 200:
                status.update(label="❌ API request failed", state="error")
                st.error("API request failed")
                return
            csv_data = io.StringIO(response.content.decode("utf-8"))
            df = pd.read_csv(csv_data)

            st.write("📍  Identifying flood micro-hotspots…")
            c4 = df[df["cluster"] == 4]
            hotspots_4 = c4.sample(n=min(4000, len(c4)), weights="inundation_weight", random_state=42)
            c2 = df[df["cluster"] == 2]
            hotspots_2 = c2.sample(n=min(3500, len(c2)), weights=(1.1 - c2["elev_norm"]), random_state=42)
            c1 = df[df["cluster"] == 1]
            hotspots_1 = c1[c1["rain_norm"] >= c1["rain_norm"].quantile(0.90)].sample(
                n=min(850, len(c1)), random_state=42)
            c3 = df[df["cluster"] == 3]
            c3_filtered = c3[
                (c3["rain_norm"] >= c3["rain_norm"].quantile(0.80)) &
                (c3["elev_norm"] <= c3["elev_norm"].quantile(0.40))
            ]
            hotspots_3 = c3_filtered.sample(n=min(500, len(c3)), random_state=42)
            hotspots = pd.concat([hotspots_4, hotspots_2, hotspots_1, hotspots_3])

            st.write("🏙️  Computing ward readiness scores…")
            wards = load_wards()
            geometry = [Point(xy) for xy in zip(hotspots["lon"], hotspots["lat"])]
            gdf_hotspots = gpd.GeoDataFrame(hotspots, geometry=geometry, crs="EPSG:4326")
            joined = gpd.sjoin(gdf_hotspots, wards, how="inner", predicate="intersects")
            exposure = joined.groupby("WardName").size().rename("hotspot_count")
            severity = joined.groupby("WardName")["inundation_weight"].quantile(0.85).rename("peak_severity")

            wards_metrics = wards.copy()
            wards_metrics["area"] = wards_metrics.geometry.area
            wards_metrics = wards_metrics.merge(exposure, on="WardName", how="left")
            wards_metrics = wards_metrics.merge(severity, on="WardName", how="left").fillna(0)
            wards_metrics["hotspot_density"] = wards_metrics["hotspot_count"] / wards_metrics["area"].replace(0, 1)

            mscaler = MinMaxScaler()
            cols = ["hotspot_count", "hotspot_density", "peak_severity"]
            wards_metrics[[f"{c}_n" for c in cols]] = mscaler.fit_transform(wards_metrics[cols])

            wards_metrics["base_readiness"] = 1 - (
                0.15 * wards_metrics["hotspot_count_n"] +
                0.50 * wards_metrics["hotspot_density_n"] +
                0.35 * wards_metrics["peak_severity_n"]
            )

            river_wards = [
                'CIVIL LINES', 'SONIA VIHAR', 'YAMUNA VIHAR', 'JAMA MASJID',
                'DARYAGANJ', 'NEW ASHOK NAGAR', 'MAYUR VIHAR PHASE-I', 'BADARPUR'
            ]
            wards_metrics["readiness_score"] = wards_metrics.apply(
                lambda x: x["base_readiness"] - (0.30 + 0.10 * x["peak_severity_n"])
                if x["WardName"] in river_wards else x["base_readiness"], axis=1)
            wards_metrics["readiness_score"] = wards_metrics["readiness_score"].clip(0.05, 1.0)

            final_map = wards.merge(
                wards_metrics[['WardName', 'readiness_score']], on="WardName", how="left")
            final_map["readiness_score"] = final_map["readiness_score"].fillna(1)
            final_map = final_map[final_map["WardName"].notna() & (final_map["WardName"] != "")]

            st.write("🗺️  Rendering GIS maps…")
            status.update(label="✅  Analysis complete!", state="complete", expanded=False)

        num_clusters = df["cluster"].nunique()
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-value">{len(df):,}</div>
                <div class="stat-label">Data points processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(hotspots):,}</div>
                <div class="stat-label">Flood micro-hotspots identified</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{num_clusters}</div>
                <div class="stat-label">GMM clusters</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{dtm}d</div>
                <div class="stat-label">Days to monsoon</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


        st.markdown("<div class='section-hdr'>🗺️ Flood Micro-Hotspot Map</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.scatter(hotspots["lon"], hotspots["lat"], s=20, c="red", alpha=0.3, edgecolors="none")
        ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik, zoom=12)
        ax.set_title("Delhi Flood Risk: OpenStreetMap Visualization (Hydrology Engine)", fontsize=16)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

        
        st.markdown("<div class='section-hdr'>🏙️ Ward-Level Flood Readiness Analysis</div>",
                    unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(12, 12))
        final_map.plot(
            column="readiness_score", cmap="RdYlGn",
            legend=True, scheme="Quantiles", k=7,
            edgecolor="#333333", linewidth=0.3,
            ax=ax2,
            legend_kwds={
                'loc': 'upper left',
                'title': "Readiness Score\n(Green=Ready, Red=Critical)"
            }
        )
        ax2.set_title("Delhi Ward-Level Pre-Monsoon Readiness Map", fontsize=18)
        ax2.axis("off")
        st.pyplot(fig2)
        plt.close(fig2)

        
        st.markdown("<div class='section-hdr'>🎯 Ward Readiness Rankings & Action Plan</div>",
                    unsafe_allow_html=True)

        rankings = generate_ward_rankings(final_map)
        critical = len(rankings[rankings["Tier"].isin(["TIER 1", "TIER 2"])])
        total = len(rankings)

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total Wards Analysed", total)
        col_b.metric("Critical Wards (Tier 1–2)", critical)
        col_c.metric("High-Risk Rate", f"{round((critical / total) * 100)}%")

        rows_html = ""
        for i, row in enumerate(rankings.itertuples(), 1):
            cfg = TIER_CONFIG.get(row.Tier, ("t7", "#006400", "—"))
            badge_cls, bar_color, action = cfg
            bar_w = max(4, round(row.readiness_score * 100))
            rows_html += f"""<tr>
                <td style="color:#666;font-size:0.78rem;">{i}</td>
                <td><strong>{row.WardName}</strong></td>
                <td><div class="score-bar-wrap">
                    <span class="score-num">{row.readiness_score:.3f}</span>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width:{bar_w}%;background:{bar_color};"></div>
                    </div>
                </div></td>
                <td><span class="tier-badge {badge_cls}">{row.Tier}</span></td>
                <td><span class="action-pill">{action}</span></td>
            </tr>"""

        st.markdown(f"""
        <table class="ward-table">
            <thead><tr>
                <th>S.NO</th><th>Ward</th><th>Readiness Score</th>
                <th>Risk Tier</th><th>Recommended Action</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        <div class="method-note">
            <strong>Scoring methodology:</strong>
            Score = 1 − (0.15 × hotspot count + 0.50 × spatial density + 0.35 × peak severity), all normalised.
            Yamuna-corridor wards: −0.30 penalty. Clipped to [0.05, 1.0].
            Tiers via 7-quantile distribution.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-hdr'>🛠️ Tier-wise Intervention & Infrastructure Plan</div>",
                    unsafe_allow_html=True)

        inter_rows = ""
        for tier_key, data in INTERVENTIONS.items():
            badge_cls = data["badge"]
            bullets = "".join(f"<li>{p}</li>" for p in data["points"])
            inter_rows += f"""<tr>
                <td style="vertical-align:top; white-space:nowrap;">
                    <span class="tier-badge {badge_cls}">{data['label']}</span>
                </td>
                <td>
                    <ul class="intervention-list">{bullets}</ul>
                </td>
            </tr>"""

        st.markdown(f"""
        <table class="ward-table">
            <thead><tr>
                <th style="width:18%;">Risk Tier</th>
                <th>Recommended Interventions & Infrastructure Actions</th>
            </tr></thead>
            <tbody>{inter_rows}</tbody>
        </table>
        <div class="method-note">
            <strong>Intervention framework reference:</strong>
            Recommendations are aligned with the IIT Delhi Drainage Master Plan for NCT of Delhi (2018),
            NDMA National Guidelines on Urban Flood Management, and NIDM July 2023 Delhi Flood Workshop outcomes.
            Actions are designed for pre-monsoon execution window (March–June).
        </div>
        """, unsafe_allow_html=True)
    
    
    st.markdown("""
    <div class="contact-section" id="info">
        <h3>📬 Project Information</h3>
        <div class="contact-grid">
            <div class="contact-card">
                <div class="contact-icon">👨‍💻</div>
                <div>
                    <div class="contact-label">Created By</div>
                    <div class="contact-value">Team NeuroCodex</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="footer">
            <p><strong>Urban Flooding & Hydrology Engine</strong> &nbsp;·&nbsp; NeuroCodex</p>
            <p>Built for Urban Cities and Government</p>
        </div>
        """, unsafe_allow_html=True)


if not st.session_state.logged_in:
    login()
else:
    hydrology_engine()
