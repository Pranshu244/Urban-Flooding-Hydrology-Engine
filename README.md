# Urban Flooding & Hydrology Engine
AI-powered GIS-integrated system that analyzes rainfall patterns, terrain elevation, and drainage data to identify urban flood micro-hotspots across the city.  

The system combines machine learning clustering with rule-based aggregation to generate **Ward-Level Pre-Monsoon Readiness Scores**, helping authorities proactively plan flood mitigation before heavy rainfall events.

---

##  Problem Statement
Urban flooding is a growing challenge in densely populated cities due to:
- Intense rainfall events
- Low-lying terrain
- Insufficient drainage infrastructure
- Rapid urbanization
Traditional flood monitoring systems are reactive and respond **after flooding occurs**.

This project proposes a **predictive hydrology intelligence engine** that detects **urban flood micro-hotspots in advance**, enabling city authorities to prepare before the monsoon season.

---

## Solution Overview
The **Urban Flooding & Hydrology Engine** processes geospatial environmental datasets and identifies potential flood-prone locations using machine learning and spatial analytics.

Key outputs include:
- Detection of **thousands of flood micro-hotspots**
- **Ward-level flood vulnerability analysis**
- Generation of **Pre-Monsoon Readiness Scores**
- GIS-based visualization of flood risk zones

This enables **data-driven flood preparedness and urban planning**.

---

## System Architecture
The system is implemented as a **three-stage pipeline**.

### Stage 1 — Machine Learning Clustering

A **Gaussian Mixture Model (GMM)** analyzes environmental variables and clusters geographic locations based on flood risk characteristics.

Input Features:

- Elevation
- Extreme rainfall levels
- Drainage presence
- Drainage density

Output:

- Spatial clusters identifying **potential flood vulnerability zones**

---

### Stage 2 — Rule-Based Flood Hotspot Identification

Cluster outputs are processed using **hydrology-inspired rules** that evaluate:

- High rainfall
- Low elevation
- Limited drainage availability
- Low drainage density

Locations satisfying these conditions are labeled as **Urban Flood Micro-Hotspots**.

---

### Stage 3 — Ward-Level Pre-Monsoon Readiness Score

Detected hotspots are aggregated at the **administrative ward level**.

Each ward receives a **Pre-Monsoon Readiness Score**, which represents its flood preparedness status.

This score helps authorities prioritize:

- Drainage maintenance
- Infrastructure upgrades
- Emergency preparedness

---

## GIS-Based Flood Risk Visualization
The system generates geospatial maps highlighting:
- Flood micro-hotspot clusters
- Ward-level flood risk zones
- Readiness score distribution across the city

These visualizations provide **clear spatial insights for decision-makers**.

---

## Dataset Description
The model uses a geospatial dataset containing environmental and hydrological variables.
### Dataset Structure

| Column | Description |
|------|-------------|
| lat | Latitude coordinate |
| lon | Longitude coordinate |
| elevation_m | Terrain elevation (meters above sea level) |
| rain_95p_mm | 95th percentile rainfall intensity (mm) |
| drain_present | Binary indicator for drainage presence (0 = no drain, 1 = drain) |
| drain_density | Density of drainage infrastructure |

### Dataset Size
- **Total Rows:** 1,913,317
- **Total Columns:** 6
- **Memory Usage:** ~87 MB

---

## Exploratory Data Analysis (EDA)
Exploratory data analysis was conducted to understand spatial patterns and relationships between environmental variables.
### 1. Data Structure
The dataset contains:
- **Geospatial coordinates:** latitude and longitude
- **Environmental variables:** elevation and rainfall
- **Drainage infrastructure indicators:** drainage presence and density
Numerical variables:
- elevation_m
- rain_95p_mm
- drain_density
Binary variable:
- drain_present (0 = no drain, 1 = drain)
---

### 2. Missing Values
The dataset was inspected for missing values.
Result:
- No significant missing values were found.
- Dataset is clean and suitable for analysis.

---

### 3. Data Visualization Insights

#### Spatial Rainfall Distribution
- Highest rainfall observed in **southwestern regions (19–20 mm)**
- High rainfall also in **northwestern regions (17–18 mm)**
- Lowest rainfall in **northeast (8–10 mm)**
- Moderate rainfall in **southeast (12–14 mm)**

#### Elevation vs Rainfall
Scatter analysis shows:
- Elevation ranges between **180–310 meters**
- Rainfall appears in clusters
- No strong linear relationship between elevation and rainfall

#### Rainfall vs Drainage Density
Hexbin visualization reveals:
- Most locations have **low drainage density (0–1)**
- Rainfall clusters around **8 mm, 13.5 mm, 18.5 mm, and 20 mm**

#### Elevation Distribution
Histogram shows:
- Elevation range: **170–320 meters**
- Majority values: **200–230 meters**
- Distribution slightly **right-skewed**

#### Rainfall vs Drainage Presence
Boxplot comparison indicates:
- Rainfall range similar for both groups
- Drain presence does not strongly influence rainfall distribution
#### Correlation Analysis
Correlation heatmap shows **very weak relationships between variables**:
- Elevation vs Rainfall: −0.032
- Rainfall vs Drain Density: −0.11
- Elevation vs Drain Density: −0.017
This indicates flood risk depends on **combined environmental factors rather than a single variable**.

---

## Technology Stack

### Machine Learning
- Python
- Scikit-learn
- Gaussian Mixture Model
  
### Data Processing
- Pandas
- NumPy
  
### Visualization
- Matplotlib
- Geospatial mapping libraries
  
### Backend
- Python API layer for model interaction
  
### Frontend
- Interactive dashboard interface for:
  - Flood hotspot visualization
  - Ward-level readiness analysis

---

## System Components
### Backend
Responsible for:
- Data preprocessing
- Machine learning clustering
- Flood hotspot detection
- Readiness score computation
- 
---

#### FastAPI

The backend exposes REST APIs for inference and monitoring.

##### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health message |
| GET | `/health` | Model status and version |
| POST | `/predict` | Run Hydrology Engine  |

---

### Frontend Dashboard
Provides visualization tools for:
- Flood micro-hotspot maps
- Ward-level flood risk maps
- Readiness score visualization
The dashboard converts complex environmental datasets into **clear spatial intelligence for decision-makers**.
---
## Data Sources
Environmental and geospatial datasets were derived from publicly available sources such as:

- Digital Elevation Models (DEM Google Earth Engine)
- Historical rainfall statistics(IMD Pune)
- Urban drainage infrastructure data
- Open geospatial datasets

These datasets enable spatial analysis of flood vulnerability across urban regions.

---

## Impact
This system enables authorities to:
- Detect **flood-prone micro-locations**
- Identify **high-risk wards before monsoon**
- Allocate **resources and maintenance efforts efficiently**
- Improve **urban flood preparedness and resilience**

---
##  Prototype Demonstration 

Below are example outputs generated by the system.

### Ward-Level Readiness Map

*(Insert screenshot here)*

---

## Contributors
Project developed as part of a hackathon prototype focused on **AI-driven urban resilience and flood preparedness**.
