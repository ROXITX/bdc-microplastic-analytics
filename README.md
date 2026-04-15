# 🌊 Ocean Microplastic Pollution Hotspot Prediction

> **Big Data Computing — DA-2 | VIT Chennai | April 2026**

A full-stack Big Data Analytics system for predicting and explaining ocean microplastic pollution hotspots using Apache PySpark, Two-Layer Machine Learning, and Graph Network Analytics.

---

## 📋 Project Overview

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Big Data Pipeline | Apache PySpark | Distributed processing of 100K+ ocean sensor records |
| Layer 1 ML | KMeans Clustering (k=5) | Discover ocean gyre zones without geographic hardcoding |
| Layer 2 ML | Random Forest Classifier | Predict hotspot zones (81.12% accuracy) |
| Graph Analytics | NetworkX + Louvain | PageRank influence, betweenness centrality, community detection |
| Dashboard | Streamlit + Plotly | Real-time decision-support visualisation |

---

## 🚀 Quick Start

```bash
# 1. Generate dataset (100,000 ocean sensor records)
python generate_actual_dataset.py

# 2. Run PySpark ML pipeline
python pipeline.py

# 3. Build graph analytics
python graph_analytics.py

# 4. Launch dashboard
streamlit run app.py
```

---

## 📁 Repository Structure

```
bdc-microplastic-analytics/
├── generate_actual_dataset.py   # Synthetic NOAA/Copernicus data generation
├── pipeline.py                   # PySpark Big Data pipeline (KMeans + RF)
├── graph_analytics.py            # NetworkX graph: PageRank, betweenness, Louvain
├── app.py                        # Streamlit dashboard (2 tabs)
├── generate_report.py            # DA-2 report generator (.docx)
├── DA2_Report_Final_WithImages.docx  # Final project report
├── screenshots/                  # Dashboard screenshots for report
│   ├── fig1_ml_dashboard.png
│   ├── fig2_graph_kpis.png
│   ├── fig3_graph_charts.png
│   └── fig4_graph_table.png
├── dashboard_data/
│   ├── predictions.json          # ML pipeline output
│   └── graph_metrics.json        # Graph analytics output
├── Project_Presentation_Guide.md
├── PySpark_Explanation_Guide.md
└── implementation_status_and_output.md
```

---

## 📊 Key Results

### ML Pipeline
| Metric | Value |
|--------|-------|
| Random Forest Accuracy | **81.12%** |
| KMeans Silhouette Score | 0.169 |
| Identified Hotspots (2K sample) | 513 |
| Top Feature | Ocean_Current_Velocity (0.457) |

### Graph Analytics
| Metric | Value |
|--------|-------|
| Graph Nodes (5°×5° cells) | 729 |
| Graph Edges | 1,731 |
| Critical Risk Zones | **110** |
| High-Risk Zones | 145 |
| Pollution Bridge Nodes | 68 |
| Graph Communities (Louvain) | 74 |
| Avg Pollution Path Length | 10.33 hops |

---

## 💡 Innovations

1. **Two-Layer ML Architecture** — KMeans cluster labels fed as features into Random Forest
2. **30-Day Spatio-Temporal Rolling Window** — PySpark Window functions for temporal current dynamics  
3. **Graph Network Analytics** — PageRank + betweenness centrality on ocean spatial proximity graph

---

## 🔧 Dependencies

```bash
pip install pyspark streamlit plotly networkx python-louvain streamlit-folium python-docx pandas numpy
```

---

## 👥 Team Members

| Name | Register Number |
|------|----------------|
| Rohit Sharma | 23MIA1104 |
| Team Member 2 | 23MIAXXXX |
| Team Member 3 | 23MIAXXXX |
