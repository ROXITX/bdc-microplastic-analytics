import streamlit as st
import folium
import json
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from streamlit_folium import st_folium
import numpy as np
import subprocess
import sys

st.set_page_config(page_title="Ocean Microplastic Analytics", layout="wide", page_icon="🌊")

# ───────────────────────────────────────────────
# GLOBAL CSS
# ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.9) 0%, rgba(30,41,59,0.9) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 16px;
    padding: 22px 20px;
    color: white;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    margin-bottom: 10px;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(79,172,254,0.25);
}
.metric-title {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #94a3b8;
    margin-bottom: 8px;
    font-weight: 600;
}
.metric-value {
    font-size: 2.1rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-value-small {
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(90deg, #4facfe, #00f2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-sub {
    font-size: 0.7rem;
    color: #64748b;
    margin-top: 4px;
}

/* Risk badges */
.badge-critical { background: linear-gradient(90deg,#ef4444,#dc2626); color:#fff; padding:3px 10px; border-radius:999px; font-size:0.72rem; font-weight:700; }
.badge-high     { background: linear-gradient(90deg,#f97316,#ea580c); color:#fff; padding:3px 10px; border-radius:999px; font-size:0.72rem; font-weight:700; }
.badge-moderate { background: linear-gradient(90deg,#eab308,#ca8a04); color:#fff; padding:3px 10px; border-radius:999px; font-size:0.72rem; font-weight:700; }
.badge-low      { background: linear-gradient(90deg,#22c55e,#16a34a); color:#fff; padding:3px 10px; border-radius:999px; font-size:0.72rem; font-weight:700; }

/* Insight cards */
.insight-card {
    background: linear-gradient(135deg, rgba(15,23,42,0.95) 0%, rgba(30,41,59,0.95) 100%);
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: #e2e8f0;
    font-size: 0.88rem;
    line-height: 1.55;
}
.insight-title {
    font-weight: 700;
    color: #f87171;
    font-size: 0.82rem;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid rgba(79,172,254,0.3);
}

div[data-testid="stTabs"] button {
    font-size: 0.92rem !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# LOAD DATA
# ───────────────────────────────────────────────
PRED_PATH  = "dashboard_data/predictions.json"
GRAPH_PATH = "dashboard_data/graph_metrics.json"

if not os.path.exists(PRED_PATH):
    st.warning("⚠️ No pipeline data found. Please run `python pipeline.py` first.")
    st.stop()

with open(PRED_PATH) as f:
    pred_data = json.load(f)

metrics = pred_data.get("metrics", {})
points  = pred_data.get("points", [])

graph_data = None
if os.path.exists(GRAPH_PATH):
    with open(GRAPH_PATH) as f:
        graph_data = json.load(f)

# ───────────────────────────────────────────────
# HEADER
# ───────────────────────────────────────────────
st.title("🌊 Ocean Microplastic Pollution Analytics")
st.markdown(
    "<span style='color:#64748b; font-size:0.9rem;'>Big Data Pipeline · PySpark ML · Graph Network Analysis · NOAA/Copernicus Features</span>",
    unsafe_allow_html=True
)

# ───────────────────────────────────────────────
# TABS
# ───────────────────────────────────────────────
tab_ml, tab_graph = st.tabs(["📊 ML Pipeline Dashboard", "🕸️ Graph Analytics"])


# ═══════════════════════════════════════════════
# TAB 1 ── ML PIPELINE DASHBOARD
# ═══════════════════════════════════════════════
with tab_ml:
    # KPI Row
    c1, c2, c3, c4 = st.columns(4)
    hotspots_count = sum(1 for p in points if p["Hotspot"] == 1)

    with c1:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-title">Model Accuracy (RF)</div>
            <div class="metric-value">{metrics.get("rf_accuracy",0)*100:.2f}%</div>
            <div class="metric-sub">Random Forest Classifier</div>
        </div>''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-title">Clustering Quality</div>
            <div class="metric-value">{metrics.get("silhouette_score",0):.3f}</div>
            <div class="metric-sub">Silhouette Score (KMeans k=5)</div>
        </div>''', unsafe_allow_html=True)
    with c3:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-title">Identified Hotspots</div>
            <div class="metric-value">{hotspots_count}</div>
            <div class="metric-sub">Top 25% concentration zones</div>
        </div>''', unsafe_allow_html=True)
    with c4:
        threshold = metrics.get("hotspot_threshold", 0)
        st.markdown(f'''<div class="metric-card">
            <div class="metric-title">Hotspot Threshold</div>
            <div class="metric-value">{threshold:.1f}</div>
            <div class="metric-sub">µg/L concentration cutoff</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_map, col_chart = st.columns([3, 2])

    with col_map:
        st.markdown('<div class="section-header">Interactive Accumulation Map</div>', unsafe_allow_html=True)
        cluster_colors = {0: '#3498db', 1: '#e74c3c', 2: '#2ecc71', 3: '#f1c40f', 4: '#9b59b6'}
        if points:
            df_map = pd.DataFrame(points)
            df_map['color'] = df_map['cluster'].map(cluster_colors).fillna('#ffffff')
            df_map['size']  = df_map['Hotspot'].apply(lambda x: 150000 if x == 1 else 15000)
            st.map(df_map, latitude='Latitude', longitude='Longitude',
                   color='color', size='size', zoom=1)
        else:
            st.warning("No data points available.")

    with col_chart:
        st.markdown('<div class="section-header">Feature Importances</div>', unsafe_allow_html=True)
        importances = metrics.get("feature_importances", {})
        df_imp = pd.DataFrame({
            "Feature": list(importances.keys()),
            "Importance": list(importances.values())
        }).sort_values(by="Importance", ascending=True)

        fig_imp = go.Figure(go.Bar(
            x=df_imp["Importance"], y=df_imp["Feature"],
            orientation='h',
            marker=dict(
                color=df_imp["Importance"],
                colorscale=[[0, "#1e3a5f"], [1, "#4facfe"]],
                showscale=False
            ),
            text=[f"{v:.3f}" for v in df_imp["Importance"]],
            textposition='outside',
            textfont=dict(color='#94a3b8', size=10)
        ))
        fig_imp.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.6)',
            font=dict(color='#e2e8f0', family='Inter'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False),
            yaxis=dict(showgrid=False),
            margin=dict(l=10, r=60, t=10, b=10),
            height=420,
        )
        st.plotly_chart(fig_imp, use_container_width=True, key="imp_chart")

    # Cluster distribution
    st.markdown('<div class="section-header">Cluster Distribution</div>', unsafe_allow_html=True)
    df_pts = pd.DataFrame(points)
    cluster_names = {0: 'North Pacific', 1: 'North Atlantic', 2: 'South Pacific', 3: 'South Atlantic', 4: 'Indian Ocean'}
    df_pts['cluster_name'] = df_pts['cluster'].map(cluster_names).fillna('Unknown')
    dist = df_pts.groupby('cluster_name').agg(
        total=('cluster', 'count'),
        hotspots=('Hotspot', 'sum'),
        avg_conc=('Microplastic_Concentration', 'mean')
    ).reset_index()

    fig_dist = px.bar(dist, x='cluster_name', y=['total', 'hotspots'],
                      barmode='group',
                      color_discrete_map={'total': '#4facfe', 'hotspots': '#ef4444'},
                      labels={'value': 'Count', 'variable': 'Type', 'cluster_name': 'Cluster'})
    fig_dist.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15,23,42,0.6)',
        font=dict(color='#e2e8f0', family='Inter'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=10, r=10, t=10, b=10),
        height=280
    )
    st.plotly_chart(fig_dist, use_container_width=True, key="dist_chart")


# ═══════════════════════════════════════════════
# TAB 2 ── GRAPH ANALYTICS
# ═══════════════════════════════════════════════
with tab_graph:

    if graph_data is None:
        st.warning("Graph analytics data not found.")
        if st.button("Run Graph Analytics Now"):
            with st.spinner("Building pollution network graph..."):
                result = subprocess.run(
                    [sys.executable, "graph_analytics.py"],
                    capture_output=True, text=True, cwd=os.getcwd()
                )
            if result.returncode == 0:
                st.success("Graph analytics complete! Reloading...")
                st.rerun()
            else:
                st.error(f"Error:\n{result.stderr}")
        st.stop()

    summary = graph_data["summary"]
    nodes   = graph_data["nodes"]
    edges   = graph_data["edges"]

    df_nodes = pd.DataFrame(nodes)
    df_edges = pd.DataFrame(edges)

    # ── KPI Row ──────────────────────────────────────────────────
    g1, g2, g3, g4, g5 = st.columns(5)
    with g1:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-title">Critical Zones</div>
            <div class="metric-value" style="background:linear-gradient(90deg,#ef4444,#dc2626);-webkit-background-clip:text;">{summary["critical_zones"]}</div>
            <div class="metric-sub">Systemic risk nodes</div>
        </div>''', unsafe_allow_html=True)
    with g2:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-title">High-Risk Zones</div>
            <div class="metric-value" style="background:linear-gradient(90deg,#f97316,#ea580c);-webkit-background-clip:text;">{summary["high_risk_zones"]}</div>
            <div class="metric-sub">Elevated spread risk</div>
        </div>''', unsafe_allow_html=True)
    with g3:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-title">Network Communities</div>
            <div class="metric-value">{summary["graph_communities"]}</div>
            <div class="metric-sub">Graph-detected groupings</div>
        </div>''', unsafe_allow_html=True)
    with g4:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-title">Pollution Bridges</div>
            <div class="metric-value">{summary["bridge_count"]}</div>
            <div class="metric-sub">Cross-zone corridor nodes</div>
        </div>''', unsafe_allow_html=True)
    with g5:
        st.markdown(f'''<div class="metric-card">
            <div class="metric-title">Avg Path Length</div>
            <div class="metric-value">{summary["avg_path_length"]}</div>
            <div class="metric-sub">Hops between grid cells</div>
        </div>''', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row A: Network Map + Insights ──────────────────────────
    col_net, col_ins = st.columns([3, 2])

    with col_net:
        st.markdown('<div class="section-header">Pollution Propagation Network — Risk Map</div>', unsafe_allow_html=True)

        RISK_COLOR = {
            "Critical": "#ef4444",
            "High":     "#f97316",
            "Moderate": "#eab308",
            "Low":      "#22c55e",
        }
        RISK_SIZE = {
            "Critical": 16,
            "High":     11,
            "Moderate": 7,
            "Low":      4,
        }

        # Draw edges first (thin grey lines)
        edge_traces = []
        # Sample edges to avoid overplotting — draw only edges between High/Critical nodes
        visible_nodes = set(df_nodes[df_nodes['risk_tier'].isin(['Critical', 'High'])]['node_id'])
        df_edges_vis = df_edges[
            df_edges['source'].isin(visible_nodes) & df_edges['target'].isin(visible_nodes)
        ]
        for _, e in df_edges_vis.iterrows():
            edge_traces.append(go.Scattergeo(
                lon=[e['source_lon'], e['target_lon'], None],
                lat=[e['source_lat'], e['target_lat'], None],
                mode='lines',
                line=dict(width=0.8, color='rgba(148,163,184,0.25)'),
                showlegend=False,
                hoverinfo='skip',
            ))

        # Node traces per risk tier
        node_traces = []
        for tier in ["Critical", "High", "Moderate", "Low"]:
            df_t = df_nodes[df_nodes['risk_tier'] == tier]
            if df_t.empty:
                continue

            hover_texts = [
                f"<b>{row['node_id']}</b><br>"
                f"Risk Tier: {row['risk_tier']}<br>"
                f"Avg Concentration: {row['avg_concentration']:.1f} µg/L<br>"
                f"PageRank Influence: {row['pagerank']:.5f}<br>"
                f"Betweenness: {row['betweenness']:.4f}<br>"
                f"Hotspot Ratio: {row['hotspot_ratio']*100:.1f}%<br>"
                f"Spread Risk Score: {row['spread_risk_score']:.4f}<br>"
                f"Community: {row['community']}"
                for _, row in df_t.iterrows()
            ]

            node_traces.append(go.Scattergeo(
                lon=df_t['lon'],
                lat=df_t['lat'],
                mode='markers',
                name=tier,
                marker=dict(
                    size=RISK_SIZE[tier],
                    color=RISK_COLOR[tier],
                    opacity=0.90,
                    line=dict(width=0.5, color='rgba(255,255,255,0.3)')
                ),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>',
            ))

        fig_net = go.Figure(data=edge_traces + node_traces)
        fig_net.update_layout(
            geo=dict(
                showland=True, landcolor='#0f172a',
                showocean=True, oceancolor='#0c1a2e',
                showcoastlines=True, coastlinecolor='#334155',
                showlakes=False,
                showframe=False,
                projection_type='natural earth',
                bgcolor='#0f172a',
            ),
            paper_bgcolor='#0f172a',
            margin=dict(l=0, r=0, t=0, b=0),
            height=480,
            legend=dict(
                bgcolor='rgba(15,23,42,0.85)',
                bordercolor='rgba(99,179,237,0.3)',
                borderwidth=1,
                font=dict(color='#e2e8f0', size=11),
                y=0.98, x=0.01
            ),
            font=dict(family='Inter', color='#e2e8f0')
        )
        st.plotly_chart(fig_net, use_container_width=True, key="network_map")

    with col_ins:
        st.markdown('<div class="section-header">Graph Insights & Actionable Findings</div>', unsafe_allow_html=True)

        # Top critical node
        top_node = df_nodes.iloc[0]
        st.markdown(f'''<div class="insight-card">
            <div class="insight-title">🔴 Highest Spread Risk Zone</div>
            Grid cell <b>{top_node["node_id"]}</b> (Lat {top_node["lat"]}, Lon {top_node["lon"]}) is the
            most systemically dangerous node with a Spread Risk Score of
            <b>{top_node["spread_risk_score"]:.4f}</b>, avg concentration of
            <b>{top_node["avg_concentration"]:.1f} µg/L</b>, and a PageRank influence of
            <b>{top_node["pagerank"]:.5f}</b>. Intervention here would impact the widest network.
        </div>''', unsafe_allow_html=True)

        # Bridge node
        top_bridge = df_nodes.sort_values('betweenness', ascending=False).iloc[0]
        st.markdown(f'''<div class="insight-card" style="border-left-color:#f97316;">
            <div class="insight-title" style="color:#fb923c;">🌉 Key Pollution Bridge</div>
            Node <b>{top_bridge["node_id"]}</b> has the highest betweenness centrality
            (<b>{top_bridge["betweenness"]:.4f}</b>) — it acts as a <b>bridge corridor</b> connecting
            otherwise separate pollution clusters. Blocking transport here would slow inter-zone spread.
        </div>''', unsafe_allow_html=True)

        # Community finding
        n_communities = summary["graph_communities"]
        n_clusters    = 5
        st.markdown(f'''<div class="insight-card" style="border-left-color:#eab308;">
            <div class="insight-title" style="color:#fbbf24;">🗺️ Hidden Community Structure</div>
            Graph community detection found <b>{n_communities} natural groupings</b> vs
            KMeans' {n_clusters} clusters. This means pollution zones have a more complex
            connectivity pattern than simple geographic proximity — some distant zones are
            linked by shared current corridors.
        </div>''', unsafe_allow_html=True)

        # Connectivity insight
        density = summary["graph_density"]
        avg_path = summary["avg_path_length"]
        st.markdown(f'''<div class="insight-card" style="border-left-color:#22c55e;">
            <div class="insight-title" style="color:#4ade80;">📡 Network Propagation Speed</div>
            With an average path length of <b>{avg_path} hops</b> and graph density of
            <b>{density:.4f}</b>, pollution can theoretically propagate from any source zone to
            any other zone in approximately <b>{avg_path:.0f} ocean-current steps</b>.
            The {summary["bridge_count"]} bridge nodes are the chokepoints.
        </div>''', unsafe_allow_html=True)

    # ── Row B: PageRank Chart + Betweenness + Top Table ────────
    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)

    with b1:
        st.markdown('<div class="section-header">Top 20 Zones by PageRank Influence Score</div>', unsafe_allow_html=True)
        df_pr = df_nodes.sort_values('pagerank', ascending=False).head(20).sort_values('pagerank', ascending=True)

        color_map = {"Critical": "#ef4444", "High": "#f97316", "Moderate": "#eab308", "Low": "#22c55e"}
        bar_colors = [color_map.get(t, "#4facfe") for t in df_pr['risk_tier']]

        fig_pr = go.Figure(go.Bar(
            x=df_pr['pagerank'],
            y=df_pr['node_id'],
            orientation='h',
            marker=dict(color=bar_colors),
            text=[f"{v:.5f}" for v in df_pr['pagerank']],
            textposition='outside',
            textfont=dict(color='#94a3b8', size=9),
            customdata=df_pr[['avg_concentration', 'hotspot_ratio', 'risk_tier']].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "PageRank: %{x:.5f}<br>"
                "Avg Conc: %{customdata[0]:.1f} µg/L<br>"
                "Hotspot Ratio: %{customdata[1]:.1%}<br>"
                "Risk Tier: %{customdata[2]}<extra></extra>"
            )
        ))
        fig_pr.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.6)',
            font=dict(color='#e2e8f0', family='Inter', size=10),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=9)),
            margin=dict(l=10, r=70, t=10, b=10),
            height=440,
        )
        st.plotly_chart(fig_pr, use_container_width=True, key="pagerank_chart")

    with b2:
        st.markdown('<div class="section-header">Top 20 Pollution Bridge Nodes (Betweenness)</div>', unsafe_allow_html=True)
        df_bt = df_nodes.sort_values('betweenness', ascending=False).head(20).sort_values('betweenness', ascending=True)
        bar_colors_bt = [color_map.get(t, "#4facfe") for t in df_bt['risk_tier']]

        fig_bt = go.Figure(go.Bar(
            x=df_bt['betweenness'],
            y=df_bt['node_id'],
            orientation='h',
            marker=dict(color=bar_colors_bt),
            text=[f"{v:.4f}" for v in df_bt['betweenness']],
            textposition='outside',
            textfont=dict(color='#94a3b8', size=9),
            customdata=df_bt[['avg_concentration', 'spread_risk_score', 'risk_tier']].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Betweenness: %{x:.4f}<br>"
                "Avg Conc: %{customdata[0]:.1f} µg/L<br>"
                "Spread Risk: %{customdata[1]:.4f}<br>"
                "Risk Tier: %{customdata[2]}<extra></extra>"
            )
        ))
        fig_bt.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.6)',
            font=dict(color='#e2e8f0', family='Inter', size=10),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False),
            yaxis=dict(showgrid=False, tickfont=dict(size=9)),
            margin=dict(l=10, r=80, t=10, b=10),
            height=440,
        )
        st.plotly_chart(fig_bt, use_container_width=True, key="betweenness_chart")

    # ── Row C: Spread Risk Scatter + Community vs Cluster ───────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-header">Concentration vs Influence — Spread Risk Scatter</div>', unsafe_allow_html=True)
        fig_sc = px.scatter(
            df_nodes,
            x='avg_concentration',
            y='pagerank',
            size='spread_risk_score',
            color='risk_tier',
            color_discrete_map=color_map,
            hover_data={
                'node_id': True,
                'betweenness': ':.4f',
                'hotspot_ratio': ':.1%',
                'community': True,
                'avg_concentration': ':.1f',
                'pagerank': ':.5f',
                'spread_risk_score': ':.4f',
            },
            labels={
                'avg_concentration': 'Avg Microplastic Concentration (µg/L)',
                'pagerank': 'PageRank Influence Score',
                'risk_tier': 'Risk Tier'
            },
            size_max=22
        )
        fig_sc.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(15,23,42,0.6)',
            font=dict(color='#e2e8f0', family='Inter'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.06)', zeroline=False),
            legend=dict(bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=20, r=20, t=20, b=20),
            height=380
        )
        st.plotly_chart(fig_sc, use_container_width=True, key="scatter_chart")

    with c2:
        st.markdown('<div class="section-header">Risk Tier Breakdown</div>', unsafe_allow_html=True)
        tier_counts = df_nodes['risk_tier'].value_counts().reset_index()
        tier_counts.columns = ['Risk Tier', 'Count']
        tier_order = ['Critical', 'High', 'Moderate', 'Low']
        tier_counts = tier_counts.set_index('Risk Tier').reindex(tier_order).dropna().reset_index()

        fig_pie = go.Figure(go.Pie(
            labels=tier_counts['Risk Tier'],
            values=tier_counts['Count'],
            hole=0.55,
            marker=dict(
                colors=["#ef4444", "#f97316", "#eab308", "#22c55e"],
                line=dict(color='#0f172a', width=2)
            ),
            textfont=dict(color='white', size=13),
            hovertemplate="<b>%{label}</b><br>Zones: %{value}<br>%{percent}<extra></extra>"
        ))
        fig_pie.add_annotation(
            text=f"<b>{len(df_nodes)}</b><br><span style='font-size:10px'>Total<br>Zones</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color='#e2e8f0'), align='center'
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#e2e8f0'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=12)),
            margin=dict(l=20, r=20, t=20, b=20),
            height=380
        )
        st.plotly_chart(fig_pie, use_container_width=True, key="pie_chart")

    # ── Row D: Critical Zones Table ──────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Critical & High-Risk Zones — Detailed Table</div>', unsafe_allow_html=True)

    df_alert = df_nodes[df_nodes['risk_tier'].isin(['Critical', 'High'])].copy()
    df_alert = df_alert[[
        'node_id', 'lat', 'lon', 'risk_tier',
        'avg_concentration', 'hotspot_ratio',
        'pagerank', 'betweenness', 'spread_risk_score', 'community'
    ]].rename(columns={
        'node_id':           'Grid Cell',
        'lat':               'Latitude',
        'lon':               'Longitude',
        'risk_tier':         'Risk Tier',
        'avg_concentration': 'Avg Conc (µg/L)',
        'hotspot_ratio':     'Hotspot %',
        'pagerank':          'PageRank',
        'betweenness':       'Betweenness',
        'spread_risk_score': 'Spread Risk Score',
        'community':         'Community'
    }).sort_values('Spread Risk Score', ascending=False).head(30)

    df_alert['Hotspot %'] = (df_alert['Hotspot %'] * 100).round(1).astype(str) + '%'
    df_alert['PageRank']  = df_alert['PageRank'].round(6)
    df_alert['Betweenness'] = df_alert['Betweenness'].round(4)
    df_alert['Spread Risk Score'] = df_alert['Spread Risk Score'].round(4)

    def style_risk(val):
        color_map_style = {'Critical': '#450a0a', 'High': '#431407'}
        bg = color_map_style.get(val, 'transparent')
        text = '#f87171' if val == 'Critical' else '#fb923c' if val == 'High' else '#e2e8f0'
        return f'background-color:{bg}; color:{text}; font-weight:700'

    styled = df_alert.style\
        .map(style_risk, subset=['Risk Tier'])\
        .format({'Spread Risk Score': '{:.4f}', 'PageRank': '{:.6f}', 'Betweenness': '{:.4f}'})\
        .background_gradient(subset=['Avg Conc (µg/L)'], cmap='YlOrRd')\
        .set_properties(**{'background-color': '#0f172a', 'color': '#e2e8f0', 'border': '1px solid #1e293b'})

    st.dataframe(styled, use_container_width=True, height=420)
