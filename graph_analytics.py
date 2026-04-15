"""
graph_analytics.py
------------------
Builds a spatial pollution propagation network from ocean sensor data
and computes graph-theoretic metrics to identify systemic high-risk zones.

Graph Model:
  - Nodes  : Grid cells (rounded lat/lon at 5° resolution)
  - Edges  : Two cells are connected if they are within ~5° of each other
             (proximity = shared current corridor)
  - Weight : Average microplastic concentration between connected cells

Computed metrics:
  - PageRank          : "Influence Score" — cells connected to many other
                        high-concentration nodes rank higher.
  - Betweenness       : Cells that act as bridges in the pollution network.
  - Degree Centrality : How connected each cell is to its neighbors.
  - Community         : Louvain community detection (graph-native clusters).
  - Spread Risk Score : Composite score combining concentration, PageRank,
                        and betweenness for actionable risk ranking.
"""

import json
import os
import math
import networkx as nx
import community as community_louvain  # python-louvain
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
PREDICTIONS_PATH = "dashboard_data/predictions.json"
OUTPUT_PATH      = "dashboard_data/graph_metrics.json"
GRID_RESOLUTION  = 5          # degrees — coarser grid = fewer nodes, cleaner graph
PROXIMITY_THRESH = 8          # degrees — cells within this range get an edge
MAX_EDGES_PER_NODE = 6        # cap to keep graph readable


def haversine_deg(lat1, lon1, lat2, lon2):
    """Great-circle distance in degrees (approx using Euclidean on lat/lon)."""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    return math.sqrt(dlat**2 + dlon**2)


def build_graph(df_grid):
    """
    Build a weighted undirected graph from grid-cell aggregates.
    Nodes  = (grid_lat, grid_lon)
    Edges  = proximity within PROXIMITY_THRESH degrees
    Weight = average of the two cells' microplastic concentration
    """
    G = nx.Graph()

    # Add nodes with attributes
    for _, row in df_grid.iterrows():
        node_id = f"{row['grid_lat']:.1f}_{row['grid_lon']:.1f}"
        G.add_node(node_id,
                   lat=row['grid_lat'],
                   lon=row['grid_lon'],
                   avg_concentration=row['avg_concentration'],
                   hotspot_ratio=row['hotspot_ratio'],
                   point_count=row['count'],
                   cluster=int(row['cluster_mode']))

    nodes = list(G.nodes(data=True))

    # Add edges based on spatial proximity; cap per-node degree
    for i, (n1, d1) in enumerate(nodes):
        neighbors_added = 0
        # Sort candidates by distance
        candidates = []
        for j, (n2, d2) in enumerate(nodes):
            if i == j:
                continue
            dist = haversine_deg(d1['lat'], d1['lon'], d2['lat'], d2['lon'])
            if dist <= PROXIMITY_THRESH:
                candidates.append((dist, n2, d2))

        # Sort by distance, keep closest MAX_EDGES_PER_NODE
        candidates.sort(key=lambda x: x[0])
        for dist, n2, d2 in candidates[:MAX_EDGES_PER_NODE]:
            if not G.has_edge(n1, n2):
                avg_weight = (d1['avg_concentration'] + d2['avg_concentration']) / 2
                G.add_edge(n1, n2, weight=avg_weight, distance=dist)

    return G


def compute_metrics(G):
    """Compute all graph analytics metrics on the graph G."""
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # PageRank — weighted by microplastic concentration on edges
    pagerank = nx.pagerank(G, weight='weight', alpha=0.85)

    # Betweenness centrality — which nodes are bridge nodes?
    betweenness = nx.betweenness_centrality(G, weight='weight', normalized=True)

    # Degree centrality
    degree_cent = nx.degree_centrality(G)

    # Community detection (Louvain)
    try:
        partition = community_louvain.best_partition(G, weight='weight', random_state=42)
    except Exception:
        # Fallback: connected components as communities
        partition = {}
        for i, comp in enumerate(nx.connected_components(G)):
            for n in comp:
                partition[n] = i

    # Clustering coefficient — how tightly a node's neighbours are connected
    clustering = nx.clustering(G, weight='weight')

    # Assemble per-node result
    results = []
    for node, data in G.nodes(data=True):
        pr    = pagerank.get(node, 0)
        bt    = betweenness.get(node, 0)
        dc    = degree_cent.get(node, 0)
        cl    = clustering.get(node, 0)
        conc  = data['avg_concentration']
        hr    = data['hotspot_ratio']

        # Spread Risk Score: normalized composite
        # High concentration + high PageRank + high betweenness = systemic risk
        spread_risk = (0.40 * conc / 100.0 +   # concentration (normalised ~0-1)
                       0.30 * pr * 10 +          # pagerank scaled
                       0.20 * bt +               # betweenness
                       0.10 * dc)                # connectivity

        results.append({
            "node_id":           node,
            "lat":               round(data['lat'], 2),
            "lon":               round(data['lon'], 2),
            "avg_concentration": round(conc, 3),
            "hotspot_ratio":     round(hr, 3),
            "point_count":       data['point_count'],
            "cluster":           data['cluster'],
            "pagerank":          round(pr, 6),
            "betweenness":       round(bt, 6),
            "degree_centrality": round(dc, 4),
            "clustering_coeff":  round(cl, 4),
            "community":         int(partition.get(node, 0)),
            "spread_risk_score": round(spread_risk, 6),
        })

    # Sort by spread risk descending
    results.sort(key=lambda x: x['spread_risk_score'], reverse=True)

    # Assign risk tier labels
    n = len(results)
    for i, r in enumerate(results):
        pct = i / max(n - 1, 1)
        if pct < 0.15:
            r['risk_tier'] = "Critical"
        elif pct < 0.35:
            r['risk_tier'] = "High"
        elif pct < 0.60:
            r['risk_tier'] = "Moderate"
        else:
            r['risk_tier'] = "Low"

    return results


def build_edge_list(G, node_results_map):
    """Return edge list enriched with risk info."""
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            "source":     u,
            "target":     v,
            "weight":     round(data.get('weight', 0), 3),
            "distance":   round(data.get('distance', 0), 2),
            "source_lat": node_results_map[u]['lat'],
            "source_lon": node_results_map[u]['lon'],
            "target_lat": node_results_map[v]['lat'],
            "target_lon": node_results_map[v]['lon'],
            "source_risk": node_results_map[u]['risk_tier'],
            "target_risk": node_results_map[v]['risk_tier'],
        })
    return edges


def compute_graph_summary(node_results, edge_list, G):
    """High-level summary stats for the dashboard KPI cards."""
    critical = [n for n in node_results if n['risk_tier'] == "Critical"]
    high     = [n for n in node_results if n['risk_tier'] == "High"]

    # Number of communities
    communities = len(set(n['community'] for n in node_results))

    # Most influential node (highest PageRank)
    top_pr  = max(node_results, key=lambda x: x['pagerank'])
    top_bt  = max(node_results, key=lambda x: x['betweenness'])
    top_rs  = node_results[0]  # already sorted by spread_risk_score

    # Average path length (largest connected component)
    if nx.is_connected(G):
        avg_path = nx.average_shortest_path_length(G, weight=None)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        sub = G.subgraph(largest_cc)
        avg_path = nx.average_shortest_path_length(sub, weight=None)

    # Pollution Bridge nodes = top 10% by betweenness AND above-median concentration
    med_conc = np.median([n['avg_concentration'] for n in node_results])
    bridges  = [n for n in node_results
                if n['betweenness'] > np.percentile([x['betweenness'] for x in node_results], 80)
                and n['avg_concentration'] > med_conc]

    return {
        "total_nodes":         len(node_results),
        "total_edges":         G.number_of_edges(),
        "critical_zones":      len(critical),
        "high_risk_zones":     len(high),
        "graph_communities":   communities,
        "avg_path_length":     round(avg_path, 2),
        "top_influence_node":  top_pr['node_id'],
        "top_bridge_node":     top_bt['node_id'],
        "top_risk_node":       top_rs['node_id'],
        "bridge_count":        len(bridges),
        "bridge_nodes":        [b['node_id'] for b in bridges[:5]],
        "graph_density":       round(nx.density(G), 4),
    }


def run_graph_analytics():
    print("=" * 60)
    print("  Graph Analytics — Ocean Pollution Network")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────
    if not os.path.exists(PREDICTIONS_PATH):
        print(f"ERROR: {PREDICTIONS_PATH} not found. Run pipeline.py first.")
        return

    with open(PREDICTIONS_PATH) as f:
        raw = json.load(f)

    points = raw['points']
    print(f"  Loaded {len(points)} sensor points")

    df = pd.DataFrame(points)

    # ── Aggregate to grid cells ─────────────────────────────────
    df['grid_lat'] = (df['Latitude'] / GRID_RESOLUTION).round() * GRID_RESOLUTION
    df['grid_lon'] = (df['Longitude'] / GRID_RESOLUTION).round() * GRID_RESOLUTION

    df_grid = df.groupby(['grid_lat', 'grid_lon']).agg(
        avg_concentration=('Microplastic_Concentration', 'mean'),
        hotspot_ratio=('Hotspot', 'mean'),
        count=('Microplastic_Concentration', 'count'),
        cluster_mode=('cluster', lambda x: x.mode()[0])
    ).reset_index()

    print(f"  Aggregated to {len(df_grid)} grid cells (5° resolution)")

    # ── Build graph ─────────────────────────────────────────────
    print("  Building spatial proximity graph...")
    G = build_graph(df_grid)

    # ── Compute metrics ─────────────────────────────────────────
    print("  Computing graph metrics (PageRank, Betweenness, Community)...")
    node_results = compute_metrics(G)
    node_map     = {n['node_id']: n for n in node_results}

    # ── Build edge list ─────────────────────────────────────────
    edge_list = build_edge_list(G, node_map)

    # ── Summary ─────────────────────────────────────────────────
    print("  Computing network summary statistics...")
    summary = compute_graph_summary(node_results, edge_list, G)

    # ── Save output ─────────────────────────────────────────────
    output = {
        "summary":  summary,
        "nodes":    node_results,
        "edges":    edge_list,
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n  [OK] Graph metrics saved -> {OUTPUT_PATH}")
    print(f"  Critical zones : {summary['critical_zones']}")
    print(f"  High-risk zones: {summary['high_risk_zones']}")
    print(f"  Communities    : {summary['graph_communities']}")
    print(f"  Top risk node  : {summary['top_risk_node']}")
    print(f"  Avg path length: {summary['avg_path_length']} hops")
    print("=" * 60)


if __name__ == "__main__":
    run_graph_analytics()
