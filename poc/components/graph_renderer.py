import networkx as nx
import plotly.graph_objects as go


def _get_node_color(node, start_events, end_events):
    if node in start_events:
        return "#2ecc71"  # green
    if node in end_events:
        return "#e74c3c"  # red
    return "#3498db"  # blue


def render_event_graph(
    event_graph,
    idx_to_event: dict,
    width: int = 800,
    height: int = 600,
) -> go.Figure:
    """Render an EventGraph as a plotly figure with colored nodes and edge labels.

    Args:
        event_graph: EventGraph instance with .graph (nx.DiGraph), .start_events, .end_events.
        idx_to_event: Mapping from event index to display name.
        width: Figure width in pixels.
        height: Figure height in pixels.

    Returns:
        plotly Figure.
    """
    G = event_graph.graph
    start_events = set(event_graph.start_events)
    end_events = set(event_graph.end_events)

    # Use hierarchical layout via graphviz if available, else spring layout
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=2.0)

    # Edge traces
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="#888"),
        hoverinfo="none",
    )

    # Edge label annotations (midpoint of each edge)
    annotations = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get("weight", 0)
        annotations.append(
            dict(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                text=f"{weight:.2f}",
                showarrow=False,
                font=dict(size=10, color="#555"),
                bgcolor="rgba(255,255,255,0.8)",
            )
        )

    # Node trace
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        label = idx_to_event.get(node, str(node))
        successors = event_graph.get_successors(node)
        succ_text = "<br>".join(
            f"  -> {idx_to_event.get(t, t)}: {p:.3f}" for t, p in successors.items()
        )
        hover = f"<b>{label}</b><br>{succ_text}" if succ_text else f"<b>{label}</b>"
        node_text.append(hover)
        node_colors.append(_get_node_color(node, start_events, end_events))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[idx_to_event.get(n, str(n)) for n in G.nodes()],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(size=30, color=node_colors, line=dict(width=2, color="white")),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            width=width,
            height=height,
            showlegend=False,
            hovermode="closest",
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=annotations,
            title="Process Event Graph",
        ),
    )
    return fig
