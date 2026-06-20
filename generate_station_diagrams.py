import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


GRAPH_TYPES = ("correlation", "top_k", "weighted", "learnable")
NON_FEATURE_COLUMNS = {"time", "element", "value"}


def load_station_summary(station_dir: Path) -> Dict:
    summary_path = station_dir / "reports" / "station_summary.json"
    if not summary_path.exists():
        return {}

    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_adjacency(adjacency_path: Path) -> pd.DataFrame:
    adjacency = pd.read_csv(adjacency_path, index_col=0)
    for col in adjacency.columns:
        for idx in adjacency.index:
            val = adjacency.at[idx, col]
            if isinstance(val, str):
                try:
                    float(val)
                except ValueError:
                    raise ValueError(f"Non-numeric value '{val}' found at cell [{idx}, {col}] in {adjacency_path}")
    adjacency = adjacency.apply(pd.to_numeric, errors="raise").fillna(0.0)

    labels = list(dict.fromkeys(list(adjacency.index) + list(adjacency.columns)))
    adjacency = adjacency.reindex(index=labels, columns=labels, fill_value=0.0)
    return adjacency


def load_node_stats(station_dir: Path, node_names: Iterable[str]) -> Dict[str, Dict[str, float]]:
    scaled_path = station_dir / "scaled" / "scaled.csv"
    cleaned_path = station_dir / "cleaned" / "cleaned.csv"
    data_path = scaled_path if scaled_path.exists() else cleaned_path

    if not data_path.exists():
        return {name: {} for name in node_names}

    data = pd.read_csv(data_path, usecols=lambda col: col not in NON_FEATURE_COLUMNS)
    stats: Dict[str, Dict[str, float]] = {}

    for name in node_names:
        if name not in data.columns:
            stats[name] = {}
            continue

        series = pd.to_numeric(data[name], errors="coerce").dropna()
        if series.empty:
            stats[name] = {}
            continue

        stats[name] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
        }

    return stats


def combined_undirected_edges(adjacency: pd.DataFrame) -> List[Tuple[str, str, float]]:
    labels = list(adjacency.index)
    edges: List[Tuple[str, str, float]] = []

    for i, source in enumerate(labels):
        for j in range(i + 1, len(labels)):
            target = labels[j]
            forward = float(adjacency.loc[source, target])
            reverse = float(adjacency.loc[target, source])

            if math.isclose(forward, 0.0, abs_tol=1e-12) and math.isclose(reverse, 0.0, abs_tol=1e-12):
                continue

            if not math.isclose(forward, 0.0, abs_tol=1e-12) and not math.isclose(reverse, 0.0, abs_tol=1e-12):
                weight = (forward + reverse) / 2.0
            else:
                weight = forward if not math.isclose(forward, 0.0, abs_tol=1e-12) else reverse

            edges.append((source, target, weight))

    return edges


def build_graph(adjacency: pd.DataFrame, graph_type: str) -> nx.Graph:
    is_directed = (graph_type == "top_k")
    graph = nx.DiGraph() if is_directed else nx.Graph()
    graph.add_nodes_from(adjacency.index)

    if is_directed:
        labels = list(adjacency.index)
        for source in labels:
            for target in labels:
                weight = float(adjacency.loc[source, target])
                if not math.isclose(weight, 0.0, abs_tol=1e-12):
                    graph.add_edge(source, target, weight=weight, abs_weight=abs(weight), sign="positive" if weight >= 0 else "negative")
    else:
        for source, target, weight in combined_undirected_edges(adjacency):
            graph.add_edge(source, target, weight=weight, abs_weight=abs(weight), sign="positive" if weight >= 0 else "negative")

    return graph


def clean_label(name: str) -> str:
    parts = name.split("_")
    lines: List[str] = []
    current = ""

    for part in parts:
        candidate = part if not current else f"{current}_{part}"
        if len(candidate) <= 18:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = part

    if current:
        lines.append(current)

    return "\n".join(lines)


def fmt(value: float, digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def node_feature_label(
    node: str,
    node_stats: Dict[str, Dict[str, float]],
    centralities: Dict[str, Dict[str, float]],
) -> str:
    stats = node_stats.get(node, {})
    centrality = centralities.get(node, {})

    return (
        f"{clean_label(node)}\n"
        f"mean {fmt(stats.get('mean', np.nan))} | std {fmt(stats.get('std', np.nan))}\n"
        f"deg {fmt(centrality.get('degree_centrality', np.nan))} | pg {fmt(centrality.get('pagerank', np.nan))}"
    )


def layout_graph(graph: nx.Graph) -> Dict[str, Tuple[float, float]]:
    if graph.number_of_nodes() == 1:
        return {next(iter(graph.nodes())): (0.0, 0.0)}

    if graph.number_of_nodes() <= 12 or graph.number_of_edges() == 0:
        return nx.circular_layout(graph)

    return nx.spring_layout(graph, seed=42, k=1.2, iterations=400, weight="abs_weight")


def draw_node_feature_annotations(
    ax: plt.Axes,
    pos: Dict[str, Tuple[float, float]],
    node_stats: Dict[str, Dict[str, float]],
    centralities: Dict[str, Dict[str, float]],
) -> None:
    for node, (x, y) in pos.items():
        stats = node_stats.get(node, {})
        centrality = centralities.get(node, {})
        text = (
            f"mean={fmt(stats.get('mean', np.nan))}, std={fmt(stats.get('std', np.nan))}\n"
            f"degree={fmt(centrality.get('degree_centrality', np.nan))}, pagerank={fmt(centrality.get('pagerank', np.nan))}"
        )

        offset_x = -0.18 if x >= 0 else 0.18
        offset_y = -0.18 if y >= 0 else 0.18
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(x + offset_x, y + offset_y),
            fontsize=8,
            color="#243447",
            ha="right" if x >= 0 else "left",
            va="top" if y >= 0 else "bottom",
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#cbd5e0", "alpha": 0.92},
            arrowprops={"arrowstyle": "-", "color": "#a0aec0", "lw": 0.8},
        )


def draw_node_feature_panel(
    ax: plt.Axes,
    node_names: List[str],
    node_stats: Dict[str, Dict[str, float]],
    centralities: Dict[str, Dict[str, float]],
) -> None:
    rows = []
    for node in node_names:
        stats = node_stats.get(node, {})
        centrality = centralities.get(node, {})
        rows.append(
            [
                node,
                fmt(stats.get("mean", np.nan)),
                fmt(stats.get("std", np.nan)),
                fmt(centrality.get("degree_centrality", np.nan)),
                fmt(centrality.get("pagerank", np.nan)),
            ]
        )

    ax.axis("off")
    ax.set_title("Node features", fontsize=12, fontweight="bold", loc="left", pad=8)
    table = ax.table(
        cellText=rows,
        colLabels=["node", "mean", "std", "degree", "pagerank"],
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.42, 0.14, 0.14, 0.14, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.25)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d8dee9")
        if row == 0:
            cell.set_facecolor("#edf2f7")
            cell.set_text_props(fontweight="bold", color="#243447")


def draw_edge_feature_panel(ax: plt.Axes, graph: nx.Graph) -> None:
    rows = []
    for source, target, data in sorted(graph.edges(data=True)):
        rows.append([source, target, fmt(data.get("weight", np.nan), 3), fmt(data.get("abs_weight", np.nan), 3), data.get("sign", "")])

    ax.axis("off")
    ax.set_title("Edge features", fontsize=12, fontweight="bold", loc="left", pad=8)

    if not rows:
        ax.text(0.0, 0.92, "No connected edges for this graph.\nIsolated nodes are still shown.", fontsize=9, va="top", color="#4a5568")
        return

    table = ax.table(
        cellText=rows,
        colLabels=["source", "target", "weight", "|weight|", "sign"],
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.30, 0.30, 0.14, 0.14, 0.12],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.15)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d8dee9")
        if row == 0:
            cell.set_facecolor("#edf2f7")
            cell.set_text_props(fontweight="bold", color="#243447")


def save_station_graph_diagram(
    station_name: str,
    graph_type: str,
    adjacency_path: Path,
    station_dir: Path,
    output_path: Path,
) -> None:
    adjacency = load_adjacency(adjacency_path)
    graph = build_graph(adjacency, graph_type)
    node_names = list(adjacency.index)
    node_stats = load_node_stats(station_dir, node_names)

    station_summary = load_station_summary(station_dir)
    graph_summary = station_summary.get("graph_structures", {}).get(graph_type, {})
    centralities = graph_summary.get("centralities", {})

    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    spec = fig.add_gridspec(nrows=2, ncols=2, width_ratios=[2.05, 1.35], height_ratios=[1.0, 1.0])
    graph_ax = fig.add_subplot(spec[:, 0])
    node_ax = fig.add_subplot(spec[0, 1])
    edge_ax = fig.add_subplot(spec[1, 1])

    pos = layout_graph(graph)
    isolated = set(nx.isolates(graph))
    connected_nodes = [node for node in graph.nodes if node not in isolated]

    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=connected_nodes,
        node_color="#3498db",
        edgecolors="#1f618d",
        linewidths=1.6,
        node_size=2400,
        alpha=0.92,
        ax=graph_ax,
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=list(isolated),
        node_color="#edf2f7",
        edgecolors="#718096",
        linewidths=1.8,
        node_size=2400,
        alpha=1.0,
        ax=graph_ax,
    )

    edges = list(graph.edges(data=True))
    if edges:
        max_weight = max(data["abs_weight"] for _, _, data in edges) or 1.0
        widths = [1.4 + 5.2 * (data["abs_weight"] / max_weight) for _, _, data in edges]
        edge_colors = ["#2b6cb0" if data["weight"] >= 0 else "#c53030" for _, _, data in edges]

        nx.draw_networkx_edges(
            graph,
            pos,
            width=widths,
            edge_color=edge_colors,
            alpha=0.70,
            arrows=isinstance(graph, nx.DiGraph),
            arrowsize=14,
            ax=graph_ax,
        )
        edge_labels = {(source, target): f"w={data['weight']:.2f}" for source, target, data in edges}
        nx.draw_networkx_edge_labels(
            graph,
            pos,
            edge_labels=edge_labels,
            font_size=8,
            font_color="#1a202c",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
            ax=graph_ax,
        )

    labels = {node: clean_label(node) for node in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8.5, font_weight="bold", font_color="#1a202c", ax=graph_ax)
    draw_node_feature_annotations(graph_ax, pos, node_stats, centralities)

    graph_ax.set_title(
        f"{station_name} - {graph_type} graph\nall {graph.number_of_nodes()} nodes kept, {graph.number_of_edges()} edges shown",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    graph_ax.text(
        0.01,
        0.01,
        "Blue edges = positive weight, red edges = negative weight. Gray nodes are isolated in this graph.",
        transform=graph_ax.transAxes,
        fontsize=9,
        color="#4a5568",
        va="bottom",
    )
    graph_ax.axis("off")
    graph_ax.set_aspect("equal")
    graph_ax.margins(0.28)

    draw_node_feature_panel(node_ax, node_names, node_stats, centralities)
    draw_edge_feature_panel(edge_ax, graph)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def iter_station_dirs(results_dir: Path) -> Iterable[Path]:
    for station_dir in sorted(results_dir.iterdir()):
        if station_dir.is_dir() and (station_dir / "graphs").is_dir():
            yield station_dir


def generate_diagrams(project_dir: Path, graph_types: Iterable[str]) -> List[Path]:
    results_dir = project_dir / "results"
    diagrams_dir = project_dir / "diagrams"
    generated: List[Path] = []

    for station_dir in iter_station_dirs(results_dir):
        station_name = station_dir.name
        for graph_type in graph_types:
            adjacency_path = station_dir / "graphs" / f"{graph_type}_adjacency.csv"
            if not adjacency_path.exists():
                continue

            output_path = diagrams_dir / station_name / f"{station_name}_{graph_type}_diagram.png"
            save_station_graph_diagram(station_name, graph_type, adjacency_path, station_dir, output_path)
            generated.append(output_path)

    return generated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-station graph diagrams that keep isolated nodes and show node/edge features.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project directory containing the results folder.",
    )
    parser.add_argument(
        "--graph-types",
        nargs="+",
        default=list(GRAPH_TYPES),
        choices=list(GRAPH_TYPES),
        help="Graph types to render.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = args.project_dir.resolve()
    generated = generate_diagrams(project_dir, args.graph_types)

    print(f"Generated {len(generated)} diagram(s) in {project_dir / 'diagrams'}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
