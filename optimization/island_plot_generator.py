#!/usr/bin/env python3
"""
Island GA Plot Generator

Generates standardized plots for island model genetic algorithm results:
- Combined plot (fitness + diversity)
- Convergence plot (per-island fitness trajectories)
- Diversity plot (island spread over time)
- Per-island diversity plot (stacked area of fitness std)
- Decision tree visualization (from best chromosome)

Usage:
    from optimization.island_plot_generator import generate_island_plots
    generate_island_plots('/path/to/results', run_name='exp_simple_very_high_ms70')
"""

import json
import matplotlib
matplotlib.use('Agg')  # Headless backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional, Dict, List, Any


# Standard color scheme
COLORS = {
    'oahu': '#2196F3',       # Blue
    'maui': '#FF9800',       # Orange
    'big_island': '#4CAF50', # Green
    'spread': '#9C27B0',     # Purple
}

# Feature name mapping for decision tree visualization
FEATURE_NAMES = {
    0: 'observed_rul',
    1: 'hours_to_major',
    2: 'hours_to_minor',
    3: 'da_line_dev'  # Signed: positive=ahead, negative=behind
}

# Fleet feature names for medium/full configs
FLEET_FEATURE_NAMES = {
    0: 'mean_observed_rul',
    1: 'min_observed_rul'
}

# Bucket labels
BUCKET_LABELS = {
    0: 'Bucket 1\n(Preventive)',
    1: 'Bucket 2\n(Wait)',
    2: 'Bucket 3\n(Phase Minor)',
    3: 'Bucket 4\n(Phase Major)',
    4: 'Bucket 5',
    5: 'Bucket 6',
    6: 'Bucket 7',
    7: 'Bucket 8'
}

# Colors for tree nodes
TREE_COLORS = {
    'decision': '#BBDEFB',      # Light blue for decision nodes
    'decision_deep': '#64B5F6', # Darker blue for deeper nodes
    'leaf_1': '#FFCDD2',        # Red - Preventive
    'leaf_2': '#C8E6C9',        # Green - Wait
    'leaf_3': '#FFE0B2',        # Orange - Phase Minor
    'leaf_4': '#E1BEE7',        # Purple - Phase Major
    'context': '#FFF9C4',       # Yellow - Context node
}


def generate_island_plots(results_dir: str, run_name: Optional[str] = None) -> None:
    """
    Generate all standard island GA plots from results directory.

    Args:
        results_dir: Path to results directory containing island_history.json and summary_report.json
        run_name: Optional run name for plot titles (defaults to directory name)
    """
    results_path = Path(results_dir)

    # Load data
    history_file = results_path / 'island_history.json'
    summary_file = results_path / 'summary_report.json'

    if not history_file.exists():
        print(f"Warning: {history_file} not found, skipping plot generation")
        return

    with open(history_file, 'r') as f:
        history = json.load(f)

    summary = {}
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)

    # Use directory name if run_name not provided
    if run_name is None:
        run_name = results_path.name

    # Create plots directory
    plots_dir = results_path / 'plots'
    plots_dir.mkdir(exist_ok=True)

    # Extract data
    generations = [h['generation'] for h in history]
    oahu = [h['oahu_best'] for h in history]
    maui = [h['maui_best'] for h in history]
    big_island = [h['big_island_best'] for h in history]

    # Calculate island spread (diversity metric)
    island_spread = []
    for h in history:
        values = [h['oahu_best'], h['maui_best'], h['big_island_best']]
        spread = max(values) - min(values)
        island_spread.append(spread)

    # Generate all plots
    _generate_combined_plot(plots_dir, generations, oahu, maui, big_island, island_spread, run_name)
    _generate_convergence_plot(plots_dir, generations, oahu, maui, big_island, run_name)
    _generate_diversity_plot(plots_dir, generations, island_spread, run_name)
    _generate_per_island_diversity_plot(plots_dir, history, generations, run_name)

    # Generate decision tree plot from best chromosome
    _generate_decision_tree_plot(results_path, plots_dir, run_name, summary)

    print(f"Plots generated in {plots_dir}")


def _generate_combined_plot(plots_dir: Path, generations: list, oahu: list, maui: list,
                            big_island: list, island_spread: list, run_name: str) -> None:
    """Generate 2-panel combined plot: fitness + diversity."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    fig.suptitle(f'Island GA - {run_name}', fontsize=14, fontweight='bold')

    # Top panel: Best fitness per island
    ax1.plot(generations, oahu, '-', color=COLORS['oahu'], linewidth=1.5, label='Oahu')
    ax1.plot(generations, maui, '-', color=COLORS['maui'], linewidth=1.5, label='Maui')
    ax1.plot(generations, big_island, '-', color=COLORS['big_island'], linewidth=1.5, label='Big Island')

    ax1.set_ylabel('Best Fitness', fontsize=12)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.set_xlim(0, max(generations))

    # Bottom panel: Island spread
    ax2.plot(generations, island_spread, '-', color=COLORS['spread'], linewidth=1.5)
    ax2.fill_between(generations, island_spread, alpha=0.3, color=COLORS['spread'])

    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Island Spread', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.set_xlim(0, max(generations))

    plt.tight_layout()
    fig.savefig(plots_dir / 'combined_plot.png', dpi=150, bbox_inches='tight')
    plt.close()


def _generate_convergence_plot(plots_dir: Path, generations: list, oahu: list,
                               maui: list, big_island: list, run_name: str) -> None:
    """Generate convergence plot showing per-island fitness trajectories."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(generations, oahu, '-', color=COLORS['oahu'], linewidth=1.5, label='Oahu')
    ax.plot(generations, maui, '-', color=COLORS['maui'], linewidth=1.5, label='Maui')
    ax.plot(generations, big_island, '-', color=COLORS['big_island'], linewidth=1.5, label='Big Island')

    ax.set_title(f'Island GA - {run_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Best Fitness', fontsize=12)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    fig.savefig(plots_dir / 'convergence_plot.png', dpi=150, bbox_inches='tight')
    plt.close()


def _generate_diversity_plot(plots_dir: Path, generations: list,
                             island_spread: list, run_name: str) -> None:
    """Generate diversity plot showing island spread over time."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(generations, island_spread, '-', color=COLORS['spread'], linewidth=1.5)
    ax.fill_between(generations, island_spread, alpha=0.3, color=COLORS['spread'])

    ax.set_title(f'Island Diversity - {run_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Island Spread (Diversity)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.tight_layout()
    fig.savefig(plots_dir / 'diversity_plot.png', dpi=150, bbox_inches='tight')
    plt.close()


def _generate_per_island_diversity_plot(plots_dir: Path, history: list,
                                        generations: list, run_name: str) -> None:
    """Generate per-island diversity plot (stacked area chart of fitness std)."""
    # Extract per-island std (with fallback for old data without std tracking)
    oahu_std = [h.get('oahu_std', 0) for h in history]
    maui_std = [h.get('maui_std', 0) for h in history]
    big_island_std = [h.get('big_island_std', 0) for h in history]

    # Only generate if we have std data
    if not (any(oahu_std) or any(maui_std) or any(big_island_std)):
        print(f"  - per_island_diversity.png (skipped - no std data)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.stackplot(generations,
                 oahu_std, maui_std, big_island_std,
                 labels=['Oahu', 'Maui', 'Big Island'],
                 colors=[COLORS['oahu'], COLORS['maui'], COLORS['big_island']],
                 alpha=0.7)

    ax.set_title(f'Per-Island Diversity - {run_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness Std (stacked)', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.set_xlim(0, max(generations))

    plt.tight_layout()
    fig.savefig(plots_dir / 'per_island_diversity.png', dpi=150, bbox_inches='tight')
    plt.close()


def _generate_decision_tree_plot(results_path: Path, plots_dir: Path,
                                  run_name: str, summary: Dict[str, Any]) -> None:
    """Generate decision tree visualization from best chromosome.

    Uses graphviz for high-quality output matching paper style.
    Falls back to matplotlib if graphviz is not available.

    Handles Simple, Medium, and Full configurations:
    - Simple: Single depth-3 tree (7 decision nodes → 8 leaves)
    - Medium: Fleet context layer + 2 subtrees
    - Full: Fleet context + 4 subtrees + early phase window
    """
    # Find best_chromosome.json (could be in root or chromosomes subdir)
    chromosome_file = results_path / 'best_chromosome.json'
    if not chromosome_file.exists():
        chromosome_file = results_path / 'chromosomes' / 'best_chromosome.json'

    if not chromosome_file.exists():
        print(f"  - decision_tree.png (skipped - no best_chromosome.json)")
        return

    with open(chromosome_file, 'r') as f:
        chromosome = json.load(f)

    config_type = chromosome.get('config_type', 'simple')

    # Try graphviz first (better quality), fall back to matplotlib
    try:
        from graphviz import Digraph
        use_graphviz = True
    except ImportError:
        use_graphviz = False
        print("  - graphviz not installed, using matplotlib fallback")

    if use_graphviz:
        if config_type == 'simple':
            _plot_simple_tree_graphviz(chromosome, plots_dir, run_name, summary)
        elif config_type == 'medium':
            _plot_medium_tree_graphviz(chromosome, plots_dir, run_name, summary)
        elif config_type == 'full':
            _plot_full_tree_graphviz(chromosome, plots_dir, run_name, summary)
        else:
            print(f"  - decision_tree.png (skipped - unknown config_type: {config_type})")
    else:
        # Matplotlib fallback
        if config_type == 'simple':
            _plot_simple_tree(chromosome, plots_dir, run_name, summary)
        elif config_type == 'medium':
            _plot_medium_tree(chromosome, plots_dir, run_name, summary)
        elif config_type == 'full':
            _plot_full_tree(chromosome, plots_dir, run_name, summary)
        else:
            print(f"  - decision_tree.png (skipped - unknown config_type: {config_type})")


# ==============================================================================
# GRAPHVIZ IMPLEMENTATIONS (preferred - matches paper style)
# ==============================================================================

def _plot_simple_tree_graphviz(chromosome: Dict[str, Any], plots_dir: Path,
                                run_name: str, summary: Dict[str, Any]) -> None:
    """Plot simple config decision tree using graphviz."""
    from graphviz import Digraph

    feature_indices = chromosome['feature_indices']
    thresholds = chromosome['thresholds']
    tree_depth = chromosome.get('tree_depth', 3)
    n_decision_nodes = 2 ** tree_depth - 1

    # Extract metrics for title
    fitness = summary.get('optimization_summary', {}).get('best_fitness', 'N/A')
    ms = summary.get('best_chromosome', {}).get('mission_success', 'N/A')
    or_val = summary.get('best_chromosome', {}).get('operational_readiness', 'N/A')
    best_gen = summary.get('optimization_summary', {}).get('best_generation', 'N/A')

    if isinstance(fitness, float):
        fitness = f"{fitness:.4f}"
    if isinstance(ms, float):
        ms = f"{ms*100:.1f}%"
    if isinstance(or_val, float):
        or_val = f"{or_val*100:.1f}%"

    dot = Digraph('Decision_Tree_Policy', comment=f'Learned Policy - {run_name}')

    # Global styling (matching paper style)
    dot.attr(rankdir='TB', splines='ortho')
    dot.attr(bgcolor='white', fontname='Helvetica', dpi='300')
    dot.attr(size='12,10')
    dot.attr(nodesep='0.5', ranksep='0.8')

    dot.attr('node', shape='box', style='rounded,filled',
             fontname='Helvetica', fontsize='10', penwidth='2.0')
    dot.attr('edge', color='#555555', penwidth='1.5', arrowsize='0.7',
             fontsize='9', fontcolor='#333333')

    # Title node
    title_text = f"""LEARNED POLICY - {run_name.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fitness: {fitness} | Gen: {best_gen}
MS: {ms} | OR: {or_val}"""

    dot.node('title', label=title_text, shape='box', style='filled',
             fillcolor='#E8F5E9', fontsize='11', fontcolor='black', penwidth='2.5')

    # Node colors by depth
    depth_colors = ['#BBDEFB', '#90CAF9', '#64B5F6']
    leaf_color = '#42A5F5'

    # Decision nodes
    for node_idx in range(n_decision_nodes):
        feat_idx = feature_indices[node_idx]
        threshold = thresholds[node_idx]
        feat_name = FEATURE_NAMES.get(feat_idx, f'var[{feat_idx}]')
        depth = _get_depth(node_idx)

        label = f"Node {node_idx}\n{feat_name}\n≤ {threshold:.1f}?"
        fontcolor = 'white' if depth >= 2 else 'black'

        dot.node(f'n{node_idx}', label=label,
                 fillcolor=depth_colors[min(depth, len(depth_colors)-1)],
                 fontcolor=fontcolor)

    # Leaf nodes (buckets)
    n_leaves = 2 ** tree_depth
    for leaf_idx in range(n_leaves):
        dot.node(f'leaf{leaf_idx}', label=f'Bucket {leaf_idx + 1}',
                 fillcolor=leaf_color, fontcolor='white')

    # Edges
    dot.edge('title', 'n0', style='invis')

    for node_idx in range(n_decision_nodes):
        left_child = 2 * node_idx + 1
        right_child = 2 * node_idx + 2

        if left_child < n_decision_nodes:
            dot.edge(f'n{node_idx}', f'n{left_child}', label='T', fontcolor='#2E7D32')
        elif left_child - n_decision_nodes < n_leaves:
            dot.edge(f'n{node_idx}', f'leaf{left_child - n_decision_nodes}', label='T', fontcolor='#2E7D32')

        if right_child < n_decision_nodes:
            dot.edge(f'n{node_idx}', f'n{right_child}', label='F', fontcolor='#C62828')
        elif right_child - n_decision_nodes < n_leaves:
            dot.edge(f'n{node_idx}', f'leaf{right_child - n_decision_nodes}', label='F', fontcolor='#C62828')

    # Render
    output_path = plots_dir / 'decision_tree'
    try:
        dot.render(str(output_path), format='png', cleanup=True)
        print(f"  - decision_tree.png (simple config, graphviz)")
    except Exception as e:
        print(f"  - decision_tree.png (graphviz error: {e}, trying matplotlib)")
        _plot_simple_tree(chromosome, plots_dir, run_name, summary)


def _plot_medium_tree_graphviz(chromosome: Dict[str, Any], plots_dir: Path,
                                run_name: str, summary: Dict[str, Any]) -> None:
    """Plot medium config decision tree using graphviz."""
    from graphviz import Digraph

    fleet_indices = chromosome.get('fleet_variable_indices', [])
    fleet_thresholds = chromosome.get('fleet_thresholds', [])
    context_subtrees = chromosome.get('context_subtrees', [])

    # Extract metrics
    fitness = summary.get('optimization_summary', {}).get('best_fitness', 'N/A')
    ms = summary.get('best_chromosome', {}).get('mission_success', 'N/A')
    or_val = summary.get('best_chromosome', {}).get('operational_readiness', 'N/A')
    best_gen = summary.get('optimization_summary', {}).get('best_generation', 'N/A')

    if isinstance(fitness, float):
        fitness = f"{fitness:.4f}"
    if isinstance(ms, float):
        ms = f"{ms*100:.1f}%"
    if isinstance(or_val, float):
        or_val = f"{or_val*100:.1f}%"

    dot = Digraph('Decision_Tree_Policy_Medium', comment=f'Medium Config - {run_name}')

    dot.attr(rankdir='TB', splines='ortho')
    dot.attr(bgcolor='white', fontname='Helvetica', dpi='300')
    dot.attr(size='16,12')
    dot.attr(nodesep='0.4', ranksep='0.7')
    dot.attr('node', shape='box', style='rounded,filled',
             fontname='Helvetica', fontsize='9', penwidth='2.0')
    dot.attr('edge', color='#555555', penwidth='1.5', arrowsize='0.6')

    # Title
    title_text = f"""LEARNED POLICY (MEDIUM) - {run_name.upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fitness: {fitness} | Gen: {best_gen} | MS: {ms} | OR: {or_val}"""

    dot.node('title', label=title_text, shape='box', style='filled',
             fillcolor='#E8F5E9', fontsize='11', penwidth='2.5')

    # Fleet context node
    if fleet_indices:
        feat_idx = fleet_indices[0]
        threshold = fleet_thresholds[0] if fleet_thresholds else 0
        feat_name = FLEET_FEATURE_NAMES.get(feat_idx, f'fleet[{feat_idx}]')

        dot.node('context0', label=f"FLEET CONTEXT\n{feat_name}\n≤ {threshold:.1f}?",
                 fillcolor='#FFF9C4', fontcolor='black', fontsize='10', penwidth='2.5')

        dot.edge('title', 'context0', style='invis')

    # Create subgraphs for left and right subtrees
    for subtree_idx, subtree in enumerate(context_subtrees[:2]):
        subtree_name = 'Left' if subtree_idx == 0 else 'Right'
        prefix = f's{subtree_idx}_'

        with dot.subgraph(name=f'cluster_{subtree_idx}') as sub:
            sub.attr(label=f'{subtree_name} Subtree (Context={subtree_idx==0})',
                     style='rounded', bgcolor='#FAFAFA')

            if subtree:
                _add_subtree_to_graph(sub, subtree, prefix)

        # Connect context to subtree root
        edge_label = 'True' if subtree_idx == 0 else 'False'
        edge_color = '#2E7D32' if subtree_idx == 0 else '#C62828'
        dot.edge('context0', f'{prefix}n0', label=edge_label, fontcolor=edge_color)

    output_path = plots_dir / 'decision_tree'
    try:
        dot.render(str(output_path), format='png', cleanup=True)
        print(f"  - decision_tree.png (medium config, graphviz)")
    except Exception as e:
        print(f"  - decision_tree.png (graphviz error: {e}, trying matplotlib)")
        _plot_medium_tree(chromosome, plots_dir, run_name, summary)


def _plot_full_tree_graphviz(chromosome: Dict[str, Any], plots_dir: Path,
                              run_name: str, summary: Dict[str, Any]) -> None:
    """Plot full config decision tree using graphviz."""
    from graphviz import Digraph

    fleet_indices = chromosome.get('fleet_variable_indices', [])
    fleet_thresholds = chromosome.get('fleet_thresholds', [])
    context_subtrees = chromosome.get('context_subtrees', [])

    fitness = summary.get('optimization_summary', {}).get('best_fitness', 'N/A')
    ms = summary.get('best_chromosome', {}).get('mission_success', 'N/A')
    or_val = summary.get('best_chromosome', {}).get('operational_readiness', 'N/A')
    best_gen = summary.get('optimization_summary', {}).get('best_generation', 'N/A')

    if isinstance(fitness, float):
        fitness = f"{fitness:.4f}"
    if isinstance(ms, float):
        ms = f"{ms*100:.1f}%"
    if isinstance(or_val, float):
        or_val = f"{or_val*100:.1f}%"

    dot = Digraph('Decision_Tree_Policy_Full', comment=f'Full Config - {run_name}')

    dot.attr(rankdir='TB', splines='ortho')
    dot.attr(bgcolor='white', fontname='Helvetica', dpi='300')
    dot.attr(size='20,14')
    dot.attr(nodesep='0.3', ranksep='0.6')
    dot.attr('node', shape='box', style='rounded,filled',
             fontname='Helvetica', fontsize='8', penwidth='1.5')
    dot.attr('edge', color='#555555', penwidth='1.2', arrowsize='0.5')

    # Title
    title_text = f"""LEARNED POLICY (FULL) - {run_name.upper()}
Fitness: {fitness} | Gen: {best_gen} | MS: {ms} | OR: {or_val}"""

    dot.node('title', label=title_text, shape='box', style='filled',
             fillcolor='#E8F5E9', fontsize='10', penwidth='2.0')

    # Context layer (2 levels, 3 nodes)
    context_colors = ['#FFF9C4', '#FFECB3']
    if len(fleet_indices) >= 1:
        feat_name = FLEET_FEATURE_NAMES.get(fleet_indices[0], f'fleet[{fleet_indices[0]}]')
        dot.node('c0', label=f"Context L0\n{feat_name}\n≤ {fleet_thresholds[0]:.1f}?",
                 fillcolor=context_colors[0], penwidth='2.0')
        dot.edge('title', 'c0', style='invis')

    if len(fleet_indices) >= 2:
        feat_name = FLEET_FEATURE_NAMES.get(fleet_indices[1], f'fleet[{fleet_indices[1]}]')
        dot.node('c1', label=f"Context L1-L\n{feat_name}\n≤ {fleet_thresholds[1]:.1f}?",
                 fillcolor=context_colors[1])
        dot.edge('c0', 'c1', label='T', fontcolor='#2E7D32')

    if len(fleet_indices) >= 3:
        feat_name = FLEET_FEATURE_NAMES.get(fleet_indices[2], f'fleet[{fleet_indices[2]}]')
        dot.node('c2', label=f"Context L1-R\n{feat_name}\n≤ {fleet_thresholds[2]:.1f}?",
                 fillcolor=context_colors[1])
        dot.edge('c0', 'c2', label='F', fontcolor='#C62828')

    # 4 subtrees
    subtree_labels = ['TT', 'TF', 'FT', 'FF']
    context_parents = ['c1', 'c1', 'c2', 'c2']
    context_edges = ['T', 'F', 'T', 'F']

    for subtree_idx, subtree in enumerate(context_subtrees[:4]):
        prefix = f's{subtree_idx}_'
        label = subtree_labels[subtree_idx]

        with dot.subgraph(name=f'cluster_{subtree_idx}') as sub:
            sub.attr(label=f'Subtree {subtree_idx+1} ({label})',
                     style='rounded', bgcolor='#FAFAFA')

            if subtree:
                _add_subtree_to_graph(sub, subtree, prefix)

        edge_color = '#2E7D32' if context_edges[subtree_idx] == 'T' else '#C62828'
        if subtree_idx < 2 and len(fleet_indices) >= 2:
            dot.edge(context_parents[subtree_idx], f'{prefix}n0',
                     label=context_edges[subtree_idx], fontcolor=edge_color)
        elif subtree_idx >= 2 and len(fleet_indices) >= 3:
            dot.edge(context_parents[subtree_idx], f'{prefix}n0',
                     label=context_edges[subtree_idx], fontcolor=edge_color)

    output_path = plots_dir / 'decision_tree'
    try:
        dot.render(str(output_path), format='png', cleanup=True)
        print(f"  - decision_tree.png (full config, graphviz)")
    except Exception as e:
        print(f"  - decision_tree.png (graphviz error: {e}, trying matplotlib)")
        _plot_full_tree(chromosome, plots_dir, run_name, summary)


def _add_subtree_to_graph(graph, subtree: Dict[str, Any], prefix: str) -> None:
    """Add a subtree's nodes and edges to a graphviz subgraph."""
    feature_indices = subtree.get('feature_indices', [])
    thresholds = subtree.get('thresholds', [])
    tree_depth = subtree.get('tree_depth', 3)
    n_decision_nodes = 2 ** tree_depth - 1

    depth_colors = ['#BBDEFB', '#90CAF9', '#64B5F6']
    leaf_color = '#42A5F5'

    # Decision nodes
    for node_idx in range(min(n_decision_nodes, len(feature_indices))):
        feat_idx = feature_indices[node_idx]
        threshold = thresholds[node_idx] if node_idx < len(thresholds) else 0
        feat_name = FEATURE_NAMES.get(feat_idx, f'v{feat_idx}')
        depth = _get_depth(node_idx)

        # Truncate long names for subtrees
        if len(feat_name) > 12:
            feat_name = feat_name[:10] + '..'

        label = f"{feat_name}\n≤ {threshold:.0f}?"
        fontcolor = 'white' if depth >= 2 else 'black'

        graph.node(f'{prefix}n{node_idx}', label=label,
                   fillcolor=depth_colors[min(depth, len(depth_colors)-1)],
                   fontcolor=fontcolor)

    # Leaf nodes
    n_leaves = 2 ** tree_depth
    for leaf_idx in range(n_leaves):
        graph.node(f'{prefix}leaf{leaf_idx}', label=f'B{leaf_idx + 1}',
                   fillcolor=leaf_color, fontcolor='white')

    # Edges
    for node_idx in range(min(n_decision_nodes, len(feature_indices))):
        left_child = 2 * node_idx + 1
        right_child = 2 * node_idx + 2

        if left_child < n_decision_nodes and left_child < len(feature_indices):
            graph.edge(f'{prefix}n{node_idx}', f'{prefix}n{left_child}')
        elif left_child - n_decision_nodes < n_leaves:
            graph.edge(f'{prefix}n{node_idx}', f'{prefix}leaf{left_child - n_decision_nodes}')

        if right_child < n_decision_nodes and right_child < len(feature_indices):
            graph.edge(f'{prefix}n{node_idx}', f'{prefix}n{right_child}')
        elif right_child - n_decision_nodes < n_leaves:
            graph.edge(f'{prefix}n{node_idx}', f'{prefix}leaf{right_child - n_decision_nodes}')


# ==============================================================================
# MATPLOTLIB FALLBACK IMPLEMENTATIONS
# ==============================================================================

def _plot_simple_tree(chromosome: Dict[str, Any], plots_dir: Path,
                      run_name: str, summary: Dict[str, Any]) -> None:
    """Plot a simple configuration decision tree (depth-3) - matplotlib fallback."""
    feature_indices = chromosome['feature_indices']
    thresholds = chromosome['thresholds']
    tree_depth = chromosome.get('tree_depth', 3)

    # For depth-3: 7 decision nodes (indices 0-6), 8 leaves (indices 7-14)
    n_decision_nodes = 2 ** tree_depth - 1

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Title with metrics
    fitness = summary.get('optimization_summary', {}).get('best_fitness', 'N/A')
    ms = summary.get('best_chromosome', {}).get('mission_success', 'N/A')
    or_val = summary.get('best_chromosome', {}).get('operational_readiness', 'N/A')

    if isinstance(fitness, float):
        fitness = f"{fitness:.4f}"
    if isinstance(ms, float):
        ms = f"{ms*100:.1f}%"
    if isinstance(or_val, float):
        or_val = f"{or_val*100:.1f}%"

    title = f"Decision Tree Policy - {run_name}\nFitness: {fitness} | MS: {ms} | OR: {or_val}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Calculate node positions for binary tree
    positions = _calculate_tree_positions(tree_depth)

    # Draw edges first (so they're behind nodes)
    for node_idx in range(n_decision_nodes):
        left_child = 2 * node_idx + 1
        right_child = 2 * node_idx + 2
        node_pos = positions[node_idx]

        if left_child < len(positions):
            child_pos = positions[left_child]
            ax.plot([node_pos[0], child_pos[0]], [node_pos[1], child_pos[1]],
                    'k-', linewidth=1.5, zorder=1)
            # Label: True (left)
            mid_x = (node_pos[0] + child_pos[0]) / 2 - 0.02
            mid_y = (node_pos[1] + child_pos[1]) / 2
            ax.text(mid_x, mid_y, 'T', fontsize=8, color='green', fontweight='bold')

        if right_child < len(positions):
            child_pos = positions[right_child]
            ax.plot([node_pos[0], child_pos[0]], [node_pos[1], child_pos[1]],
                    'k-', linewidth=1.5, zorder=1)
            # Label: False (right)
            mid_x = (node_pos[0] + child_pos[0]) / 2 + 0.02
            mid_y = (node_pos[1] + child_pos[1]) / 2
            ax.text(mid_x, mid_y, 'F', fontsize=8, color='red', fontweight='bold')

    # Draw decision nodes
    for node_idx in range(n_decision_nodes):
        x, y = positions[node_idx]
        feat_idx = feature_indices[node_idx]
        threshold = thresholds[node_idx]
        feat_name = FEATURE_NAMES.get(feat_idx, f'feat_{feat_idx}')

        # Node color based on depth
        depth = _get_depth(node_idx)
        color = TREE_COLORS['decision'] if depth < 2 else TREE_COLORS['decision_deep']

        # Draw node box
        box = mpatches.FancyBboxPatch(
            (x - 0.055, y - 0.035), 0.11, 0.07,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            facecolor=color, edgecolor='black', linewidth=1.5, zorder=2
        )
        ax.add_patch(box)

        # Node text
        label = f"{feat_name}\n≤ {threshold:.1f}?"
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold', zorder=3)

    # Draw leaf nodes (buckets)
    n_leaves = 2 ** tree_depth
    leaf_colors = [TREE_COLORS['leaf_1'], TREE_COLORS['leaf_2'],
                   TREE_COLORS['leaf_3'], TREE_COLORS['leaf_4']] * 2

    for leaf_idx in range(n_leaves):
        node_idx = n_decision_nodes + leaf_idx
        if node_idx < len(positions):
            x, y = positions[node_idx]
            bucket_num = leaf_idx + 1

            box = mpatches.FancyBboxPatch(
                (x - 0.04, y - 0.03), 0.08, 0.06,
                boxstyle="round,pad=0.01,rounding_size=0.01",
                facecolor=leaf_colors[leaf_idx % len(leaf_colors)],
                edgecolor='black', linewidth=1.5, zorder=2
            )
            ax.add_patch(box)

            ax.text(x, y, f"B{bucket_num}", ha='center', va='center',
                    fontsize=9, fontweight='bold', zorder=3)

    # Add legend for features used
    _add_feature_legend(ax, feature_indices, thresholds, tree_depth)

    plt.tight_layout()
    fig.savefig(plots_dir / 'decision_tree.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - decision_tree.png (simple config)")


def _plot_medium_tree(chromosome: Dict[str, Any], plots_dir: Path,
                      run_name: str, summary: Dict[str, Any]) -> None:
    """Plot a medium configuration decision tree (context layer + 2 subtrees)."""
    fleet_indices = chromosome.get('fleet_variable_indices', [])
    fleet_thresholds = chromosome.get('fleet_thresholds', [])
    context_subtrees = chromosome.get('context_subtrees', [])

    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    # Title
    fitness = summary.get('optimization_summary', {}).get('best_fitness', 'N/A')
    ms = summary.get('best_chromosome', {}).get('mission_success', 'N/A')
    or_val = summary.get('best_chromosome', {}).get('operational_readiness', 'N/A')

    if isinstance(fitness, float):
        fitness = f"{fitness:.4f}"
    if isinstance(ms, float):
        ms = f"{ms*100:.1f}%"
    if isinstance(or_val, float):
        or_val = f"{or_val*100:.1f}%"

    fig.suptitle(f"Decision Tree Policy (Medium) - {run_name}\n"
                 f"Fitness: {fitness} | MS: {ms} | OR: {or_val}",
                 fontsize=14, fontweight='bold')

    # Left panel: Context layer
    ax0 = axes[0]
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis('off')
    ax0.set_title("Fleet Context Layer", fontsize=12, fontweight='bold')

    # Draw context node
    if fleet_indices:
        feat_idx = fleet_indices[0]
        threshold = fleet_thresholds[0] if fleet_thresholds else 0
        feat_name = FLEET_FEATURE_NAMES.get(feat_idx, f'fleet_{feat_idx}')

        box = mpatches.FancyBboxPatch(
            (0.35, 0.55), 0.3, 0.15,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor=TREE_COLORS['context'], edgecolor='black', linewidth=2
        )
        ax0.add_patch(box)
        ax0.text(0.5, 0.625, f"{feat_name}\n≤ {threshold:.1f}?",
                 ha='center', va='center', fontsize=10, fontweight='bold')

        # Arrows to subtrees
        ax0.annotate('', xy=(0.25, 0.35), xytext=(0.4, 0.55),
                     arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax0.text(0.25, 0.4, 'True\n→ Left Tree', ha='center', fontsize=9, color='green')

        ax0.annotate('', xy=(0.75, 0.35), xytext=(0.6, 0.55),
                     arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax0.text(0.75, 0.4, 'False\n→ Right Tree', ha='center', fontsize=9, color='red')

    # Middle and right panels: Subtrees
    for idx, (ax, subtree) in enumerate(zip(axes[1:], context_subtrees[:2])):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(f"{'Left' if idx == 0 else 'Right'} Subtree", fontsize=12, fontweight='bold')

        if subtree:
            _draw_subtree_on_axis(ax, subtree)

    plt.tight_layout()
    fig.savefig(plots_dir / 'decision_tree.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - decision_tree.png (medium config)")


def _plot_full_tree(chromosome: Dict[str, Any], plots_dir: Path,
                    run_name: str, summary: Dict[str, Any]) -> None:
    """Plot a full configuration decision tree (context layer + 4 subtrees)."""
    # For full config, create a larger figure with context + 4 subtrees
    fig = plt.figure(figsize=(20, 12))

    # Title
    fitness = summary.get('optimization_summary', {}).get('best_fitness', 'N/A')
    ms = summary.get('best_chromosome', {}).get('mission_success', 'N/A')
    or_val = summary.get('best_chromosome', {}).get('operational_readiness', 'N/A')

    if isinstance(fitness, float):
        fitness = f"{fitness:.4f}"
    if isinstance(ms, float):
        ms = f"{ms*100:.1f}%"
    if isinstance(or_val, float):
        or_val = f"{or_val*100:.1f}%"

    fig.suptitle(f"Decision Tree Policy (Full) - {run_name}\n"
                 f"Fitness: {fitness} | MS: {ms} | OR: {or_val}",
                 fontsize=14, fontweight='bold')

    # Create subplot grid: context on top, 4 subtrees below
    ax_context = fig.add_subplot(2, 1, 1)
    ax_context.set_xlim(0, 1)
    ax_context.set_ylim(0, 1)
    ax_context.axis('off')
    ax_context.set_title("Fleet Context Layer (2 levels)", fontsize=12, fontweight='bold')

    # Draw context nodes
    fleet_indices = chromosome.get('fleet_variable_indices', [])
    fleet_thresholds = chromosome.get('fleet_thresholds', [])

    if len(fleet_indices) >= 3:
        # Root context node
        feat_name = FLEET_FEATURE_NAMES.get(fleet_indices[0], f'fleet_{fleet_indices[0]}')
        box = mpatches.FancyBboxPatch((0.4, 0.7), 0.2, 0.15,
            boxstyle="round,pad=0.02", facecolor=TREE_COLORS['context'],
            edgecolor='black', linewidth=2)
        ax_context.add_patch(box)
        ax_context.text(0.5, 0.775, f"{feat_name}\n≤ {fleet_thresholds[0]:.1f}?",
                        ha='center', va='center', fontsize=9, fontweight='bold')

        # Second level context nodes
        for i, (x_pos, label) in enumerate([(0.25, 'Left'), (0.75, 'Right')]):
            if i + 1 < len(fleet_indices):
                feat_name = FLEET_FEATURE_NAMES.get(fleet_indices[i+1], f'fleet_{fleet_indices[i+1]}')
                box = mpatches.FancyBboxPatch((x_pos - 0.1, 0.35), 0.2, 0.15,
                    boxstyle="round,pad=0.02", facecolor=TREE_COLORS['context'],
                    edgecolor='black', linewidth=2)
                ax_context.add_patch(box)
                ax_context.text(x_pos, 0.425, f"{feat_name}\n≤ {fleet_thresholds[i+1]:.1f}?",
                                ha='center', va='center', fontsize=9, fontweight='bold')

    # Subtree axes (2x2 grid below)
    context_subtrees = chromosome.get('context_subtrees', [])
    subtree_titles = ['Subtree 1 (TT)', 'Subtree 2 (TF)', 'Subtree 3 (FT)', 'Subtree 4 (FF)']

    for idx in range(4):
        ax = fig.add_subplot(2, 4, 5 + idx)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(subtree_titles[idx], fontsize=10)

        if idx < len(context_subtrees) and context_subtrees[idx]:
            _draw_subtree_on_axis(ax, context_subtrees[idx])

    plt.tight_layout()
    fig.savefig(plots_dir / 'decision_tree.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  - decision_tree.png (full config)")


def _draw_subtree_on_axis(ax, subtree: Dict[str, Any]) -> None:
    """Draw a subtree on the given axis."""
    feature_indices = subtree.get('feature_indices', [])
    thresholds = subtree.get('thresholds', [])
    tree_depth = subtree.get('tree_depth', 3)

    n_decision_nodes = 2 ** tree_depth - 1
    positions = _calculate_tree_positions(tree_depth)

    # Draw edges
    for node_idx in range(min(n_decision_nodes, len(feature_indices))):
        left_child = 2 * node_idx + 1
        right_child = 2 * node_idx + 2
        node_pos = positions[node_idx]

        if left_child < len(positions):
            child_pos = positions[left_child]
            ax.plot([node_pos[0], child_pos[0]], [node_pos[1], child_pos[1]],
                    'k-', linewidth=1, zorder=1)

        if right_child < len(positions):
            child_pos = positions[right_child]
            ax.plot([node_pos[0], child_pos[0]], [node_pos[1], child_pos[1]],
                    'k-', linewidth=1, zorder=1)

    # Draw decision nodes
    for node_idx in range(min(n_decision_nodes, len(feature_indices))):
        x, y = positions[node_idx]
        feat_idx = feature_indices[node_idx]
        threshold = thresholds[node_idx] if node_idx < len(thresholds) else 0
        feat_name = FEATURE_NAMES.get(feat_idx, f'f{feat_idx}')

        depth = _get_depth(node_idx)
        color = TREE_COLORS['decision'] if depth < 2 else TREE_COLORS['decision_deep']

        box = mpatches.FancyBboxPatch(
            (x - 0.07, y - 0.04), 0.14, 0.08,
            boxstyle="round,pad=0.01", facecolor=color,
            edgecolor='black', linewidth=1, zorder=2
        )
        ax.add_patch(box)

        label = f"{feat_name[:8]}\n≤{threshold:.0f}"
        ax.text(x, y, label, ha='center', va='center', fontsize=6, fontweight='bold', zorder=3)

    # Draw leaves
    n_leaves = 2 ** tree_depth
    leaf_colors = [TREE_COLORS['leaf_1'], TREE_COLORS['leaf_2'],
                   TREE_COLORS['leaf_3'], TREE_COLORS['leaf_4']] * 2

    for leaf_idx in range(n_leaves):
        node_idx = n_decision_nodes + leaf_idx
        if node_idx < len(positions):
            x, y = positions[node_idx]
            box = mpatches.FancyBboxPatch(
                (x - 0.035, y - 0.025), 0.07, 0.05,
                boxstyle="round,pad=0.01", facecolor=leaf_colors[leaf_idx % len(leaf_colors)],
                edgecolor='black', linewidth=1, zorder=2
            )
            ax.add_patch(box)
            ax.text(x, y, f"B{leaf_idx+1}", ha='center', va='center', fontsize=7, fontweight='bold', zorder=3)


def _calculate_tree_positions(depth: int) -> List[tuple]:
    """Calculate (x, y) positions for all nodes in a binary tree.

    Returns positions for decision nodes (depth levels 0 to depth-1)
    followed by leaf nodes (at level depth).
    """
    positions = []

    # Decision nodes (levels 0 to depth-1)
    for level in range(depth):
        n_nodes_at_level = 2 ** level
        y = 0.9 - (level * 0.25)  # Descend from top

        for i in range(n_nodes_at_level):
            # Spread nodes evenly across the width
            x = (i + 0.5) / n_nodes_at_level
            positions.append((x, y))

    # Leaf nodes (at level depth)
    n_leaves = 2 ** depth
    y = 0.9 - (depth * 0.25)

    for i in range(n_leaves):
        x = (i + 0.5) / n_leaves
        positions.append((x, y))

    return positions


def _get_depth(node_idx: int) -> int:
    """Get the depth of a node given its index (0-indexed, breadth-first)."""
    if node_idx == 0:
        return 0
    import math
    return int(math.floor(math.log2(node_idx + 1)))


def _add_feature_legend(ax, feature_indices: List[int], thresholds: List[float],
                        tree_depth: int) -> None:
    """Add a legend showing feature usage statistics."""
    n_decision_nodes = 2 ** tree_depth - 1

    # Count feature usage
    feature_counts = {}
    for i in range(min(n_decision_nodes, len(feature_indices))):
        feat_idx = feature_indices[i]
        feat_name = FEATURE_NAMES.get(feat_idx, f'feat_{feat_idx}')
        if feat_name not in feature_counts:
            feature_counts[feat_name] = {'count': 0, 'thresholds': []}
        feature_counts[feat_name]['count'] += 1
        if i < len(thresholds):
            feature_counts[feat_name]['thresholds'].append(thresholds[i])

    # Create legend text
    legend_lines = ["Features Used:"]
    for feat_name, info in sorted(feature_counts.items(), key=lambda x: -x[1]['count']):
        thresh_str = ', '.join([f"{t:.1f}" for t in info['thresholds'][:3]])
        if len(info['thresholds']) > 3:
            thresh_str += ', ...'
        legend_lines.append(f"  • {feat_name} ({info['count']}x): {thresh_str}")

    legend_text = '\n'.join(legend_lines)

    # Add text box
    props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', bbox=props, family='monospace')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate island GA plots')
    parser.add_argument('results_dir', help='Path to results directory')
    parser.add_argument('--name', help='Run name for plot titles')
    args = parser.parse_args()

    generate_island_plots(args.results_dir, run_name=args.name)
