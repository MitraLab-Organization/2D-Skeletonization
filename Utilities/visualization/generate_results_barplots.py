#!/usr/bin/env python3
"""
Generate bar plots of the evaluation metrics for PMD and STP.

Each bar is the score over all tiles pooled (the value reported in Table 1);
the dots overlaid on it are the same metric for each individual tile.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Embed fonts as TrueType so text in the PDF output stays editable
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.size'] = 14

# Base paths - derive from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

# Method label -> evaluation folder name (same order as Table 1)
METHODS = [
    ('bwskel', 'bwskel_evaluation'),
    ('Diff. Skel', 'diffskel_evaluation'),
    ('neuTube', 'neutube_evaluation'),
    ('VESS', 'vess_evaluation'),
    ('PHDF', 'phd_evaluation'),
    ('DM2D', 'dm2d_evaluation'),
]

# Table column, per-tile column, colour
METRICS = [
    ('Precision', 'precision', '#2a78d6'),
    ('Recall', 'recall', '#eb6834'),
    ('F-Score', 'f_score', '#1baf7a'),
    ('IoU', 'iou', '#eda100'),
]


def load_per_tile(outputs_dir):
    """Per-tile metrics for each method, keyed by method label."""
    per_tile = {}
    for label, folder in METHODS:
        path = os.path.join(outputs_dir, folder, 'per_image_metrics.csv')
        if os.path.exists(path):
            per_tile[label] = pd.read_csv(path)
        else:
            print(f"Warning: {path} not found")
    return per_tile


def create_bar_plot(pooled, per_tile, title, output_path):
    methods = [m for m, _ in METHODS if m in set(pooled['Method'])]
    if not methods:
        print(f"Skipping {title} - no data available")
        return

    pooled = pooled.set_index('Method').loc[methods]
    rng = np.random.default_rng(0)  # fixed jitter so reruns give identical figures

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(methods))
    width = 0.2

    for i, (column, tile_column, color) in enumerate(METRICS):
        offset = (i - 1.5) * width
        ax.bar(x + offset, pooled[column], width * 0.9, label=column, color=color,
               edgecolor='none', zorder=2)
        for j, method in enumerate(methods):
            if method not in per_tile:
                continue
            values = per_tile[method][tile_column].to_numpy()
            jitter = rng.uniform(-width * 0.25, width * 0.25, size=len(values))
            ax.scatter(x[j] + offset + jitter, values, s=12, color='#222222',
                       edgecolor='white', linewidth=0.5, zorder=3)

    n_tiles = sorted({len(df) for df in per_tile.values()})
    n_text = ' or '.join(str(n) for n in n_tiles)
    ax.set_title(f'{title} (bars: all tiles pooled; dots: individual tiles, n = {n_text})',
                 fontsize=14, pad=12)
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, color='#dddddd', linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', ncol=4, frameon=False, bbox_to_anchor=(0, -0.08))
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.splitext(output_path)[0] + '.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    figures_dir = os.path.join(BASE_DIR, 'results', 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    for dataset in ('pmd', 'stp'):
        table_path = os.path.join(BASE_DIR, 'results', 'tables', f'results_{dataset.upper()}.csv')
        if not os.path.exists(table_path):
            print(f"Skipping {dataset.upper()} barplot - {table_path} not found")
            continue
        pooled = pd.read_csv(table_path)
        per_tile = load_per_tile(os.path.join(BASE_DIR, 'outputs', dataset))
        create_bar_plot(pooled, per_tile, f'{dataset.upper()} dataset',
                        os.path.join(figures_dir, f'results_{dataset.upper()}_barplot.png'))

    print("Done!")


if __name__ == '__main__':
    main()
