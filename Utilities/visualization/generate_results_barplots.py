#!/usr/bin/env python3
"""Generate bar plots for PMD and STP results - research compliant style"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Base paths - derive from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Up to WholeBrainProject

# Research-compliant style
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']  # Green, Blue, Red, Purple
metrics = ['Precision', 'Recall', 'F-Score', 'IoU']

def create_bar_plot(df, title, output_path):
    if df.empty or len(df) == 0:
        print(f"Skipping {title} - no data available")
        return False
        
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(df['Method']))
    width = 0.18
    
    for i, metric in enumerate(metrics):
        bars = ax.bar(x + i*width, df[metric], width, label=metric, color=colors[i], 
                      edgecolor='black', linewidth=1)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14, fontweight='bold', rotation=45)
    
    ax.set_xlabel('Method', fontsize=24, fontweight='bold')
    ax.set_ylabel('Score', fontsize=24, fontweight='bold')
    ax.set_title(title, fontsize=24, fontweight='bold', pad=20)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df['Method'], fontsize=20, fontweight='bold')
    ax.tick_params(axis='y', labelsize=18)
    ax.set_ylim(0, 1.2)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=16, framealpha=0.9, ncol=4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    return True

def read_csv_safe(path):
    """Read CSV file, returning empty DataFrame if file is empty or missing"""
    try:
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            return pd.DataFrame()
        df = pd.read_csv(path)
        if df.empty:
            print(f"Warning: {path} is empty")
        return df
    except Exception as e:
        print(f"Warning: Could not read {path}: {e}")
        return pd.DataFrame()

# Read data
pmd_path = os.path.join(BASE_DIR, 'results/tables/results_PMD.csv')
stp_path = os.path.join(BASE_DIR, 'results/tables/results_STP.csv')

pmd = read_csv_safe(pmd_path)
stp = read_csv_safe(stp_path)

# Create plots
os.makedirs(os.path.join(BASE_DIR, 'results/figures'), exist_ok=True)

if not pmd.empty:
    create_bar_plot(pmd, 'PMD Dataset', 
                    os.path.join(BASE_DIR, 'results/figures/results_PMD_barplot.png'))
else:
    print("Skipping PMD barplot - no data")

if not stp.empty:
    create_bar_plot(stp, 'STP Dataset',
                    os.path.join(BASE_DIR, 'results/figures/results_STP_barplot.png'))
else:
    print("Skipping STP barplot - no data")

print("Done!")
