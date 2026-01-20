#!/usr/bin/env python3
"""
Tabulate evaluation results from all methods into CSV and PNG tables.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

# Base paths - derive from script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # Up to WholeBrainProject

# Methods and their evaluation subdirectories
def get_methods(outputs_dir):
    return {
        'bwskel': os.path.join(outputs_dir, 'bwskel_evaluation'),
        'Diff. Skel': os.path.join(outputs_dir, 'diffskel_evaluation'),
        'neutube': os.path.join(outputs_dir, 'neutube_evaluation'),
        'VESS': os.path.join(outputs_dir, 'vess_evaluation'),
        'PHDF': os.path.join(outputs_dir, 'phd_evaluation'),
        'DM2D': os.path.join(outputs_dir, 'dm2d_evaluation'),
    }

def load_metrics(eval_dir):
    json_path = os.path.join(eval_dir, 'summary.json')
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['overall_metrics']

def create_table(methods_dict, output_csv, title):
    records = []
    
    for method, path in methods_dict.items():
        metrics = load_metrics(path)
        if metrics:
            rec = {
                'Method': method,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F-Score': metrics['f_score'],
                'IoU': metrics['iou'],
                'Dice': metrics['dice']
            }
            records.append(rec)
        else:
            print(f"Warning: No summary found for {method} at {path}")
            
    df = pd.DataFrame(records)
    # Reorder columns
    cols = ['Method', 'Precision', 'Recall', 'F-Score', 'IoU', 'Dice']
    df = df[cols] if not df.empty else df
    
    # Save CSV (only if there's data)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if df.empty:
        print(f"No data for {title} - skipping CSV generation")
    else:
        df.to_csv(output_csv, index=False)
        print(f"Saved table to {output_csv}")
    return df

def save_table_image(df, title, filename):
    if df.empty:
        return
        
    fig, ax = plt.subplots(figsize=(10, 2 + len(df)*0.5))
    ax.axis('tight')
    ax.axis('off')
    
    # Format floats
    cell_text = []
    for row in df.itertuples(index=False):
        formatted_row = [row[0]] + [f"{x:.4f}" for x in row[1:]]
        cell_text.append(formatted_row)
        
    table = ax.table(cellText=cell_text, colLabels=df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Bold headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.title(title, fontsize=16, pad=20, weight='bold')
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved image to {filename}")

def main():
    results_dir = os.path.join(BASE_DIR, 'results', 'tables')
    
    # PMD
    print("Generating PMD Table...")
    pmd_outputs = os.path.join(BASE_DIR, 'outputs', 'pmd')
    pmd_methods = get_methods(pmd_outputs)
    df_pmd = create_table(pmd_methods, os.path.join(results_dir, 'results_PMD.csv'), 'PMD Dataset Evaluation')
    save_table_image(df_pmd, 'PMD Dataset Results', os.path.join(results_dir, 'results_PMD.png'))
    
    # STP
    print("\nGenerating STP Table...")
    stp_outputs = os.path.join(BASE_DIR, 'outputs', 'stp')
    stp_methods = get_methods(stp_outputs)
    df_stp = create_table(stp_methods, os.path.join(results_dir, 'results_STP.csv'), 'STP Dataset Evaluation')
    save_table_image(df_stp, 'STP Dataset Results', os.path.join(results_dir, 'results_STP.png'))

if __name__ == "__main__":
    main()
