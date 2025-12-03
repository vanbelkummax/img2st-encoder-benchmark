#!/usr/bin/env python3
"""Generate benchmark comparison figures from results JSON."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Style settings
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

def load_results(repo_root: Path) -> dict:
    """Load results from JSON file."""
    results_path = repo_root / "results" / "encoder_loocv_comparison.json"
    with open(results_path) as f:
        return json.load(f)

def create_bar_chart(results: dict, output_dir: Path):
    """Main encoder comparison bar chart."""
    encoders = ['DenseNet\n(random)', 'DenseNet\n(frozen)', 'DenseNet\n(fine-tuned)',
                'H-optimus-0', 'UNI2-h', 'Virchow2']

    # Extract from JSON
    r = results['results']
    pccs = [
        0.070,  # random init (from summary)
        r['densenet_frozen']['pearson_r']['mean'],
        r['densenet_finetuned']['pearson_r']['mean'],
        r['h_optimus_frozen']['pearson_r']['mean'],
        r['uni2h_frozen']['pearson_r']['mean'],
        r['virchow2_frozen']['pearson_r']['mean'],
    ]
    stds = [
        0.02,
        r['densenet_frozen']['pearson_r']['std'],
        r['densenet_finetuned']['pearson_r']['std'],
        r['h_optimus_frozen']['pearson_r']['std'],
        r['uni2h_frozen']['pearson_r']['std'],
        r['virchow2_frozen']['pearson_r']['std'],
    ]

    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#17becf']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(encoders, pccs, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1)

    ax.set_ylabel('Pearson Correlation (PCC)')
    ax.set_title('Encoder Comparison for Spatial Transcriptomics Prediction\n(LOOCV across 3 CRC patients, 200 epochs)')
    ax.set_ylim(0, 0.45)
    ax.axhline(y=pccs[1], color='orange', linestyle='--', alpha=0.5, label='DenseNet frozen baseline')

    for bar, pcc in zip(bars, pccs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{pcc:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.legend(loc='upper left')
    plt.tight_layout()

    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'encoder_comparison.{fmt}', bbox_inches='tight')
    plt.close()
    print(f"Created encoder_comparison.pdf/png")

def create_grouped_bar_chart(results: dict, output_dir: Path):
    """Grouped comparison: frozen vs fine-tuned."""
    r = results['results']

    encoders = ['DenseNet-121', 'H-optimus-0', 'UNI2-h', 'Virchow2']
    frozen = [
        r['densenet_frozen']['pearson_r']['mean'],
        r['h_optimus_frozen']['pearson_r']['mean'],
        r['uni2h_frozen']['pearson_r']['mean'],
        r['virchow2_frozen']['pearson_r']['mean'],
    ]
    finetuned = [r['densenet_finetuned']['pearson_r']['mean'], None, None, None]

    x = np.arange(len(encoders))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, frozen, width, label='Frozen', color='#1f77b4', edgecolor='black')

    ft_vals = [v if v else 0 for v in finetuned]
    bars2 = ax.bar(x + width/2, ft_vals, width, label='Fine-tuned', color='#2ca02c', edgecolor='black')
    bars2[1].set_alpha(0.2)
    bars2[2].set_alpha(0.2)
    bars2[3].set_alpha(0.2)

    ax.set_ylabel('Pearson Correlation (PCC)')
    ax.set_title('Frozen vs Fine-tuned Encoder Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(encoders)
    ax.legend()
    ax.set_ylim(0, 0.45)

    for bar, val in zip(bars1, frozen):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9)
    if finetuned[0]:
        ax.text(bars2[0].get_x() + bars2[0].get_width()/2, bars2[0].get_height() + 0.01,
                f'{finetuned[0]:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'encoder_comparison_grouped.{fmt}', bbox_inches='tight')
    plt.close()
    print(f"Created encoder_comparison_grouped.pdf/png")

def create_simple_comparison(results: dict, output_dir: Path):
    """Simple frozen-only comparison."""
    r = results['results']

    encoders = ['DenseNet-121\n(ImageNet)', 'H-optimus-0\n(Pathology)', 'UNI2-h\n(Pathology)', 'Virchow2\n(Pathology)']
    pccs = [
        r['densenet_frozen']['pearson_r']['mean'],
        r['h_optimus_frozen']['pearson_r']['mean'],
        r['uni2h_frozen']['pearson_r']['mean'],
        r['virchow2_frozen']['pearson_r']['mean'],
    ]
    stds = [
        r['densenet_frozen']['pearson_r']['std'],
        r['h_optimus_frozen']['pearson_r']['std'],
        r['uni2h_frozen']['pearson_r']['std'],
        r['virchow2_frozen']['pearson_r']['std'],
    ]
    colors = ['#ff7f0e', '#1f77b4', '#9467bd', '#17becf']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(encoders, pccs, yerr=stds, capsize=5, color=colors, edgecolor='black')

    ax.set_ylabel('Pearson Correlation (PCC)')
    ax.set_title('Frozen Encoder Comparison (Fair Comparison)')
    ax.set_ylim(0, 0.45)

    for bar, pcc in zip(bars, pccs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{pcc:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'encoder_simple.{fmt}', bbox_inches='tight')
    plt.close()
    print(f"Created encoder_simple.pdf/png")

def create_params_vs_pcc(results: dict, output_dir: Path):
    """Scatter: model size vs performance."""
    r = results['results']

    models = {
        'DenseNet-121': (8, r['densenet_frozen']['pearson_r']['mean'], '#ff7f0e'),
        'H-optimus-0': (1100, r['h_optimus_frozen']['pearson_r']['mean'], '#1f77b4'),
        'UNI2-h': (681, r['uni2h_frozen']['pearson_r']['mean'], '#9467bd'),
        'Virchow2': (632, r['virchow2_frozen']['pearson_r']['mean'], '#17becf'),
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, (params, pcc, color) in models.items():
        ax.scatter(params, pcc, s=200, c=color, edgecolors='black', linewidth=1.5, label=name, zorder=5)
        ax.annotate(name, (params, pcc), xytext=(10, 10), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Parameters (millions)')
    ax.set_ylabel('Pearson Correlation (PCC)')
    ax.set_title('Model Size vs Performance (Frozen Encoders)')
    ax.set_xscale('log')
    ax.set_xlim(5, 2000)
    ax.set_ylim(0.28, 0.38)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'params_vs_pcc.{fmt}', bbox_inches='tight')
    plt.close()
    print(f"Created params_vs_pcc.pdf/png")

def create_per_fold_bars(results: dict, output_dir: Path):
    """Per-patient consistency."""
    r = results['results']

    patients = ['P1', 'P2', 'P5']
    data = {
        'DenseNet (frozen)': [r['densenet_frozen']['fold_results'][f'test_{p}']['pearson_r'] for p in patients],
        'DenseNet (fine-tuned)': [r['densenet_finetuned']['fold_results'][f'test_{p}']['pearson_r'] for p in patients],
        'H-optimus-0': [r['h_optimus_frozen']['fold_results'][f'test_{p}']['pearson_r'] for p in patients],
        'UNI2-h': [r['uni2h_frozen']['fold_results'][f'test_{p}']['pearson_r'] for p in patients],
        'Virchow2': [r['virchow2_frozen']['fold_results'][f'test_{p}']['pearson_r'] for p in patients],
    }

    x = np.arange(len(patients))
    width = 0.15
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#17becf']

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (name, vals) in enumerate(data.items()):
        ax.bar(x + i*width, vals, width, label=name, color=colors[i], edgecolor='black')

    ax.set_ylabel('Pearson Correlation (PCC)')
    ax.set_title('Per-Patient Performance Consistency')
    ax.set_xticks(x + width*2)
    ax.set_xticklabels(patients)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_ylim(0.25, 0.40)

    plt.tight_layout()
    for fmt in ['pdf', 'png']:
        fig.savefig(output_dir / f'per_fold_consistency.{fmt}', bbox_inches='tight')
    plt.close()
    print(f"Created per_fold_consistency.pdf/png")


if __name__ == "__main__":
    # Resolve paths relative to script location
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "figures"
    output_dir.mkdir(exist_ok=True)

    print(f"Loading results from {repo_root / 'results'}...")
    results = load_results(repo_root)

    print("Generating figures...")
    create_bar_chart(results, output_dir)
    create_grouped_bar_chart(results, output_dir)
    create_simple_comparison(results, output_dir)
    create_params_vs_pcc(results, output_dir)
    create_per_fold_bars(results, output_dir)

    print("\nAll figures generated!")
