"""
Visualization Script for Lottery Ticket Hypothesis Results
Author: Research Project for NMIMS Tech Trends
Description: Generates charts and graphs to visualize pruning experiment results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style for professional-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11


def load_results(filename='results/experiment_results.json'):
    """Load experimental results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def plot_accuracy_vs_pruning(results, save_path='images/accuracy_vs_pruning.png'):
    """
    Create a line plot showing accuracy vs pruning percentage.
    This is the main visualization demonstrating the Lottery Ticket Hypothesis.
    """
    pruning_pcts = [r['pruning_percentage'] for r in results['pruning_results']]
    accuracies = [r['accuracy'] for r in results['pruning_results']]
    baseline = results['baseline_accuracy']
    
    plt.figure(figsize=(12, 7))
    
    # Plot accuracy line
    plt.plot(pruning_pcts, accuracies, 'o-', linewidth=2.5, markersize=8, 
             color='#2E86AB', label='Pruned Network Accuracy')
    
    # Plot baseline as horizontal line
    plt.axhline(y=baseline, color='#A23B72', linestyle='--', linewidth=2, 
                label=f'Baseline Accuracy ({baseline:.2f}%)')
    
    # Add 95% accuracy threshold line
    plt.axhline(y=95, color='#F18F01', linestyle=':', linewidth=1.5, 
                label='95% Accuracy Threshold', alpha=0.7)
    
    # Annotations
    plt.xlabel('Pruning Percentage (%)', fontsize=13, fontweight='bold')
    plt.ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    plt.title('Neural Network Accuracy vs Pruning Percentage\n(Lottery Ticket Hypothesis Demonstration)', 
              fontsize=15, fontweight='bold', pad=20)
    
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('images', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_parameter_reduction(results, save_path='images/parameter_reduction.png'):
    """
    Visualize the reduction in model parameters with pruning.
    Shows how much smaller the model becomes.
    """
    pruning_pcts = [r['pruning_percentage'] for r in results['pruning_results']]
    remaining_params = [r['remaining_parameters'] for r in results['pruning_results']]
    total_params = results['pruning_results'][0]['total_parameters']
    
    plt.figure(figsize=(12, 7))
    
    # Create bar chart
    bars = plt.bar(range(len(pruning_pcts)), remaining_params, 
                   color=sns.color_palette("viridis", len(pruning_pcts)))
    
    # Add value labels on bars
    for i, (bar, params) in enumerate(zip(bars, remaining_params)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{params:,}\n({100*params/total_params:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Pruning Percentage (%)', fontsize=13, fontweight='bold')
    plt.ylabel('Number of Parameters', fontsize=13, fontweight='bold')
    plt.title('Model Size Reduction Through Pruning\n(Remaining Parameters vs Pruning Level)', 
              fontsize=15, fontweight='bold', pad=20)
    
    plt.xticks(range(len(pruning_pcts)), [f'{p}%' for p in pruning_pcts])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_accuracy_difference(results, save_path='images/accuracy_difference.png'):
    """
    Show the difference in accuracy compared to baseline.
    Helps visualize how much accuracy is lost (or gained) with pruning.
    """
    pruning_pcts = [r['pruning_percentage'] for r in results['pruning_results']]
    acc_diffs = [r['accuracy_difference'] for r in results['pruning_results']]
    
    plt.figure(figsize=(12, 7))
    
    # Color bars based on positive/negative difference
    colors = ['#06A77D' if diff >= 0 else '#D62246' for diff in acc_diffs]
    bars = plt.bar(range(len(pruning_pcts)), acc_diffs, color=colors, alpha=0.7)
    
    # Add value labels
    for bar, diff in zip(bars, acc_diffs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{diff:+.2f}%',
                ha='center', va='bottom' if diff >= 0 else 'top', fontsize=10)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.xlabel('Pruning Percentage (%)', fontsize=13, fontweight='bold')
    plt.ylabel('Accuracy Difference from Baseline (%)', fontsize=13, fontweight='bold')
    plt.title('Impact of Pruning on Model Accuracy\n(Difference from Baseline Performance)', 
              fontsize=15, fontweight='bold', pad=20)
    
    plt.xticks(range(len(pruning_pcts)), [f'{p}%' for p in pruning_pcts])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_combined_metrics(results, save_path='images/combined_metrics.png'):
    """
    Create a dual-axis plot showing both accuracy and model size.
    This gives a comprehensive view of the trade-off.
    """
    pruning_pcts = [r['pruning_percentage'] for r in results['pruning_results']]
    accuracies = [r['accuracy'] for r in results['pruning_results']]
    remaining_params = [r['remaining_parameters'] for r in results['pruning_results']]
    total_params = results['pruning_results'][0]['total_parameters']
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot accuracy on left y-axis
    color1 = '#2E86AB'
    ax1.set_xlabel('Pruning Percentage (%)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Test Accuracy (%)', color=color1, fontsize=13, fontweight='bold')
    line1 = ax1.plot(pruning_pcts, accuracies, 'o-', color=color1, linewidth=2.5, 
                     markersize=8, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Plot model size on right y-axis
    ax2 = ax1.twinx()
    color2 = '#F18F01'
    ax2.set_ylabel('Model Size (% of Original)', color=color2, fontsize=13, fontweight='bold')
    model_sizes = [100 * p / total_params for p in remaining_params]
    line2 = ax2.plot(pruning_pcts, model_sizes, 's--', color=color2, linewidth=2.5, 
                     markersize=8, label='Model Size')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', fontsize=11)
    
    plt.title('Accuracy vs Model Size Trade-off\n(Demonstrating Efficiency Gains from Pruning)', 
              fontsize=15, fontweight='bold', pad=20)
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def create_summary_table(results, save_path='images/results_table.png'):
    """
    Create a visual table summarizing all results.
    """
    data = []
    for r in results['pruning_results']:
        data.append([
            f"{r['pruning_percentage']}%",
            f"{r['accuracy']:.2f}%",
            f"{r['accuracy_difference']:+.2f}%",
            f"{r['remaining_parameters']:,}",
            f"{100 * r['remaining_parameters'] / r['total_parameters']:.1f}%"
        ])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=data,
                    colLabels=['Pruning %', 'Accuracy', 'Δ from Baseline', 
                              'Parameters', '% of Original'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.15, 0.15, 0.2, 0.25, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    plt.title('Experimental Results Summary\n(Lottery Ticket Hypothesis on MNIST Dataset)', 
              fontsize=16, fontweight='bold', pad=20)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    # Load results
    print("\nLoading experimental results...")
    results = load_results()
    
    # Generate all plots
    print("\nCreating visualizations...")
    plot_accuracy_vs_pruning(results)
    plot_parameter_reduction(results)
    plot_accuracy_difference(results)
    plot_combined_metrics(results)
    create_summary_table(results)
    
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("Check the 'images/' directory for output files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
