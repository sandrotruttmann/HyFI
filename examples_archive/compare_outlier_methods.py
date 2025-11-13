#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparison script for outlier detection methods (DBSCAN vs LOF vs Isolation Forest)

This script compares the performance of DBSCAN, Local Outlier Factor (LOF),
and Isolation Forest algorithms for outlier detection in hypocenter datasets.

@author: Sandro Truttmann
@date: 2025
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hyfi.core.fault_network import DBSCAN_outlier_detection, LOF_outlier_detection, IsolationForest_outlier_detection


def compare_outlier_methods(hypo_file, hypo_sep=',', 
                           lof_n_neighbors=None, lof_contamination='auto',
                           if_n_estimators=100, if_contamination=0.05, if_random_state=42):
    """
    Compare DBSCAN, LOF, and Isolation Forest outlier detection methods on the same dataset.
    
    Parameters
    ----------
    hypo_file : str
        Path to hypocenter data file
    hypo_sep : str
        Separator for the data file
    lof_n_neighbors : int or None
        Number of neighbors for LOF (None for auto-tuning)
    lof_contamination : float or 'auto'
        Expected proportion of outliers for LOF
    if_n_estimators : int
        Number of trees for Isolation Forest
    if_contamination : float or 'auto'
        Expected proportion of outliers for Isolation Forest (default: 0.05 = 5%, conservative)
    if_random_state : int
        Random seed for Isolation Forest
    """
    print("="*70)
    print("OUTLIER DETECTION METHOD COMPARISON: DBSCAN vs LOF vs Isolation Forest")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from: {hypo_file}")
    df = pd.read_csv(hypo_file, sep=hypo_sep, dtype={'ID': str}, usecols=range(24))
    
    # Extract date information
    df['Date'] = pd.to_datetime(pd.DataFrame({
        'year': df['YR'],
        'month': df['MO'],
        'day': df['DY'],
        'hour': df['HR'],
        'minute': df['MI'],
        'second': df['SC']
    }))
    
    print(f"Dataset size: {len(df)} events")
    print(f"Spatial extent: X=[{df['X'].min():.0f}, {df['X'].max():.0f}], "
          f"Y=[{df['Y'].min():.0f}, {df['Y'].max():.0f}], "
          f"Z=[{df['Z'].min():.0f}, {df['Z'].max():.0f}]")
    
    # Test DBSCAN
    print("\n" + "="*70)
    df_dbscan = df.copy()
    df_dbscan = DBSCAN_outlier_detection(df_dbscan)
    dbscan_outliers = df_dbscan[df_dbscan['clust_labels'] == -1]
    dbscan_outlier_count = len(dbscan_outliers)
    
    # Test LOF
    print("\n" + "="*70)
    df_lof = df.copy()
    df_lof = LOF_outlier_detection(df_lof, 
                                    n_neighbors=lof_n_neighbors,
                                    contamination=lof_contamination)
    lof_outliers = df_lof[df_lof['clust_labels'] == -1]
    lof_outlier_count = len(lof_outliers)
    
    # Test Isolation Forest
    print("\n" + "="*70)
    df_iforest = df.copy()
    df_iforest = IsolationForest_outlier_detection(df_iforest,
                                                    n_estimators=if_n_estimators,
                                                    contamination=if_contamination,
                                                    random_state=if_random_state)
    iforest_outliers = df_iforest[df_iforest['clust_labels'] == -1]
    iforest_outlier_count = len(iforest_outliers)
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\nDBSCAN outliers:          {dbscan_outlier_count:4d} ({100*dbscan_outlier_count/len(df):5.2f}%)")
    print(f"LOF outliers:             {lof_outlier_count:4d} ({100*lof_outlier_count/len(df):5.2f}%)")
    print(f"Isolation Forest outliers: {iforest_outlier_count:4d} ({100*iforest_outlier_count/len(df):5.2f}%)")
    
    # Find agreement and disagreement
    dbscan_outlier_ids = set(dbscan_outliers['ID'])
    lof_outlier_ids = set(lof_outliers['ID'])
    iforest_outlier_ids = set(iforest_outliers['ID'])
    
    all_three = dbscan_outlier_ids & lof_outlier_ids & iforest_outlier_ids
    any_method = dbscan_outlier_ids | lof_outlier_ids | iforest_outlier_ids
    only_dbscan = dbscan_outlier_ids - lof_outlier_ids - iforest_outlier_ids
    only_lof = lof_outlier_ids - dbscan_outlier_ids - iforest_outlier_ids
    only_iforest = iforest_outlier_ids - dbscan_outlier_ids - lof_outlier_ids
    dbscan_lof = (dbscan_outlier_ids & lof_outlier_ids) - iforest_outlier_ids
    dbscan_iforest = (dbscan_outlier_ids & iforest_outlier_ids) - lof_outlier_ids
    lof_iforest = (lof_outlier_ids & iforest_outlier_ids) - dbscan_outlier_ids
    
    print(f"\nAgreement:")
    print(f"  All three methods:    {len(all_three):4d} ({100*len(all_three)/len(df):5.2f}%)")
    print(f"  DBSCAN & LOF only:    {len(dbscan_lof):4d} ({100*len(dbscan_lof)/len(df):5.2f}%)")
    print(f"  DBSCAN & IForest only: {len(dbscan_iforest):4d} ({100*len(dbscan_iforest)/len(df):5.2f}%)")
    print(f"  LOF & IForest only:    {len(lof_iforest):4d} ({100*len(lof_iforest)/len(df):5.2f}%)")
    print(f"  Only DBSCAN:          {len(only_dbscan):4d} ({100*len(only_dbscan)/len(df):5.2f}%)")
    print(f"  Only LOF:             {len(only_lof):4d} ({100*len(only_lof)/len(df):5.2f}%)")
    print(f"  Only IForest:         {len(only_iforest):4d} ({100*len(only_iforest)/len(df):5.2f}%)")
    
    consensus_rate = len(all_three) / max(len(any_method), 1) * 100
    print(f"  Consensus rate:       {consensus_rate:5.2f}% (all three agree / any detected)")
    
    # Create visualization
    fig = plt.figure(figsize=(18, 10))
    
    # Plot 1: XY view
    ax1 = fig.add_subplot(231)
    ax1.scatter(df['X'], df['Y'], c='lightgray', s=20, alpha=0.5, label='Inliers')
    if len(all_three) > 0:
        mask = df['ID'].isin(all_three)
        ax1.scatter(df.loc[mask, 'X'], df.loc[mask, 'Y'], 
                   c='red', s=80, marker='*', label=f'All 3 ({len(all_three)})', edgecolors='black', linewidths=0.5)
    if len(only_dbscan) > 0:
        mask = df['ID'].isin(only_dbscan)
        ax1.scatter(df.loc[mask, 'X'], df.loc[mask, 'Y'], 
                   c='blue', s=50, marker='^', label=f'DBSCAN only ({len(only_dbscan)})')
    if len(only_lof) > 0:
        mask = df['ID'].isin(only_lof)
        ax1.scatter(df.loc[mask, 'X'], df.loc[mask, 'Y'], 
                   c='green', s=50, marker='s', label=f'LOF only ({len(only_lof)})')
    if len(only_iforest) > 0:
        mask = df['ID'].isin(only_iforest)
        ax1.scatter(df.loc[mask, 'X'], df.loc[mask, 'Y'], 
                   c='orange', s=50, marker='D', label=f'IForest only ({len(only_iforest)})')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Map View (XY)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: XZ view (cross-section)
    ax2 = fig.add_subplot(232)
    ax2.scatter(df['X'], df['Z'], c='lightgray', s=20, alpha=0.5, label='Inliers')
    if len(all_three) > 0:
        mask = df['ID'].isin(all_three)
        ax2.scatter(df.loc[mask, 'X'], df.loc[mask, 'Z'], 
                   c='red', s=80, marker='*', label=f'All 3 ({len(all_three)})', edgecolors='black', linewidths=0.5)
    if len(only_dbscan) > 0:
        mask = df['ID'].isin(only_dbscan)
        ax2.scatter(df.loc[mask, 'X'], df.loc[mask, 'Z'], 
                   c='blue', s=50, marker='^', label=f'DBSCAN only ({len(only_dbscan)})')
    if len(only_lof) > 0:
        mask = df['ID'].isin(only_lof)
        ax2.scatter(df.loc[mask, 'X'], df.loc[mask, 'Z'], 
                   c='green', s=50, marker='s', label=f'LOF only ({len(only_lof)})')
    if len(only_iforest) > 0:
        mask = df['ID'].isin(only_iforest)
        ax2.scatter(df.loc[mask, 'X'], df.loc[mask, 'Z'], 
                   c='orange', s=50, marker='D', label=f'IForest only ({len(only_iforest)})')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Cross-section (XZ)')
    ax2.invert_yaxis()
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: YZ view
    ax3 = fig.add_subplot(233)
    ax3.scatter(df['Y'], df['Z'], c='lightgray', s=20, alpha=0.5, label='Inliers')
    if len(all_three) > 0:
        mask = df['ID'].isin(all_three)
        ax3.scatter(df.loc[mask, 'Y'], df.loc[mask, 'Z'], 
                   c='red', s=80, marker='*', label=f'All 3 ({len(all_three)})', edgecolors='black', linewidths=0.5)
    if len(only_dbscan) > 0:
        mask = df['ID'].isin(only_dbscan)
        ax3.scatter(df.loc[mask, 'Y'], df.loc[mask, 'Z'], 
                   c='blue', s=50, marker='^', label=f'DBSCAN only ({len(only_dbscan)})')
    if len(only_lof) > 0:
        mask = df['ID'].isin(only_lof)
        ax3.scatter(df.loc[mask, 'Y'], df.loc[mask, 'Z'], 
                   c='green', s=50, marker='s', label=f'LOF only ({len(only_lof)})')
    if len(only_iforest) > 0:
        mask = df['ID'].isin(only_iforest)
        ax3.scatter(df.loc[mask, 'Y'], df.loc[mask, 'Z'], 
                   c='orange', s=50, marker='D', label=f'IForest only ({len(only_iforest)})')
    ax3.set_xlabel('Y (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Cross-section (YZ)')
    ax3.invert_yaxis()
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: LOF scores histogram
    ax4 = fig.add_subplot(234)
    if 'lof_score' in df_lof.columns:
        inlier_scores = df_lof[df_lof['clust_labels'] == 0]['lof_score']
        outlier_scores = df_lof[df_lof['clust_labels'] == -1]['lof_score']
        
        bins = np.linspace(df_lof['lof_score'].min(), df_lof['lof_score'].max(), 50)
        ax4.hist(inlier_scores, bins=bins, alpha=0.6, label='LOF Inliers', color='blue')
        if len(outlier_scores) > 0:
            ax4.hist(outlier_scores, bins=bins, alpha=0.6, label='LOF Outliers', color='red')
        ax4.set_xlabel('LOF Score')
        ax4.set_ylabel('Count')
        ax4.set_title('LOF Score Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Isolation Forest scores histogram
    ax5 = fig.add_subplot(235)
    if 'isolation_score' in df_iforest.columns:
        inlier_scores = df_iforest[df_iforest['clust_labels'] == 0]['isolation_score']
        outlier_scores = df_iforest[df_iforest['clust_labels'] == -1]['isolation_score']
        
        bins = np.linspace(df_iforest['isolation_score'].min(), df_iforest['isolation_score'].max(), 50)
        ax5.hist(inlier_scores, bins=bins, alpha=0.6, label='IForest Inliers', color='blue')
        if len(outlier_scores) > 0:
            ax5.hist(outlier_scores, bins=bins, alpha=0.6, label='IForest Outliers', color='red')
        ax5.set_xlabel('Isolation Score')
        ax5.set_ylabel('Count')
        ax5.set_title('Isolation Forest Score Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Venn-style comparison
    ax6 = fig.add_subplot(236)
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection
    
    # Create a simple bar chart showing overlap
    categories = ['All 3', 'DBSCAN\n& LOF', 'DBSCAN\n& IForest', 'LOF\n& IForest', 
                 'Only\nDBSCAN', 'Only\nLOF', 'Only\nIForest']
    counts = [len(all_three), len(dbscan_lof), len(dbscan_iforest), len(lof_iforest),
             len(only_dbscan), len(only_lof), len(only_iforest)]
    colors = ['red', 'purple', 'brown', 'teal', 'blue', 'green', 'orange']
    
    bars = ax6.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax6.set_ylabel('Number of Outliers')
    ax6.set_title('Method Agreement Breakdown')
    ax6.grid(True, alpha=0.3, axis='y')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_file = hypo_file.replace('.csv', '_outlier_comparison_3methods.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    
    # Show plot
    plt.show()
    
    # Save detailed comparison to CSV
    comparison_file = hypo_file.replace('.csv', '_outlier_comparison_3methods.csv')
    df_comparison = df.copy()
    df_comparison['DBSCAN_outlier'] = df_comparison['ID'].isin(dbscan_outlier_ids)
    df_comparison['LOF_outlier'] = df_comparison['ID'].isin(lof_outlier_ids)
    df_comparison['IForest_outlier'] = df_comparison['ID'].isin(iforest_outlier_ids)
    df_comparison['lof_score'] = df_lof['lof_score']
    df_comparison['isolation_score'] = df_iforest['isolation_score']
    df_comparison['outlier_count'] = (df_comparison['DBSCAN_outlier'].astype(int) + 
                                     df_comparison['LOF_outlier'].astype(int) + 
                                     df_comparison['IForest_outlier'].astype(int))
    df_comparison.to_csv(comparison_file, index=False)
    print(f"Detailed comparison saved to: {comparison_file}")
    
    return df_dbscan, df_lof, df_iforest


if __name__ == '__main__':
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare DBSCAN, LOF, and Isolation Forest outlier detection methods')
    parser.add_argument('hypo_file', type=str, help='Path to hypocenter data file')
    parser.add_argument('--sep', type=str, default=',', help='Data file separator (default: comma)')
    parser.add_argument('--lof-neighbors', type=int, default=None, 
                       help='Number of neighbors for LOF (default: auto-tuned)')
    parser.add_argument('--lof-contamination', type=str, default='auto',
                       help='LOF contamination parameter (default: auto)')
    parser.add_argument('--if-estimators', type=int, default=100,
                       help='Number of trees for Isolation Forest (default: 100)')
    parser.add_argument('--if-contamination', type=str, default='0.05',
                       help='Isolation Forest contamination parameter (default: 0.05 = 5%, conservative)')
    parser.add_argument('--if-random-state', type=int, default=42,
                       help='Random seed for Isolation Forest (default: 42)')
    
    args = parser.parse_args()
    
    # Convert contamination to float if it's not 'auto'
    lof_contamination = args.lof_contamination
    if lof_contamination != 'auto':
        try:
            lof_contamination = float(lof_contamination)
        except ValueError:
            print(f"Warning: Invalid LOF contamination value '{lof_contamination}', using 'auto'")
            lof_contamination = 'auto'
    
    if_contamination = args.if_contamination
    if if_contamination != 'auto':
        try:
            if_contamination = float(if_contamination)
        except ValueError:
            print(f"Warning: Invalid IForest contamination value '{if_contamination}', using 'auto'")
            if_contamination = 'auto'
    
    # Run comparison
    df_dbscan, df_lof, df_iforest = compare_outlier_methods(
        args.hypo_file,
        hypo_sep=args.sep,
        lof_n_neighbors=args.lof_neighbors,
        lof_contamination=lof_contamination,
        if_n_estimators=args.if_estimators,
        if_contamination=if_contamination,
        if_random_state=args.if_random_state
    )
