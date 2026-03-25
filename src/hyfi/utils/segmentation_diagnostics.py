"""
Diagnostic plots for catalog segmentation.

Generates visualisations that help tune ``dbscan_eps`` and
``temporal_window_days`` before committing to a full HyFI run.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* that has a ``Date`` column (datetime)."""
    df = df.copy()
    if 'Date' not in df.columns:
        df['Date'] = pd.to_datetime(pd.DataFrame({
            'year':   df['YR'], 'month': df['MO'], 'day':    df['DY'],
            'hour':   df['HR'], 'minute': df['MI'], 'second': df['SC']
        }))
    else:
        df['Date'] = pd.to_datetime(df['Date'])
    return df


def _cluster_color_map(unique_labels, cmap_name='tab20'):
    """Return a dict mapping cluster_label -> RGBA colour array."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # noise (-1) always gets light grey
    valid = sorted(lbl for lbl in unique_labels if lbl != -1)
    cmap = cm.get_cmap(cmap_name, max(len(valid), 1))
    color_map = {lbl: cmap(i) for i, lbl in enumerate(valid)}
    color_map[-1] = (0.75, 0.75, 0.75, 0.4)   # noise → transparent grey
    return color_map


def _sequence_color_map(sequence_names, cmap_name='tab20'):
    """Return a dict mapping sequence_name -> RGBA colour."""
    import matplotlib.cm as cm
    non_outlier = [n for n in sequence_names if n != 'Z_outliers']
    cmap = cm.get_cmap(cmap_name, max(len(non_outlier), 1))
    color_map = {n: cmap(i) for i, n in enumerate(non_outlier)}
    color_map['Z_outliers'] = (0.75, 0.75, 0.75, 0.5)
    return color_map


def _spatial_extent_m(df: pd.DataFrame) -> float:
    """Bounding-box diagonal in metres (using X, Y, Z coords)."""
    if len(df) < 2:
        return 0.0
    dx = df['X'].max() - df['X'].min()
    dy = df['Y'].max() - df['Y'].min()
    dz = df['Z'].max() - df['Z'].min() if 'Z' in df.columns else 0.0
    return float(np.sqrt(dx**2 + dy**2 + dz**2))


def _temporal_span_days(df: pd.DataFrame) -> float:
    """Date range in days."""
    df = _ensure_date_column(df)
    span = (df['Date'].max() - df['Date'].min()).total_seconds() / 86400.0
    return max(span, 0.0)


def _elongated_axis(catalog: pd.DataFrame) -> str:
    """Return 'X' or 'Y', whichever has the larger spatial spread in *catalog*."""
    x_range = catalog['X'].max() - catalog['X'].min()
    y_range = catalog['Y'].max() - catalog['Y'].min()
    return 'X' if x_range >= y_range else 'Y'


# ---------------------------------------------------------------------------
# Per-step diagnostic figure
# ---------------------------------------------------------------------------

def plot_segmentation_step_diagnostics(
    catalog: pd.DataFrame,
    cluster_labels: np.ndarray,
    step_name: str,
    step_level: str,
    clustering_params: Dict[str, Any],
    features: List[str],
    output_dir: str,
) -> None:
    """
    Save a multi-panel diagnostic figure for **one** segmentation step.

    Parameters
    ----------
    catalog : pd.DataFrame
        Events fed into this step (may be a subset of the full catalog).
    cluster_labels : np.ndarray
        Integer cluster labels from the clustering algorithm (-1 = noise).
    step_name : str
        Human-readable step name (used in title and filename).
    step_level : str
        Single letter ('A', 'B', …) identifying the hierarchy level.
    clustering_params : dict
        The full dict passed to the clustering algorithm (eps, temporal_window_days, …).
    features : list of str
        Feature names used ('spatial', 'temporal', 'magnitude').
    output_dir : str
        Root output directory; figures are saved into a ``diagnostics/`` sub-folder.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.lines as mlines
    except ImportError:
        logger.warning("matplotlib not available – skipping segmentation diagnostics")
        return

    diag_dir = Path(output_dir) / 'diagnostics'
    diag_dir.mkdir(parents=True, exist_ok=True)

    catalog = _ensure_date_column(catalog).copy()
    catalog['_label'] = cluster_labels

    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = int((cluster_labels == -1).sum())

    eps = clustering_params.get('dbscan_eps', None)
    temporal_window_days = clustering_params.get('temporal_window_days', None)
    use_temporal = 'temporal' in features
    use_spatial = 'spatial' in features
    dim = clustering_params.get('cluster_dimension', '3d')
    horiz_col = _elongated_axis(catalog)

    color_map = _cluster_color_map(unique_labels)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        f"Segmentation Step {step_level} — \"{step_name}\"\n"
        f"Input events: {len(catalog)}  |  Clusters found: {n_clusters}  |  Noise: {n_noise} ({100*n_noise/max(len(catalog),1):.1f}%)",
        fontsize=13, fontweight='bold'
    )

    # ── Panel 1: Map view (X-Y) ────────────────────────────────────────────
    ax = axes[0, 0]
    for lbl in sorted(unique_labels, key=lambda x: (x == -1, x)):
        mask = catalog['_label'] == lbl
        subset = catalog[mask]
        label_str = 'noise' if lbl == -1 else f'cluster {lbl}'
        zorder = 1 if lbl == -1 else 2
        s = 6 if lbl == -1 else 10
        ax.scatter(subset['X'], subset['Y'],
                   c=[color_map[lbl]] * len(subset),
                   s=s, zorder=zorder, label=label_str, rasterized=True)

    # Cluster centroids
    for lbl in sorted(unique_labels):
        if lbl == -1:
            continue
        mask = catalog['_label'] == lbl
        cx, cy = catalog.loc[mask, 'X'].mean(), catalog.loc[mask, 'Y'].mean()
        ax.scatter(cx, cy, marker='*', s=150, c=[color_map[lbl]], edgecolors='k',
                   linewidths=0.5, zorder=5)

    # eps scale circle in lower-left corner
    if eps is not None and use_spatial:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # Re-get after scatter
        ax.autoscale()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        cx_ref = xlim[0] + 0.10 * (xlim[1] - xlim[0])
        cy_ref = ylim[0] + 0.10 * (ylim[1] - ylim[0])
        circle = mpatches.Circle((cx_ref, cy_ref), radius=eps,
                                  fill=False, edgecolor='black',
                                  linewidth=1.5, linestyle='--', zorder=6)
        ax.add_patch(circle)
        ax.text(cx_ref, cy_ref - eps * 1.25,
                f'eps={eps:.0f} m', ha='center', va='top', fontsize=7,
                color='black', bbox=dict(fc='white', ec='none', alpha=0.7))

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Map view (X–Y)', fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Depth section (X-Z) ──────────────────────────────────────
    ax = axes[0, 1]
    if 'Z' in catalog.columns:
        for lbl in sorted(unique_labels, key=lambda x: (x == -1, x)):
            mask = catalog['_label'] == lbl
            subset = catalog[mask]
            zorder = 1 if lbl == -1 else 2
            s = 6 if lbl == -1 else 10
            ax.scatter(subset['X'], subset['Z'],
                       c=[color_map[lbl]] * len(subset),
                       s=s, zorder=zorder, rasterized=True)
        # Centroids
        for lbl in sorted(unique_labels):
            if lbl == -1:
                continue
            mask = catalog['_label'] == lbl
            cx, cz = catalog.loc[mask, 'X'].mean(), catalog.loc[mask, 'Z'].mean()
            ax.scatter(cx, cz, marker='*', s=150, c=[color_map[lbl]],
                       edgecolors='k', linewidths=0.5, zorder=5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m, depth)')
        ax.set_title('Depth section (X–Z)', fontsize=10)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No Z column available', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Depth section (N/A)', fontsize=10)

    # ── Panel 3: Space-Time (Date vs elongated axis) ──────────────────────
    ax = axes[0, 2]
    for lbl in sorted(unique_labels, key=lambda x: (x == -1, x)):
        mask = catalog['_label'] == lbl
        subset = catalog[mask]
        zorder = 1 if lbl == -1 else 2
        s = 6 if lbl == -1 else 10
        ax.scatter(subset[horiz_col],
                   subset['Date'].values.astype('datetime64[s]').astype(float) / 86400.0,
                   c=[color_map[lbl]] * len(subset),
                   s=s, zorder=zorder, rasterized=True)

    # Show temporal_window_days as a labelled horizontal scale bar (at top-right)
    if use_temporal and temporal_window_days is not None:
        ylim = ax.get_ylim()
        xlim = ax.get_xlim()
        y_bar = ylim[1] - 0.05 * (ylim[1] - ylim[0])
        x_bar_end = xlim[1] - 0.02 * (xlim[1] - xlim[0])
        x_bar_start = x_bar_end - (xlim[1] - xlim[0]) * 0.15
        ax.annotate('', xy=(x_bar_start, y_bar), xytext=(x_bar_start, y_bar - temporal_window_days),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        ax.text((x_bar_start + x_bar_end) / 2, y_bar - temporal_window_days / 2,
                f'{temporal_window_days} d', ha='right', va='center', fontsize=7,
                color='black', bbox=dict(fc='white', ec='none', alpha=0.7))

    # Format Y axis as dates
    _date_epoch = catalog['Date'].min()
    def _day_fmt(x, pos):
        try:
            d = _date_epoch + pd.Timedelta(days=x)
            return d.strftime('%Y-%m-%d')
        except Exception:
            return ''

    import matplotlib.ticker as mticker
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_day_fmt))
    ax.set_xlabel(f'{horiz_col} (m)')
    ax.set_ylabel('Date')
    ax.set_title(f'Space-Time map ({horiz_col} vs Date)', fontsize=10)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: Event count per cluster ──────────────────────────────────
    ax = axes[1, 0]
    cluster_ids = sorted(lbl for lbl in unique_labels if lbl != -1)
    counts = [int((catalog['_label'] == lbl).sum()) for lbl in cluster_ids]
    bar_colors = [color_map[lbl] for lbl in cluster_ids]
    bars = ax.bar(range(len(cluster_ids)), counts, color=bar_colors, edgecolor='k', linewidth=0.5)
    ax.set_xticks(range(len(cluster_ids)))
    ax.set_xticklabels([f'C{lbl}' for lbl in cluster_ids], fontsize=8)
    ax.set_ylabel('Event count')
    ax.set_title('Events per cluster', fontsize=10)
    ax.axhline(clustering_params.get('min_cluster_size', 0), color='red', linestyle='--',
               linewidth=1, label=f'min_size={clustering_params.get("min_cluster_size", "?")}')
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(cnt), ha='center', va='bottom', fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    # ── Panel 5: Temporal span per cluster ────────────────────────────────
    ax = axes[1, 1]
    spans = []
    for lbl in cluster_ids:
        mask = catalog['_label'] == lbl
        spans.append(_temporal_span_days(catalog[mask]))
    bars = ax.bar(range(len(cluster_ids)), spans, color=bar_colors, edgecolor='k', linewidth=0.5)
    if temporal_window_days is not None:
        ax.axhline(temporal_window_days, color='blue', linestyle='--', linewidth=1,
                   label=f'temporal_window={temporal_window_days} d')
    ax.set_xticks(range(len(cluster_ids)))
    ax.set_xticklabels([f'C{lbl}' for lbl in cluster_ids], fontsize=8)
    ax.set_ylabel('Duration (days)')
    ax.set_title('Temporal span per cluster', fontsize=10)
    for bar, v in zip(bars, spans):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{v:.0f}', ha='center', va='bottom', fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    # ── Panel 6: Spatial extent per cluster ───────────────────────────────
    ax = axes[1, 2]
    extents = []
    for lbl in cluster_ids:
        mask = catalog['_label'] == lbl
        extents.append(_spatial_extent_m(catalog[mask]) / 1000.0)   # → km
    bars = ax.bar(range(len(cluster_ids)), extents, color=bar_colors, edgecolor='k', linewidth=0.5)
    if eps is not None:
        ax.axhline(eps / 1000.0, color='darkgreen', linestyle='--', linewidth=1,
                   label=f'eps={eps:.0f} m')
    ax.set_xticks(range(len(cluster_ids)))
    ax.set_xticklabels([f'C{lbl}' for lbl in cluster_ids], fontsize=8)
    ax.set_ylabel('Bounding-box diagonal (km)')
    ax.set_title('Spatial extent per cluster (bbox diagonal)', fontsize=10)
    for bar, v in zip(bars, extents):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{v:.1f}', ha='center', va='bottom', fontsize=7)
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    # Legend patch for noise
    noise_patch = mpatches.Patch(facecolor=(0.75, 0.75, 0.75, 0.6), label=f'noise ({n_noise})')
    centroid_marker = mlines.Line2D([], [], marker='*', color='k', markersize=8,
                                    linestyle='None', label='centroid')
    fig.legend(handles=[noise_patch, centroid_marker],
               loc='lower center', ncol=2, fontsize=8, framealpha=0.8)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    safe_name = step_name.replace(' ', '_').replace('/', '_')
    out_path = diag_dir / f'step_{step_level}_{safe_name}_diagnostics.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [diagnostics] Saved step diagnostic: {out_path}")


# ---------------------------------------------------------------------------
# Final overview figure
# ---------------------------------------------------------------------------

def plot_segmentation_overview(
    catalog: pd.DataFrame,
    sequences: Dict[str, pd.DataFrame],
    output_dir: str,
) -> None:
    """
    Save a final overview figure showing **all sequences** after segmentation.

    Parameters
    ----------
    catalog : pd.DataFrame
        Full input catalog (all events, used for background reference).
    sequences : dict
        Mapping of sequence name → DataFrame as returned by
        ``multi_step_catalog_segmentation``.
    output_dir : str
        Root output directory; figure is saved as
        ``diagnostics/segmentation_overview.png``.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available – skipping segmentation overview")
        return

    diag_dir = Path(output_dir) / 'diagnostics'
    diag_dir.mkdir(parents=True, exist_ok=True)

    catalog = _ensure_date_column(catalog)
    horiz_col = _elongated_axis(catalog)
    seq_names = list(sequences.keys())
    color_map = _sequence_color_map(seq_names)

    n_seqs = len([n for n in seq_names if n != 'Z_outliers'])
    n_outliers = len(sequences.get('Z_outliers', pd.DataFrame()))
    total = sum(len(v) for v in sequences.values())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Segmentation Overview — {n_seqs} sequences  |  {n_outliers} outliers  |  {total} total events",
        fontsize=14, fontweight='bold'
    )

    # Ensure all sequence DFs have dates
    seq_dfs = {name: _ensure_date_column(df) for name, df in sequences.items()}

    # ── Panel 1: Final map (X-Y) ───────────────────────────────────────────
    ax = axes[0, 0]
    # Background: all events lightly
    ax.scatter(catalog['X'], catalog['Y'], s=4, c='0.85', zorder=1, rasterized=True)
    for name in seq_names:
        df = seq_dfs[name]
        col = color_map[name]
        zorder = 2 if name != 'Z_outliers' else 1
        s = 10 if name != 'Z_outliers' else 5
        ax.scatter(df['X'], df['Y'], c=[col] * len(df), s=s, zorder=zorder,
                   label=f'{name} ({len(df)})', rasterized=True)
        if name != 'Z_outliers' and len(df) > 0:
            ax.scatter(df['X'].mean(), df['Y'].mean(),
                       marker='*', s=180, c=[col], edgecolors='k',
                       linewidths=0.5, zorder=6)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Map view – final sequences (X–Y)', fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Space-Time map (elongated axis vs Date) ──────────────────
    ax = axes[0, 1]
    t0 = catalog['Date'].min()
    catalog_days = (catalog['Date'] - t0).dt.total_seconds() / 86400.0
    ax.scatter(catalog[horiz_col], catalog_days, s=4, c='0.85', zorder=1, rasterized=True)
    for name in seq_names:
        df = seq_dfs[name]
        col = color_map[name]
        zorder = 2 if name != 'Z_outliers' else 1
        s = 10 if name != 'Z_outliers' else 5
        df_days = (df['Date'] - t0).dt.total_seconds() / 86400.0
        ax.scatter(df[horiz_col], df_days, c=[col] * len(df), s=s, zorder=zorder,
                   rasterized=True)
        if name != 'Z_outliers' and len(df) > 0:
            ax.scatter(df[horiz_col].mean(), df_days.mean(),
                       marker='*', s=180, c=[col], edgecolors='k',
                       linewidths=0.5, zorder=6)

    import matplotlib.ticker as _mticker
    _date_epoch = t0
    def _day_fmt(x, pos):
        try:
            return (_date_epoch + pd.Timedelta(days=x)).strftime('%Y-%m-%d')
        except Exception:
            return ''
    ax.yaxis.set_major_formatter(_mticker.FuncFormatter(_day_fmt))
    ax.set_xlabel(f'{horiz_col} (m)')
    ax.set_ylabel('Date')
    ax.set_title(f'Space-Time map – final sequences ({horiz_col} vs Date)', fontsize=10)
    ax.tick_params(axis='y', labelsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Timeline (horizontal bars per sequence) ──────────────────
    ax = axes[1, 0]
    sorted_seqs = sorted(
        [(n, seq_dfs[n]) for n in seq_names if n != 'Z_outliers'],
        key=lambda kv: kv[1]['Date'].min() if len(kv[1]) > 0 else pd.Timestamp.max
    )
    for i, (name, df) in enumerate(sorted_seqs):
        if len(df) == 0:
            continue
        t_start = df['Date'].min()
        t_end = df['Date'].max()
        col = color_map[name]
        ax.barh(i, (t_end - t_start).total_seconds() / 86400.0,
                left=(t_start - catalog['Date'].min()).total_seconds() / 86400.0,
                color=col, edgecolor='k', linewidth=0.5, height=0.7)
        # Mark centroid time
        t_center = t_start + (t_end - t_start) / 2
        ax.scatter((t_center - catalog['Date'].min()).total_seconds() / 86400.0,
                   i, marker='|', s=80, c='k', zorder=5)
        ax.text(
            (t_end - catalog['Date'].min()).total_seconds() / 86400.0 + 1,
            i, f'{name} ({len(df)})',
            va='center', fontsize=7
        )

    # Outliers: mark as scattered individual ticks
    if 'Z_outliers' in seq_dfs and len(seq_dfs['Z_outliers']) > 0:
        out_df = seq_dfs['Z_outliers']
        days_offset = (out_df['Date'] - catalog['Date'].min()).dt.total_seconds() / 86400.0
        y_out = len(sorted_seqs)
        ax.scatter(days_offset, [y_out] * len(out_df),
                   c='0.6', s=5, marker='|', zorder=2)
        ax.text(days_offset.max() + 1, y_out,
                f'Z_outliers ({len(out_df)})', va='center', fontsize=7)

    ax.set_yticks([])
    ax.set_xlabel(f'Days since {catalog["Date"].min().strftime("%Y-%m-%d")}')
    ax.set_title('Sequence timeline (horizontal extent = duration)', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # ── Panel 4: Event counts & sequence properties ────────────────────────
    ax = axes[1, 1]
    all_names_sorted = [n for n, _ in sorted_seqs] + (
        ['Z_outliers'] if 'Z_outliers' in sequences else []
    )
    counts = [len(sequences[n]) for n in all_names_sorted]
    bar_colors = [color_map[n] for n in all_names_sorted]
    bars = ax.bar(range(len(all_names_sorted)), counts,
                  color=bar_colors, edgecolor='k', linewidth=0.5)
    ax.set_xticks(range(len(all_names_sorted)))
    ax.set_xticklabels(all_names_sorted, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Event count')
    ax.set_title('Event count per sequence', fontsize=10)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(cnt), ha='center', va='bottom', fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    # Legend
    legend_handles = []
    for name in seq_names:
        patch = mpatches.Patch(facecolor=color_map[name], edgecolor='k',
                               linewidth=0.5, label=f'{name} ({len(sequences[name])})')
        legend_handles.append(patch)
    ax.legend(handles=legend_handles, fontsize=7, loc='upper right',
              framealpha=0.8, ncol=max(1, len(legend_handles) // 10))

    plt.tight_layout()

    out_path = diag_dir / 'segmentation_overview.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [diagnostics] Saved segmentation overview: {out_path}")


# ---------------------------------------------------------------------------
# Convenience: parameter sensitivity summary
# ---------------------------------------------------------------------------

def plot_parameter_reference(
    catalog: pd.DataFrame,
    segmentation_steps: List,
    output_dir: str,
) -> None:
    """
    Save a quick one-page summary showing the key parameter scales (eps, temporal
    window) overlaid on the raw catalog to help judge parameter choices.

    Parameters
    ----------
    catalog : pd.DataFrame
        Full input catalog.
    segmentation_steps : list of SegmentationStep
        All steps from the config.
    output_dir : str
        Root output directory.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        return

    catalog = _ensure_date_column(catalog)
    horiz_col = _elongated_axis(catalog)
    diag_dir = Path(output_dir) / 'diagnostics'
    diag_dir.mkdir(parents=True, exist_ok=True)

    n_steps = len(segmentation_steps)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Parameter reference — raw catalog with eps / temporal_window_days scales',
                 fontsize=12, fontweight='bold')

    # Colours for steps
    import matplotlib.cm as cm
    step_cmap = cm.get_cmap('Set1', max(n_steps, 1))

    # ── Left: Map view with eps circles ───────────────────────────────────
    ax = axes[0]
    ax.scatter(catalog['X'], catalog['Y'], s=4, c='0.7', zorder=1, rasterized=True,
               label='all events')

    # Place each step's eps circle at the data centroid, stacked slightly offset
    cx = catalog['X'].mean()
    cy = catalog['Y'].mean()
    for i, step in enumerate(segmentation_steps):
        if not hasattr(step, 'dbscan_eps'):
            continue
        eps_val = step.dbscan_eps
        use_spatial = 'spatial' in getattr(step, 'features', ['spatial'])
        if not use_spatial:
            continue
        col = step_cmap(i)
        circle = mpatches.Circle(
            (cx, cy), radius=eps_val,
            fill=False, edgecolor=col, linewidth=2, linestyle='-', zorder=4,
            label=f'{step.step_name}: eps={eps_val:.0f} m'
        )
        ax.add_patch(circle)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Spatial: eps reference circles at data centroid', fontsize=10)
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ── Right: Space-Time with temporal_window_days bands ─────────────────
    ax = axes[1]
    days_offset = (catalog['Date'] - catalog['Date'].min()).dt.total_seconds() / 86400.0
    ax.scatter(catalog[horiz_col], days_offset, s=4, c='0.7', zorder=1, rasterized=True,
               label='all events')

    for i, step in enumerate(segmentation_steps):
        if not hasattr(step, 'temporal_window_days'):
            continue
        tw = step.temporal_window_days
        use_temporal = 'temporal' in getattr(step, 'features', [])
        if not use_temporal:
            continue
        col = step_cmap(i)
        # Draw a shaded horizontal band of height temporal_window_days at the top
        y_top = days_offset.max()
        ax.axhspan(y_top - tw, y_top, alpha=0.15, color=col,
                   label=f'{step.step_name}: window={tw} d')
        ax.annotate(
            '', xy=(catalog[horiz_col].min(), y_top),
            xytext=(catalog[horiz_col].min(), y_top - tw),
            arrowprops=dict(arrowstyle='<->', color=col, lw=2)
        )

    ax.set_xlabel(f'{horiz_col} (m)')
    ax.set_ylabel(f'Days since {catalog["Date"].min().strftime("%Y-%m-%d")}')
    ax.set_title('Temporal: window reference bands', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = diag_dir / 'parameter_reference.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [diagnostics] Saved parameter reference: {out_path}")
