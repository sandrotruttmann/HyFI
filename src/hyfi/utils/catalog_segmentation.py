# Catalog segmentation for multi-sequence analysis

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS
from typing import Tuple, List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def multi_step_catalog_segmentation(catalog: pd.DataFrame,
                                  segmentation_steps: List,
                                  final_outlier_handling: str = 'keep',
                                  max_outlier_ratio: float = 0.3) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Multi-step segmentation for full earthquake catalogs.
    
    Sequences are named with hierarchical levels:
    - Level A: First segmentation step (e.g., A1, A2, A3...)
    - Level B: Second segmentation step (e.g., B1, B2, B3...)  
    - Level C: Third segmentation step (e.g., C1, C2, C3...)
    - Z_outliers: Final unprocessed outliers
    
    Parameters
    ----------
    catalog : pd.DataFrame
        Full earthquake catalog with required columns
    segmentation_steps : List[SegmentationStep]
        List of segmentation steps to apply sequentially
    final_outlier_handling : str
        How to handle final outliers ('keep', 'discard', 'merge_largest')
    max_outlier_ratio : float
        Maximum allowed ratio of outliers to total events
        
    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]
        Dictionary of sequence DataFrames and detailed results from each step
    """
    
    logger.info(f"Starting multi-step catalog segmentation with {len(segmentation_steps)} steps")
    logger.info(f"Input catalog size: {len(catalog)} events")
    
    # Create step level mapping (A, B, C, ...)
    step_level_names = [chr(ord('A') + i) for i in range(len(segmentation_steps))]
    
    all_sequences = {}
    step_results = {}
    remaining_data = catalog.copy()
    remaining_data['original_index'] = remaining_data.index
    
    total_clustered = 0
    
    for step_idx, step in enumerate(segmentation_steps):
        step_level = step_level_names[step_idx]
        logger.info(f"\n--- Segmentation Step {step_level}: {step.step_name} ---")
        logger.info(f"Input data for this step: {len(remaining_data)} events")
        
        if len(remaining_data) < step.min_cluster_size:
            logger.info(f"Remaining data too small for segmentation (< {step.min_cluster_size}), stopping")
            break
        
        # Apply clustering to remaining data
        step_sequences, cluster_labels = advanced_catalog_segmentation(
            remaining_data,
            method=step.method,
            features=step.features,
            **_step_to_clustering_params(step)
        )
        
        # Process clusters from this step
        step_sequence_count = 0
        new_remaining_data = pd.DataFrame()
        
        for sequence_name, sequence_data in step_sequences.items():
            if sequence_name == 'noise':
                # Handle outliers according to step configuration
                if step.process_outliers and step.outlier_handling == 'next_step':
                    new_remaining_data = pd.concat([new_remaining_data, sequence_data], ignore_index=True)
                    logger.info(f"Passing {len(sequence_data)} outliers to next step")
                elif step.outlier_handling == 'merge_smallest':
                    # Merge with smallest existing cluster
                    if all_sequences:
                        smallest_sequence = min(all_sequences.keys(), 
                                             key=lambda k: len(all_sequences[k]))
                        all_sequences[smallest_sequence] = pd.concat([
                            all_sequences[smallest_sequence], sequence_data
                        ], ignore_index=True)
                        logger.info(f"Merged {len(sequence_data)} outliers with {smallest_sequence}")
                    else:
                        # No existing clusters, keep as outliers for final handling
                        new_remaining_data = pd.concat([new_remaining_data, sequence_data], ignore_index=True)
                elif step.outlier_handling == 'discard':
                    logger.info(f"Discarded {len(sequence_data)} outliers from {step.step_name}")
                else:  # Keep outliers for final handling
                    new_remaining_data = pd.concat([new_remaining_data, sequence_data], ignore_index=True)
            else:
                # Add sequence with step level prefix and continuous numbering (A1, A2, B1, B2, etc.)
                step_sequence_count += 1
                sequence_key = f"{step_level}{step_sequence_count}"
                all_sequences[sequence_key] = sequence_data
                total_clustered += len(sequence_data)
                logger.info(f"Found sequence {sequence_key}: {len(sequence_data)} events")
        
        # Store step results
        step_results[step.step_name] = {
            'input_size': len(remaining_data),
            'sequences_found': step_sequence_count,
            'events_clustered': sum(len(sequence_data) for name, sequence_data in step_sequences.items() 
                                  if name != 'noise'),
            'outliers': len(step_sequences.get('noise', pd.DataFrame())),
            'clustering_method': step.method,
            'clustering_features': step.features,
            'quality_metrics': evaluate_clustering_quality(
                remaining_data, cluster_labels, 
                _prepare_clustering_features(remaining_data, step.features, _step_to_clustering_params(step))
            ) if len(cluster_labels) > 0 else {}
        }
        
        
        # Update remaining data for next step
        remaining_data = new_remaining_data.reset_index(drop=True)
        
        # Stop if no outliers to process further
        if len(remaining_data) == 0:
            logger.info("No remaining outliers, stopping segmentation")
            break
    
    # Handle final outliers
    if len(remaining_data) > 0:
        final_outliers = _handle_final_outliers(
            remaining_data, all_sequences, final_outlier_handling
        )
        
        if final_outliers is not None and len(final_outliers) > 0:
            all_sequences['Z_outliers'] = final_outliers
    
    # Check outlier ratio
    outlier_count = len(all_sequences.get('Z_outliers', pd.DataFrame()))
    outlier_ratio = outlier_count / len(catalog)
    
    if outlier_ratio > max_outlier_ratio:
        logger.warning(f"High outlier ratio: {outlier_ratio:.2%} (threshold: {max_outlier_ratio:.2%})")
    
    # Summary
    logger.info(f"\n=== Multi-Step Segmentation Summary ===")
    logger.info(f"Total input events: {len(catalog)}")
    logger.info(f"Total sequences found: {len([k for k in all_sequences.keys() if k != 'Z_outliers'])}")
    logger.info(f"Total events clustered: {total_clustered}")
    logger.info(f"Final outliers: {outlier_count} ({outlier_ratio:.2%})")
    
    for sequence_name, sequence_data in all_sequences.items():
        logger.info(f"  {sequence_name}: {len(sequence_data)} events")
    
    # Compile overall results
    overall_results = {
        'step_results': step_results,
        'total_sequences': len([k for k in all_sequences.keys() if k != 'Z_outliers']),
        'total_events_clustered': total_clustered,
        'final_outliers': outlier_count,
        'outlier_ratio': outlier_ratio,
        'steps_executed': len(step_results)
    }
    
    return all_sequences, overall_results


def _step_to_clustering_params(step) -> Dict[str, Any]:
    """Convert SegmentationStep to clustering parameters dictionary."""
    return {
        'dbscan_eps': step.dbscan_eps,
        'dbscan_min_samples': step.dbscan_min_samples,
        'dbscan_metric': step.dbscan_metric,
        'hdbscan_min_cluster_size': step.hdbscan_min_cluster_size,
        'hdbscan_min_samples': step.hdbscan_min_samples,
        'optics_min_samples': step.optics_min_samples,
        'optics_max_eps': step.optics_max_eps,
        'optics_cluster_method': step.optics_cluster_method,
        'optics_xi': step.optics_xi,
        'temporal_window_days': step.temporal_window_days,
        'spatial_weight': step.spatial_weight,
        'min_cluster_size': step.min_cluster_size,
        'cluster_dimension': step.cluster_dimension
    }


def _handle_final_outliers(outlier_data: pd.DataFrame, 
                          existing_sequences: Dict[str, pd.DataFrame],
                          handling_method: str) -> Optional[pd.DataFrame]:
    """Handle final outliers according to specified method."""
    
    if handling_method == 'discard':
        logger.info("Discarding final outliers")
        return None
    elif handling_method == 'merge_largest' and existing_sequences:
        # Merge with largest existing cluster
        largest_sequence = max(existing_sequences.keys(), 
                            key=lambda k: len(existing_sequences[k]))
        existing_sequences[largest_sequence] = pd.concat([
            existing_sequences[largest_sequence], outlier_data
        ], ignore_index=True)
        logger.info(f"Merged {len(outlier_data)} final outliers with {largest_sequence}")
        return None
    else:  # 'keep' or fallback
        logger.info("Keeping final outliers as separate group")
        return outlier_data


def advanced_catalog_segmentation(catalog: pd.DataFrame, 
                               method: str = 'dbscan',
                               features: List[str] = ['spatial'],
                               **clustering_params) -> Tuple[Dict[str, pd.DataFrame], np.ndarray]:
    """
    Advanced segmentation for full earthquake catalogs.
    
    Parameters
    ----------
    catalog : pd.DataFrame
        Full earthquake catalog with required columns
    method : str
        Clustering method ('dbscan', 'hdbscan', 'temporal', 'spatial_temporal')
    features : List[str]
        Features to use for segmentation ('spatial', 'temporal', 'magnitude')
    **clustering_params : dict
        Method-specific clustering parameters
        
    Returns
    -------
    Tuple[Dict[str, pd.DataFrame], np.ndarray]
        Dictionary of sequence DataFrames and cluster labels array
    """
    
    logger.info(f"Starting catalog segmentation with method: {method}")
    logger.info(f"Input catalog size: {len(catalog)} events")
    logger.info(f"Clustering features: {features}")
    
    # Prepare features for segmentation
    feature_matrix = _prepare_clustering_features(catalog, features, clustering_params)
    
    # Apply clustering algorithm
    if method == 'dbscan':
        cluster_labels = _apply_dbscan_clustering(feature_matrix, clustering_params)
    elif method == 'hdbscan':
        cluster_labels = _apply_hdbscan_clustering(feature_matrix, clustering_params)
    elif method == 'optics':
        cluster_labels = _apply_optics_clustering(feature_matrix, clustering_params)
    elif method == 'temporal':
        cluster_labels = _apply_temporal_clustering(catalog, clustering_params)
    elif method == 'spatial_temporal':
        cluster_labels = _apply_spatial_temporal_clustering(feature_matrix, clustering_params)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Organize results into sequences
    sequences = _organize_sequences(catalog, cluster_labels, clustering_params.get('min_cluster_size', 20))
    
    logger.info(f"Segmentation completed. Found {len(sequences)} sequences:")
    for sequence_name, sequence_data in sequences.items():
        logger.info(f"  {sequence_name}: {len(sequence_data)} events")
    
    return sequences, cluster_labels


def _prepare_clustering_features(catalog: pd.DataFrame, 
                                features: List[str], 
                                params: Dict[str, Any]) -> np.ndarray:
    """Prepare feature matrix for segmentation."""
    
    feature_arrays = []
    cluster_dimension = params.get('cluster_dimension', '3d')
    
    if 'spatial' in features:
        # Spatial coordinates (already transformed)
        if cluster_dimension == '2d':
            # Use only X, Y (ignore depth Z)
            spatial_coords = catalog[['X', 'Y']].values
            logger.info(f"Added 2D spatial features (X,Y only): shape {spatial_coords.shape}")
        else:
            # Use full 3D coordinates
            spatial_coords = catalog[['X', 'Y', 'Z']].values
            logger.info(f"Added 3D spatial features (X,Y,Z): shape {spatial_coords.shape}")
        
        # Use raw coordinates without normalization (old implementation style)
        feature_arrays.append(spatial_coords)
    
    if 'temporal' in features:
        # Convert time to numerical representation
        if 'Date' not in catalog.columns:
            # Create Date column if not exists
            catalog['Date'] = pd.to_datetime(pd.DataFrame({
                'year': catalog['YR'],
                'month': catalog['MO'],
                'day': catalog['DY'],
                'hour': catalog['HR'],
                'minute': catalog['MI'],
                'second': catalog['SC']
            }))
        
        time_numeric = catalog['Date'].astype(np.int64) // 10**9  # Unix timestamp
        time_normalized = (time_numeric - time_numeric.mean()) / time_numeric.std()
        feature_arrays.append(time_normalized.values.reshape(-1, 1))
        
        logger.info(f"Added temporal features: shape {time_normalized.values.reshape(-1, 1).shape}")
    
    if 'magnitude' in features:
        # Use magnitude if available
        mag_cols = ['ML', 'Mw', 'MAG']  # Common magnitude column names
        mag_col = None
        
        for col in mag_cols:
            if col in catalog.columns:
                mag_col = col
                break
        
        if mag_col:
            mag_values = catalog[mag_col].values
            # Handle NaN values
            mag_values = np.nan_to_num(mag_values, nan=mag_values[~np.isnan(mag_values)].mean())
            mag_normalized = (mag_values - mag_values.mean()) / mag_values.std()
            feature_arrays.append(mag_normalized.reshape(-1, 1))
            
            logger.info(f"Added magnitude features ({mag_col}): shape {mag_normalized.reshape(-1, 1).shape}")
        else:
            logger.warning("Magnitude feature requested but no magnitude column found")
    
    if not feature_arrays:
        raise ValueError("No valid features found for segmentation")
    
    feature_matrix = np.hstack(feature_arrays)
    logger.info(f"Final feature matrix shape: {feature_matrix.shape}")
    
    return feature_matrix


def _apply_dbscan_clustering(features: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply DBSCAN clustering algorithm."""
    
    eps = params.get('dbscan_eps', 1000.0)
    min_samples = params.get('dbscan_min_samples', 10)
    metric = params.get('dbscan_metric', 'euclidean')
    
    logger.info(f"DBSCAN parameters: eps={eps}, min_samples={min_samples}, metric={metric}")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    cluster_labels = dbscan.fit_predict(features)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    logger.info(f"DBSCAN results: {n_clusters} clusters, {n_noise} noise points")
    
    return cluster_labels


def _apply_hdbscan_clustering(features: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply HDBSCAN clustering algorithm."""
    
    min_cluster_size = params.get('hdbscan_min_cluster_size', 15)
    min_samples = params.get('hdbscan_min_samples', None)
    metric = params.get('dbscan_metric', 'euclidean')
    
    logger.info(f"HDBSCAN parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric={metric}")
    
    hdbscan = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric
    )
    cluster_labels = hdbscan.fit_predict(features)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    logger.info(f"HDBSCAN results: {n_clusters} clusters, {n_noise} noise points")
    
    return cluster_labels


def _apply_optics_clustering(features: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply OPTICS clustering algorithm.
    
    OPTICS (Ordering Points To Identify the Clustering Structure) is particularly
    good for variable density data and elongated/linear structures. It's similar to
    HDBSCAN but uses reachability distance for better handling of linear features.
    """
    
    min_samples = params.get('optics_min_samples', 10)
    max_eps = params.get('optics_max_eps', np.inf)  # Maximum eps for neighborhood
    metric = params.get('dbscan_metric', 'euclidean')
    cluster_method = params.get('optics_cluster_method', 'xi')  # 'xi' or 'dbscan'
    xi = params.get('optics_xi', 0.05)  # Steepness threshold for xi method
    
    logger.info(f"OPTICS parameters: min_samples={min_samples}, max_eps={max_eps}, "
                f"metric={metric}, cluster_method={cluster_method}, xi={xi}")
    
    optics = OPTICS(
        min_samples=min_samples,
        max_eps=max_eps,
        metric=metric,
        cluster_method=cluster_method,
        xi=xi
    )
    
    cluster_labels = optics.fit_predict(features)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    logger.info(f"OPTICS results: {n_clusters} clusters, {n_noise} noise points")
    
    return cluster_labels


def _apply_temporal_clustering(catalog: pd.DataFrame, params: Dict[str, Any]) -> np.ndarray:
    """Apply time-based clustering using sliding windows."""
    
    window_days = params.get('temporal_window_days', 30)
    
    logger.info(f"Temporal clustering with {window_days}-day windows")
    
    if 'Date' not in catalog.columns:
        catalog['Date'] = pd.to_datetime(pd.DataFrame({
            'year': catalog['YR'],
            'month': catalog['MO'],
            'day': catalog['DY'],
            'hour': catalog['HR'],
            'minute': catalog['MI'],
            'second': catalog['SC']
        }))
    
    time_series = catalog['Date']
    time_windows = pd.date_range(
        start=time_series.min(),
        end=time_series.max(),
        freq=f"{window_days}D"
    )
    
    cluster_labels = np.full(len(catalog), -1)
    
    for i, (start_time, end_time) in enumerate(zip(time_windows[:-1], time_windows[1:])):
        mask = (time_series >= start_time) & (time_series < end_time)
        if mask.sum() > 0:  # Only assign label if there are events in this window
            cluster_labels[mask] = i
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    logger.info(f"Temporal clustering results: {n_clusters} time windows")
    
    return cluster_labels


def _apply_spatial_temporal_clustering(features: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """Apply combined spatial-temporal clustering with weighting."""
    
    spatial_weight = params.get('spatial_weight', 0.7)
    temporal_weight = 1.0 - spatial_weight
    
    logger.info(f"Spatial-temporal clustering: spatial_weight={spatial_weight}, temporal_weight={temporal_weight}")
    
    # Weight features (assuming first 3 columns are spatial, rest are temporal)
    features_weighted = features.copy()
    n_spatial = 3  # X, Y, Z
    
    if features.shape[1] >= n_spatial:
        features_weighted[:, :n_spatial] *= spatial_weight
        if features.shape[1] > n_spatial:
            features_weighted[:, n_spatial:] *= temporal_weight
    
    # Apply DBSCAN to weighted features
    return _apply_dbscan_clustering(features_weighted, params)


def _organize_sequences(catalog: pd.DataFrame, 
                      cluster_labels: np.ndarray, 
                      min_cluster_size: int) -> Dict[str, pd.DataFrame]:
    """Organize segmentation results into separate DataFrames."""
    
    sequences = {}
    catalog_with_labels = catalog.copy()
    catalog_with_labels['cluster_id'] = cluster_labels
    
    unique_labels = np.unique(cluster_labels)
    
    for label in unique_labels:
        if label == -1:
            # Handle noise/outliers
            cluster_mask = catalog_with_labels['cluster_id'] == label
            sequence_data = catalog_with_labels[cluster_mask].copy()
            if len(sequence_data) > 0:
                sequences['noise'] = sequence_data.reset_index(drop=True)
        else:
            # Handle regular clusters
            cluster_mask = catalog_with_labels['cluster_id'] == label
            sequence_data = catalog_with_labels[cluster_mask].copy()
            
            # Only keep clusters that meet minimum size requirement
            if len(sequence_data) >= min_cluster_size:
                sequences[f"cluster_{label}"] = sequence_data.reset_index(drop=True)
            else:
                logger.info(f"Cluster {label} too small ({len(sequence_data)} events), merging with noise")
                # Merge small clusters with noise
                if 'noise' not in sequences:
                    sequences['noise'] = sequence_data.reset_index(drop=True)
                else:
                    sequences['noise'] = pd.concat([sequences['noise'], sequence_data], ignore_index=True)
    
    return sequences


def evaluate_clustering_quality(catalog: pd.DataFrame, 
                               cluster_labels: np.ndarray,
                               features: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the quality of clustering results.
    
    Parameters
    ----------
    catalog : pd.DataFrame
        Original catalog
    cluster_labels : np.ndarray
        Cluster labels from clustering algorithm
    features : np.ndarray
        Feature matrix used for segmentation
        
    Returns
    -------
    Dict[str, float]
        Dictionary of clustering quality metrics
    """
    
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    # Remove noise points for quality evaluation
    non_noise_mask = cluster_labels != -1
    if non_noise_mask.sum() < 2:
        return {"error": "Too few non-noise points for evaluation"}
    
    clean_features = features[non_noise_mask]
    clean_labels = cluster_labels[non_noise_mask]
    
    if len(np.unique(clean_labels)) < 2:
        return {"error": "Need at least 2 clusters for evaluation"}
    
    metrics = {}
    
    try:
        # Silhouette score (higher is better, range [-1, 1])
        metrics['silhouette_score'] = silhouette_score(clean_features, clean_labels)
        
        # Calinski-Harabasz score (higher is better)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(clean_features, clean_labels)
        
        # Davies-Bouldin score (lower is better)
        metrics['davies_bouldin_score'] = davies_bouldin_score(clean_features, clean_labels)
        
        # Additional custom metrics
        metrics['n_clusters'] = len(np.unique(clean_labels))
        metrics['n_noise_points'] = (cluster_labels == -1).sum()
        metrics['noise_ratio'] = metrics['n_noise_points'] / len(cluster_labels)
        
        # Cluster size statistics
        cluster_sizes = [np.sum(clean_labels == label) for label in np.unique(clean_labels)]
        metrics['avg_cluster_size'] = np.mean(cluster_sizes)
        metrics['std_cluster_size'] = np.std(cluster_sizes)
        metrics['min_cluster_size'] = np.min(cluster_sizes)
        metrics['max_cluster_size'] = np.max(cluster_sizes)
        
    except Exception as e:
        logger.error(f"Error computing clustering quality metrics: {e}")
        metrics['error'] = str(e)
    
    return metrics

