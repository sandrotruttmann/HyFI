#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS
This script provides an example of how to combine the different modules to perform 'hypocenter-based 3D imaging of active faults'.
Each of the modules can be turned on and off by specifing the respective boolean argument (true/false)

Please cite: Truttmann et al. (2023). Hypocenter-based 3D Imaging of Active Faults: Method and Applications in the Southwestern Swiss Alps.

@author: Sandro Truttmann
@contact: sandro.truttmann@gmail.com
@license: GPL-3.0
@date: April 2023
@version: 0.1.1
"""

# Clear variables
globals().clear()

# Import external modules
import datetime
import time
import numpy as np
import os
import sys

# Import modules of the provided toolbox
sys.path.insert(0, './src')
import fault_network, model_validation, stress_analysis, auto_class, utilities, visualisation, cluster_identification


# ##########################    Input parameters     ###########################
input_params = {
    ###     General settings
    'project_title' : 'Réclère',                               # Project title
    ###     Hypocenter input file
    'hypo_file' : '/home/sandro/ownCloud/SPEKTRUM_DevOps/02_Projekte/0038_swisstopo_Tektonik/Analyse/hypo_fault_imaging/CH_cluster/SECOS_20250305_HyFI_format.csv',        # File location
    'hypo_sep' : ',',                                                 # Separator
    ###     Output directory
    'out_dir' : '/home/sandro/ownCloud/SPEKTRUM_DevOps/02_Projekte/0038_swisstopo_Tektonik/Analyse/hypo_fault_imaging/CH_cluster/',
    ###     "Cluster identification" module settings
    'from_year': 2000,                     # only consider hypocenters from this year onwards
    'DBSCAN_reg' : True,                    # detect outliers with DBSCAN clustering (recommended for regional datasets)
    'max_dist_reg' : 500,                  # DBSCAN: maximum distance [m] between two samples for one to be considered as in the neighborhood of the other
    'min_samples_reg' : 15,                 # DBSCAN: number of samples in a neighborhood for a point to be considered as a core point
    'clust_alg_reg' : 'auto',               # DBSCAN: The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors
    'leaf_size_reg' : 30,                   # DBSCAN: Leaf size passed to BallTree or cKDTree

    # ###     "Cluster identification" module settings
    # 'DBSCAN_reg' : True,                    # detect outliers with HDBSCAN clustering (recommended for regional datasets)
    # 'min_cluster_size_reg' : 15,            # HDBSCAN: minimum number of samples in a group for that group to be considered a cluster.
    # 'min_samples_reg' : None,                 # HDBSCAN: number of samples in a neighborhood for a point to be considered as a core point
    # 'cluster_selection_epsilon_reg': 100,   # HDBSCAN: maximum distance [m] between two samples for one to be considered as in the neighborhood of the other
    # 'clust_alg_reg' : 'auto',               # 
    # 'leaf_size_reg' : 40,                   # cluster_selection_epsilon

    ###     "Fault network reconstruction" module settings
    'n_mc' : 1,                      # number of Monte Carlo simulations
    'r_nn' : 100,                       # search radius [m] of nearest neighbor search
    'dt_nn' : 999999,                    # search time window [h]
    'mag_type' : 'ML',                  # magnitude type: 'ML' or 'Mw'
    ###     "Model Validation" module settings
    'validation_bool' : False,
    'foc_file' : '/home/sandro/ownCloud/SPEKTRUM_DevOps/02_Projekte/0038_swisstopo_Tektonik/Analyse/hypo_fault_imaging/Reclere/FM_reclere.csv',
    'foc_sep' : ',',
    'foc_mag_check' : True,             # check focal magnitude (recommended)
    'foc_loc_check' : True,             # check focal location (recommended)
    ###     "Automatic Classification" module settings
    'autoclass_bool' : False,
    'n_clusters' : 2,                   # number of expected classes
    'algorithm' : 'vmf_soft',           # clustering algorithm
    'rotation' : True,                  # rotate poles before analysis (recommended for vertical faults)
    ###     "Fault Stress Analysis" module settings
    'stress_bool' : False,
    'S1_trend' : 301,                   # σ1 trend
    'S1_plunge' : 23,                   # σ1 plunge
    'S3_trend' : 43,                    # σ3 trend
    'S3_plunge' : 26,                   # σ3 plunge
    'stress_R' : 0.35,                  # Stress shape ratio
    'PP' : 0,                           # Pore pressure
    'fric_coeff' : 0.75                 # Friction coefficient
}

###############################################################################
# Start the timer
start = time.time()
print('')
print('###   HYPOCENTER-BASED 3D IMAGING OF ACTIVE FAULTS   ###')
print('Calculation started...')
print('')


###############################################################################
# Clustering
(data_input, data_output, data_input_outliers) = cluster_identification.cluster_identification(input_params)

# Add reg_clust labels to data_input
data_input['reg_clust'] = data_output['reg_clust']

# Save data_input and data_input_outliers to csv
data_input.to_csv(os.path.join(input_params['out_dir'], 'data_input.csv'), index=False)
data_input_outliers.to_csv(os.path.join(input_params['out_dir'], 'data_input_outliers.csv'), index=False)

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# # drop columns where _X == 0
# ax.scatter(data_input_outliers['_X'], data_input_outliers['_Y'], c='black', s=10, alpha=0.2)
# ax.scatter(data_input['_X'], data_input['_Y'], c=data_output['reg_clust'], s=10, cmap='viridis')
# ax.set_aspect('equal')
# plt.show()


# ###############################################################################
# # Fault network reconstruction
# (data_input, data_input_outliers, data_output,
#  df_per_X, df_per_Y, df_per_Z) = fault_network.faultnetwork3D(input_params)
 
# ###############################################################################
# # Model Validation
# data_input, data_output = model_validation.focal_validation(input_params,
#                                                             data_input,
#                                                             data_output)

# ###############################################################################
# # Automatic Classification
# data_output = auto_class.auto_classification(input_params,
#                                              data_output)

# ###############################################################################
# # Fault Stress Analysis
# data_output, S2_trend, S2_plunge = stress_analysis.fault_stress(input_params,
#                                                                 data_output)

# ###############################################################################
# # Visualisation
# visualisation.model_3d(input_params, data_input, data_input_outliers, data_output)
# visualisation.faults_stereoplot(input_params, data_output)

# ###############################################################################
# # Save model output data
# utilities.save_data(input_params, data_input, data_input_outliers, data_output,
#                     df_per_X, df_per_Y, df_per_Z)

# ###############################################################################
# # Stop the timer
# end = time.time()
# runtime = datetime.timedelta(seconds=(end - start))
# print('')
# print('')
# print('Calculation done!')
# print('Model runtime: ', str(runtime))


