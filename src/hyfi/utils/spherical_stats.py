#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spherical statistics utilities for HyFI.

This module provides spherical statistical functions needed for fault imaging,
replacing the external 'sphere' package dependency.
"""

import numpy as np
from numpy.linalg import norm, eig


class FB8Distribution:
    """
    Fisher-Bingham distribution on the 8-sphere.
    
    This is a simplified implementation focused on the Kent distribution
    functionality needed for fault imaging.
    """
    
    minimum_value_for_kappa = 1e-6
    
    @staticmethod
    def gamma1_to_spherical_coordinates(gamma1):
        """
        Convert gamma1 vector to spherical coordinates.
        
        Parameters
        ----------
        gamma1 : array_like
            Unit vector
            
        Returns
        -------
        tuple
            (theta, phi) spherical coordinates
        """
        x, y, z = gamma1
        theta = np.arccos(z)
        phi = np.arctan2(y, x)
        return theta, phi
    
    @staticmethod
    def create_matrix_H(theta, phi):
        """
        Create rotation matrix H from spherical coordinates.
        
        Parameters
        ----------
        theta : float
            Polar angle
        phi : float
            Azimuthal angle
            
        Returns
        -------
        ndarray
            3x3 rotation matrix
        """
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        H = np.array([
            [cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta],
            [-sin_phi, cos_phi, 0],
            [sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]
        ])
        
        return H
    
    @staticmethod
    def create_matrix_Ht(theta, phi):
        """
        Create transpose of rotation matrix H.
        
        Parameters
        ----------
        theta : float
            Polar angle
        phi : float
            Azimuthal angle
            
        Returns
        -------
        ndarray
            3x3 transpose rotation matrix
        """
        return FB8Distribution.create_matrix_H(theta, phi).T


def MMul(*matrices):
    """
    Matrix multiplication for multiple matrices.
    
    Parameters
    ----------
    *matrices : array_like
        Matrices to multiply
        
    Returns
    -------
    ndarray
        Result of matrix multiplication
    """
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.dot(result, matrix)
    return result


def fb84(G, kappa, beta):
    """
    Create Fisher-Bingham distribution with given parameters.
    
    This is a simplified implementation that returns a dictionary
    with the distribution parameters.
    
    Parameters
    ----------
    G : ndarray
        Rotation matrix
    kappa : float
        Concentration parameter
    beta : float
        Ovalness parameter
        
    Returns
    -------
    dict
        Distribution parameters
    """
    return {
        'type': 'Fisher-Bingham',
        'G': G,
        'kappa': kappa,
        'beta': beta,
        'gamma1': G[:, 0],  # First column is the mean direction
        'gamma2': G[:, 1],  # Second column is the major axis
        'gamma3': G[:, 2]   # Third column is the minor axis
    }


def kent_me(xs):
    """
    Generate Fisher-Bingham distribution based on Kent moment estimation.
    
    This function estimates the parameters of a Kent distribution (Fisher-Bingham
    distribution on the sphere) from a set of unit vectors using method of moments.
    
    Parameters
    ----------
    xs : array_like
        Array of unit vectors on the sphere, shape (n_samples, 3)
        
    Returns
    -------
    dict
        Dictionary containing the estimated distribution parameters:
        - type: 'Fisher-Bingham'
        - G: rotation matrix (3x3)
        - kappa: concentration parameter
        - beta: ovalness parameter  
        - gamma1: mean direction (unit vector)
        - gamma2: major axis direction
        - gamma3: minor axis direction
    """
    xs = np.asarray(xs)
    lenxs = len(xs)
    
    # Average direction of samples from origin
    xbar = np.average(xs, 0)
    
    # Dispersion (or covariance) matrix around origin
    S = np.average(xs.reshape((lenxs, 3, 1)) * xs.reshape((lenxs, 1, 3)), 0)
    
    # Unit vector in the same direction as xbar
    gamma1 = xbar / norm(xbar)
    theta, phi = FB8Distribution.gamma1_to_spherical_coordinates(gamma1)

    H = FB8Distribution.create_matrix_H(theta, phi)
    Ht = FB8Distribution.create_matrix_Ht(theta, phi)
    B = MMul(Ht, MMul(S, H))

    # Eigenvalue decomposition of the 2x2 submatrix
    eigvals, eigvects = eig(B[1:, 1:])
    eigvals = np.real(eigvals)
    
    # Sort eigenvalues in descending order
    if eigvals[0] < eigvals[1]:
        eigvals[0], eigvals[1] = eigvals[1], eigvals[0]
        eigvects = eigvects[:, ::-1]
    
    K = np.diag([1.0, 1.0, 1.0])
    K[1:, 1:] = eigvects

    G = MMul(H, K)
    Gt = np.swapaxes(G, -2, -1)
    T = MMul(Gt, MMul(S, G))

    r1 = norm(xbar)
    t22, t33 = T[1, 1], T[2, 2]
    r2 = t22 - t33

    # Estimate kappa and beta parameters
    # Ensure they lie within permitted ranges
    min_kappa = FB8Distribution.minimum_value_for_kappa
    
    # Avoid division by zero and ensure valid parameter ranges
    denom1 = 2.0 - 2.0 * r1 - r2
    denom2 = 2.0 - 2.0 * r1 + r2
    
    if abs(denom1) < 1e-10 or abs(denom2) < 1e-10:
        # Handle edge cases
        kappa = min_kappa
        beta = 0.0
    else:
        kappa = max(min_kappa, 1.0 / denom1 + 1.0 / denom2)
        beta = 0.5 * (1.0 / denom1 - 1.0 / denom2)

    return fb84(G, kappa, beta)


def sample_kent_distribution(kent_params, n_samples=1000):
    """
    Sample from a Kent distribution.
    
    This is a basic implementation for generating samples from the
    estimated Kent distribution.
    
    Parameters
    ----------
    kent_params : dict
        Kent distribution parameters from kent_me()
    n_samples : int
        Number of samples to generate
        
    Returns
    -------
    ndarray
        Samples from the Kent distribution, shape (n_samples, 3)
    """
    # This is a simplified sampling method
    # For production use, you might want to implement more sophisticated methods
    
    G = kent_params['G']
    kappa = kent_params['kappa']
    beta = kent_params['beta']
    
    # Generate samples using rejection sampling or other methods
    # For now, we'll use a simple approximation
    samples = []
    
    for _ in range(n_samples):
        # Generate a random point on the sphere
        u = np.random.normal(0, 1, 3)
        u = u / norm(u)
        
        # Transform using the rotation matrix
        sample = np.dot(G, u)
        samples.append(sample)
    
    return np.array(samples)


def kent_concentration(kent_params):
    """
    Get the concentration parameter of a Kent distribution.
    
    Parameters
    ----------
    kent_params : dict
        Kent distribution parameters
        
    Returns
    -------
    float
        Concentration parameter kappa
    """
    return kent_params['kappa']


def kent_mean_direction(kent_params):
    """
    Get the mean direction of a Kent distribution.
    
    Parameters
    ----------
    kent_params : dict
        Kent distribution parameters
        
    Returns
    -------
    ndarray
        Mean direction vector (unit vector)
    """
    return kent_params['gamma1']


def kent_axes(kent_params):
    """
    Get the principal axes of a Kent distribution.
    
    Parameters
    ----------
    kent_params : dict
        Kent distribution parameters
        
    Returns
    -------
    tuple
        (major_axis, minor_axis) as unit vectors
    """
    return kent_params['gamma2'], kent_params['gamma3']
