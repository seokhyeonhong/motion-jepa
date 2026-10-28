# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from nv-tlabs/kimodo e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954.
# Batched-only evaluation adapters; see NOTICE.md for changes.
import logging
import numpy as np
from scipy import linalg
log = logging.getLogger(__name__)

def calculate_activation_statistics(activations: np.ndarray):
    """Compute ``(mean, covariance)`` over a stack of latent activations."""
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6):
    """Numerically stable Frechet (FID-style) distance between two Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # scipy >= 1.18 dropped sqrtm's `disp` argument (and its error-estimate return); a non-finite
    # result is caught below either way
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        log.warning(
            "fid calculation produces singular product; adding %s to diagonal of cov estimates",
            eps,
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
