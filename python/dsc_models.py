"""DSC model ports from MATLAB."""

from __future__ import annotations

from typing import Any, Iterable, Tuple

import numpy as np


def dsc_convolution_ssvd(
    concentration_array: Any,
    aif: Iterable[float],
    delta_t: float,
    kh: float,
    rho: float,
    psvd: float,
    method: int = 1,
) -> Tuple[Any, Any, Any]:
    """Port of `dsc/DSC_convolution_sSVD.m` for core map outputs.

    This implementation reproduces the numerical core (`CBF`, `CBV`, `MTT`)
    and intentionally omits NIfTI file writing side effects.
    """
    conc = np.asarray(concentration_array, dtype=float)
    if conc.ndim not in (3, 4):
        raise ValueError(f"concentration_array must be 3D or 4D, got ndim={conc.ndim}")

    if conc.ndim == 4:
        dimx, dimy, dimz, dimt = conc.shape
        is_4d = True
    else:
        dimx, dimy, dimt = conc.shape
        dimz = 1
        is_4d = False

    if dimt < 2:
        raise ValueError("concentration_array time dimension must be >= 2")

    aif_vec = np.asarray(list(aif), dtype=float).reshape(-1)
    if aif_vec.size < dimt:
        raise ValueError(f"AIF length must be >= time dimension ({dimt}), got {aif_vec.size}")
    aif_vec = aif_vec[:dimt]

    delta_t = float(delta_t)
    kh = float(kh)
    rho = float(rho)
    psvd = float(psvd)

    if rho == 0.0:
        raise ValueError("rho must be non-zero")

    time_vect = np.arange(dimt, dtype=float) * delta_t
    aif_int = float(np.trapezoid(aif_vec, x=time_vect))

    # Match MATLAB lower-triangularized AIF construction exactly.
    a_matrix = np.zeros((dimt, dimt), dtype=float)
    for i in range(dimt):
        for j in range(dimt):
            if i == j:
                a_matrix[i, j] = (delta_t * (4.0 * aif_vec[i - j] + aif_vec[i - j + 1])) / 6.0
            elif j < i and i == (dimt - 1) and j == 0:
                a_matrix[i, j] = (delta_t * (aif_vec[i - j - 1] + 4.0 * aif_vec[i - j] + aif_vec[dimt - 1])) / 6.0
            elif j < i:
                a_matrix[i, j] = (delta_t * (aif_vec[i - j - 1] + 4.0 * aif_vec[i - j] + aif_vec[i - j + 1])) / 6.0

    u_mat, singular_values, v_h = np.linalg.svd(a_matrix, full_matrices=True)
    threshold = float(np.max(singular_values)) * (psvd / 100.0)
    singular_values = np.where(singular_values < threshold, 0.0, singular_values)
    inv_s = np.zeros_like(singular_values)
    nz_mask = singular_values != 0.0
    inv_s[nz_mask] = 1.0 / singular_values[nz_mask]
    v_mat = v_h.T

    if dimz == 1:
        cbf = np.zeros((dimx, dimy), dtype=float)
        cbv = np.zeros((dimx, dimy), dtype=float)
        mtt = np.zeros((dimx, dimy), dtype=float)
    else:
        cbf = np.zeros((dimx, dimy, dimz), dtype=float)
        cbv = np.zeros((dimx, dimy, dimz), dtype=float)
        mtt = np.zeros((dimx, dimy, dimz), dtype=float)

    kh_over_rho = 100.0 * (kh / rho)

    for k in range(dimz):
        for j in range(dimy):
            for i in range(dimx):
                if is_4d:
                    c = conc[i, j, k, :].reshape(-1)
                else:
                    c = conc[i, j, :].reshape(-1)

                # b = V * W * (U' * c), where W = diag(1./S) after thresholding.
                b = v_mat @ (inv_s * (u_mat.T @ c))

                c_int = float(np.trapezoid(c, x=time_vect))
                max_b = float(np.max(b))

                if method == 1:
                    cbf_val = kh_over_rho * max_b
                    cbv_val = kh_over_rho * (c_int / aif_int)
                    mtt_val = cbv_val / cbf_val
                elif method == 2:
                    # MATLAB method-2 branch references `b_index` (undefined); use `b`.
                    r_int = float(np.trapezoid(b, x=time_vect))
                    cbf_val = kh_over_rho * max_b
                    cbv_val = kh_over_rho * max_b * r_int
                    mtt_val = r_int
                else:
                    raise ValueError(f"Unsupported method: {method}")

                if dimz == 1:
                    cbf[i, j] = cbf_val
                    cbv[i, j] = cbv_val
                    mtt[i, j] = mtt_val
                else:
                    cbf[i, j, k] = cbf_val
                    cbv[i, j, k] = cbv_val
                    mtt[i, j, k] = mtt_val

    return cbf.tolist(), cbv.tolist(), mtt.tolist()
