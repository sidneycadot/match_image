"""This module implements the 'alpha_beta_fit' function."""

from typing import Optional, Tuple

import numpy as np
import scipy.signal

def _correlate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Correlate two arrays. The resulting array is the dot product of all entries in a with the curresponding entries in b,
       for each position ob b in a.
    """
    return scipy.signal.correlate(a , b, 'valid')


def alpha_beta_fit(p: np.ndarray, q: np.ndarray, wp: Optional[np.ndarray]=None, wq: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ This function matches a template 'q' to each position of an image 'p'.
    
    For each position, it returns three values: alpha, beta, and weighted_sum_of_residuals_squared.
    
    For each position, this provides the information of how close an optimally offsetted and scaled version
    of the template 'q' is to the image 'p' at that particular position.

    The match is 'optimal' in the sense that the following expression:
    
        weighted_sum_of_residuals_squared = SUM( wp_i * wq_i *(p_coutout_i - (alpha + beta * q_i)) ** 2

    ... is minimized.

    Note:
        This function works in 
    Returns:
        A tuple consisting of the optimal values for alpha, optimal values for beta, and the weighted_sum_of_residuals_squared, for each position.

    """

    if wp is None:
        wp = np.ones_like(p)

    if wq is None:
        wq = np.ones_like(q)

    if (p.shape != wp.shape):
        raise ValueError("wp must be the same shape as p.")

    if (q.shape != wq.shape):
        raise ValueError("wq must be the same shape as q.")

    p_weighted  = wp * p          # element-wise
    q_weighted  = wq * q          # element-wise
    qq_weighted = q_weighted * q  # element-wise
    pp_weighted = p_weighted * p  # element-wise

    dot_wp  = _correlate(p_weighted , wq         )
    dot_wq  = _correlate(wp         , q_weighted )
    dot_wpq = _correlate(p_weighted , q_weighted )
    dot_wqq = _correlate(wp         , qq_weighted)
    dot_wpp = _correlate(pp_weighted, wq         )
    sum_w   = _correlate(wp         , wq         )

    alpha = (dot_wpq * dot_wq - dot_wp  * dot_wqq) / (dot_wq * dot_wq - dot_wqq * sum_w)
    beta  = (dot_wp  * dot_wq - dot_wpq * sum_w  ) / (dot_wq * dot_wq - dot_wqq * sum_w)

    weighted_sum_of_residuals_squared = alpha * alpha * sum_w - 2 * alpha * dot_wp + dot_wpp + 2 * alpha * beta * dot_wq - 2 * beta * dot_wpq + beta * beta * dot_wqq

    return (alpha, beta, weighted_sum_of_residuals_squared)
