#! /usr/bin/env python3

import numpy as np
import scipy.signal

def calculate_alpha_beta_residual_of_cutout(p, wp, q, wq):
    """ This calculates the optimal alpha and beta, and the corresponding sum_of_residuals_squared,
        of the given image and template, considering the given weighing factors for the image and template.
    """

    assert p.shape == wp.shape
    assert q.shape == wq.shape
    assert p.shape == q.shape

    w = wp * wq # weight of each pixel is the element-wise multiplication of the template and image weights.

    dot_wp  = np.sum(w * p)
    dot_wq  = np.sum(w * q)
    dot_wpq = np.sum(w * p * q)
    dot_wqq = np.sum(w * q * q)
    sum_w   = np.sum(w)

    alpha = (dot_wpq * dot_wq - dot_wp  * dot_wqq) / (dot_wq * dot_wq - dot_wqq * sum_w)
    beta  = (dot_wp  * dot_wq - dot_wpq * sum_w  ) / (dot_wq * dot_wq - dot_wqq * sum_w)

    residuals = (p - (alpha + beta * q))

    sum_of_residuals_squared = np.sum(residuals * residuals * wp * wq)

    return (alpha, beta, sum_of_residuals_squared)


def calculate_alpha_beta_residual(p, q, wp=None, wq=None):
    """ This calculates the optimal alpha and beta, and the corresponding sum_of_residuals_squared,
        of the given image p and template q, considering the given weighting factors for the image and template.
    """

    print(p.shape)
    print(q.shape)
    print(wp.shape)
    print(wq.shape)

    if wp is None:
        wp = np.ones_like(p)

    if wq is None:
        wq = np.ones_like(q)

    assert p.shape == wp.shape
    assert q.shape == wq.shape

    p_weighted = wp * p          # element-wise
    q_weighted = wq * q          # element-wise
    qq_weighted = q_weighted * q # element-wise
    pp_weighted = p_weighted * p # element-wise

    dot_wp  = scipy.signal.correlate(p_weighted , wq          , 'valid')
    dot_wq  = scipy.signal.correlate(wp         , q_weighted  , 'valid')
    dot_wpq = scipy.signal.correlate(p_weighted , q_weighted  , 'valid')
    dot_wqq = scipy.signal.correlate(wp         , qq_weighted , 'valid')
    sum_w   = scipy.signal.correlate(wp         , wq          , 'valid')
    dot_wpp = scipy.signal.correlate(pp_weighted, wq          , 'valid')

    alpha = (dot_wpq * dot_wq - dot_wp  * dot_wqq) / (dot_wq * dot_wq - dot_wqq * sum_w)
    beta  = (dot_wp  * dot_wq - dot_wpq * sum_w  ) / (dot_wq * dot_wq - dot_wqq * sum_w)

    sum_of_residuals_squared = alpha * alpha * sum_w - 2 * alpha * dot_wp + dot_wpp + 2 * alpha * beta * dot_wq - 2 * beta * dot_wpq + beta * beta * dot_wqq

    return (alpha, beta, sum_of_residuals_squared)


def test_2d_match_offset_and_scale():

    image_width     = 800
    image_height    = 600

    image_size = (image_height, image_width)

    template_width  = 80
    template_height = 60

    template_size = (template_height, template_width)

    rng = np.random.default_rng(0)

    image        = rng.normal(100.0, 15.0, image_size)
    image_weight = rng.random(image_size)

    template        = (image[20:20+template_height, 24:24+template_width] - 13.0) / 17.0
    template_weight = rng.random(template_size)

    (alpha, beta, sum_of_residuals_squared) = calculate_alpha_beta_residual_of_cutout(image[19:19+template_height, 23:23+template_width], image_weight[19:19+template_height, 23:23+template_width], template, template_weight)

    print("cutout - alpha:", alpha)
    print("cutout - beta:", beta)
    print("cutout - sum_of_residuals_squared:", sum_of_residuals_squared)

    (alpha, beta, sum_of_residuals_squared) = calculate_alpha_beta_residual(image, template, image_weight, template_weight)
    
    print("element (19, 23):", alpha[19][23])
    print("element (19, 23):", beta[19][23])
    print("element (19, 23):", sum_of_residuals_squared[19][23])

    print(sum_of_residuals_squared.shape)

def main():
    test_2d_match_offset_and_scale()

if __name__ == "__main__":
    main()
