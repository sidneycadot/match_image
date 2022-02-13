#! /usr/bin/env python3

import numpy as np

from alpha_beta_fit import alpha_beta_fit


def alpha_beta_fit_cutout(p, q, wp, wq):
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


def test_alpha_beta_fit_1d():

    image_width    = 800
    image_size     = (image_width, )

    template_width = 80
    template_size  = (template_width, )

    # Make random image.

    rng = np.random.default_rng(0)

    image        = rng.normal(0.0, 15.0, image_size)
    image_weight = rng.random(image_size)

    # image == 13 + 17 * template

    template        = (image[19:19+template_width] - 13.0) / 17.0
    template_weight = rng.random(template_size)

    # Make cutouts.

    image_coutout       = image       [19:19+template_width]
    image_weight_cutout = image_weight[19:19+template_width]

    # Determine fit for the cutout.

    print("*** 1D test ***")
    print()

    (alpha_cutout, beta_cutout, weighted_sum_of_residuals_squared_cutout) = alpha_beta_fit_cutout(
        image_coutout, template, image_weight_cutout,template_weight)

    print("cutout - alpha:", alpha_cutout)
    print("cutout - beta:", beta_cutout)
    print("cutout - weighted_sum_of_residuals_squared:", weighted_sum_of_residuals_squared_cutout)

    (alpha, beta, weighted_sum_of_residuals_squared) = alpha_beta_fit(
        image, template, image_weight, template_weight)

    print("element (19, ) - alpha:", alpha[19])
    print("element (19, ) - beta:", beta[19])
    print("element (19, ) - sum_of_residuals_squared:", weighted_sum_of_residuals_squared[19])
    print()

    epsilon = 1e-10

    assert np.all(weighted_sum_of_residuals_squared >= -0.001)
    assert np.abs(alpha_cutout - alpha[19]) < epsilon
    assert np.abs(beta_cutout - beta[19]) < epsilon
    assert np.abs(weighted_sum_of_residuals_squared_cutout - weighted_sum_of_residuals_squared[19])

    print("Test succeeded.")
    print()


def test_alpha_beta_fit_2d():

    image_width     = 800
    image_height    = 600

    image_size = (image_height, image_width)

    template_width  = 80
    template_height = 60

    template_size = (template_height, template_width)

    # Make random image.

    rng = np.random.default_rng(0)

    image        = rng.normal(100.0, 15.0, image_size)
    image_weight = rng.random(image_size)

    # image == 13 + 17 * template

    template        = (image[19:19+template_height, 23:23+template_width] - 13.0) / 17.0
    template_weight = rng.random(template_size)

    # Make cutouts.

    image_coutout       = image       [19:19+template_height, 23:23+template_width]
    image_weight_cutout = image_weight[19:19+template_height, 23:23+template_width]

    (alpha_cutout, beta_cutout, weighted_sum_of_residuals_squared_cutout) = alpha_beta_fit_cutout(
        image_coutout, template, image_weight_cutout,template_weight)

    (alpha, beta, weighted_sum_of_residuals_squared) = alpha_beta_fit(
        image, template, image_weight, template_weight)

    print("*** 2D test ***")
    print()

    print("cutout - alpha:", alpha_cutout)
    print("cutout - beta:", beta_cutout)
    print("cutout - weighted_sum_of_residuals_squared:", weighted_sum_of_residuals_squared_cutout)
    print()
    print("element (19, 23) - alpha:", alpha[19][23])
    print("element (19, 23) - beta:", beta[19][23])
    print("element (19, 23) - weighted_sum_of_residuals_squared:", weighted_sum_of_residuals_squared[19][23])
    print()

    epsilon = 1e-10

    assert np.all(weighted_sum_of_residuals_squared >= -0.001)
    assert np.abs(alpha_cutout - alpha[19][23]) < epsilon
    assert np.abs(beta_cutout - beta[19][23]) < epsilon
    assert np.abs(weighted_sum_of_residuals_squared_cutout - weighted_sum_of_residuals_squared[19][23])

    print("Test succeeded.")
    print()


def main():
    test_alpha_beta_fit_1d()
    test_alpha_beta_fit_2d()


if __name__ == "__main__":
    main()
