#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from alpha_beta_fit import alpha_beta_fit

filename = "solvay.jpg"

solvay = np.asarray(Image.open(filename)) / 255.0

einstein = solvay[1100:1250,1580:1690]

(alpha, beta, weighted_sum_of_residuals_squared) = alpha_beta_fit(solvay, einstein)

plt.subplot(231)
plt.title('solvay.jpg')
plt.imshow(solvay, cmap='gray')
plt.colorbar()

plt.subplot(232)
plt.title('einstein')
plt.imshow(einstein, cmap='gray')
plt.colorbar()

plt.subplot(234)
plt.title('offset')
plt.imshow(alpha, cmap='seismic')
plt.colorbar()

plt.subplot(235)
plt.title('scale')
plt.imshow(beta, cmap='seismic')
plt.colorbar()

plt.subplot(236)
plt.title('∑R²')
plt.imshow(weighted_sum_of_residuals_squared, cmap='hot')
plt.colorbar()

plt.show()
