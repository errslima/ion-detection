# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:11:08 2019

@author: Lima
"""

import matplotlib.pyplot as plt
import functions as fct

# A sample graph showing white noise.
# Syntax: whitenoise(steps, amplitude)
# steps = number of data points that are generated
# amplitude = average amplitude of all data points
x, y = fct.whitenoise(3000, 1)
plt.plot(x, y)
plt.show()