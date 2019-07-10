# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:40:46 2019

@author: Lima
"""

import matplotlib.pyplot as plt
import functions as fct

# A sample graph showing pink noise.
# Syntax: whitenoise(steps, amplitude)
# steps = number of data points that are generated
# amplitude = average amplitude of all data points
x, y = fct.pinknoise(3000, 1)
plt.plot(x, y)
plt.show()