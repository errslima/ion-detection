# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:13:41 2019

@author: Lima
"""

import matplotlib.pyplot as plt
import functions as fct

# A sample graph showing an ion detector signal without noise.
# Syntax: signal(steps, electrodes, amplitude, frequency, time, offset, centered)
# steps = Total generated data points
# electrodes = Number of electrodes, assumed each electrode takes 0.2 microseconds to pass
# amplitude = Amplitude without dimension, usually between 0 and 1
# frequency = Frequency in MHz
# time = Time in microseconds
# offset = Offset in amounts of (Amplitude / 100)
# centered = 0 to not center the signal, other character to center the signal at t=0
x, y = fct.signal(3000, 4, 1, 5, 3, 0, 1)
plt.plot(x, y)
plt.show()