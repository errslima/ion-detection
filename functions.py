# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:21:12 2019

@author: Lima
"""

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider, RadioButtons
import scipy.signal
# from scipy.stats import norm
import math
import colorednoise as cn


def whitenoise(steps, amplitude):
    OutputNoise = np.random.normal(0, amplitude, steps)
    OutputAxis = range(steps)
    return OutputAxis, OutputNoise

def pinknoise(steps, amplitude):
    OutputNoise = cn.powerlaw_psd_gaussian(1, steps) * amplitude
    OutputAxis = range(steps)
    return OutputAxis, OutputNoise
    
# Generate a simple simulation of the signal that an ion detector outputs
# SignalElectrodes = Number of electrodes, assumed each electrode takes 0.2 microseconds to pass
# SignalAmplitude = Amplitude without dimension, usually between 0 and 1
# SignalFrequency = Frequency in MHz
# 3000 steps, 3 microseconds, no offset, not centered
def signal_simple(electrodes, amplitude, frequency):
    Tsig = (electrodes * 3000) / (5 * 3)
    T1 = (3000 / 2) - (Tsig / 2)
    phase = (2 * np.pi * frequency) * T1 * 3 / 3000
    OutputSignal = np.ones(3000)
    OutputTime = np.array(np.linspace(0, 3, num=3000))

    for j in range(3000):
        if j < T1:
            OutputSignal[j] = 0
        elif j > (T1 + Tsig):
            OutputSignal[j] = 0
        else:
            OutputSignal[j] = amplitude + (amplitude * math.sin((2 * np.pi * frequency * j * 3 / 3000) - (0.5 * np.pi) - phase))
    return OutputTime, OutputSignal

# Generate a simulation of the signal that an ion detector outputs
# SignalSteps = Total generated datapoints
# SignalElectrodes = Number of electrodes, assumed each electrode takes 0.2 microseconds to pass
# SignalAmplitude = Amplitude without dimension, usually between 0 and 1
# SignalFrequency = Frequency in MHz
# SignalTime = Time in microseconds
# SignalOffset = Offset in amounts of (Amplitude / 100)
# SignalCentered = 0 to not center the signal, other character to center the signal at t=0
def signal(steps, electrodes, amplitude, frequency, time, offset, centered):
    Tsig = (electrodes * steps) / (5 * time)
    T1 = (steps / 2) - (Tsig / 2)
    phase = (2 * np.pi * frequency) * T1 * time / steps
    OutputSignal = np.ones(steps)
    OutputTime = np.array(np.linspace(0, time, num=steps))

    for j in range(steps):
        if j < T1:
            if centered == 0:
                OutputSignal[j] = - offset / 100
            else:
                OutputSignal[j] = - (offset / 100) + amplitude + (amplitude * math.sin((2 * np.pi * frequency * j * time / steps) - (0.5 * np.pi) - phase))
        elif j > (T1 + Tsig):
            if centered == 0:
                OutputSignal[j] = - offset / 100
            else:
                OutputSignal[j] = - (offset / 100) + amplitude + (amplitude * math.sin((2 * np.pi * frequency * j * time / steps) - (0.5 * np.pi) - phase))
        else:
            if centered == 0:
                OutputSignal[j] = - (offset / 100) + amplitude + (amplitude * math.sin((2 * np.pi * frequency * j * time / steps) - (0.5 * np.pi) - phase))
            else:
                OutputSignal[j] = - offset / 100
    return OutputTime, OutputSignal


# Generate a square window function
# WindowSteps = Total generated datapoints
# WindowWidth = Width of the generated window in microseconds
# WindowTime = Total signal time in microseconds
def rectangularwindow(steps, width, time):
    OutputWindow = np.ones(steps)
    OutputTime = np.array(np.linspace(0, time, num=steps))
    for i in range(steps):
        if i < (steps / 2) - ((width * steps) / (time * 2)):
            OutputWindow[i] = 0
        elif i > (steps / 2) + ((width * steps) / (time * 2)):
            OutputWindow[i] = 0
        else:
            OutputWindow[i] = 1
    return OutputTime, OutputWindow


def chebywindow(steps, width, time):
    OutputTime = np.array(np.linspace(0, time, num=steps))
    OutputWindow = scipy.signal.chebwin(steps, width, sym=True)
    return OutputTime, OutputWindow

def fourier(steps, electrodes, amplitude, frequency, time, offset, width, white, pink):
    xsignal = signal(steps, electrodes, amplitude, frequency, time, offset, 0)[1]
    xwhitenoise = whitenoise(steps, white)[1]
    xpinknoise = pinknoise(steps, pink)[1]
    window = rectangularwindow(steps, width, time)[1]
    combinedsignal = (xsignal + xwhitenoise + xpinknoise) * window
    OutputFFT = np.abs(np.fft.rfft(combinedsignal)) / time
    OutputFrequency = np.linspace(0, (steps / 2 + 1) / time, num=int(steps / 2 + 1))
    return OutputFrequency, OutputFFT

def fourieraverage(number, steps, electrodes, amplitude, frequency, time, offset, width, white, pink):
    OutputAverage = fourier(steps, electrodes, amplitude, frequency, time, offset, width, white, pink)[1]
    OutputAxis = fourier(steps, electrodes, amplitude, frequency, time, offset, width, white, pink)[0]
    OutputArray = np.ones((number, len(OutputAxis)))
    for o in range(number):
        data = fourier(steps, electrodes, amplitude, frequency, time, offset, width, white, pink)[1]
        pre = (o * OutputAverage) + data
        OutputAverage = pre / (o + 1)
        OutputArray[o] = data
    return OutputAxis, OutputAverage, OutputArray

def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)

def probabilitydist(number, steps, electrodes, amplitude, frequency, time, offset, width, white, pink):
    x, y, a = fourieraverage(number, steps, electrodes, amplitude, frequency, time, offset, width, white, pink)
    xrange = range(1000)
    dist = np.ones(1000)
    for i in range(number):
        idx = (np.abs(x - 10.55)).argmin()
        value = int(clamp(round((a[i][idx] * 1),0), 0, 999))
        dist[value] += 1
    return xrange, dist

# pinknoise test
x, y = probabilitydist(60000, 3000, 5, 2.25, 10.55, 3, 0, 1.5, 1, 7.4)
plt.figure()
plt.plot(x,y)
plt.show()

# Generate an array of window functions, multiply with signal and take fourier transform
# def WindowArray(WaSteps, WaRange, WaElectrodes, WaAmplitude, WaFrequency, WaTime, WaOffset):
#    WaArray = np.ones((WaRange, WaSteps))
#    WaSignal = Signal(WaSteps, WaElectrodes, WaAmplitude, WaFrequency, WaTime, WaOffset)[1]
#    OutputFFTArray = np.ones((WaRange, int(WaSteps / 2 + 1)))
#    for i in range (WaRange):
#        WaArray[i] = WaSignal * Window(WaSteps, i * (WaTime / WaRange), WaTime)[1]
#        OutputFFTArray[i] = np.abs(np.fft.rfft(WaArray[i])) / WaTime
#    OutputFrequency = np.linspace(0, (WaSteps / 2 + 1) / WaTime, num=int(WaSteps / 2 + 1))
#    return OutputFrequency, OutputFFTArray

# Generate an graph showing the change in frequency at different window function lenghts
# def WindowRelation(WrSteps, WrRange, WrElectrodes, WrAmplitude, WrFrequency, WrTime, WrOffset, WrTarget):
#    OutputPeaks = np.ones(WrRange)
#    OutputFrequency = WindowArray(WrSteps, WrRange, WrElectrodes, WrAmplitude, WrFrequency, WrTime, WrOffset)[1]
#    idx = (np.abs(OutputFrequency - WrTarget)).argmin()
#    for i in range(WrRange):
#        OutputPeaks[i] = WindowArray(WrSteps, WrRange, WrElectrodes, WrAmplitude, WrFrequency, WrTime, WrOffset)[0][i][idx]
#    return OutputPeaks, OutputFrequency


# stim, sig = signal(3000, 5, 1, 5, 3, 0)
# wtime, win = window(1, 3000, 1, 3)
# freq, fft = fourier(3000, 5, 1, 5, 3, 0, 1, 1)

# fig = plt.figure(figsize=(16, 6))
# plt.subplot(131)
# m, = plt.plot(stim, sig)
# plt.subplot(132)
# n, = plt.plot(wtime, win)
# plt.subplot(133)
# l, = plt.plot(freq, fft)

# ax = plt.axis([0, 10, 0, 200])
# axWW = plt.axes([0.4, 0.01, 0.226, 0.05])
# winwidth = Slider(axWW, 'Window width', 0, 300, valinit=100)
#axE = plt.axes([0.12, 0.01, 0.16, 0.05])
#enum = Slider(axE, 'Electrodes', 0, 10, valinit=5)
#axr = plt.axes([0.91, 0.5, 0.15, 0.15])
#radio = RadioButtons(axr, ('Rectangle', 'Chebyshev'))

# def update_w(val):
#     WW = winwidth.val / 100
#     four = fourier(3000, 5, 1, 5, 3, 0, WW, 1)[1]
#     l.set_ydata(fourier)
#     plt.subplot(133)
#     plt.ylim([np.amin(four), np.amax(four)])
#     n.set_ydata(window(winfunc, 3000, WW, 3)[1])
#     fig.canvas.draw_idle()

# def update_w(val):
#    radwin = radio.value_selected
#    winfunc = 1
#    if radwin == "Rectangle":
#        WW = winwidth.val / 10
#        winfunc = 1
#    elif radwin == "Chebyshev":
#        WW = winwidth.val
#        winfunc = 2
#    else:
#        WW = winwidth.val / 10
#        winfunc = 1
#    four = fourier(3000, 5, 1, 5, 3, 0, (WW / 100), winfunc)[1]
#    l.set_ydata(fourier)
#    plt.subplot(133)
#    plt.ylim([np.amin(four), np.amax(four)])
#    n.set_ydata(window(winfunc, 3000, (WW / 100), 3)[1])
#    fig.canvas.draw_idle()


# def update_e(val):
#    E = enum.val
#    y1data = signal(3000, E, 1, 5, 3, 0)[1]
#    radwin = radio.value_selected
#    winfunc = 1
#    if radwin == "Rectangle":
#        WW = winwidth.val / 10
#        winfunc = 1
#    elif radwin == "Chebyshev":
#        WW = winwidth.val
#        winfunc = 2
#    else:
#        WW = winwidth.val / 10
#        winfunc = 1
#    y2data = fourier(3000, E, 1, 5, 3, 0, (WW / 100), winfunc)[1]
#    m.set_ydata(y1data)
#    l.set_ydata(y2data)
#    fig.canvas.draw_idle()


# def update_win(label):
#    rwin = {'Rectangle': "Rectangle", 'Chebyshev': "Chebyshev"}
#    winfunc = 1
#    if rwin == "Rectangle":
#        WW = winwidth.val / 10
#        winfunc = 1
#    elif rwin == "Chebyshev":
#        WW = winwidth.val
#        winfunc = 2
#    else:
#        WW = winwidth.val / 10
#        winfunc = 1
#    E = enum.val
#    y1data = window(winfunc, 3000, (WW / 100), 3)[1]
#    y2data = fourier(3000, E, 1, 5, 3, 0, (WW / 100), winfunc)[1]
#    n.set_ydata(y1data)
#    l.set_ydata(y2data)
#    fig.canvas.draw_idle()

# radio.on_clicked(update_win)
# winwidth.on_changed(update_w)
# enum.on_changed(update_e)

# plt.show()
