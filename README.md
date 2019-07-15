# Function repository for an Image Charge Detection Simulation

These code samples are a summary of the functions used to generate simulated data for the image charge detector as described in [this article](https://www.nature.com/articles/s41598-018-28167-6) (Paul Räcke, 2018).

An overview of some example outputs from the different functions:

### The signal

Generated by signal.py, using an include to functions.py.

[Simulated image charge signal](https://i.imgur.com/ySLI06Q.png)

The fourier transform of this signal is also calculated.

[Imgur](https://i.imgur.com/4P9Sd0o.png)

### Two types of noise

In the simulation we consider white noise and pink noise as artefacts from the detector electronics.

[Pink noise](https://i.imgur.com/GJiivyN.png)

[White noise](https://i.imgur.com/259lBrG.png)


### The probability density function from multiple simulations with and without signal

Without signal:

[Imgur](https://i.imgur.com/SQkwwlB.png)


With signal:

[Imgur](https://i.imgur.com/SQkwwlB.png)


Using this data, a cutoff can be made to guarantee a 99% true positive detection rate:

[Imgur](https://i.imgur.com/xUc5iSJ.png)
