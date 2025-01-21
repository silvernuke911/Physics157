# Task 0 - Make sure everyone is able to at least import these packages
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# import scipy
# import skimage
# import sklearn

# Latex font
# def latexfont():
#     plt.rcParams.update({
#         'text.usetex':True,
#         'font.family':'serif',
#         'font.size':12
#     })
# latexfont()

# Task 1
def logistic(x0, r):
    counter = 0
    x_n = x0
    tolerance = 1e-6  # Correct tolerance for 10^-6
    max_steps = int(1e6)  # Correct number of steps for 10^6
    while counter < max_steps:
        x_n1 = r * x_n * (1 - x_n)
        dif = abs(x_n1 - x_n)
        if dif < tolerance:  # Exit if the difference is below tolerance
            print(f"Fixed point reached after {counter + 1} steps: {x_n1}")
            return
        x_n = x_n1
        counter += 1
    # If the loop completes, print the last value
    print(f"Did not converge after {max_steps} steps. Last value: {x_n}")
    return
logistic(0.56, 1)

# Task 2
# Part A

#setting the bins and samples
N = int(10e6)
n_bins = 100
#sampling
sample = np.random.normal(0,1,N)

#Graphing
counts, bins = np.histogram(sample,n_bins)
plt.stairs(counts, bins, zorder = 2, linewidth = 2)
plt.xlim(-5,5)
plt.grid()
plt.title('$10^6$ Samples from a Normal Distribution')
plt.xlabel('x')
plt.ylabel('count')
plt.show()

# Part B
#define the extents and the bins
dx = 10e-6
x = np.arange(-5, 5 + dx, dx)
n_bins = 100
n_samp = int(10e4)
# Generate the pdf
y = np.cos(x) ** 2
normalization_constant = np.sum(y * dx)
pdf = y / normalization_constant

#sampling
samples = np.random.choice(x, size = n_samp, p = pdf / np.sum(pdf))

#graphing
plt.hist(samples,n_bins, color = 'b', alpha = 0.75)
plt.plot(x,np.cos(x)**2 * n_samp/(n_bins/2), color = 'k')
plt.title(r'$10^6$ Samples from a $\cos^2(x)$ PDF')
plt.xlabel('x')
plt.ylabel('count')
plt.xlim(-5,5)
plt.grid(True)
plt.show()

# Task 3 A1
# Define the grid
x = np.linspace(-2, 2, 500)  # x-coordinates
y = np.linspace(-2, 2, 500)  # y-coordinates
X, Y = np.meshgrid(x, y)  # Create a meshgrid

# Defining circle centers
c1 = (0, 2 / 3)
c2 = (-0.5, -1 / 3)
c3 = (0.5, -1 / 3)

# Equations of the circles (radius = 1)
C1 = (X - c1[0])**2 + (Y - c1[1])**2
C2 = (X - c2[0])**2 + (Y - c2[1])**2
C3 = (X - c3[0])**2 + (Y - c3[1])**2

# Plotting the contours of the circles
plt.contour(X, Y, C1, levels=[1], colors='r')  # Circle 1
plt.contour(X, Y, C2, levels=[1], colors='g')  # Circle 2
plt.contour(X, Y, C3, levels=[1], colors='b')  # Circle 3

# Plotting
plt.xlabel('x')
plt.ylabel('y')
plt.title('Overlapping Circles (Venn Diagram)')
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.grid(alpha=0.3)
plt.gca().set_aspect('equal')
plt.show()

# Task3A2
# Equations of the circles (radius = 1)
C1 = (X - c1[0])**2 + (Y - c1[1])**2 <= 1
C2 = (X - c2[0])**2 + (Y - c2[1])**2 <= 1
C3 = (X - c3[0])**2 + (Y - c3[1])**2 <= 1

# Create RGB image by stacking the binary masks (Initialize all zeroes)
rgb_image = np.zeros((X.shape[0], X.shape[1], 3))  

# Assigning each circle to a color channel: rgb
rgb_image[..., 0] = C1  # R
rgb_image[..., 1] = C2  # G
rgb_image[..., 2] = C3  # B

# Plotting the overlapping RGB circles
plt.figure(figsize=(6, 6))
plt.imshow(rgb_image, extent=(-2, 2, -2, 2))
plt.xlabel('x')
plt.ylabel('y')
plt.title('Overlapping Circles (RGB Venn Diagram)')
plt.gca().set_aspect('equal')
plt.show()

# Task 3B
# Square box aperture
box_size = 0.01
x = np.linspace(-1,1,200)
y = np.linspace(-1,1,200)
X, Y = np.meshgrid(x, y)  

# Create mask for the square box aperture
mask = (X >= -box_size) & (X <= box_size) & (Y >= -box_size) & (Y <= box_size)
# mask = (X**2 + Y**2 <= box_size**2)

# Display the mask 
plt.imshow(mask, extent=(-1, 1, -1, 1), cmap='gray')
plt.title('Square Box Aperture')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Fourier transform plot

fft_result = np.fft.fft2(mask) # compute the 2D Fourier transform of the mask
fft_shifted = np.fft.fftshift(fft_result) # shift the zero frequency component to the center
magnitude = np.abs(fft_shifted) # compute the magnitude (absolute value) of the fourier transform
log_magnitude = np.log1p(magnitude) # apply a logarithmic scale to the magnitude (log1p avoids log(0))

# Plot the result 
plt.figure(figsize=(6, 6))
plt.imshow(log_magnitude, cmap='inferno', extent=(-1, 1, -1, 1))
plt.colorbar(label='Log Magnitude',)
plt.title('Log Magnitude of Fourier Transform of the Aperture')
plt.xlabel('x')
plt.ylabel('y')
plt.show()