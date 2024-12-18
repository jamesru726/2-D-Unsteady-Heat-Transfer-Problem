import numpy as np
import matplotlib.pyplot as plt
import math
import os
import cv2
import sys

# Makes Video
def Video(phi):
    # Directory for saving frames
    os.makedirs("frames", exist_ok=True)

    # Save each frame as an image
    for t in range(Steps//Skip):
        plt.figure()
        x1 = np.linspace(-L_x/2, L_x/2, x) * 100
        y1 = np.linspace(L_y/2, -L_y/2, y) * 100
        X, Y = np.meshgrid(x1, y1)

        # Contour Plot
        plt.contourf(X, Y, phi[t*Skip, :, :], levels = x, cmap='jet')
        contour = plt.colorbar(label='Temperature (C)')
        plt.title(f"{NAME}: {mat1} and {mat2}\nTime step: {del_t}s, Steps: {Steps}, Skipped Frames: {Skip - 1} \nTime: {time1[t*Skip]:.2f}s")
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.savefig(f"frames/frame_{t:03d}.png")
        plt.close()

    # Combine images into a video using OpenCV
    frame_files = sorted([f"frames/{f}" for f in os.listdir("frames") if f.endswith(".png")])
    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape

    video = cv2.VideoWriter(f'{name}_{mat2}_{del_t}_{Steps}_Steps_{x}x{y}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), Speed, (width, height))

    for file in frame_files:
        video.write(cv2.imread(file))

    video.release()

# Explicit Euler
def FE():
    # Each time step
    for k in range(1,Steps + 1):

        # Column index
        for i in range(1, y-1):
            
            # Row index
            for j in range(1, x-1):
                
                # Multiply Each Coefficient to its corresponding phi value
                phi[k, i, j] = A_N[i - 1, j - 1] * phi[k - 1, i - 1, j] + A_W[i - 1, j - 1] * phi[k - 1, i, j - 1] + A_CF[i - 1, j - 1] * phi[k - 1, i, j] + A_E[i - 1, j - 1] * phi[k - 1, i, j + 1] + A_S[i - 1, j - 1] * phi[k - 1, i + 1, j]
                
                # If any value doesn't make physical sense, exit code
                if phi[k, i, j] > np.max([T_N, T_E, T_S, T_W, T_init]):
                    print('Solution Diverges')
                    sys.exit(1)

    return phi
        

def BE(phi):

    phi_old = 0  # Initialize phi_old

    # Time step loop for Implicit Euler
    for k in range(Steps):

        # Copy the previous time step values for current time step initial guess
        phi[k + 1, :, :] = phi[k, :, :].copy()

        # Iterative loop for Gauss Seidel
        for iter in range(max_iter):

            # Copy current GS iteration for calculating residual
            phi_old = phi[k + 1, :, :].copy()
            
            # Column Index
            for i in range(1, y - 1): 
                
                # Row Index
                for j in range(1, x - 1):
                    
                    # Store current b value in a temp variable to not alter b values
                    temp_b = phi[k, i, j]

                    # Add all A coeffcients with the corresponding guess
                    # In normal GS, it's normally subtraction, but the plus accounts for the negative A neighbor coeffcients
                    temp_b += A_N[i - 1, j - 1] * phi[k + 1, i - 1, j]
                    temp_b += A_W[i - 1, j - 1] * phi[k + 1, i, j - 1]
                    temp_b += A_E[i - 1, j - 1] * phi[k + 1, i, j + 1]
                    temp_b += A_S[i - 1, j - 1] * phi[k + 1, i + 1, j]

                    # Divide by Center value
                    phi[k + 1, i, j] = temp_b / A_CB[i - 1, j - 1]  

            if relax == 1:
                # Relaxation
                phi[k + 1, :, :] = lambda1 * phi + (1 - lambda1) * phi_old

            # Find residual
            res = np.max(np.abs(phi[k + 1, :, :] - phi_old))
            
            # Check is residual is less than tolerance
            if res < tol:

                break
    
    return phi

# Assinging A matrix coefficients in for upper material
# Use ceil for dealing with odd cases
# Refer to the end of the report for the explanation of colors
def A_Coeff():

    index = math.ceil((y-2)/2)

    for i in range(1, index):
        
        # GREEN
        A_N[i - 1, i:-i] = A_S[i - 1, i:-i] = k1 * inv_delta_y * tau1 
        A_E[i - 1, i:-i] = A_W[i - 1, i:-i] = k1 * inv_delta_x * tau1
        A_CF[i - 1, i:-i] = A_C1F
        A_CB[i - 1, i:-i] = A_C1B

        # PINK Left side Mat 2
        A_S[i, i - 1] = ((k2 - k1)/4 + k2) * inv_delta_y * tau2
        A_N[i, i - 1] = (-(k2 - k1)/4 + k2) * inv_delta_y * tau2
        A_E[i, i - 1] = ((k1 - k2)/4 + k2) * inv_delta_x * tau2
        A_W[i, i - 1] = (-(k1 - k2)/4 + k2) * inv_delta_x * tau2
        A_CF[i, i - 1] = A_C2F
        A_CB[i, i - 1] = A_C2B

        # PINK Right side Mat 2
        A_S[i, -i] = ((k2 - k1)/4 + k2) * inv_delta_y * tau2
        A_N[i, -i] = (-(k2 - k1)/4 + k2) * inv_delta_y * tau2
        A_E[i, -i] = ((k2 - k1)/4 + k2) * inv_delta_x * tau2
        A_W[i, -i] = (-(k2 - k1)/4 + k2) * inv_delta_x * tau2
        A_CF[i, -i] = A_C2F
        A_CB[i, -i] = A_C2B

    for i in range(index):

        # LIGHT BLUE LEFT
        A_S[i, i] = ((k2 - k1)/4 + k1) * inv_delta_y * tau1
        A_N[i, i] = (-(k2 - k1)/4 + k1) * inv_delta_y * tau1
        A_E[i, i] = ((k2 - k1)/4 + k1) * inv_delta_x * tau1
        A_W[i, i] = (-(k2 - k1)/4 + k1) * inv_delta_x * tau1
        A_CF[i, i] = A_C1F
        A_CB[i, i] = A_C1B

        # LIGHT BLUE RIGHT
        A_S[i, -i - 1] = ((k2 - k1)/4 + k1) * inv_delta_y * tau1
        A_N[i, -i - 1] = (-(k2 - k1)/4 + k1) * inv_delta_y * tau1
        A_E[i, -i - 1] = ((k1 - k2)/4 + k1) * inv_delta_x * tau1
        A_W[i, -i - 1] = (-(k1 - k2)/4 + k1) * inv_delta_x * tau1
        A_CF[i, -i - 1] = A_C1F
        A_CB[i, -i - 1] = A_C1B

    # BLUE CENTER
    A_E[index, index - 1] = A_W[index, index - 1] = A_W[index, -index] = A_E[index, -index] =  k2 * inv_delta_x * tau2
    A_N[index, index - 1] = A_N[index, -index] = (-(k2 - k1)/4 + k2) * inv_delta_y * tau2
    A_S[index, index - 1] = A_S[index, -index] = ((k2 - k1)/4 + k2) * inv_delta_y * tau2
    A_CF[index, index - 1] = A_CF[index, -index] = A_C2F
    A_CB[index, index - 1] = A_CB[index, -index] = A_C2B

def Plot(phi):
    x1 = np.linspace(-L_x/2, L_x/2, x) * 100
    y1 = np.linspace(L_y/2, -L_y/2, y) * 100
    X, Y = np.meshgrid(x1, y1)
    Z = phi[Steps, :, :]

    # Contour Plot
    plt.contourf(X, Y, Z, levels = x, cmap='jet')
    # Colobar label, tick, and temperature range adjustments
    contour = plt.colorbar(label='Temperature (C)')

    # Title and Axis labels
    plt.title(f'{mat1} and {mat2}: Unsteady 2-D Heat Conduction\nTime Step: {del_t} - Steps: {Steps}, Time: {time1[-1]} s')
    plt.xlabel('x length (cm)')
    plt.ylabel('y length (cm)')

# Function for Probes
def Probe():
    plt.plot(time1, phi_F[:, 1, math.floor(y/2)], label = 'Explicit: Probe at top face')
    plt.plot(time1, phi_F[:, math.floor(y/2), math.floor(y/2)], label = 'Explicit: Probe at center')
    plt.plot(time1, phi_F[:, -2, math.floor(y/2)], label = 'Explicit: Probe at bottom face')

    plt.plot(time1, phi_B[:, 1, math.floor(y/2)], label = 'Implicit: Probe at top face')
    plt.plot(time1, phi_B[:, math.floor(y/2), math.floor(y/2)], label = 'Implicit: Probe at center')
    plt.plot(time1, phi_B[:, -2, math.floor(y/2)], label = 'Implicit: Probe at bottom face')

    plt.title('Temperature Probes along x = 0 cm')
    plt.xlabel('Time (s)')
    plt.ylabel('Temperature')
    plt.ylim([10, 25])
    plt.legend()
    plt.grid(True)

# Grid Size X x Y
x = 50
y = 50

# Initialize values for calculations
L_x = 0.1 # meters (10 cm)
L_y = 0.1 # meters (10 cm)

# Initial inner temperature
T_init = 20

# Boundary conditions
# [N, E, S, W]
T_N = 10
T_E = 35
T_S = 20
T_W = 35

# Choose 1 for Explicit Euler
# -1 for Implicit Euler with video
# 0 for both with probes.
Solver = -1

# Gauss Seidel
max_iter = 300
tol = 1e-3
relax = 0 # 1 for relaxation, else for no relaxation
lambda1 = 1 # How much to relax, only active is relax is set to 1

# 1 to make video
# Anything else to not make video
Animation = 1

# Metal Properties
# 304 Stainless Steel
mat1 = "SS" 
rho1 = 7900 # Density K/m^3
cp1 = 477 # Specific Heat J/kg-K
k1 = 14.9 # Thermal Conductivity W/m*K

# Material 2
mat2 = "Beryllium" # Material Name
rho2 = 1850 # Density K/m^3
cp2 = 1825 # Specific Heat J/kg-K
k2 = 200 # Thermal Conductivity W/m*K

# Time Step
del_t = 0.04 # Delta_T
Steps = 1000 # Number of steps

# Skip Frames: calculates all frames, but only uses the nth frame to save time for creating video
Skip = 10
Vid_time = 10 # Length of time of the video in seconds

Speed = np.floor(Steps/ Vid_time / Skip)  # For dcalculating frames per second

# Time array for plotting
time1 = np.arange(0, del_t*(Steps + 1), del_t)

xticks = np.arange(-5, 6, 1)
yticks = np.arange(-5, 6, 1)

# Solve for delta_x and delta_y
delta_x = L_x/(x)
delta_y = L_y/(y)

# Initialize inv_delta so they wouldn't be repeatedly calculated in the for loops
inv_delta_x = 1/delta_x**2
inv_delta_y = 1/delta_y**2

# Some values for Calculating A coefficients
tau1 = del_t / (rho1 * cp1)
tau2 = del_t / (rho2 * cp2)

A_C1F = -2 * k1 * tau1 * (inv_delta_x + inv_delta_y) + 1
A_C2F = -2 * k2 * tau2 * (inv_delta_x + inv_delta_y) + 1

A_C1B = 2 * k1 * tau1 * (inv_delta_x + inv_delta_y) + 1
A_C2B = 2 * k2 * tau2 * (inv_delta_x + inv_delta_y) + 1

A_NS2 = k2 * inv_delta_y * tau2
A_EW2 = k2 * inv_delta_x * tau2

################################################################################################

# Initialize phi
phi = np.zeros((Steps + 1, x, y))

# Initialize North, West, South, and West boundaries
phi[0,:,:] = T_init
phi[:, 0, 0:y] = T_N
phi[:, 0:x, x-1] = T_E
phi[:, 0:x, 0] = T_W
phi[:, y-1, 0:x] = T_S

# Initialize A matrix Coefficients
A_CF = np.full((x-2, y-2), A_C2F)
A_CB = np.full((x-2, y-2), A_C2B)
A_N = np.full((x-2, y-2), A_NS2)
A_E = np.full((x-2, y-2), A_EW2)
A_S = np.full((x-2, y-2), A_NS2)
A_W = np.full((x-2, y-2), A_EW2)

# Build A Arrays
A_Coeff()

# Pick Solver

if Solver == 1:

    NAME = "Forward Euler"
    name = "Explicit"
    phi_F = FE()

    Plot(phi_F)

    if Animation == 1:
        # Create Video and Export it to directory
        Video(phi_F)

elif Solver == -1:

    NAME = "Backward Euler with Gauss Seidel"
    name = "Implicit"
    phi_B = BE(phi)

    Plot(phi_B)

    if Animation == 1:
        # Create Video and Export it to directory
        Video(phi_B)

elif Solver == 0:
    NAME = "Forward Euler"
    name = "Explicit"
    phi_F = FE()

    plt.figure(1)
    Plot(phi_F)
    # Initialize phi
    phi = np.zeros((Steps + 1, x, y))

    # Reinitialize North, West, South, and West boundaries
    phi[0,:,:] = T_init
    phi[:, 0, 0:y] = T_N
    phi[:, 0:x, x-1] = T_E
    phi[:, 0:x, 0] = T_W
    phi[:, y-1, 0:x] = T_S

    NAME = "Backward Euler with Gauss Seidel"
    name = "Implicit"
    phi_B = BE(phi)

    plt.figure(2)
    Plot(phi_B)

    plt.figure(3)
    Probe()

plt.show()





