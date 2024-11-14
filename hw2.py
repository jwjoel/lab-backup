import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

data = np.loadtxt('waypoints-1.csv', delimiter=',')
x = data[:, 0]
y = data[:, 1]

N = len(x)
dt = 1.0 / 30 
t = np.arange(N) * dt

plt.figure(figsize=(8, 6))
plt.plot(x, y, 'k-', label='Ground Truth')
plt.title('Ground Truth')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.savefig('ground_truth.png')

sample_rates = [10, 1, 0.2]
interpolation_methods = ['linear', 'quadratic', 'cubic']

for sr in sample_rates:
    step = int(30 / sr)
    t_obs = t[::step]
    x_obs = x[::step]
    y_obs = y[::step]
    
    mask = (t >= t_obs[0]) & (t <= t_obs[-1])
    t_interp = t[mask]
    x_true = x[mask]
    y_true = y[mask]
    
    for method_name in interpolation_methods:
        f_x = interp1d(t_obs, x_obs, kind=method_name)
        f_y = interp1d(t_obs, y_obs, kind=method_name)
        
        x_interp = f_x(t_interp)
        y_interp = f_y(t_interp)
        
        error = np.sum(np.sqrt((x_interp - x_true)**2 + (y_interp - y_true)**2)) # cumulative error
        
        plt.figure(figsize=(8, 6))
        plt.plot(x_true, y_true, 'k-', label='Ground Truth')
        plt.plot(x_interp, y_interp, 'r--', label='Interpolated')
        plt.title(f'{method_name} Interpolation / {sr} Hz / Error = {error:.2f}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)
        plt.savefig(method_name + "-" + str(sr) + ".png")
        
        print(f'Error for {method_name} at {sr} hz: {error:.2f}')
