import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define the differential equation dy/dx = f(x, y)
def f(x, y):
    return 1 / (x**2 * (1 - y))

# Exact solution for comparison
def exact_solution(x):
    return 1 - np.sqrt(2 * (1 + 1/x))

# Define the initial condition
x0 = 1.0  # Starting from x = 1
y0 = -1.0
h = -0.05  # Negative step size (since we're moving towards x = 0)
x_end = 0.05  # Ending at x = 0.05 (approaching 0 from the right)
N = int((x_end - x0) / h)  # Number of steps

# Generate the x values from x0 down to x = x_end
x_values = np.arange(x0, x_end + h/2, h)  # Include x_end

# Euler's Method
def euler_method(f, x0, y0, h):
    x_euler = [x0]
    y_euler = [y0]
    while x_euler[-1] + h >= x_end:
        x_next = x_euler[-1] + h
        y_next = y_euler[-1] + h * f(x_euler[-1], y_euler[-1])
        x_euler.append(x_next)
        y_euler.append(y_next)
    return np.array(x_euler), np.array(y_euler)

# Fourth-Order Runge-Kutta Method
def runge_kutta_method(f, x0, y0, h):
    x_rk = [x0]
    y_rk = [y0]
    while x_rk[-1] + h >= x_end:
        x_n = x_rk[-1]
        y_n = y_rk[-1]
        k1 = f(x_n, y_n)
        k2 = f(x_n + h/2, y_n + h/2 * k1)
        k3 = f(x_n + h/2, y_n + h/2 * k2)
        k4 = f(x_n + h, y_n + h * k3)
        y_next = y_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        x_next = x_n + h
        x_rk.append(x_next)
        y_rk.append(y_next)
    return np.array(x_rk), np.array(y_rk)

def modified_midpoint(f, x0, y0, h, n):
    h_mid = h / n
    x = x0
    y = y0
    y_mid = y0
    y_mid_prev = y0
    for _ in range(n):
        y_mid_prev = y_mid
        y_mid += h_mid * f(x, y_mid)
        x += h_mid
    return y_mid

# Bulirsch-Stoer Method Implementation
def bulirsch_stoer_method(f, x0, y0, x_end, h_init, eps=1e-6):
    h = h_init
    x_values = [x0]
    y_values = [y0]
    
    x = x0
    y = y0
    
    while x > x_end:
        if x + h < x_end:
            h = x_end - x
        success = False
        while not success:
            n = 1
            # Create arrays for Richardson extrapolation
            y_seq = []
            # Modified midpoint method with n steps
            y_mid = modified_midpoint(f, x, y, h, n)
            y_seq.append(y_mid)
            # Initial extrapolation table
            error_estimate = np.inf
            for k in range(1, 10):  # Max 10 iterations for extrapolation
                n *= 2
                y_mid = modified_midpoint(f, x, y, h, n)
                y_seq.append(y_mid)
                # Richardson extrapolation
                for j in range(k, 0, -1):
                    factor = n / (n - 1)
                    y_seq[j-1] = y_seq[j] + (y_seq[j] - y_seq[j-1]) / (factor**(2*(j)) - 1)
                error_estimate = abs(y_seq[0] - y_seq[1])
                if error_estimate < eps:
                    success = True
                    break
            if not success:
                h /= 2  # Reduce step size and retry
            else:
                x += h
                y = y_seq[0]
                x_values.append(x)
                y_values.append(y)
                h *= 1.5  # Increase step size for efficiency
    return np.array(x_values), np.array(y_values)

# Compute numerical solutions
x_euler, y_euler = euler_method(f, x0, y0, h)
x_rk, y_rk = runge_kutta_method(f, x0, y0, h)
x_bs, y_bs = bulirsch_stoer_method(f, x0, y0, x_end, h_init=h)

# Compute exact solution
x_exact = np.linspace(x0, x_end, 1000)
y_exact = exact_solution(x_exact)

# Plot the results
plt.figure(figsize=(12, 8))
plt.plot(x_exact, y_exact, 'k-', label='Exact Solution')
plt.plot(x_euler, y_euler, 'bo-', label="Euler's Method", markersize=4)
plt.plot(x_rk, y_rk, 'gs-', label='Runge-Kutta 4th Order', markersize=4)
plt.plot(x_bs, y_bs, 'r^-', label='Bulirsch-Stoer Method', markersize=6)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Numerical Solutions of the Differential Equation')
plt.legend()
plt.grid(True)
plt.savefig("1.png")

# Compute errors at several x values
x_targets = [0.05, 0.1, 0.2, 0.5, 0.8, 1.0]

print(f"{'x':>5} | {'Exact y':>12} | {'Euler y':>12} | {'Error Euler':>12} | {'RK4 y':>12} | {'Error RK4':>12} | {'Bulirsch-Stoer y':>17} | {'Error BS':>12}")
print('-' * 100)

for x_target in x_targets:
    if x_target < x_end or x_target > x0:
        continue  # Skip out of range values

    exact_y_at_target = exact_solution(x_target)

    # Interpolated numerical solutions at x = x_target
    euler_interpolator = interp1d(x_euler, y_euler, kind='cubic', fill_value="extrapolate")
    rk_interpolator = interp1d(x_rk, y_rk, kind='cubic', fill_value="extrapolate")
    bs_interpolator = interp1d(x_bs, y_bs, kind='cubic', fill_value="extrapolate")

    euler_y_at_target = euler_interpolator(x_target)
    rk_y_at_target = rk_interpolator(x_target)
    bs_y_at_target = bs_interpolator(x_target)

    # Compute absolute errors
    error_euler = abs(euler_y_at_target - exact_y_at_target)
    error_rk = abs(rk_y_at_target - exact_y_at_target)
    error_bs = abs(bs_y_at_target - exact_y_at_target)

    print(f"{x_target:5.2f} | {exact_y_at_target:12.6f} | {euler_y_at_target:12.6f} | {error_euler:12.6f} | {rk_y_at_target:12.6f} | {error_rk:12.6f} | {bs_y_at_target:17.6f} | {error_bs:12.6f}")
