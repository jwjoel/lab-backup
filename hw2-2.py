import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Function to interpolate
def f(x):
    return np.cos(x)

# Verify linear interpolation
def verify_linear_interpolation():
    # Required maximum error
    max_error_allowed = 1e-6

    # Calculated h from previous solution
    h_linear = np.sqrt(8 * max_error_allowed)
    N_linear = int(np.ceil(np.pi / h_linear))

    # Recalculate h to fit the interval exactly
    h_linear = np.pi / N_linear

    # Generate the interpolation table
    x_table = np.linspace(0, np.pi, N_linear + 1)
    y_table = f(x_table)

    # Testing points (dense sampling for error calculation)
    x_test = np.linspace(0, np.pi, 100000)  # High-resolution points for testing
    y_true = f(x_test)

    # Linear interpolation
    y_interp = np.interp(x_test, x_table, y_table)

    # Compute the error
    error = np.abs(y_true - y_interp)
    max_error = np.max(error)

    print(f"Linear Interpolation:")
    print(f"Spacing h: {h_linear}")
    print(f"Number of table entries: {len(x_table)}")
    print(f"Maximum error: {max_error}")

    # Assert to verify that the maximum error is within the allowed limit
    assert max_error <= max_error_allowed, "Maximum error exceeds allowed error!"

# Verify quadratic interpolation
def verify_quadratic_interpolation():
    # Required maximum error
    max_error_allowed = 1e-6

    # Calculated h from previous solution
    h_quad = (6 * max_error_allowed)**(1/3)
    N_quad = int(np.ceil(np.pi / h_quad))

    # Recalculate h to fit the interval exactly
    h_quad = np.pi / N_quad

    # Generate the interpolation table
    x_table = np.linspace(0, np.pi, N_quad + 1)
    y_table = f(x_table)

    # Testing points (dense sampling for error calculation)
    x_test = np.linspace(0, np.pi, 100000)  # High-resolution points for testing
    y_true = f(x_test)

    # Quadratic interpolation
    # Using 'quadratic' interpolation with interp1d
    quadratic_interp = interp1d(x_table, y_table, kind='quadratic')

    y_interp = quadratic_interp(x_test)

    # Compute the error
    error = np.abs(y_true - y_interp)
    max_error = np.max(error)

    print(f"Quadratic Interpolation:")
    print(f"Spacing h: {h_quad}")
    print(f"Number of table entries: {len(x_table)}")
    print(f"Maximum error: {max_error}")

    # Assert to verify that the maximum error is within the allowed limit
    assert max_error <= max_error_allowed, "Maximum error exceeds allowed error!"

# Run verifications
verify_linear_interpolation()
verify_quadratic_interpolation()
