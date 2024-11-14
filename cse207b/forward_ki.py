import numpy as np

def forward_kinematics(q0, q1, q2, L0=5, L1=10, L2=10, L3=10):
    """
    Compute the forward kinematics for a 3-DOF planar robot arm.
    
    Parameters:
    q0, q1, q2 : float
        Joint angles in degrees.
    L0, L1, L2, L3 : float
        Link lengths in cm. Default values are given as per the problem statement.
    
    Returns:
    (x_e, y_e) : tuple of float
        The (x, y) position of the end effector.
    """
    # Convert angles from degrees to radians
    q0_rad = np.radians(q0)
    q1_rad = np.radians(q1)
    q2_rad = np.radians(q2)
    
    # Compute the position of the end effector
    x_e = L1 * np.sin(q0_rad) + L2 * np.sin(q0_rad + q1_rad) + L3 * np.sin(q0_rad + q1_rad + q2_rad)
    y_e = L0 + L1 * np.cos(q0_rad) + L2 * np.cos(q0_rad + q1_rad) + L3 * np.cos(q0_rad + q1_rad + q2_rad)
    
    return x_e, y_e

# Check the position when all joint angles are 0 degrees
q0 = -40
q1 = 87.96
q2 = 67.87

# Compute the end effector position
x_e, y_e = forward_kinematics(q0, q1, q2)

print(f"End effector position when all joints are at 0 degrees: x_e = {x_e:.2f} cm, y_e = {y_e:.2f} cm")
