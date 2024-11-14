import numpy as np
from scipy.optimize import fsolve

L0 = 5
L1 = L2 = L3 = 10 

# since we only need single valid solution, we assume q0 is a fixed joint at 40 deg to simplify
q0_fixed_deg = -40
q0_fixed_rad = np.radians(-40)

def equations(vars):
    q1, q2 = vars
    
    # solve the forward kinematics equations with two vars
    x = (L1 * np.cos(q0_fixed_rad) +
         L2 * np.cos(q0_fixed_rad + q1) +
         L3 * np.cos(q0_fixed_rad + q1 + q2))
    y = (L0 + L1 * np.sin(q0_fixed_rad) +
         L2 * np.sin(q0_fixed_rad + q1) +
         L3 * np.sin(q0_fixed_rad + q1 + q2))
    
    eq1 = x - 10
    eq2 = y - 15
    
    return [eq1, eq2]

# solve the equations
solution = fsolve(equations, [np.radians(30.0), np.radians(30.0)])
q1_sol, q2_sol = solution

q1_deg = np.degrees(q1_sol)
q2_deg = np.degrees(q2_sol)
q0_deg = q0_fixed_deg 

print(f"q0 = {q0_deg:.2f} degrees")
print(f"q1 = {q1_deg:.2f} degrees")
print(f"q2 = {q2_deg:.2f} degrees")
