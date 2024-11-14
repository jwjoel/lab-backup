import numpy as np
from numpy.linalg import svd, inv
from numpy.linalg import norm

# Constants
SQUARE_SIZE = 25  # mm
NUM_CORNERS_X = 8
NUM_CORNERS_Y = 6
NUM_POINTS = NUM_CORNERS_X * NUM_CORNERS_Y  # 48

def read_image_points(filename):
    """
    Reads the image points from the file.
    Returns a list of numpy arrays, one for each image.
    """
    with open(filename, 'r') as file:
        lines = file.readlines()

    # Each image has NUM_POINTS lines
    total_points = len(lines)
    num_images = total_points // NUM_POINTS

    image_points = []
    for i in range(num_images):
        points = []
        for j in range(NUM_POINTS):
            idx = i * NUM_POINTS + j
            u_str, v_str = lines[idx].strip().split()
            u, v = float(u_str), float(v_str)
            points.append([u, v])
        image_points.append(np.array(points))

    return image_points

def generate_world_points():
    """
    Generates the world coordinates for the checkerboard corners.
    Origin is at the bottom-left corner.
    Returns a numpy array of shape (NUM_POINTS, 2).
    """
    world_points = []
    for i in range(NUM_CORNERS_Y):
        for j in range(NUM_CORNERS_X):
            x = j * SQUARE_SIZE
            y = i * SQUARE_SIZE
            world_points.append([x, y])
    return np.array(world_points)

def normalize_points(points):
    """
    Normalize a set of points so that the centroid is at the origin
    and the average distance from the origin is sqrt(2) (for 2D points).
    Returns the normalization matrix and the normalized points.
    """
    # Compute the centroid
    centroid = np.mean(points, axis=0)
    # Shift the origin to the centroid
    shifted_points = points - centroid
    # Compute the average distance to the origin
    mean_distance = np.mean(np.sqrt(np.sum(shifted_points**2, axis=1)))
    # Compute the scaling factor
    scale = np.sqrt(2) / mean_distance
    # Construct the normalization matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    # Apply normalization
    num_points = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_points, 1))))
    normalized_points = (T @ homogeneous_points.T).T
    # Remove the homogeneous coordinate
    normalized_points = normalized_points[:, :2]
    return T, normalized_points

def construct_M_matrix(image_points, world_points):
    """
    Constructs the matrix M for homography computation using normalized points.
    """
    N = world_points.shape[0]
    M = []

    # Normalize the points
    T_image, norm_image_points = normalize_points(image_points)
    T_world, norm_world_points = normalize_points(world_points)

    for i in range(N):
        x, y = norm_world_points[i, 0], norm_world_points[i, 1]
        u, v = norm_image_points[i, 0], norm_image_points[i, 1]

        row1 = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]
        row2 = [0, 0, 0, x, y, 1, -v*x, -v*y, -v]

        M.append(row1)
        M.append(row2)

    M = np.array(M)
    return M, T_image, T_world

def compute_homography(M, T_image, T_world):
    """
    Computes the homography matrix using SVD and de-normalizes it.
    """
    # Perform SVD
    U, S, Vh = svd(M)
    # Homography is the last column of V (or row of Vh)
    h = Vh[-1, :]
    # Reshape to 3x3 matrix
    H_normalized = h.reshape((3, 3))
    # De-normalize the homography
    H = inv(T_image) @ H_normalized @ T_world
    # Normalize so that H[2,2] = 1
    H = H / H[2, 2]
    return H

def project_points(H, world_points):
    """
    Projects world points to image points using the homography matrix H.
    """
    N = world_points.shape[0]
    # Convert to homogeneous coordinates
    homogeneous_world_points = np.hstack((world_points, np.ones((N, 1))))
    # Project points
    projected_points = H @ homogeneous_world_points.T
    # Convert back to inhomogeneous coordinates
    projected_points = projected_points / projected_points[2, :]
    projected_points = projected_points[:2, :].T
    return projected_points

def compute_reprojection_error(image_points, projected_points):
    """
    Computes the reprojection error between observed and projected points.
    """
    errors = norm(image_points - projected_points, axis=1)
    total_error = np.sum(errors)
    return total_error

def construct_v_ij(h_i, h_j):
    """
    Constructs the vector v_{ij} used to form the constraints for intrinsic parameters.
    """
    v_ij = np.array([
        h_i[0]*h_j[0],                               # h_i1 * h_j1
        h_i[0]*h_j[1] + h_i[1]*h_j[0],               # h_i1 * h_j2 + h_i2 * h_j1
        h_i[1]*h_j[1],                               # h_i2 * h_j2
        h_i[2]*h_j[0] + h_i[0]*h_j[2],               # h_i3 * h_j1 + h_i1 * h_j3
        h_i[2]*h_j[1] + h_i[1]*h_j[2],               # h_i3 * h_j2 + h_i2 * h_j3
        h_i[2]*h_j[2],                               # h_i3 * h_j3
    ])
    return v_ij

def compute_intrinsic_parameters(L):
    """
    Computes the intrinsic parameters from matrix L.
    """
    # Solve L b = 0 using SVD
    U, S, Vh = svd(L)
    b = Vh[-1, :]
    # Normalize b (scale invariant)
    b = b / b[-1]

    # Form the matrix B from vector b
    B11 = b[0]
    B12 = b[1]
    B22 = b[2]
    B13 = b[3]
    B23 = b[4]
    B33 = b[5]

    # Compute intrinsic parameters using Zhang's method (Appendix B)
    v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
    lambda_ = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
    if B11 == 0 or lambda_ / B11 < 0:
        raise ValueError("Invalid value encountered while computing alpha. Check data normalization.")

    alpha = np.sqrt(lambda_ / B11)
    beta = np.sqrt(lambda_ * B11 / (B11*B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lambda_
    u0 = gamma * v0 / alpha - B13 * alpha**2 / lambda_

    # Form the intrinsic matrix A
    A = np.array([
        [alpha, gamma, u0],
        [0,     beta,  v0],
        [0,     0,     1]
    ])

    return A, [alpha, beta, gamma, u0, v0]


import numpy as np
import matplotlib.pyplot as plt

def compute_extrinsic_parameters(H, A):
    """
    Computes the rotation matrix R and translation vector t from homography H and intrinsic matrix A.
    """
    # Compute the inverse of A
    A_inv = np.linalg.inv(A)
    # Compute normalized homography
    H_normalized = A_inv @ H
    # Extract columns
    h1 = H_normalized[:, 0]
    h2 = H_normalized[:, 1]
    h3 = H_normalized[:, 2]
    # Compute scale factor lambda
    lambda_ = 1 / np.linalg.norm(h1)
    # Compute rotation vectors
    r1 = lambda_ * h1
    r2 = lambda_ * h2
    r3 = np.cross(r1, r2)
    # Form the approximate rotation matrix
    R_approx = np.stack((r1, r2, r3), axis=1)
    # Orthonormalize R using SVD
    U, _, Vt = np.linalg.svd(R_approx)
    R = U @ Vt
    # Ensure determinant of R is +1
    if np.linalg.det(R) < 0:
        R = -R
    # Compute translation vector
    t = lambda_ * h3
    return R, t

def compute_camera_center(R, t):
    """
    Computes the camera center C from R and t such that R C + t = 0.
    """
    C = -R.T @ t
    return C

def plot_camera_trajectory(camera_centers):
    """
    Plots the camera trajectory in 3D space.
    """
    camera_centers = np.array(camera_centers)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(camera_centers[:, 0], camera_centers[:, 1], camera_centers[:, 2], marker='o')
    ax.set_title('Traj')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig("img.png")


def main():
    # Step 1: Read image points
    image_points_list = read_image_points('imgpoints.txt')

    # Step 2: Generate world points
    world_points = generate_world_points()

    total_reprojection_error = 0.0
    homographies = []

    # Step 3: Process each image and compute homographies
    num_images = len(image_points_list)
    for i in range(num_images):
        image_points = image_points_list[i]

        # Step 3a: Construct M matrix with normalization
        M, T_image, T_world = construct_M_matrix(image_points, world_points)

        # Step 3b: Compute homography matrix H
        H = compute_homography(M, T_image, T_world)
        homographies.append(H)

        # Step 3c: Project world points using homography
        projected_points = project_points(H, world_points)

        # Step 3d: Compute reprojection error
        error = compute_reprojection_error(image_points, projected_points)
        total_reprojection_error += error

        print(f'Image {i+1}: Reprojection Error = {error:.4f} pixels')

    print(f'\nTotal Reprojection Error for all images: {total_reprojection_error:.4f} pixels')

    # Step 4: Compute intrinsic parameters
    # Initialize matrix L (2n x 6)
    L = []

    for H in homographies:
        h1 = H[:, 0]
        h2 = H[:, 1]
        # Compute v12, v11, and v22
        v12 = construct_v_ij(h1, h2)
        v11 = construct_v_ij(h1, h1)
        v22 = construct_v_ij(h2, h2)
        # Form the two equations and append to L
        L.append(v12)
        L.append((v11 - v22))

    L = np.array(L)

    # Step 5: Compute intrinsic parameters
    A, params = compute_intrinsic_parameters(L)
    alpha, beta, gamma, u0, v0 = params

    print('\nEstimated Intrinsic Parameters:')
    print(f'alpha (focal length in x) = {alpha}')
    print(f'beta  (focal length in y) = {beta}')
    print(f'gamma (skew)             = {gamma}')
    print(f'u0    (principal point x) = {u0}')
    print(f'v0    (principal point y) = {v0}')

    print('\nIntrinsic Matrix A:')
    print(A)

    # List to store camera centers
    camera_centers = []

    # Step 6: Compute extrinsic parameters for each image
    print('\nComputing Extrinsic Parameters and Camera Trajectory...')
    for i, H in enumerate(homographies):
        # Compute extrinsic parameters
        R, t = compute_extrinsic_parameters(H, A)
        # Compute camera center
        C = compute_camera_center(R, t)
        camera_centers.append(C)
        print(f'Image {i+1}:')
        print('Rotation Matrix R:')
        print(R)
        print('Translation Vector t:')
        print(t)
        print('Camera Center C:')
        print(C)
        print('-' * 50)

    # Step 7: Plot camera trajectory
    plot_camera_trajectory(camera_centers)

if __name__ == '__main__':
    main()
