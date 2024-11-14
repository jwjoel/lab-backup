import numpy as np
from numpy.linalg import svd
from numpy.linalg import norm

# Constants
SQUARE_SIZE = 25  # mm
NUM_CORNERS_X = 8
NUM_CORNERS_Y = 6
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
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

def construct_M_matrix(image_points, world_points):
    """
    Constructs the matrix M for homography computation.
    """
    N = world_points.shape[0]
    M = []

    for i in range(N):
        x, y = world_points[i, 0], world_points[i, 1]
        u, v = image_points[i, 0], image_points[i, 1]

        row1 = [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]
        row2 = [0, 0, 0, x, y, 1, -v*x, -v*y, -v]

        M.append(row1)
        M.append(row2)

    M = np.array(M)
    return M

def compute_homography(M):
    """
    Computes the homography matrix using SVD.
    """
    # Perform SVD
    U, S, Vh = svd(M)
    # Homography is the last column of V (or row of Vh)
    h = Vh[-1, :]
    # Reshape to 3x3 matrix
    H = h.reshape((3, 3))
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

def compute_individual_errors(image_points, projected_points):
    errors = np.linalg.norm(image_points - projected_points, axis=1)
    return errors

import matplotlib.pyplot as plt

def visualize_reprojection(image_points, projected_points, image_index):
    plt.figure()
    plt.scatter(image_points[:, 0], image_points[:, 1], color='red', label='Observed Points')
    plt.scatter(projected_points[:, 0], projected_points[:, 1], color='blue', marker='+', label='Projected Points')
    plt.legend()
    plt.title(f'Reprojection for Image {image_index + 1}')
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    filename = f'reprojection_image_{image_index + 1}.png'
    plt.savefig(filename)

def main():
    # Step 1: Read image points
    image_points_list = read_image_points('imgpoints.txt')

    # Step 2: Generate world points
    world_points = generate_world_points()

    total_reprojection_error = 0.0

    # Step 3: Process each image
    num_images = len(image_points_list)
    for i in range(num_images):
        image_points = image_points_list[i]

        # Step 3a: Construct M matrix
        M = construct_M_matrix(image_points, world_points)

        # Step 3b: Compute homography matrix H
        H = compute_homography(M)

        # Step 3c: Project world points using homography
        projected_points = project_points(H, world_points)

        errors = compute_individual_errors(image_points, projected_points)
        max_error = np.max(errors)
        mean_error = np.mean(errors)
        print(f'Max Error: {max_error:.4f} pixels, Mean Error: {mean_error:.4f} pixels')


        visualize_reprojection(image_points, projected_points, 1)

        # Step 3d: Compute reprojection error
        error = compute_reprojection_error(image_points, projected_points)
        total_reprojection_error += error

        print(f'Image {i+1}: Reprojection Error = {error:.4f} pixels')

    print(f'\nTotal Reprojection Error for all images: {total_reprojection_error:.4f} pixels')

if __name__ == '__main__':
    main()
