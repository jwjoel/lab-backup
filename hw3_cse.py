import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import time

def load_point_cloud(filename):
    """
    Loads the point cloud data from a file into a NumPy array.
    Assumes the file is in ASCII format with one point per line: x y z
    """
    point_cloud = np.loadtxt(filename)
    return point_cloud

def fit_plane(points):
    """
    Fits a plane to a set of points using Singular Value Decomposition (SVD).
    Returns the plane normal vector and a point on the plane (the centroid).
    """
    centroid = np.mean(points, axis=0)
    uu, dd, vv = np.linalg.svd(points - centroid)
    normal = vv[2, :]
    return normal, centroid

def plane_model(point, normal):
    """
    Returns the plane equation coefficients (a, b, c, d) given a normal vector and a point on the plane.
    The plane equation is ax + by + cz + d = 0
    """
    a, b, c = normal
    x0, y0, z0 = point
    d = - (a * x0 + b * y0 + c * z0)
    return a, b, c, d

def point_plane_distance(points, plane_coeffs):
    """
    Calculates the perpendicular distances from a set of points to a plane.
    Plane coefficients are (a, b, c, d) for the plane equation: ax + by + cz + d = 0
    """
    a, b, c, d = plane_coeffs
    numerator = np.abs(a * points[:,0] + b * points[:,1] + c * points[:,2] + d)
    denominator = np.sqrt(a**2 + b**2 + c**2)
    distances = numerator / denominator
    return distances

def ransac_plane_fit(points, threshold, max_iterations=1000):
    """
    Performs RANSAC to fit a plane to the given points.
    Returns the best plane coefficients and the indices of inlier points.
    """
    best_plane = None
    best_inliers = []
    best_inlier_count = 0
    N = len(points)

    for iteration in range(max_iterations):
        # Randomly sample 3 unique points
        indices = np.random.choice(N, 3, replace=False)
        sample_points = points[indices]

        # Check for colinear points
        if np.linalg.matrix_rank(sample_points - sample_points[0]) < 2:
            continue  # Skip degenerate samples

        # Fit a plane to the sample points
        normal, centroid = fit_plane(sample_points)
        plane_coeffs = plane_model(centroid, normal)

        # Compute distances from all points to the plane
        distances = point_plane_distance(points, plane_coeffs)

        # Identify inliers
        inlier_indices = np.where(distances < threshold)[0]
        inlier_count = len(inlier_indices)

        # Update the best plane if current one has more inliers
        if inlier_count > best_inlier_count:
            best_plane = plane_coeffs
            best_inliers = inlier_indices
            best_inlier_count = inlier_count

    return best_plane, best_inliers

def extract_planes(point_cloud, distance_threshold=0.05, min_inliers_ratio=0.01):
    """
    Iteratively extracts planes from the point cloud using RANSAC.
    Returns a list of plane models and their corresponding inlier points.
    """
    remaining_points = point_cloud.copy()
    total_points = len(point_cloud)
    planes = []
    plane_inliers = []

    iteration = 0
    while True:
        iteration += 1
        print(f"\nIteration {iteration}:")
        # Fit a plane using RANSAC
        plane_coeffs, inlier_indices = ransac_plane_fit(remaining_points, threshold=distance_threshold)

        if plane_coeffs is None or len(inlier_indices) == 0:
            print("No plane found.")
            break

        inlier_ratio = len(inlier_indices) / total_points
        print(f"Plane found with {len(inlier_indices)} inliers (Inlier ratio: {inlier_ratio:.4f})")

        if inlier_ratio < min_inliers_ratio:
            print("Inlier ratio below threshold. Stopping iteration.")
            break

        # Store the plane and its inliers
        planes.append(plane_coeffs)
        plane_inliers.append(remaining_points[inlier_indices])

        # Remove inliers from the point cloud
        remaining_points = np.delete(remaining_points, inlier_indices, axis=0)

        print(f"Remaining points: {len(remaining_points)}")
        if len(remaining_points) < 3:
            print("Not enough points for further plane extraction.")
            break

    return planes, plane_inliers, remaining_points

def visualize_planes(plane_inliers, remaining_points=None):
    """
    Visualizes the extracted planes and the remaining points using Matplotlib.
    """
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cm

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Assign a random color to each plane
    num_planes = len(plane_inliers)
    colors = cm.rainbow(np.linspace(0, 1, num_planes))

    for idx, inliers in enumerate(plane_inliers):
        ax.scatter(inliers[:,0], inliers[:,1], inliers[:,2], color=colors[idx], s=1)

    # Plot remaining points in grey
    if remaining_points is not None and len(remaining_points) > 0:
        ax.scatter(remaining_points[:,0], remaining_points[:,1], remaining_points[:,2], color='grey', s=1)

    ax.set_title('Extracted Planes')
    plt.show()

def main():
    # Load point cloud data
    filename = 'CSE.asc'  # Replace with your point cloud filename
    point_cloud = load_point_cloud(filename)
    print(f"Loaded point cloud with {len(point_cloud)} points.")

    # Set RANSAC parameters
    distance_threshold = 0.05  # Adjust based on point cloud scale and noise
    min_inliers_ratio = 0.01   # Minimum ratio of inliers to total points

    start_time = time.time()

    # Extract planes
    planes, plane_inliers, remaining_points = extract_planes(
        point_cloud,
        distance_threshold=distance_threshold,
        min_inliers_ratio=min_inliers_ratio
    )

    print(planes)

    end_time = time.time()
    print(f"\nPlane extraction completed in {end_time - start_time:.2f} seconds.")
    print(f"Extracted {len(planes)} planes.")

    # Visualize the extracted planes
    visualize_planes(plane_inliers, remaining_points)

if __name__ == "__main__":
    main()
