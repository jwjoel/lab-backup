import numpy as np

def fit_plane_ransac(points, iterations=2000, threshold=0.01):
    """
    Fit a plane to the point cloud using the RANSAC algorithm.
    
    Args:
        points: numpy array of shape (N, 3), where N is the number of points.
        iterations: number of iterations to run RANSAC.
        threshold: distance threshold to consider a point as an inlier.
    
    Returns:
        best_plane: tuple containing the plane parameters (a, b, c, d) of the plane equation ax + by + cz + d = 0.
        best_inliers: indices of the inlier points in the original dataset.
    """
    best_inliers = []
    best_plane = None
    n_points = points.shape[0]
    
    for _ in range(iterations):
        # Randomly select 3 different points to define a plane
        idx_samples = np.random.choice(n_points, 3, replace=False)
        sample_points = points[idx_samples]
        
        # Compute the plane normal vector
        p1, p2, p3 = sample_points
        v1 = p2 - p1
        v2 = p3 - p1
        plane_normal = np.cross(v1, v2)
        
        # Skip if the points are colinear
        if np.linalg.norm(plane_normal) == 0:
            continue
        
        plane_normal_unit = plane_normal / np.linalg.norm(plane_normal)
        d = -np.dot(plane_normal_unit, p1)
        
        # Compute distances from all points to the plane
        distances = np.abs(np.dot(points, plane_normal_unit) + d)
        
        # Identify inliers
        inliers = np.where(distances < threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (*plane_normal_unit, d)
    
    return best_plane, best_inliers

# Load the point cloud data
empty_table_points = np.loadtxt('Empty2.asc')
cluttered_table_points = np.loadtxt('TableWithObjects2.asc')

# Estimate the plane parameters for the empty table
plane_empty, inliers_empty = fit_plane_ransac(empty_table_points, iterations=1000, threshold=0.01)
print("Empty Table Plane Parameters (a, b, c, d):", plane_empty)
print("Number of inliers (Empty Table):", len(inliers_empty))

# Estimate the plane parameters for the cluttered table
plane_cluttered, inliers_cluttered = fit_plane_ransac(cluttered_table_points, iterations=1000, threshold=0.01)
print("Cluttered Table Plane Parameters (a, b, c, d):", plane_cluttered)
print("Number of inliers (Cluttered Table):", len(inliers_cluttered))

# Optional: Extract the inlier points (table surface)
table_surface_points_cluttered = cluttered_table_points[inliers_cluttered]

# Optional: Remove the table plane points to get the objects
objects_points = np.delete(cluttered_table_points, inliers_cluttered, axis=0)
