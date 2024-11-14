import open3d as o3d
import numpy as np
import copy

# Load point cloud from a file
# Replace 'cse_building_outdoor.ply' with the path to your point cloud file
cse_points = np.loadtxt('CSE.asc')

# 将 numpy 数组转换为 Open3D 点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cse_points)

# Make a copy of the point cloud to work on
pcd_working = copy.deepcopy(pcd)

# Initialize variables to store the plane models and point clouds
plane_models = []
plane_point_clouds = []

# Set RANSAC parameters
distance_threshold = 0.05    # Maximum distance a point can be from the plane to be considered an inlier
ransac_n = 3                 # Number of points to sample for plane fitting
num_iterations = 1000        # Number of RANSAC iterations
min_inlier_ratio = 0.01      # Minimum ratio of inliers to total points for a plane to be considered valid

print("Starting plane extraction...")

# Iterative plane extraction
while True:
    # Segment the largest plane from the point cloud
    plane_model, inliers = pcd_working.segment_plane(distance_threshold=distance_threshold,
                                                     ransac_n=ransac_n,
                                                     num_iterations=num_iterations)
    # Check if the number of inliers is sufficient
    inlier_ratio = len(inliers) / len(pcd_working.points)
    if inlier_ratio < min_inlier_ratio:
        print("No more significant planes detected.")
        break

    # Extract inlier and outlier point clouds
    plane_pcd = pcd_working.select_by_index(inliers)
    plane_pcd.paint_uniform_color(np.random.rand(3))  # Assign a random color to the plane
    remaining_pcd = pcd_working.select_by_index(inliers, invert=True)

    # Store the plane model and point cloud
    plane_models.append(plane_model)
    plane_point_clouds.append(plane_pcd)

    # Update the working point cloud
    pcd_working = remaining_pcd

    # Print plane equation and inlier statistics
    [a, b, c, d] = plane_model
    print(f"Detected plane: {a:.4f} x + {b:.4f} y + {c:.4f} z + {d:.4f} = 0")
    print(f"Number of inliers: {len(inliers)}, Remaining points: {len(pcd_working.points)}")

print("Plane extraction completed.")

# Visualize the extracted planes
print("Visualizing the extracted planes...")
o3d.visualization.draw_geometries(plane_point_clouds)

# Optionally, visualize the remaining points that were not part of any plane
if len(pcd_working.points) > 0:
    print("Visualizing the remaining points not part of any plane...")
    pcd_working.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([pcd_working])
