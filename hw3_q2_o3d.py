import open3d as o3d
import numpy as np

# 加载点云数据
empty_table_points = np.loadtxt('Empty2.asc')
cluttered_table_points = np.loadtxt('TableWithObjects2.asc')

# 将 numpy 数组转换为 Open3D 点云对象
empty_table_pcd = o3d.geometry.PointCloud()
empty_table_pcd.points = o3d.utility.Vector3dVector(empty_table_points)

cluttered_table_pcd = o3d.geometry.PointCloud()
cluttered_table_pcd.points = o3d.utility.Vector3dVector(cluttered_table_points)

# 定义一个函数来进行 RANSAC 平面拟合并打印结果
def fit_plane_and_print(pcd, name):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.015,
                                             ransac_n=3,
                                             num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"{name} 平面模型方程为: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    # 提取内点和平面
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])  # 红色表示拟合平面的内点
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    # # 可视化结果
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
    #                                   window_name=f'{name} RANSAC 平面拟合',
    #                                   point_show_normal=False)

# 对两个点云进行平面拟合并打印结果
fit_plane_and_print(empty_table_pcd, "Empty Table")
fit_plane_and_print(cluttered_table_pcd, "Cluttered Table")
