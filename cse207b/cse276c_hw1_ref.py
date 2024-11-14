import cv2
import numpy as np
import glob

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
        image_points.append(np.array(points, dtype=np.float32))
    return image_points

def generate_world_points():
    """
    Generates the world coordinates for the checkerboard corners.
    Origin is at the bottom-left corner.
    Returns a numpy array of shape (NUM_POINTS, 3).
    """
    objp = np.zeros((NUM_POINTS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:NUM_CORNERS_X, 0:NUM_CORNERS_Y].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE  # Scale to actual square size
    return objp

def main():
    # Step 1: Read image points
    image_points_list = read_image_points('imgpoints.txt')

    # Step 2: Generate world points
    objp = generate_world_points()

    # Prepare data for calibration
    objpoints = []  # 3d point in real world space (list per image)
    imgpoints = []  # 2d points in image plane (list per image)

    # Number of images
    num_images = len(image_points_list)

    for i in range(num_images):
        # Append object points (same for all images)
        objpoints.append(objp)

        # Append corresponding image points
        imgpoints.append(image_points_list[i])

    # Assume image size (you need to set this correctly)
    # If you have the image size (width, height), set it here.
    # For example, image_size = (640, 480)
    # Alternatively, extract it from your data
    # We'll compute the max u and v values from image points
    all_u = np.concatenate([pts[:, 0] for pts in imgpoints])
    all_v = np.concatenate([pts[:, 1] for pts in imgpoints])
    width = int(np.max(all_u)) + 1
    height = int(np.max(all_v)) + 1
    image_size = (width, height)

    # Camera calibration
    # We assume zero distortion coefficients to match your ideal pinhole model
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None,
        flags=cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
    )

    # Reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        
        # 调整 imgpoints2 的形状
        imgpoints2 = np.squeeze(imgpoints2)
        
        # 确保数据类型匹配
        imgpoints2 = imgpoints2.astype(np.float32)
        imgpoints[i] = imgpoints[i].astype(np.float32)
        
        # 计算重投影误差
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        print(f"Image {i+1}: Reprojection Error = {error:.4f} pixels")


    print(f"\nTotal Reprojection Error for all images: {total_error:.4f} pixels")

    # Intrinsic parameters
    print('\nEstimated Intrinsic Parameters:')
    print('Camera Matrix (Intrinsic Matrix):')
    print(mtx)

    # Since we assumed zero distortion, dist should be zeros
    print('\nDistortion Coefficients:')
    print(dist)

    # For comparison with your parameters
    alpha = mtx[0, 0]
    beta = mtx[1, 1]
    gamma = mtx[0, 1]
    u0 = mtx[0, 2]
    v0 = mtx[1, 2]

    print('\nParameters:')
    print(f'alpha (focal length in x) = {alpha}')
    print(f'beta  (focal length in y) = {beta}')
    print(f'gamma (skew)             = {gamma}')
    print(f'u0    (principal point x) = {u0}')
    print(f'v0    (principal point y) = {v0}')

if __name__ == '__main__':
    main()
