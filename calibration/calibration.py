import cv2
import glob 
import numpy as np
import cv2
import copy
import imageio
import glob
import os

script_dir = os.path.dirname(os.path.abspath(__file__))


def get_chessboard_points(chessboard_shape, dx, dy):
    points = np.empty((0, 3), dtype=np.float32)  # Garantizamos dtype np.float32
    for i in range(chessboard_shape[1]):
        for j in range(chessboard_shape[0]):
            points = np.vstack((points, np.array([dx*i, dy*j, 0], dtype=np.float32)))
    return points


def calibration():
    #1. loading the images
    imgs_path = glob.glob(f'{script_dir}/calibration_photos/*.jpg')  

    imgs = [imageio.imread(filename) for filename in imgs_path]

    #2. Obtaining the corners 
    pattern_size = (7, 7)
    corners = [cv2.findChessboardCorners((img), pattern_size) for img in imgs]

    # 3. Obtaining the calibration
    chessboard_points = get_chessboard_points(pattern_size, 30, 30)
    valid_corners = [cor[1] for cor in corners if cor[0]]
    valid_corners = np.asarray(valid_corners, dtype=np.float32)

    real_points = get_chessboard_points(pattern_size, 30, 30)
    object_points = np.asarray([real_points for i in range(len(valid_corners))], dtype=np.float32)
    image_points = np.asarray(valid_corners, dtype=np.float32)

    #calibrating
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, (350, 350), None, None)

    # Obtain extrinsics
    extrinsics = list(map(lambda rvec, tvec: np.hstack((cv2.Rodrigues(rvec)[0], tvec)), rvecs, tvecs))

    #5. Save outputs
    np.savez(f'{script_dir}/calib', intrinsic=intrinsics, extrinsic=extrinsics, distortion=dist_coeffs)


def undistort_image(image, intrinsics, dist_coeffs):

    h,  w = image.shape[:2]

    new_intrinsics, roi = cv2.getOptimalNewCameraMatrix(intrinsics, dist_coeffs, (w,h), 1, (w,h))
    undistorted = cv2.undistort(image, intrinsics, dist_coeffs, None, new_intrinsics)
    # crop the image
    x, y, w, h = roi
    return undistorted[y:y+h, x:x+w]



if __name__ == "__main__":
    calibration()