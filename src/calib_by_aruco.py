import cv2
import cv2.aruco as aruco
import numpy as np
from glob import glob

id_1 = input("Which do you want to move ?: ")
id_2 = input("Which is base ? : ")
marker_size = 390 # mm

# Load camera matrix and distortion coefficients 
camera_matrix_1 = np.load(f"matrix/cam_int{id_1}.npy")
dist_coeff_1 = np.load(f"matrix/cam_dist{id_1}.npy")
camera_matrix_2 = np.load(f"matrix/cam_int{id_2}.npy")
dist_coeff_2 = np.load(f"matrix/cam_dist{id_2}.npy")

print("camera_matrix_1:\n",camera_matrix_1)
print("dist_coeff_1:\n",dist_coeff_1)
print("camera_matrix_2:\n",camera_matrix_2)
print("dist_coeff_2:\n",dist_coeff_2)

image1_paths = glob(f'images/color_{id_1}.png')
image2_paths = glob(f'images/color_{id_2}.png')

############################################


# Find ArUco markers in the image
def find_aruco_markers(image, marker_size=6, total_markers=250, draw=True):
    gray = cv2.cvtColor(image, cv2.color_BGR2GRAY)
    key = getattr(aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
    aruco_dict = aruco.getPredefinedDictionary(key)
    aruco_params = aruco.DetectorParameters()
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    if draw:
        aruco.drawDetectedMarkers(image, corners, ids)

    return corners, ids

# Estimate the pose of the ArUco markers
def estimate_pose(image, corners, ids, camera_matrix, dist_coeff):
    if corners and len(corners) > 0 and ids is not None and len(ids) > 0:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeff)
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(image, camera_matrix, dist_coeff, rvec, tvec, 0.5)
        return rvecs, tvecs
    return None, None

# Calculate the relative pose of the camera
def relative_camera_pose(rvec1, tvec1, rvec2, tvec2):
    R1, _ = cv2.Rodrigues(rvec1)
    R2, _ = cv2.Rodrigues(rvec2)
    R_rel = np.dot(R2, R1.T)
    t_rel = tvec2 - np.dot(R_rel, tvec1)
    return R_rel, t_rel

# Convert the relative pose to the format used in Unity
def convert_for_unity(R, T):
    R_unity = R
    T_unity = -T
    return R_unity, T_unity

# Save the relative transformation matrices
def save_rt_matrices(R, T):
    R,T = convert_for_unity(R, T)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T

    print("4x4 Transformation Mtx :\n",transformation_matrix)
    np.save(f'Transformation_{id_1}_{id_2}.npy', transformation_matrix)
    
def main():
    for image1_path, image2_path in zip(image1_paths, image2_paths):
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        corners1, ids1 = find_aruco_markers(image1)
        corners2, ids2 = find_aruco_markers(image2)

        if ids1 is not None and ids2 is not None:
            rvecs1, tvecs1 = estimate_pose(image1, corners1, ids1, camera_matrix_1, dist_coeff_1)
            rvecs2, tvecs2 = estimate_pose(image2, corners2, ids2, camera_matrix_2, dist_coeff_2)

            common_ids = np.intersect1d(ids1.flatten(), ids2.flatten())

            for id in common_ids:
                idx1 = np.where(ids1 == id)[0][0]
                idx2 = np.where(ids2 == id)[0][0]
                R, T = relative_camera_pose(rvecs1[idx1], tvecs1[idx1][0], rvecs2[idx2], tvecs2[idx2][0])
                print(f"Relative Rotation Matrix from Camera 1 to Camera 2 for Marker {id}: \n{R}")
                print(f"Relative Translation Vector from Camera 1 to Camera 2 for Marker {id}: \n{T}")

                save_rt_matrices(R, T)

            cv2.imshow("Image 1", image1)
            cv2.imshow("Image 2", image2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No common markers found !")
            print("Hint: Is the room too dark ? Try reshooting the photo or apply gamma correction to the image !")

if __name__ == "__main__":
    main()