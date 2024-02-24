import numpy as np
from pyk4a import PyK4A, CalibrationType

def start_device(device_id):
    """
    Start the K4A device with the given device ID.
    """
    device = PyK4A(device_id=device_id)
    try:
        device.start()
        return device
    except Exception as e:
        print(f"Failed to start device {device_id}: {e}")
        return None

def get_calibration_data(device):
    """
    Retrieve the calibration data from the device.
    """
    try:
        calibration = device.calibration
        camera_matrix = np.array(calibration.get_camera_matrix(CalibrationType.COLOR))
        distortion_coefficients = np.array(calibration.get_distortion_coefficients(CalibrationType.COLOR))
        return camera_matrix, distortion_coefficients
    except Exception as e:
        print(f"Failed to get calibration data: {e}")
        return None, None

def save_data(data, filename):
    """
    Save the given data to a file.
    """
    np.save(filename, data)
    print(f"{filename} saved.")

def main():
    device1 = start_device(0)
    device2 = start_device(1)

    if device1 and device2:
        cam_matrix1, cam_dist1 = get_calibration_data(device1)
        cam_matrix2, cam_dist2 = get_calibration_data(device2)

        if cam_matrix1 is not None and cam_dist1 is not None:
            save_data(cam_matrix1, 'matrix/cam_int1.npy')
            save_data(cam_dist1, 'matrix/cam_dist1.npy')

        if cam_matrix2 is not None and cam_dist2 is not None:
            save_data(cam_matrix2, 'matrix/cam_int2.npy')
            save_data(cam_dist2, 'matrix/cam_dist2.npy')

    if device1:
        device1.stop()
    if device2:
        device2.stop()

if __name__ == "__main__":
    main()
