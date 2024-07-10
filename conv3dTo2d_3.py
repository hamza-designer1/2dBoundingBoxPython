import numpy as np
import matplotlib.pyplot as plt

# Global variable for camera calibration matrix
CAM_CALIB = np.matrix(np.identity(3))

def setup_cam_calib(VIEW_WIDTH, VIEW_HEIGHT, VIEW_FOV):
    global CAM_CALIB
    CAM_CALIB[0, 2] = VIEW_WIDTH / 2.0
    CAM_CALIB[1, 2] = VIEW_HEIGHT / 2.0
    CAM_CALIB[0, 0] = CAM_CALIB[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))

def transform_point(point, sensorLocation, sensorRotation, VIEW_WIDTH, VIEW_HEIGHT, VIEW_FOV):
    """
    Transforms a 3D point from world coordinates to 2D image coordinates using sensor (camera) location and rotation.
    """
    def get_matrix(transform):
        """
        Creates a transformation matrix from location and rotation.
        """
        rotation = transform[1]
        location = transform[0]

        rotationYaw = rotation[2]     # rot z
        rotationRoll = rotation[0]    # rot x
        rotationPitch = rotation[1]   # rot y

        locationX = location[0]
        locationY = location[1]
        locationZ = location[2]

        c_y = np.cos(np.radians(rotationYaw))
        s_y = np.sin(np.radians(rotationYaw))
        c_r = np.cos(np.radians(rotationRoll))
        s_r = np.sin(np.radians(rotationRoll))
        c_p = np.cos(np.radians(rotationPitch))
        s_p = np.sin(np.radians(rotationPitch))

        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = locationX
        matrix[1, 3] = locationY
        matrix[2, 3] = locationZ
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

    def world_to_sensor(cords, sensorLocation, sensorRotation):
        """
        Transforms world coordinates to sensor.
        """
        sensorTransform = (sensorLocation, sensorRotation)
        sensor_world_matrix = get_matrix(sensorTransform)
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    # Setup camera calibration matrix
    setup_cam_calib(VIEW_WIDTH, VIEW_HEIGHT, VIEW_FOV)

    # Transform point to homogeneous coordinates
    point_homogeneous = np.ones((4, 1))
    point_homogeneous[:3, 0] = point

    # Transform point from world to sensor coordinates
    sensor_point = world_to_sensor(point_homogeneous, sensorLocation, sensorRotation)

    # Transform sensor coordinates to 2D image coordinates
    cords_x_y_z = sensor_point[:3, :]
    cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
    bbox = np.transpose(np.dot(CAM_CALIB, cords_y_minus_z_x))
    img_coords = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2]], axis=0)

    return img_coords

def plot_point_on_image(img_path, projected_coords):
    """
    Plots a point on an image based on projected 2D coordinates.
    """
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.scatter(projected_coords[0], projected_coords[1], color='red', marker='o')
    plt.title('Projected Point on Image')
    plt.xlim(0, img.shape[1])
    plt.ylim(img.shape[0], 0)  # Reverse y-axis to match image coordinates
    plt.show()

# Example usage
pose_values = (6.0, 2.0, 3.0)
camera_location = (5.0, 3.0, 1.5)
camera_rotation = (0.0, -10.0, 0.0)

VIEW_WIDTH = 1280
VIEW_HEIGHT = 720
VIEW_FOV = 90

projected_coords = transform_point(pose_values, camera_location, camera_rotation, VIEW_WIDTH, VIEW_HEIGHT, VIEW_FOV)
print("Projected 2D coordinates:", projected_coords)

image_path = r"E:\python_code\2dBoundingBoxPython\Data\RGB\img_2.jpeg"  # Example image path where coordinates will be plotted
plot_point_on_image(image_path, projected_coords[0])
