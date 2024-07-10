import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 

# Setup your camera parameters (example values)
VIEW_WIDTH = 1920
VIEW_HEIGHT = 1080
VIEW_FOV = 90.0

# Initialize the camera calibration matrix
CAM_CALIB = np.identity(3)
CAM_CALIB[0, 2] = VIEW_WIDTH / 2.0
CAM_CALIB[1, 2] = VIEW_HEIGHT / 2.0
CAM_CALIB[0, 0] = CAM_CALIB[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))

def get_matrix(location, rotation):
    """
    Creates a transformation matrix from location and rotation.
    """
    rotationYaw, rotationPitch, rotationRoll = rotation
    locationX, locationY, locationZ = location

    c_y = np.cos(np.radians(rotationYaw))
    s_y = np.sin(np.radians(rotationYaw))
    c_r = np.cos(np.radians(rotationRoll))
    s_r = np.sin(np.radians(rotationRoll))
    c_p = np.cos(np.radians(rotationPitch))
    s_p = np.sin(np.radians(rotationPitch))

    matrix = np.identity(4)
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

def transform_point(point, location, rotation, inverse=False):
    """
    Transforms a 3D point using a given location and rotation.
    """
    transform_matrix = get_matrix(location, rotation)
    if inverse:
        transform_matrix = np.linalg.inv(transform_matrix)
    point_homogeneous = np.append(point, 1)
    transformed_point = np.dot(transform_matrix, point_homogeneous)
    return transformed_point[:3]  # Take the first 3 elements

def project_point_to_image(point, sensorLocation, sensorRotation):
    """
    Projects a 3D point to 2D image coordinates.
    """
    # Transform point from vehicle to world coordinates (assuming the point is in vehicle coordinates)
    world_point = transform_point(point, sensorLocation, sensorRotation)  # Assuming vehicle is at the origin

    # Transform point from world to sensor (camera) coordinates
    sensor_point = transform_point(world_point, sensorLocation, sensorRotation, inverse=True)

    # Project the 3D sensor coordinates to 2D image coordinates
    cords_y_minus_z_x = np.array([sensor_point[1], -sensor_point[2], sensor_point[0]])
    bbox = np.dot(CAM_CALIB, cords_y_minus_z_x)
    image_coords = [bbox[0] / bbox[2], bbox[1] / bbox[2]]
    return image_coords

def read_and_print_file(directory, filename,convert):

    try:
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            lines = file.readlines()
            all_rows = []
            for line in lines:
                values = line.strip().split()
                if(convert):
                    xyz_tuple_cm= tuple(map(float, values))
                    xyz_tuple=tuple(element / one_meters for element in xyz_tuple_cm)
                else:
                    xyz_tuple=tuple(map(float, values))
                #yz_tuple = tuple(map(float, values))  # Convert strings to floats and make a tuple
                all_rows.append(xyz_tuple)
            #print(all_rows)
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return all_rows
def write_coordinates_to_file(coordinates, output_file):
    
    """
    Writes coordinates to a text file, each coordinate pair in a new line.
    
    Parameters:
    - coordinates (list of tuples): List containing coordinate tuples (x, y).
    - output_file (str): Path to the output text file.
    """
    try:
        with open(output_file, 'w') as file:
            for coord in coordinates:
                file.write(f"{coord[0]}, {coord[1]}\n")
        print(f"Coordinates written to {output_file} successfully.")
    except Exception as e:
        print(f"Error writing coordinates to {output_file}: {e}")       
        
def read_coordinates_from_file(input_file):
    """
    Reads coordinates from a text file.
    
    Parameters:
    - input_file (str): Path to the input text file containing coordinates.
    
    Returns:
    - list of tuples: List containing coordinate tuples (x, y).
    """
    coordinates = []
    try:
        with open(input_file, 'r') as file:
            for line in file:
                x, y = map(float, line.strip().split(','))
                coordinates.append((x, y))
        print(f"Coordinates read from {input_file} successfully.")
    except FileNotFoundError:
        print(f"The file {input_file} was not found.")
    except Exception as e:
        print(f"Error reading coordinates from {input_file}: {e}")
    return coordinates

def plot_coordinates_on_image(coordinates, image_path):
    """
    Plots coordinates on an image.
    
    Parameters:
    - coordinates (list of tuples): List containing coordinate tuples (x, y).
    - image_path (str): Path to the image file to plot coordinates on.
    """
    try:
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.axis('off')  # Turn off axis labels
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinate system
        plt.scatter([coord[0] for coord in coordinates], [coord[1] for coord in coordinates], color='red', s=10)
        plt.title('Image with Coordinates')
        plt.show()
    except FileNotFoundError:
        print(f"The image file {image_path} was not found.")
    except Exception as e:
        print(f"Error plotting coordinates on {image_path}: {e}")
        
        
        
# image_coords = project_point_to_image(point_3d, sensor_location, sensor_rotation)
# print("2D Image Coordinates:", image_coords)
# Function to read a text file and print its contents

# Specify the path to the text file
file_path_pose = r'E:\python_code\2dBoundingBoxPython\Data\Pose'
file_path_loc = r'E:\python_code\2dBoundingBoxPython\Data\CameraLocation'
file_path_rot = r'E:\python_code\2dBoundingBoxPython\Data\CameraRotation'
input_file = "output_coordinates.txt"  # Example input file containing coordinates
image_path = r"E:\python_code\2dBoundingBoxPython\Data\RGB\img_2.jpeg"  # Example image path where coordinates will be plotted
output_file = "output_coordinates.txt"
coords=[]
one_meters=100
file_names_pos=os.listdir(file_path_pose)
#print("poses:",file_names_pos)
file_names_loc=os.listdir(file_path_loc)
#print("locations:",file_names_loc)
file_names_rot=os.listdir(file_path_rot)
#print("rotations:",file_names_rot)
for filename_pos,filename_loc,filename_rot in zip(file_names_pos,file_names_loc,file_names_rot):
    print(filename_pos)
    posses=read_and_print_file(file_path_pose,filename_pos,True)
    print("posses:",posses,"\n")
    print(filename_loc)
    locations=read_and_print_file(file_path_loc,filename_loc,True)
    print("locations:",locations,"\n")
    print(filename_rot)
    rotations=read_and_print_file(file_path_rot,filename_rot,False)
    print("rotations:",rotations,"\n")
    
    for pose in posses:
        print("list_pose:",list(pose))
        print("list_loc:",list(locations[0]))
        print("list_rot:",list(rotations[0]))
        coord=project_point_to_image(list(pose), list(locations[0]), list(rotations[0]))
        print("cords:",coord)
        coords.append(tuple(coord))
    write_coordinates_to_file(coords,output_file)
    
    break ## only first frame data just for testing purpose
    

# coordinates = read_coordinates_from_file(input_file)

# # Plot coordinates on the image
# plot_coordinates_on_image(coordinates, image_path)

