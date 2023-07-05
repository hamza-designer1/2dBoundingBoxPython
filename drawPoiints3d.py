import cv2
import ast

# Load the image
frameNum=1
file_path = 'VehicleBBox/bbox'+str(frameNum) +".txt"
img_path='custom_data/img_00000'+str(frameNum)  +'.jpeg'
image = cv2.imread(img_path)



# Read the file containing the list of tuples
with open(file_path, 'r') as file:
    lines = file.readlines()

# Process each line
for line in lines:
    # Convert the line to a list of tuples
    coordinates = ast.literal_eval(line)

    # Plot the 3D bounding box as a rectangle
    for i in range(4):
        cv2.line(image, coordinates[i], coordinates[(i + 1) % 4], color=(0, 0, 255), thickness=2)
        cv2.line(image, coordinates[i + 4], coordinates[((i + 1) % 4) + 4], color=(0, 0, 255), thickness=2)
        cv2.line(image, coordinates[i], coordinates[i + 4], color=(0, 0, 255), thickness=2)

# Display the image with plotted bounding boxes
cv2.imshow('Image with Bounding Boxes', image)
# Get the width and height of the image
height, width = image.shape[:2]
print("image width",width)
print("image height",height)
cv2.waitKey(0)
cv2.destroyAllWindows()
