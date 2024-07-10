# # list1=[1,2,3]
# # list2=[4,5]
# # list3=[7,8,9]

# # for i, j, k in zip(list1, list2, list3):
# #     print(i)
# #     print(j)
# #     print(k)


# list1=[]
# a=[1,2,3]
# list1.append(tuple(a))
# print(list1)

# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg 


# # Example usage:
# coordinates = [(0, 0), (300, 400), (500, 600)]  # Replace with your actual coordinates
# image_path = r"E:\python_code\2dBoundingBoxPython\Data\RGB\img_2.jpeg"  # Example image path where coordinates will be plotted

# def plot_coordinates_on_image(coordinates, image_path):
#     # Load the image
#     img = plt.imread(image_path)
    
#     # Plot the image
#     plt.imshow(img)
    
#     # Plot the coordinates
#     x_coords, y_coords = zip(*coordinates)
#     plt.scatter(x_coords, y_coords, color='red', marker='x')
    
#     # # Invert the y-axis to match the image coordinate system
#     # plt.gca().invert_yaxis()
    
#     # Show the plot
#     plt.show()
    
# plot_coordinates_on_image(coordinates, image_path)

a=(10,20,30)
divisor=10
b = tuple(element / divisor for element in a)
print(b)