
import ast
import numpy  as np
import glob
import os
import shutil 
import sys

import random
import cv2
import time
import argparse
import re
# Specify the path to the text file
file_path = "testVehicleBBox/bbox1.txt"

VIEW_WIDTH =1920            #1920//2
VIEW_HEIGHT =1080               #1080//2
VIEW_FOV = 90

BB_COLOR = (248, 64, 24)
WBB_COLOR = (0, 0, 255)
vehicle_bbox_record = False
pedestrian_bbox_record = False
count = 0



CAM_CALIB = np.identity(3)


##############

main_dir='E:/ue5_2_project/Altroverse_LA_Cin/Altroverse_LA/Saved/data/'
dir_rgb = 'custom_data/'
dir_seg = 'SegmentationImage/'
dir_pbbox = 'PedestrianBBox/'
dir_vbbox = 'VehicleBBox/'
colorFolder=main_dir+r"carColors/"
if not os.path.exists(main_dir+dir_rgb):
    os.makedirs(main_dir+dir_rgb)
if not os.path.exists(main_dir+dir_seg):
    os.makedirs(main_dir+dir_seg)
if not os.path.exists(main_dir+dir_pbbox):
    os.makedirs(main_dir+dir_pbbox)
if not os.path.exists(main_dir+dir_vbbox):
    print(f"Creating this directory  {main_dir+dir_vbbox}  !!!!!!!!!!!!!!!!!!")
    os.makedirs(main_dir+dir_vbbox)
VBB_COLOR = (0, 0, 255)
WBB_COLOR = (255, 0, 0)

rgb_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
seg_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype="i")
area_info = np.zeros(shape=[VIEW_HEIGHT, VIEW_WIDTH, 3], dtype=np.uint8)
index_count = 0

# Brings Images and Bounding Box Information
def reading_data(index):
    global rgb_info, seg_info,main_dir
    v_data = []
    w_data = []
    k = 0
    w = 0


    rgb_img = cv2.imread(main_dir+"custom_data/"+"img_"+"%.6d"%index + '.jpeg', cv2.IMREAD_COLOR)
    # seg_img = cv2.imread('SegmentationImage/seg'+ str(index)+ '.png', cv2.IMREAD_COLOR)

   
    seg_img = cv2.imread(main_dir+"SegmentationImage/"+"img_"+"%.6d"%index + '.png', cv2.IMREAD_COLOR)

    if rgb_img is not None:
        print("RGB image read")
    else:
        print("RGB image not read !!!!!!!!!")

    if seg_img is not None:
        print("seg_img image read")
    else:
        print("seg_img image not read !!!!!!!!!")

    if str(rgb_img) != "None" and str(seg_img) != "None":
        # Vehicle
        with open(main_dir+'VehicleBBox/bbox'+ str(index)+".txt", 'r') as fin:
            v_bounding_box_rawdata = fin.read()

        v_bounding_box_data = re.findall(r"-?\d+", v_bounding_box_rawdata)
        v_line_length = len(v_bounding_box_data) / 16 
        #############################################################################################
        v_line_length=int(v_line_length)
        #print("v_bounding_box_data:",len(v_bounding_box_data),"TYPE:",type(len(v_bounding_box_data)))
        #print("v_line_length:",v_line_length,"TYPE:",type(v_line_length))
        #########################################################################
        v_bbox_data = [[0 for col in range(8)] for row in range(v_line_length)] 

        for i in range(int(len(v_bounding_box_data)/2)):
            j = i*2
            v_data.append(tuple((int(v_bounding_box_data[j]), int(v_bounding_box_data[j+1]))))

        for i in range(int(len(v_bounding_box_data)/16)):
            for j in range(8):
                v_bbox_data[i][j] = v_data[k]
                k += 1

        # Walker (Pedestrian)
####################################################################################################        
        # with open(main_dir+'PedestrianBBox/bbox'+ str(index), 'r') as w_fin:
        #     w_bounding_box_rawdata = w_fin.read()

        # w_bounding_box_data = re.findall(r"-?\d+", w_bounding_box_rawdata)
        # w_line_length = len(w_bounding_box_data) / 16 

        # w_bb_data = [[0 for col in range(8)] for row in range(int(w_line_length))] 

        # for i in range(int(len(w_bounding_box_data)/2)):
        #     j = i*2
        #     w_data.append(tuple((int(w_bounding_box_data[j]), int(w_bounding_box_data[j+1]))))

        # for i in range(int(len(w_bounding_box_data)/16)):
        #     for j in range(8):
        #         w_bb_data[i][j] = w_data[w]
        #         w += 1
###############################################################################################################
        origin_rgb_info = rgb_img
        rgb_info = rgb_img
        seg_info = seg_img
        #return v_bbox_data, v_line_length, w_bb_data, w_line_length 
        return v_bbox_data, v_line_length

    else:
        return False
# Converts 8 Vertices to 4 Vertices
def converting(bounding_boxes, line_length):
    points_array = []
    line_length=int(line_length)
    bb_4data = [[0 for col in range(4)] for row in range(line_length)]
    k = 0
    for i in range(line_length):
        points_array_x = []
        points_array_y = []      
        for j in range(8):
            points_array_x.append(bounding_boxes[i][j][0])
            points_array_y.append(bounding_boxes[i][j][1])

            max_x = max(points_array_x)
            min_x = min(points_array_x)
            max_y = max(points_array_y)
            min_y = min(points_array_y)           

        points_array.append(tuple((min_x, min_y)))
        points_array.append(tuple((max_x, min_y)))
        points_array.append(tuple((max_x, max_y)))
        points_array.append(tuple((min_x, max_y)))

    for i in range(line_length):
        for j in range(int(len(points_array)/line_length)):
            bb_4data[i][j] = points_array[k]
            k += 1  

    return bb_4data

def readColors(frameNum):
    global main_dir,colorFolder
    fileName="frame"+str(frameNum)
    colorList=[]
    file = open(colorFolder+fileName+".txt", "r")
    for line in file:
        # Convert the line from a string to a list object
        colorTuple = eval(line.strip())
        colorList.append(colorTuple)
        # for tuple in my_list:
        #     print(tuple)
        
        # Do something with the list
        print(colorTuple) 
    file.close()
    return colorList

# Gets Object's Bounding Box Area
def object_area(data,vehiclesColors):
    global area_info
    global seg_info
    global rgb_info
    global Vehicle_COLOR
    global Vehicles_COLORS
    Vehicles_COLORS=vehiclesColors
    carIndex=0
    area_info = np.zeros(shape=[VIEW_HEIGHT, VIEW_WIDTH, 3], dtype=np.uint8)

    print("total cars in the scene :",len(data))
    for vehicle_area in data:
        print("CAR INDEX !!!!!!!!!! ",carIndex)
        array_x = []
        array_y = []
        Vehicle_COLOR=Vehicles_COLORS[carIndex]
        carIndex+=1
        
        
        for i in range(4):
           array_x.append(vehicle_area[i][0])
        for j in range(4):
           array_y.append(vehicle_area[j][1])

        for i in range(4):
            if array_x[i] <= 0:
                array_x[i] = 1
            elif array_x[i] >= VIEW_WIDTH:
                array_x[i] = VIEW_WIDTH -1
        for j in range(4):
            if array_y[j] <= 0:
                array_y[j] = 1
            elif array_y[j] >= VIEW_HEIGHT:
                array_y[j] = VIEW_HEIGHT -1
       
        min_x = min(array_x) 
        max_x = max(array_x) 
        min_y = min(array_y) 
        max_y = max(array_y) 
        array = [min_x, max_x, min_y, max_y]
        #print(f"---------> area_info  {Vehicle_COLOR}")
        #sys.exit()
        print("check the color!!!!!!!!!!!!!!",filtering(array, Vehicle_COLOR))
        print("color to check !!!!!!!!!!!!!!",Vehicle_COLOR[0])
        #print("color in !!!!!!!!!!!!!!",seg_info[605, 605][0])
        if filtering(array, Vehicle_COLOR): 
            cv2.rectangle(area_info, (min_x, min_y), (max_x, max_y), Vehicle_COLOR, -1)
            cv2.rectangle(rgb_info, (min_x, min_y), (max_x, max_y), Vehicle_COLOR, 2)
            # Display the image
            cv2.imshow("Line", rgb_info)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Fits Bounding Box to the Object
def fitting_x(x1, x2, range_min, range_max, color):
    global seg_info
    state = False
    cali_point = 0
    if (x1 < x2):
        for search_point in range(x1, x2):
            for range_of_points in range(range_min, range_max):
                if seg_info[range_of_points, search_point][2] == color[0] and seg_info[range_of_points, search_point][1] == color[1] and seg_info[range_of_points, search_point][0] == color[2]:
                    print("*****--->Fitting x******** ")
                    print("value to comape ",seg_info[range_of_points, search_point][2])
                    print("vehicle color value ",color[0])
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    else:
        for search_point in range(x1, x2, -1):
            for range_of_points in range(range_min, range_max):
                if seg_info[range_of_points, search_point][2] == color[0] and seg_info[range_of_points, search_point][1] == color[1] and seg_info[range_of_points, search_point][0] == color[2]:
                    print("*****--->Fitting x******** ")
                    print("value to comape ",seg_info[range_of_points, search_point][2])
                    print("vehicle color value ",color[0])                    
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    return cali_point

def fitting_y(y1, y2, range_min, range_max, color):
    global seg_info
    state = False
    cali_point = 0
    if (y1 < y2):
        for search_point in range(y1, y2):
            for range_of_points in range(range_min, range_max):
                if seg_info[search_point, range_of_points][2] == color[0] and seg_info[search_point, range_of_points][1] == color[1] and seg_info[search_point, range_of_points][0] == color[2] :
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    else:
        for search_point in range(y1, y2, -1):
            for range_of_points in range(range_min, range_max):
                if seg_info[search_point, range_of_points][2] == color[0]:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    return cali_point

# Removes small objects that obstruct to learning
def small_objects_excluded(array, bb_min):
    diff_x = array[1]- array[0]
    diff_y = array[3] - array[2]
    if (diff_x > bb_min and diff_y > bb_min):
        return True
    return False

# Filters occluded objects
def post_occluded_objects_excluded(array, color):
    global seg_info
    top_left = seg_info[array[2]+1, array[0]+1][0]
    top_right = seg_info[array[2]+1, array[1]-1][0] 
    bottom_left = seg_info[array[3]-1, array[0]+1][0] 
    bottom_right = seg_info[array[3]-1, array[1]-1][0]
    if top_left == color[0] and top_right == color[0] and bottom_left == color[0] and bottom_right == color[0]:
        return False

    return True

def pre_occluded_objects_excluded(array, area_image, color):
    top_left = area_image[array[2]-1, array[0]-1][0]
    top_right = area_image[array[2], array[1]+1][0] 
    bottom_left = area_image[array[3]+1, array[1]-1][0] 
    bottom_right = area_image[array[3]+1, array[0]+1][0]
    if top_left == color[0] and top_right == color[0] and bottom_left == color[0] and bottom_right == color[0]:
        return False

    return True

# Filters objects not in the scene
def filtering(array, color):
    global seg_info
    # print("color value of segmentation to compare[-_-!]--->",)
    for y in range(array[2], array[3]):
        for x in range(array[0], array[1]):

            # print("color in !!!!!!!!!!!!!!",seg_info[y, x][2])
            if seg_info[y, x][2] == color[0] and seg_info[y, x][1] == color[1] and seg_info[y, x][0] == color[2]:   #seg_info[y, x][0]
                print("color value of segmentation to compare at index 2[-_-!]--->",seg_info[y, x][2])
                print("color value of segmentation to compare at index 1[-_-!]--->",seg_info[y, x][1])
                print("color value of segmentation to compare at index 0[-_-!]--->",seg_info[y, x][0])                
                print("color that is being compared at index 0 !",color[0])
                print("color that is being compared at index 1 !",color[1])                   
                print("color that is being compared at index 2 !",color[2])
                return True
    return False
# Processes Post-Processing
def processing(img, v_data, w_data, index,vehiclesColors):
    global seg_info, area_info
    global Vehicle_COLOR
    global Vehicles_COLORS
    global main_dir
    vehicle_class = 0
    walker_class = 1
    carIndex=0
    Vehicles_COLORS=vehiclesColors

    object_area(v_data,vehiclesColors)
    f = open(main_dir+"custom_data/image"+str(index) + ".txt", 'w')

    # Vehicle
    for v_bbox in v_data:
        #print("car number-->",carIndex)
        Vehicle_COLOR=Vehicles_COLORS[carIndex]
        carIndex+=1
        array_x = []
        array_y = []
        for i in range(4):
           array_x.append(v_bbox[i][0])
           #print(v_bbox[i][0])
        for j in range(4):
           array_y.append(v_bbox[j][1])
           #print(v_bbox[j][1])

        for i in range(4):
            if array_x[i] <= 0:
                array_x[i] = 1
            elif array_x[i] >= VIEW_WIDTH - 1:
                array_x[i] = VIEW_WIDTH - 2
        for j in range(4):
            if array_y[j] <= 0:
                array_y[j] = 1
            elif array_y[j] >= VIEW_HEIGHT - 1:
                array_y[j] = VIEW_HEIGHT - 2
       
        min_x = min(array_x) 
        max_x = max(array_x) 
        min_y = min(array_y) 
        max_y = max(array_y) 
        v_bb_array = [min_x, max_x, min_y, max_y]
        center_x = (min_x + max_x)//2
        center_y = (min_y + max_y)//2
        print("pre_occluded_objects_excluded-------------->",pre_occluded_objects_excluded(v_bb_array, area_info, Vehicle_COLOR))
        print("filtering-------------->",filtering(v_bb_array, Vehicle_COLOR))
        
        if filtering(v_bb_array, Vehicle_COLOR) and pre_occluded_objects_excluded(v_bb_array, area_info, Vehicle_COLOR): 
            cali_min_x = fitting_x(min_x, max_x, min_y, max_y, Vehicle_COLOR)
            cali_max_x = fitting_x(max_x, min_x, min_y, max_y, Vehicle_COLOR)
            cali_min_y = fitting_y(min_y, max_y, min_x, max_x, Vehicle_COLOR)
            cali_max_y = fitting_y(max_y, min_y, min_x, max_x, Vehicle_COLOR)
            v_cali_array = [cali_min_x, cali_max_x, cali_min_y, cali_max_y]
            print("small_objects_excluded-------------->",small_objects_excluded(v_cali_array, 10))
            print("post_occluded_objects_excluded-------------->",post_occluded_objects_excluded(v_cali_array, Vehicle_COLOR))
            if small_objects_excluded(v_cali_array, 10) and post_occluded_objects_excluded(v_cali_array, Vehicle_COLOR):
                darknet_x = float((cali_min_x + cali_max_x) // 2) / float(VIEW_WIDTH)
                darknet_y = float((cali_min_y + cali_max_y) // 2) / float(VIEW_HEIGHT)
                darknet_width = float(cali_max_x - cali_min_x) / float(VIEW_WIDTH)
                darknet_height= float(cali_max_y - cali_min_y) / float(VIEW_HEIGHT)

                f.write(str(vehicle_class) + ' ' + str("%0.6f" % darknet_x) + ' ' + str("%0.6f" % darknet_y) + ' ' + 
                str("%0.6f" % darknet_width) + ' ' + str("%0.6f" % darknet_height) + "\n")

                cv2.line(img, (cali_min_x, cali_min_y), (cali_max_x, cali_min_y), VBB_COLOR, 2)
                cv2.line(img, (cali_max_x, cali_min_y), (cali_max_x, cali_max_y), VBB_COLOR, 2)
                cv2.line(img, (cali_max_x, cali_max_y), (cali_min_x, cali_max_y), VBB_COLOR, 2)
                cv2.line(img, (cali_min_x, cali_max_y), (cali_min_x, cali_min_y), VBB_COLOR, 2)

    # # Walker (Pedestrian)
    # object_area(w_data)

    # for wbbox in w_data:
    #     array_x = []
    #     array_y = []

    #     for i in range(4):
    #        array_x.append(wbbox[i][0])
    #     for j in range(4):
    #        array_y.append(wbbox[j][1])

    #     for i in range(4):
    #         if array_x[i] <= 0:
    #             array_x[i] = 1
    #         elif array_x[i] >= VIEW_WIDTH - 1:
    #             array_x[i] = VIEW_WIDTH - 2
    #     for j in range(4):
    #         if array_y[j] <= 0:
    #             array_y[j] = 1
    #         elif array_y[j] >= VIEW_HEIGHT - 1:
    #             array_y[j] = VIEW_HEIGHT - 2
       
    #     min_x = min(array_x) 
    #     max_x = max(array_x) 
    #     min_y = min(array_y) 
    #     max_y = max(array_y)
    #     w_bb_array = [min_x, max_x, min_y, max_y]
    #     if filtering(w_bb_array, Walker_COLOR) and pre_occluded_objects_excluded(w_bb_array, area_info, Walker_COLOR): 
    #         cali_min_x = fitting_x(min_x, max_x, min_y, max_y, Walker_COLOR)
    #         cali_max_x = fitting_x(max_x, min_x, min_y, max_y, Walker_COLOR)
    #         cali_min_y = fitting_y(min_y, max_y, min_x, max_x, Walker_COLOR)
    #         cali_max_y = fitting_y(max_y, min_y, min_x, max_x, Walker_COLOR)
    #         w_cali_array = [cali_min_x, cali_max_x, cali_min_y, cali_max_y]

    #         if small_objects_excluded(w_cali_array, 7) and post_occluded_objects_excluded(w_cali_array, Walker_COLOR):
    #             darknet_x = float((cali_min_x + cali_max_x) // 2) / float(VIEW_WIDTH)
    #             darknet_y = float((cali_min_y + cali_max_y) // 2) / float(VIEW_HEIGHT)
    #             darknet_width = float(cali_max_x - cali_min_x) / float(VIEW_WIDTH)
    #             darknet_height= float(cali_max_y - cali_min_y) / float(VIEW_HEIGHT)

    #             f.write(str(walker_class) + ' ' + str("%0.6f" % darknet_x) + ' ' + str("%0.6f" % darknet_y) + ' ' + 
    #             str("%0.6f" % darknet_width) + ' ' + str("%0.6f" % darknet_height) + "\n")

    #             cv2.line(img, (cali_min_x, cali_min_y), (cali_max_x, cali_min_y), WBB_COLOR, 2)
    #             cv2.line(img, (cali_max_x, cali_min_y), (cali_max_x, cali_max_y), WBB_COLOR, 2)
    #             cv2.line(img, (cali_max_x, cali_max_y), (cali_min_x, cali_max_y), WBB_COLOR, 2)
    #             cv2.line(img, (cali_min_x, cali_max_y), (cali_min_x, cali_min_y), WBB_COLOR, 2)

    f.close()
    #cv2.imwrite(main_dir+'draw_bounding_box/image'+str(index)+'.png', img)            # write Cordinates on image !!!!!!!!!!!!!!!!
def generate():
    global rgb_info
    global index_count
    global dataEA
    global main_dir
    dataEA = len(os.walk(main_dir+'VehicleBBox/').__next__()[2])
    print("dataEA-------------->",dataEA)
    #print("Different COlors------!---!--->",vehiclesColors) #implemented a read function!!!!!!!!!!!
    #train = open("my_data/train.txt", 'w')

    for i in range(dataEA + 1):
        v_four_points=[]
        w_four_points=[]
        if reading_data(i) != False:
            print("Generate file running ......cordinate file read correctly !!!!!")
            v_four_points = converting(reading_data(i)[0], reading_data(i)[1])
            #w_four_points = converting(reading_data(i)[2], reading_data(i)[3])
            print("four points----------------->",v_four_points,"\nshape:",len(v_four_points))
            vehiclesColors=readColors(i)
            print("vehiclesColors----------------->",vehiclesColors,"\nshape:",len(vehiclesColors))
            processing(rgb_info, v_four_points, w_four_points, i,vehiclesColors)               # if u want to draw 2dBBox on new image !!!
            #train.write(str('custom_data/image'+str(i) + '.png') + "\n")
            index_count = index_count + 1
            print("Correct file number",i)
        else:
            print("Generate file running ......cordinate file could not be read  !!!!!!!!!!!!")
            print("InCorrect file number",i)
    #train.close()
    print(index_count)
    return True

################################################################### EXTRACT.py##########################################################################################################
def getData():
    # Read the data from the text file
    with open(file_path, "r") as file:
        file_data = file.read()

    # Remove unnecessary characters and split the lines
    lines = file_data.splitlines()



    VehiclesBoxLocations=ast.literal_eval(lines[0])
    VehiclesBoxRotations=ast.literal_eval(lines[1])
    agentLoc=ast.literal_eval(lines[2])
    agentRot=ast.literal_eval(lines[3])
    Extends=ast.literal_eval(lines[4])
    VehiclesLocations=ast.literal_eval(lines[5])
    print("VehiclesBoxLocations-->",VehiclesBoxLocations)
    print("VehiclesBoxRotations-->",VehiclesBoxRotations)
    print("agentLoc-->",agentLoc)
    print("agentRot-->",agentRot)
    print("Extends-->",Extends)
    print("VehiclesLocations-->",VehiclesLocations)
    return VehiclesBoxLocations ,VehiclesBoxRotations,agentLoc,agentRot,Extends,VehiclesLocations
def setup_cam_calib():

        
        CAM_CALIB[0, 2] = VIEW_WIDTH / 2.0
        CAM_CALIB[1, 2] = VIEW_HEIGHT / 2.0
        CAM_CALIB[0, 0] = CAM_CALIB[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))

def get_bounding_boxes(vehiclesBoxLocations,vehiclesBoxRotations,sensorLocation,sensorRotation,vehiclesBoxExtends,vehiclesLocations):
    """
    Creates 3D bounding boxes based on carla vehicle list and camera.
    """
    bounding_boxes=[]
    for i in range(len(vehiclesBoxLocations)):
        vehicleBoxLocation=vehiclesBoxLocations[i]
        vehicleBoxRotation=vehiclesBoxRotations[i]
        sensorLocation=sensorLocation
        sensorRotation=sensorRotation
        vehicleBoxExtend=vehiclesBoxExtends[i]
        vehicleLocation=vehiclesLocations[i]
        bounding_boxes.append(get_bounding_box(vehicleBoxLocation,vehicleBoxRotation,sensorLocation,sensorRotation,vehicleBoxExtend,vehicleLocation))
    #bounding_boxes = [VehicleBoundingBoxes.get_bounding_box(vehicleBoxLocation, sensorLocation,vehiclesBoxExtends) for vehicleBoxLocation in vehiclesBoxLocations]
    # filter objects behind camera
    bad_boxes = [i for i, bb in enumerate(bounding_boxes) if not all(bb[:, 2] > 0)]
    #bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
    print("Index of cars behind the camera++++++++++---->",bad_boxes)
    print("Clean Box +++---++++---->",bounding_boxes)
    return bounding_boxes ,bad_boxes

def get_bounding_box(vehicleBoxLocation,vehicleBoxRotation,sensorLocation,sensorRotation,vehicleBoxExtend,vehicleLocation):
    """
    Returns 3D bounding box for a vehicle based on camera view.
    """

    bb_cords = create_bb_points(vehicleBoxExtend)
    # print("<--------------------------bb_cords-------------------------->")
    # print("bb_cords shape",np.shape(bb_cords))
    # print(bb_cords)
    # print("<--------------------------bb_cords-------------------------->")
    cords_x_y_z = vehicle_to_sensor(bb_cords, vehicleBoxLocation,vehicleBoxRotation, sensorLocation,sensorRotation,vehicleLocation)[:3, :]
    cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
    bbox = np.transpose(np.dot(CAM_CALIB, cords_y_minus_z_x))
    camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)



    # print("<-------------------------------camera_bbox...............................>")
    # print("shape of cords",np.shape(camera_bbox))
    # print(camera_bbox)
    # print("<----------------------------------------------...............................>")
    return camera_bbox
def create_bb_points(vehicleBoxExtend):
    """
    Returns 3D bounding box for a vehicle.
    """
    # extent[0]=extent.x
    # extent[1]=extent.y
    # extent[2]=extent.z
    cords = np.zeros((8, 4))
    extent = vehicleBoxExtend
    # cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
    # cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
    # cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
    # cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
    # cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
    # cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
    # cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
    # cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])

    cords[0, :] = np.array([extent[0], extent[1], -extent[2], 1])
    cords[1, :] = np.array([-extent[0], extent[1], -extent[2], 1])
    cords[2, :] = np.array([-extent[0], -extent[1], -extent[2], 1])
    cords[3, :] = np.array([extent[0], -extent[1], -extent[2], 1])
    cords[4, :] = np.array([extent[0], extent[1], extent[2], 1])
    cords[5, :] = np.array([-extent[0], extent[1], extent[2], 1])
    cords[6, :] = np.array([-extent[0], -extent[1], extent[2], 1])
    cords[7, :] = np.array([extent[0], -extent[1], extent[2], 1])
    ##print("CORDS---------->",cords)
    return cords

def vehicle_to_sensor(cords, vehicleBoxLocation,vehicleBoxRotation, sensorLocation,sensorRotation,vehicleLocation):
    """
    Transforms coordinates of a vehicle bounding box to sensor.
    """

    world_cord = vehicle_to_world(cords, vehicleBoxLocation,vehicleBoxRotation,vehicleLocation)
    sensor_cord = world_to_sensor(world_cord, sensorLocation,sensorRotation)
    #print("sensor_cord",sensor_cord)
    return sensor_cord

def vehicle_to_world(cords, vehicleBoxLocation,vehicleBoxRotation,vehicleLocation):
    """
    Transforms coordinates of a vehicle bounding box to world.
    """
    
    bb_transform = (vehicleBoxLocation,(0,0,0))
    vehicleTransform = (vehicleLocation,vehicleBoxRotation)
    bb_vehicle_matrix = get_matrix(bb_transform)
    vehicle_world_matrix = get_matrix(vehicleTransform)
    
    # bb_transform = carla.Transform(vehicle.bounding_box.location)
    # bb_vehicle_matrix = VehicleBoundingBoxes.get_matrix(bb_transform)
    # vehicle_world_matrix = VehicleBoundingBoxes.get_matrix(vehicle.get_transform())

    bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
    # print("<-------------------------------bb_world_matrix...............................>")
    # print("bb_world_matrix",bb_world_matrix)
    # print("<----------------------------------------------...............................>")
    
    world_cords = np.dot(bb_world_matrix, np.transpose(cords))
    
    # print("<-------------------------------world_cords...............................>")
    # print("shape of cords",np.shape(world_cords))
    # print(world_cords)
    # print("<----------------------------------------------...............................>")

    # print("<-------------------bb_vehicle_matrix------------------->")
    # print(bb_vehicle_matrix)
    # print("<------------------------------------------------->")
    return world_cords

def world_to_sensor(cords, sensorLocation,sensorRotation):
    """
    Transforms world coordinates to sensor.
    """


    sensorTransform=(sensorLocation,sensorRotation)
    #sensor_world_matrix = VehicleBoundingBoxes.get_matrix(sensor.get_transform())
    sensor_world_matrix = get_matrix(sensorTransform)
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    sensor_cords = np.dot(world_sensor_matrix, cords)


    # print("<-------------------------------cords...............................>")
    # print("shape of cords",np.shape(sensor_cords))
    # print(sensor_cords)
    # print("<----------------------------------------------...............................>")
    return sensor_cords


def get_matrix(transform):
    """
    Creates matrix from carla transform.
    """

    rotation = transform[1]
    location = transform[0]

    rotationYaw=rotation[2]     # rot z
    rotationRoll=rotation[0]    # rot x
    rotationPitch=rotation[1]   # rot y


    locationX=location[0]
    locationY=location[1]
    locationZ=location[2]

    c_y = np.cos(np.radians(rotationYaw))
    s_y = np.sin(np.radians(rotationYaw))
    c_r = np.cos(np.radians(rotationRoll))
    s_r = np.sin(np.radians(rotationRoll))
    c_p = np.cos(np.radians(rotationPitch))
    s_p = np.sin(np.radians(rotationPitch))
    # print("<-------------------bb_vehicle_matrix------------------->")
    # print("rotationYaw in deg",rotationYaw)
    # print("rotationRoll in deg",rotationRoll)
    # print("rotationPitch in deg",rotationPitch)
    # print("rotationYaw in radians",np.radians(rotationYaw))
    # print("rotationRoll in radians",np.radians(rotationRoll))
    # print("rotationPitch in radians",np.radians(rotationPitch))
    # print("locationX",locationX)
    # print("locationY",locationY)
    # print("locationZ",locationZ)

    # print("c_y",c_y)
    # print("s_y",s_y)
    # print("c_r",c_r)
    # print("s_r",s_r)
    # print("c_p",c_p)
    # print("s_p",s_p)
    # print("<------------------------------------------------->")
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

def draw_bounding_boxes(bounding_boxes,frameNum):
    """
    Draws bounding boxes on pygame display.
    """
    global vehicle_bbox_record
    global count
    count=frameNum
    vehicle_bbox_record=True
    

    # bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
    # bb_surface.set_colorkey((0, 0, 0))

    if vehicle_bbox_record == True:
        f = open("testVehicleBBox/bbox"+str(count), 'w')
        #print("VehicleBoundingBox")
    for bbox in bounding_boxes:
        points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
        

        if vehicle_bbox_record == True:
            f.write(str(points)+"\n")
    
    if vehicle_bbox_record == True:
        f.close()
        vehicle_bbox_record = False
def run(usrFrame,frameNum,vehiclesBoxLocations,vehiclesBoxRotations,sensorLocation,sensorRotation,vehiclesBoxExtends,vehiclesLocations):
    
    global vehicle_bbox_record
    global count
    vehicle_bbox_record=True
    count=frameNum
    print("frame number:",count)
    print("User Frame:",usrFrame)

    
    setup_cam_calib()
    print(CAM_CALIB)
    bounding_boxes=[]
    bounding_boxes ,badBoundingBoxesIndex = get_bounding_boxes(vehiclesBoxLocations,vehiclesBoxRotations,sensorLocation,sensorRotation,vehiclesBoxExtends,vehiclesLocations)
    print("<-------------------------------bounding_boxes...............................>")
    print("shape of bounding_boxes",np.shape(bounding_boxes))
    print(bounding_boxes)
    print("<----------------------------------------------...............................>")
    # pedestrian_bounding_boxes = PedestrianBoundingBoxes.get_bounding_boxes(pedestrians, self.camera)

    
    draw_bounding_boxes(bounding_boxes,frameNum)

    #return badBoundingBoxesIndex


generate()
# VehiclesBoxLocations ,VehiclesBoxRotations,agentLoc,agentRot,Extends,VehiclesLocations=getData()
# run(1,1,VehiclesBoxLocations,VehiclesBoxRotations,agentLoc,agentRot,Extends,VehiclesLocations)