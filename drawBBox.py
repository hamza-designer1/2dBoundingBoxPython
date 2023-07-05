
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




import cv2
frameNum=0
file_path = 'custom_data/image'+str(frameNum)+".txt"
#file_path = 'image0.txt'  # Replace with the actual file path
def process_darknet_boxes(file_path):
    VIEW_WIDTH = 1920  # Replace with the desired VIEW_WIDTH value
    VIEW_HEIGHT = 1080  # Replace with the desired VIEW_HEIGHT value

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split(' ')
        if len(parts) != 5:
            print('Invalid box data:', line)
            continue

        walker_class = int(parts[0])
        darknet_x = float(parts[1])
        darknet_y = float(parts[2])
        darknet_width = float(parts[3])
        darknet_height = float(parts[4])

        cali_min_x = int(darknet_x * VIEW_WIDTH - darknet_width * VIEW_WIDTH / 2)
        cali_min_y = int(darknet_y * VIEW_HEIGHT - darknet_height * VIEW_HEIGHT / 2)
        cali_max_x = int(darknet_x * VIEW_WIDTH + darknet_width * VIEW_WIDTH / 2)
        cali_max_y = int(darknet_y * VIEW_HEIGHT + darknet_height * VIEW_HEIGHT / 2)

        print('cali_min_x:', cali_min_x)
        print('cali_min_y:', cali_min_y)
        print('cali_max_x:', cali_max_x)
        print('cali_max_y:', cali_max_y)

        # Perform the desired operations with the cali_min_x, cali_min_y, cali_max_x, cali_max_y values
        imgPath='custom_data/img_00000'+str(frameNum)  +'.jpeg'
        img = cv2.imread(imgPath)  # Replace with the path to your image
        cv2.line(img, (cali_min_x, cali_min_y), (cali_max_x, cali_min_y), (0, 255, 0), 2)
        cv2.line(img, (cali_max_x, cali_min_y), (cali_max_x, cali_max_y), (0, 255, 0), 2)
        cv2.line(img, (cali_max_x, cali_max_y), (cali_min_x, cali_max_y), (0, 255, 0), 2)
        cv2.line(img, (cali_min_x, cali_max_y), (cali_min_x, cali_min_y), (0, 255, 0), 2)

        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

process_darknet_boxes(file_path)
