# import cv2 to capture videofeed
import cv2
import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3 , 640)
camera.set(4 , 480)

# loading the mountain image
mountain = cv2.imread('C:/Users/shlok/OneDrive/Desktop/Coding Class/Python/Projects/Anywhere Booth/mountain.png')

# resizing the mountain image as 640 X 480
mountain = cv2.resize(mountain, (640,480)) 
mountain = mountain.astype(np.uint8)

while True:

    # read a frame from the attached camera
    status , frame = camera.read()

    # if we got the frame successfully
    if status:

        # flip it
        frame = np.flip(frame , axis=1)

        # converting the image to RGB for easy processing
        hsv = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)

        # creating thresholds
        #using colors for my wall, you can use any
        lower_bound = np.array([17, 100, 100])
        upper_bound = np.array([37, 255, 255])

        # thresholding image
        mask1 = cv2.inRange(hsv, lower_bound, upper_bound)
        # inverting the mask

        #same here, use any colors you want
        lower_bound = np.array([187, 100, 100])
        upper_bound = np.array([207, 255, 255])
        mask2 = cv2.inRange(hsv, lower_bound, upper_bound)

        mask1 = mask1+mask2
        # bitwise and operation to extract foreground / person
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))
        # final image
        mask2 = cv2.bitwise_not(mask1)

        frame = frame.astype(np.uint8)
        mountain = mountain.astype(np.uint8)
        mask1 = mask1.astype(np.uint8)

        res_1 = cv2.bitwise_and(frame, frame, mask=mask2)
        res2 = cv2.bitwise_and(mountain, mountain, mask=mask1)

        # show it
        final_output = cv2.addWeighted(res_1, 1, res2, 1, 0)
        cv2.imshow('Anywhere Photo Booth!' , final_output)

        # wait of 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code  ==  32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()
