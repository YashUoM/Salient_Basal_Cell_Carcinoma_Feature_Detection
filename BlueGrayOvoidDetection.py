# -*- coding: utf-8 -*-
"""
Created on Wed May 12 22:00:56 2021

@author: Yashoda
"""

import numpy as np
import cv2
import os


def binarySegmentation(image):
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 15

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.01

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)
        
    kernel = np.ones((7, 7), np.uint8)
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh2 = cv2.threshold(image1, 100, 255, cv2.THRESH_BINARY)
    
    cv2.imshow("original", image)
    cv2.imshow("Binary image", thresh2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    closing = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    closing_binary_keypoints = detector.detect(closing)
    
    
    cv2.imshow("Binary image", thresh2)
    cv2.imshow("After Closing", closing)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    closing_binary_im_with_keypoints = cv2.drawKeypoints(closing,
                                                         closing_binary_keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow("original", image)
    cv2.imshow("Blob Mask", closing_binary_im_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return closing_binary_im_with_keypoints


def BlueGrayOvoidDetection(image):
    # binary boundary
    boundaries = [
        ([0, 0, 0], [255, 255, 255]),
    ]

    # blue boundary
    lower = [115, 45, 55]
    upper = [165, 140, 120]

    bluelower = np.array(lower, dtype="uint8")
    blueupper = np.array(upper, dtype="uint8")

    binary = binarySegmentation(image)
    if not os.path.exists('./BCC/ovoidMask'):
        os.makedirs('./BCC/ovoidMask')

    cv2.imwrite("./BCC/ovoidMask/ovoidmask.jpg", binary)

    image2 = cv2.imread("./BCC/ovoidMask/ovoidmask.jpg")

    for (lower, upper) in boundaries:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(image2, lower, upper)

        output = cv2.bitwise_or(image, image2, mask=mask)
        
        cv2.imshow("original", image)
        cv2.imshow("Blob Extraction", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if not os.path.exists('./BCC'):
            os.makedirs('./BCC')
        if not os.path.exists('./BCC/out'):
            os.makedirs('./BCC/out')

        hsvOutput = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
        
        cv2.imwrite("./BCC/ovoidMask/hsvout.jpg", hsvOutput)
        mask = cv2.inRange(hsvOutput, bluelower, blueupper)
        blueOutput = cv2.bitwise_and(hsvOutput, hsvOutput, mask=mask)
        cv2.imwrite("./BCC/ovoidMask/ovoidmask.jpg", mask)

        output = cv2.bitwise_or(image, image2, mask=mask)
        
        cv2.imshow("Blob Mask",binary)
        cv2.imshow("Blue Gray Ovoid Mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        rgbOut = cv2.cvtColor(cv2.cvtColor(
            blueOutput, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)
        cv2.imwrite('./BCC/out/BlueGrayOvoid.jpg', rgbOut)
        
        cv2.imshow("original", image)
        cv2.imshow("detected Blue Gray Ovoid", rgbOut)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        number_of_blue_pix = np.sum(rgbOut != 0)
        width, height, _ = rgbOut.shape
        
        print("Area", number_of_blue_pix)
        if(200 < number_of_blue_pix < 27000):
            blueGayOvoidPresence = "Detected"
        else:
            blueGayOvoidPresence = "Not Detected"

        return blueGayOvoidPresence

# blue gay
# BlueGrayPath = r"C:\Users\Yashoda\.spyder-py3\final Individual model\final - Copy\all\ISIC_0026337.jpg"
BlueGrayPath = r"C:\Users\Yashoda\.spyder-py3\final Individual model\final - Copy\all\ISIC_0026845.jpg"

# # NBG
# BlueGrayPath = r"C:\Users\Yashoda\.spyder-py3\final Individual model\final - Copy\all\ISIC_0024448.jpg"


BlueGray = cv2.imread(BlueGrayPath)
    
BlueGrayPresent = BlueGrayOvoidDetection(BlueGray)

print("Blue Gray ", BlueGrayPresent)




#BG ISIC_0026845
# not bg ISIC_0028155