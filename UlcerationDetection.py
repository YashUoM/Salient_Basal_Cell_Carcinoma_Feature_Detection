# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 13:01:21 2020

@author: Yashoda
"""
# Built-in imports

import numpy as np
import cv2
import os


def segmentationimage(image):

    if not os.path.exists('./BCC/bin'):
        os.makedirs('./BCC/bin')

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    cv2.imwrite('./BCC/bin/gray.jpg', cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    
    b, g, r = cv2.split(image)

    cv2.imwrite('./BCC/bin/green.jpg', g)

    cv2.imwrite('./BCC/bin/blue.jpg', b)
    cv2.imwrite('./BCC/bin/red.jpg', r)
    
    cv2.imshow("original", image)
    cv2.imshow("Green Channel", g)
    cv2.imshow("Blue Channel", b)
    cv2.imshow("Red Channel", r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    subracted = cv2.cvtColor((g-r), cv2.COLOR_GRAY2BGR)

    cv2.imwrite('./BCC/bin/sub.jpg', subracted)
    
    cv2.imshow("original", image)
    cv2.imshow("Sutracted", subracted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sub = cv2.imread('./BCC/bin/sub.jpg', 0)

    ret, thresh2 = cv2.threshold(sub, 170, 255, cv2.THRESH_BINARY_INV)

    cv2.imwrite('./BCC/bin/thresh2.jpg', thresh2)
    
    
    

    im_thresh_gray = cv2.bitwise_and(sub, thresh2)
    cv2.imwrite('./BCC/bin/thresh_gray.jpg', im_thresh_gray)
    
    cv2.imshow("original", image)
    cv2.imshow("threshold", thresh2)
    cv2.imshow("threshold 2", im_thresh_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    mask3 = cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR)  # 3 channel mask

    im_thresh_color = cv2.cvtColor(
        cv2.bitwise_and(img, mask3), cv2.COLOR_BGR2RGB)
    
    cv2.imshow("original", image)
    cv2.imshow("Before Color Filter", im_thresh_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
   # filtering out by red color
    lower = [1, 0, 20]
    upper = [60, 40, 200]

    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")
    mask = cv2.inRange(im_thresh_color, lower, upper)
    
    cv2.imshow("original", image)
    cv2.imshow("Before Color Filter", im_thresh_color)
    cv2.imshow("Color mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    output = cv2.bitwise_and(im_thresh_color, im_thresh_color, mask=mask)
    
    cv2.imshow("original", image)
    cv2.imshow("Before Color Filter", im_thresh_color)
    cv2.imshow("After Color Filter", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return output
    # Replace the contours with the white color same as the background


def UlcerationDetection(image):

    segmentedImage = segmentationimage(image)

    if not os.path.exists('./BCC'):
        os.makedirs('./BCC')
    if not os.path.exists('./BCC/out'):
        os.makedirs('./BCC/out')
    cv2.imwrite('./BCC/out/Ulceration.jpg', segmentedImage)

    # extracting numerical data
    number_of_red_pix = np.sum(segmentedImage != 0)
    width, height, _ = segmentedImage.shape
    
    print("Area", number_of_red_pix)
    if(50 < number_of_red_pix < 16000):
        ulcerPresent = "Detected"
    else:
        ulcerPresent = "Not Detected"

    return ulcerPresent


def removeHair(image):
    # Convert to grayscale
    img = image
    grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Create kernel & perform blackhat filtering
    kernel = cv2.getStructuringElement(1, (22, 22))
    blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)

    # Create contours & inpaint
    ret, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    result = cv2.inpaint(img, thresh, 1, cv2.INPAINT_TELEA)

    return result

 #ulcer
UlcerPath = r"C:\Users\Yashoda\.spyder-py3\final Individual model\final - Copy\all\ISIC_0024799.jpg"
# UlcerPath = r"C:\Users\Yashoda\.spyder-py3\final Individual model\final - Copy\all\ISIC_0032536.jpg"

# not ulcer
# UlcerPath = r"C:\Users\Yashoda\.spyder-py3\final Individual model\final - Copy\all\ISIC_0030687.jpg"

ulcer = cv2.imread(UlcerPath)
ulcerPresent = UlcerationDetection(ulcer)
   
print("Ulceration ", ulcerPresent)

