# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 22:15:00 2020

@author: Yashoda
"""

import cv2
import numpy as np
import os


def extract_bv(image):
    arborizing = image
    
    b, green_arborizing, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced_green_arborizing = clahe.apply(green_arborizing)

    # applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(contrast_enhanced_green_arborizing, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (23, 23)), iterations=1)
    f4 = cv2.subtract(R3, contrast_enhanced_green_arborizing)
    f5 = clahe.apply(f4)
    
    cv2.imshow("After alternate sequential filtering", f4)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./BCC/arbMask/ASF.jpg", f4)
    cv2.imwrite("./BCC/arbMask/Clahe.jpg", f5)
    cv2.imshow("Clahe applied", f5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # removing very small contours through area parameter noise removal
    ret, f6 = cv2.threshold(f5, 10, 255, cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    contours, hierarchy = cv2.findContours(
        f6.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret, fin = cv2.threshold(im, 15, 255, cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    
    cv2.imshow("Area Parameter Noise Removal", newfin)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imwrite("./BCC/arbMask/areaparameter.jpg", newfin)
    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    # vessels and also in an interval of area

    arborizing_eroded = cv2.bitwise_not(newfin)
    xmask = np.ones(arborizing.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(
        arborizing_eroded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"
        else:
            shape = "veins"
        if(shape == "circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)
    
    cv2.imshow("blob mask", xmask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    finimage = cv2.bitwise_and(
        arborizing_eroded, arborizing_eroded, mask=xmask)
    blood_vessels = cv2.bitwise_not(finimage)
    kernel = np.ones((3, 3), np.uint8)
    blood_vessels = cv2.dilate(blood_vessels, kernel)
    blood_vessels = cv2.erode(blood_vessels, kernel)
    return blood_vessels


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


def ArborizingDetection(image):

    b_lower = [0, 0, 0]
    b_upper = [255, 255, 255]
    redBoundaries = [
        ([150, 93, 161], [170, 189, 202]),
        ([171, 127, 166], [180, 165, 166])
    ]
    kernel = np.ones((7, 7), np.uint8)
    
    cv2.imshow("original Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    hairRemoved = removeHair(image)
    cv2.imshow("Hair Removed Image", hairRemoved)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    bloodvessel = extract_bv(hairRemoved)
    if not os.path.exists('./BCC/arbMask'):
        os.makedirs('./BCC/arbMask')
    
    
    cv2.imshow("original Image", image)
    cv2.imshow("Binary mask level 1", bloodvessel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("./BCC/arbMask/arbmask.jpg", bloodvessel)

    bv = cv2.imread("./BCC/arbMask/arbmask.jpg")
   

    binarylower = np.array(b_lower, dtype="uint8")
    binaryupper = np.array(b_upper, dtype="uint8")
    mask = cv2.inRange(bv, binarylower, binaryupper)
    output = cv2.bitwise_or(image, bv, mask=mask)
    
    cv2.imshow("original Image", image)
    cv2.imshow("extracted vessels level 1", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # extracting bloodVessels again for higher accuracy

    bloodvesselLevel2 = extract_bv(output)
    
    cv2.imshow("Binary mask level 1", bloodvessel)
    cv2.imshow("Binary mask level 2", bloodvesselLevel2)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    cv2.imwrite("./BCC/arbMask/arbmask2.jpg", bloodvesselLevel2)

    bv2 = cv2.imread("./BCC/arbMask/arbmask2.jpg")
    maskLevel2 = cv2.inRange(bv2, binarylower, binaryupper)
    outputLevel2 = cv2.bitwise_or(output, bv2, mask=maskLevel2)
    
    cv2.imshow("original Image", image)
    cv2.imshow("extracted vessels level 2", outputLevel2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    redVesselMask = np.ones(image.shape)[:, :, 0].astype("uint8")*255

    for (lower, upper) in redBoundaries:

        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        red_mask = cv2.inRange(outputLevel2, lower, upper)
        redVesselMask = cv2.addWeighted(redVesselMask, 1, red_mask, 1, 0.0)

    result = cv2.bitwise_and(outputLevel2, outputLevel2, mask=redVesselMask)
    number_of_red_pix = np.sum(result != 255)

    if not os.path.exists('./BCC'):
        os.makedirs('./BCC')
    if not os.path.exists('./BCC/out'):
        os.makedirs('./BCC/out')
    cv2.imwrite('./BCC/out/arborizing.jpg', result)
    
    cv2.imshow("original Image", image)
    
    cv2.imshow("final Output", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    number_of_red_pix = np.sum(result != 255)
    width, height, _ = result.shape
    
    print("Area", number_of_red_pix)

    if(1000 < number_of_red_pix):
        arborizingVesselsPresent = "Detected"
    else:
        arborizingVesselsPresent = "Not Detected"

    return arborizingVesselsPresent

# arb
ArbPath = r"C:\Users\Yashoda\.spyder-py3\final Individual model\final - Copy\all\ISIC_0070853.jpg"
# ArbPath = r"C:\Users\Yashoda\.spyder-py3\final Individual model\final - Copy\all\ISIC_0068781.jpg"

# # not arb
# ArbPath = r"C:\Users\Yashoda\.spyder-py3\final Individual model\final - Copy\all\ISIC_0026117.jpg"

arb = cv2.imread(ArbPath)
ArbPresent = ArborizingDetection(arb)
print("Arborizing ", ArbPresent)


# arb ISIC_0068781
# not Arb ISIC_0026117