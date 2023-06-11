import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
import package_901476.Data as data

# Sift Functions 
def SIFTAnalysisOnMHI(mhi):
    sift = cv.SIFT_create()
    # Detect keypoints and compute descriptors
    mhi=cv.convertScaleAbs(mhi)
    keypoints, descriptors = sift.detectAndCompute(mhi, None)
    # Draw keypoints 
    mhi_with_keypoints = cv.drawKeypoints(mhi, keypoints, None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Display keypoints
    data.showImageForXms("Keypoints", mhi_with_keypoints, 5)
    data.showImageForXms("Frame",mhi,1)
    
    return keypoints, descriptors

def SIFTAnalysisOnMHI_Array(MHIArray):
    print("Sift Analysis on MHI Array")
    print("Generating SIFT KEYPOINTS\nThis will take some time...")
    keypoints = []
    descriptors = []
    nums=[]
    x = 0
    for label in MHIArray:
        sublist_keypoints = []
        sublist_descriptors = []
        sublist_nums = []
        for mhi in label:
            keypoint, descriptor = SIFTAnalysisOnMHI(mhi)
            if descriptor is not None:
                sublist_keypoints.extend(keypoint)
                sublist_descriptors.extend(descriptor)
                sublist_nums.extend([x]*len(descriptor))
            
        x += 1
        keypoints.append(sublist_keypoints)
        descriptors.append(sublist_descriptors)
        nums.append(sublist_nums)
        print("\nLen of sublist_descriptors",len(sublist_descriptors))
        print("Len of sublist_keypoints",len(sublist_keypoints))
    print("\nDone - SIFT keypoint and descriptors arrays created\n")

    cv.destroyAllWindows()
    return keypoints, descriptors, nums


