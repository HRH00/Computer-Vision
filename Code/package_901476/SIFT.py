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
    data.showImageForXms("Keypoints", mhi_with_keypoints, 1)
    data.showImageForXms("Frame",mhi,1)
    
    return keypoints, descriptors

def SIFTAnalysisOnMHI_Array(MHIArray):
    print("Sift Analysis on MHI Array")
    print("Generating SIFT KEYPOINTS\nThis will take some time...")
    keypoints = []
    descriptors = []
    numerical_labels = []
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
                sublist_nums.extend([x] * len(keypoint))
                numerical_labels.append(x)
        x += 1
        keypoints.append(sublist_keypoints)
        descriptors.append(sublist_descriptors)
        numerical_labels.extend(sublist_nums)
    print("\nDone - SIFT keypoint and descriptors arrays created\n")

    cv.destroyAllWindows()
    return keypoints, descriptors, numerical_labels



def compute_image_features(keypoints, descriptors, k):
    features = []
    print("computing image features")
    kmeans = KMeans(n_clusters=k,n_init=k)
    
    for i in range(len(keypoints)):
        label_features = []
        for image_index in range(len(keypoints[i])):
            sift_features = descriptors[i][image_index]
            sift_features = np.array(sift_features)
            sift_features = sift_features.reshape(-1, 1)
            kmeans.fit(sift_features)
            cluster_centers = kmeans.cluster_centers_
            label_features.append(cluster_centers)
        features.append(label_features)
        
    features = features[:1000]    
    return features

