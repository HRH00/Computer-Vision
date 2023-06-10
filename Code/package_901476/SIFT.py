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
    x=0
    for label in MHIArray:
        sublist_keypoints = [] 
        sublist_descriptors = [] 
        sublist_nums = [] 
        for mhi in label:
            numerical_labels.append(x)
            keypoint, descriptor = SIFTAnalysisOnMHI(mhi)
            if descriptor is not None:
                sublist_keypoints.append(keypoint)
                sublist_descriptors.append(descriptor)
                sublist_nums.append(x)
        x+=1
        keypoints.append(sublist_keypoints)
        numerical_labels.append(sublist_nums)
        descriptors.append(sublist_descriptors)
    print("\nDone - SIFT keypoint and descriptors arrays created\n")
    dataTuple = (keypoints, descriptors, numerical_labels)
    
    cv.destroyAllWindows()
    return(dataTuple)

def compute_image_features(keypoints, descriptors, k):
    features = []
    kmeans = KMeans(n_clusters=k)
    descriptors = [descriptor for sublist in descriptors if sublist is not None for descriptor in sublist]
    descriptors_Vstacked = np.vstack(descriptors)
    kmeans.fit(descriptors_Vstacked)
    
    for image_keypoints, image_descriptors in zip(keypoints, descriptors):
        if image_descriptors is None:
            features.append(np.zeros(k))
            continue
        descriptors = [descriptor for descriptor in image_descriptors if descriptor is not None]
        stacked_image_descriptors = np.vstack(descriptors)
        if len(stacked_image_descriptors) == 0:
            features.append(np.zeros(k))
            continue
        predicted_clusters = kmeans.predict(stacked_image_descriptors)
        histogram, _ = np.histogram(predicted_clusters, bins=range(k+1))
        normalized_histogram = histogram / len(image_keypoints)
        features.append(normalized_histogram)
    
    return np.array(features)


#def compute_image_features(keypoints, k):
#    image_features = []
#    kmeans = KMeans(n_clusters=k)
#    kmeans.fit(descriptors)
#    
#    for image_descriptors in descriptors:
#        if image_descriptors is None:
#            image_features.append(np.zeros(k))
#            continue
#        image_descriptors = [descriptor for sublist in image_descriptors for descriptor in sublist]
#        if len(image_descriptors) == 0:
#            image_features.append(np.zeros(k))
#            continue
#        image_descriptors = np.array(image_descriptors)
#        predicted_clusters = kmeans.predict(image_descriptors)
#        histogram, _ = np.histogram(predicted_clusters, bins=range(k+1))
#        normalized_histogram = histogram / len(image_descriptors)
#        image_features.append(normalized_histogram)
#    return np.array(image_features)
