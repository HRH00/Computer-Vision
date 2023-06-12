import cv2 as cv
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import package_901476.Data as data
import package_901476.Hog as hog
import scipy


# initial sift function, not used anymore 
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
                sublist_keypoints.append(keypoint)
                sublist_descriptors.append(descriptor)
                sublist_nums.append(x)
            
        x += 1
        keypoints.append(sublist_keypoints)
        descriptors.append(sublist_descriptors)
        nums.append(sublist_nums)
        print("\nLen of sublist_descriptors",len(sublist_descriptors))
        print("Len of sublist_keypoints",len(sublist_keypoints))
    print("\nDone - SIFT keypoint and descriptors arrays created\n")

    cv.destroyAllWindows()
    return keypoints, descriptors, nums


def doSift(MHI_array):
      
    descriptors = []
    flat_mhi = [] # flatten
    Sift = cv.SIFT_create()
    int_label = 0
        
    
    for lab in MHI_array:
        for image in lab:
            kpoint, descriptor = Sift.detectAndCompute(image, None)
            img_with_keypoints = cv.drawKeypoints(image, kpoint, None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            data.showImageForXms("Keypoints", img_with_keypoints, 5)
            data.showImageForXms("Frame",image,1)
            descriptors.append(descriptor)
            flat_mhi.append(image)
            int_label += 1
    
    
    #stack vertically   
    descriptors = np.vstack(descriptors)
    
    
    #convert ot float32 for kmeans
    float_descriptor = descriptors.astype(np.float32)
    
    clusters = 100
    
    print("True on NAN Error: ",np.any(np.isnan(float_descriptor))) # Return True if any NaN values)
    print("True on INF Error: ",np.any(np.isinf(float_descriptor))) # Return True if any inf values)
    voc, var = scipy.cluster.vq.kmeans(float_descriptor, clusters, 1)
    
    
    
    image_features = np.zeros((len(MHI_array), clusters), "float32")
    for i in range(len(MHI_array)):
        words, distance = scipy.cluster.vq.vq(float_descriptor, voc)
        for w in words:
            image_features[i][w] += 1
    
    num_of_ocs = np.sum((image_features > 0), axis=0) # vectorisation
    inverse_doc_Frequency = np.array(np.log((1.0*len(MHI_array)+1) / (1.0*num_of_ocs + 1)), 'float32')

    # Weight the image_features with idf
    image_features = image_features * inverse_doc_Frequency

    scaler = StandardScaler().fit(image_features)
    image_features = scaler.transform(image_features)
    
  # Create labels
    num_labels = []
    for i, lab in enumerate(image_features):
        subnum = [i]*len(lab)  # All descriptors from this image get the same label 'i'
        num_labels.extend(subnum)  # Using extend instead of append
    num_labels = np.array(num_labels)

    num_labels=np.array(num_labels).reshape(-1)
    features = np.array(image_features).reshape(-1,1)    # reshape to (600,)
    
    cv.destroyAllWindows()
    
    return features, num_labels
    
