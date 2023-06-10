import os
import pickle
import cv2 as cv
import numpy as np
from package_901476 import Data as data

##You need to set the following paths to the location of the data on your machine
dir_path = os.path.dirname(os.path.realpath(__file__))
PATH_TO_DATA = os.path.join(dir_path,"Datastore","Supplied")

PATH_TO_MHI_STORAGE = os.path.join("q") #it ot in use
FILE_EXTENTION = ".avi" 

# Sift Functions 
def SIFTAnalysisOnMHI(mhi):
    sift = cv.SIFT_create()
    # Detect keypoints and compute descriptors
    mhi=cv.convertScaleAbs(mhi)
    keypoints, descriptors = sift.detectAndCompute(mhi, None)
    # Draw keypoints 
    mhi_with_keypoints = cv.drawKeypoints(mhi, keypoints, None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Display keypoints
    showImageForXms("Keypoints", mhi_with_keypoints, 1)
    return keypoints, descriptors, mhi_with_keypoints

def SIFTAnalysisOnMHI_Array(MHIArray):
    print("Sift Analysis on MHI Array")
    keypoints = [] 
    descriptors = [] 
    keypoints_images = [] 

    for label in MHIArray:
        sublist_keypoints = [] 
        sublist_descriptors = [] 
        sublist_keypoints_images = [] 
        for mhi in label:
            keypoint, descriptor, mhi_keypoints_image = SIFTAnalysisOnMHI(mhi)
            if descriptor is not None:
                sublist_keypoints.append(keypoint)
                sublist_descriptors.append(descriptor)
                sublist_keypoints_images.append(mhi_keypoints_image)

        keypoints.append(sublist_keypoints)
        descriptors.append(sublist_descriptors)
        keypoints_images.append(sublist_keypoints_images)
    print("Done\n")
    cv.destroyAllWindows()
    return(keypoints, descriptors, keypoints_images)

def compute_image_features(kmeans, descriptors, k):
    image_features = []
    for image_descriptors in descriptors:
        if image_descriptors is None:
            image_features.append(np.zeros(k))
            continue
        image_descriptors = [descriptor for sublist in image_descriptors for descriptor in sublist]
        if len(image_descriptors) == 0:
            image_features.append(np.zeros(k))
            continue
        image_descriptors = np.array(image_descriptors)
        predicted_clusters = kmeans.predict(image_descriptors)
        histogram, _ = np.histogram(predicted_clusters, bins=range(k+1))
        normalized_histogram = histogram / len(image_descriptors)
        image_features.append(normalized_histogram)
    return np.array(image_features)



def main():
    labels=(data.getLabels(PATH_TO_DATA))
    label_indexes = [index for index, label in enumerate(labels)]
    print("Labels: ", label_indexes)
    filePathArray = data.getFilePaths(PATH_TO_DATA, labels) 
    
    ##smaller sample for testing
    #filePathArray=getSubsetOfFilePathArray(filePathArray, 20)

    
    #MHI Variables
    Min_Delta = 50  
    Max_Delta  = 100
    MHI_DURATION= 1
    MHI_array=getMHIFromFilePathArray(filePathArray, Min_Delta,Max_Delta, MHI_DURATION)
    
    
    print("MHI ARRAY", type(MHI_array[0]))
    
    
    
    #keypoints, descriptors, mhi_with_keypoints_images = SIFTAnalysisOnMHI_Array(MHI_array)
    #print("Calculating all_descriptors Variable")   
    #all_descriptors = np.concatenate([descriptor.flatten() for image_descriptors in descriptors for descriptor in image_descriptors if descriptor is not None], axis=0)
    #all_descriptors = all_descriptors.reshape(-1, 128)

    # Extract HoG features
    print("DONE\n\nExtracting HoG features")
    hog_features, numerical_labels = extract_hog_features(MHI_array)
    
    print("Hog Features length  : ", len(hog_features))
    print("Numerical label length  : ", len(numerical_labels))

    print("DONE\n\nTraining SVM classifier")
    # Train SVM classifier
    svm_classifier, X_test, y_test = train_svm(hog_features, numerical_labels)

    print("DONE\n\nTesting SVM Classifier")
    # Test SVM classifier
    test_svm(svm_classifier, X_test, y_test)
    print(y_test)




    cwd = os.getcwd()  
    path = os.path.join(cwd,"SIFT_svm_classifier.pkl")
    print("Saving\n",path,"\n")

    with open(path, 'wb') as f:
        pickle.dump((svm_classifier,X_test,y_test), f)
    
    cwd = os.getcwd()  
    path = os.path.join(cwd,"Coursework","SIFT_svm_classifier.pkl")
    print("Saving\n",path,"\n")
   
    with open(path, 'wb') as f:
        pickle.dump((svm_classifier,X_test,y_test), f)


    print("DONE\n\nSaving Classifier")
    name=('SIFT_svm_classifier.pkl')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dir_path,name)
    with open(path, 'wb') as f:
        pickle.dump((svm_classifier,X_test,y_test), f)


    # Create the label mapping dictionary
    
    
    
    label_mapping = {numerical_labels: label for numerical_labels, label in zip(label_indexes, labels)}
#    # Save the SVM classifier
#    with open('svm_classifier.pkl', 'wb') as f:
#        pickle.dump(svm_classifier, f)
#
#    # Save the label encoding mapping
#    with open('./Coursework/SIFT_svm_classifier.pkl', 'wb') as f:
#        pickle.dump(label_mapping, f)
#
    #k = 6
    #print("Calculating kmeans")   
    #kmeans = KMeans(n_clusters=k, random_state=0,n_init=10).fit(hog_features)
    #print("DONE\n")
    #print("Calculating Image features")
    #image_features = compute_image_features(kmeans, descriptors, k)
    #print("DONE\n")
    #print("training SVM")  
    #svm_classifier, X_test, y_test = train_svm(image_features, label_indexes)
    #print("DONE\n")
    #print("Testing SVM")
    #test_svm(svm_classifier, X_test, y_test)
    #print("DONE\n")
    cv.destroyAllWindows()
    print("\nDONE - exiting program")
    
if __name__=="__main__":
    main()


