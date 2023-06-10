


import os
import pickle
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import sys
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans



##You need to set the following paths to the location of the data on your machine
dir_path = os.path.dirname(os.path.realpath(__file__))
PATH_TO_DATA = os.path.join(dir_path,"Datastore","Supplied")

PATH_TO_MHI_STORAGE = os.path.join("q") #iqqn ot in use
FILE_EXTENTION = ".avi" 

##DEBUGGING FUNCTIONS

def showChannelFromPath(video_path, channel_index):
    print("showing Channel",channel_index,"from path",video_path)
    
    # Open the video file
    cap = cv.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(" opening video file", video_path)
        return
    showShape=True
    while True:
        success, frame = cap.read()
        if success: 
            # Split the frame into color channels
            channels = cv.split(frame)

            # Extract the specified channel based on the index
            channel = channels[channel_index]

            if showShape:
                    print("Showchannelfrompath - Channel Shape",channel.shape)
                    showShape=False
            
            # Display the channel
            cv.imshow("Channel", channel)

            # Check for 'q' key press to quit
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("releasing video")
            cap.release()
            cv.destroyAllWindows()
            break
def getSubsetOfFilePathArray(filePathArray, samplesPerFeature):
    subsetData=[]
    for feature in filePathArray:
        featureList=[]
        for video in feature[:samplesPerFeature]:
            featureList.append(video)
        subsetData.append(featureList)    
        
    return subsetData  


###File Gathering init Function
def getLabels():# Creates a list of labels, sourced from each child directory names in the specified path 

    directories = []
    try:
        for item in os.listdir(PATH_TO_DATA):
            item_path = os.path.join(PATH_TO_DATA, item)
            if os.path.isdir(item_path):
                directories.append(item)                
    except Exception as e:
        print(", No data found\n\n",e)
    else:
        print("Labels:", directories)
    return directories
def enumerateLabels(labels):#Creates a list of integers which corrolate with the labels list, 

    
    LabelIntegers = [i for i in range(len(labels))] # list aprehention

    if LabelIntegers== []:
        print("CANNOT ENumerate, no labels found\n\n")    
    else:
        print("Labels enumerated:",LabelIntegers,"\n")
        
    return LabelIntegers          
def getFilePaths(PATH_TO_DATA, labels):#creates a 2D array of paths, the first index corrolates with each label integer
    all_data=[]  
    
    for label in labels:
        label_data=[]
        label_path = os.path.join(PATH_TO_DATA,label)
        for item in os.listdir(label_path):
            item_path = os.path.join(label_path, item)    
            if item_path.endswith(FILE_EXTENTION):
                label_data.append(item_path)
        print("Found",len(label_data),"Files ending in",FILE_EXTENTION,"for Label,",label)
        all_data.append(label_data)   


    for data in all_data:
        if not (data):
            fileError()                       
       
    if not all_data:
        print("Error: No files found in at least one label directory")
        
    return all_data

    
def getLabelNameFromInteger(label_int,labels):
    return labels[label_int]
def getLabelIntFromName(label_name,labels):
    return labels.index(label_name)
def fileError():
    print("\033[91mBad file path, no data found\n")
    print(PATH_TO_DATA,"\nContains no .avi, has directory with no .avi or has wrong file structure\033[0m")
    print("\033[91mupdate PATH_TO_DATA with correct path\033[0m")
    sys.exit()
###def getFramesGreyscale(video_path):

    # Open the video file
    
    cap = cv.VideoCapture(video_path)
    frames=[]

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(" opening video file",video_path)
        return   

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        frames.append(frame)     

    cap.release()

    print("Frames extracted from ",video_path)
    
    return frames
def getFirstFrame(video_path):
    # Open the video file
    cap = cv.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(" opening video file",video_path)
        return   

    success, frame = cap.read()
    if success:
        return frame     
    else:
        print("Can not find a frame")   
        return None

    cap.release()

### Output Graphics Functions
def showChannelfromVideoFrames(video, channel_index):
    print("showing Channel",channel_index,"from video frames")
    for frame in video:
        # Split the frame into color channels
        channels = cv.split(frame)

        # Extract the specified channel based on the index
        channel = channels[channel_index]
        Windowsname="Channel"+str(channel_index)
        # Display the channel
        cv.imshow(Windowsname, channel)

        # Check for 'q' key press to quit
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
def showVideo(videopath):
    video = cv.VideoCapture(videopath)
    while True:
        success, frame = video.read()
        if success:
                # Extract the specified channel based on the index
            Windowsname="Video"
            # Display the channel
            cv.imshow(Windowsname,frame)

            # Check for 'q' key press to quit
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            video.release()
            cv.destroyAllWindows()
            break
def showImageUntilKeyPress(windowName,image):
    while True:
        cv.imshow(windowName,image)

        # Check for 'q' key press to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
def showImageForXms(windowName,image,showForMS):##Display a single image for a specified number of milliseconds
    cv.imshow(windowName,image)
    if showForMS == 0:
        return
    cv.waitKey(showForMS)

## MOTION HISTORY IMAGE FUNCTIONS
def getMHIFromFilePathArray(filePathArray, MIN_DELTA, MAX_DELTA, MHI_DURATION):  
#generates the motion history from a video file path   
    print("\nCalculating Motion History Image from the file path array")
    MHI_array=[]
    i=0
    for row in filePathArray:
        Label_MHI=[]
        labels =[]
        for path in row:
            MHI=getMHIFromVideo(path, MIN_DELTA, MAX_DELTA, MHI_DURATION)
            Label_MHI.append(MHI)
            

        MHI_array.append(Label_MHI)
    print("Done - MHI array created\n")
    return (MHI_array)

  

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


    labels=(getLabels())
    label_indexes = [index for index, label in enumerate(labels)]
    print("Labels: ", label_indexes)
    filePathArray = getFilePaths(PATH_TO_DATA, labels) 
    
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


