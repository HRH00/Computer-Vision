


import os
import pickle
import cv2 as cv
import numpy as np
import sys
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



##You need to set the following paths to the location of the data on your machine
dir_path = os.path.dirname(os.path.realpath(__file__))
PATH_TO_DATA = os.path.join(dir_path,"Datastore","Supplied")
FILE_EXTENTION = ".avi" 

##DEBUGGING FUNCTIONS

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

def getMHIFromVideo(video_path, MIN_DELTA, MAX_DELTA, MHI_DURATION):
    # Create a VideoCapture 
    cap = cv.VideoCapture(video_path)

    # Get frame 1 of video
    ret, frame = cap.read()

    # Convert to grayscale
    prev_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Init mhi motion history image
    h, w = prev_frame.shape[:2]
    mhi = np.zeros((h, w), np.float32)

    while True:
        # Read the next frame
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to grayscale
        curr_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Compute asd difference
        frame_diff = cv.absdiff(curr_frame, prev_frame)

        # binary motion image
        _, motion_mask = cv.threshold(frame_diff, MIN_DELTA, MAX_DELTA, cv.THRESH_BINARY)

        # Update the motion history image
        timestamp = cv.getTickCount() / cv.getTickFrequency()
        cv.motempl.updateMotionHistory(motion_mask, mhi, timestamp, MHI_DURATION)

        # Update the previous frame
        prev_frame = curr_frame

    # Release the VideoCapture object
    cap.release()


    # Normalize to [0,255] and convert type to uint8
    mhi_uint8 = cv.normalize(mhi, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    # Apply a colormap if you want to visualize the MHI
    mhi_colored = cv.applyColorMap(mhi_uint8, cv.COLORMAP_JET)

    # Return the uint8 MHI
    return mhi_uint8

  

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

def get_hog_features(image):
    winSize = (64,64)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                            histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    hist = hog.compute(image,winStride,padding,locations)
    return hist


def extract_hog_features(images):
    features = []
    labels = []
    x=0
    for lab in images:

        for img in lab:
    
            hog_features = get_hog_features(img)
    #        features.sort(key=lambda x: x.distance)
            features.append(hog_features)
            labels.append(x)
            print("HOG FEATURES", hog_features)
        x+=1    

    return (features, labels)

#convert face labels into numerical representations
def train_svm(features, labels):
    # Split data into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

    # Train SVM classifier
    svm_classifier = svm.SVC()
    svm_classifier.fit(X_train, y_train)

    return svm_classifier, X_test, y_test

# Step 5: Testing/Evaluation
def test_svm(svm_classifier, X_test, y_test):
    print("Testing SVM classifier")
    predictions = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    print("Done\n")
    
    

def main():
    labels=(getLabels())
    filePathArray = getFilePaths(PATH_TO_DATA, labels) 
    
    ##smaller sample for debugging
    
    #filePathArray=getSubsetOfFilePathArray(filePathArray, 20)

    #MHI Variables
    Min_Delta = 50  
    Max_Delta  = 100
    MHI_DURATION= 1
    MHI_array=getMHIFromFilePathArray(filePathArray, Min_Delta,Max_Delta, MHI_DURATION)
           
    # Extract HoG features
    print("DONE\n\nExtracting HoG features")
    hog_features, numerical_labels = extract_hog_features(MHI_array)
    
    print("DONE\n\nTraining SVM classifier")
    # Train SVM classifier
    svm_classifier, X_test, y_test = train_svm(hog_features, numerical_labels)

    print("DONE\n\nSaving Classifier")
    # Save the SVM classifier
    with open('./Coursework/Hog_svm_classifier.pkl', 'wb') as f:
        pickle.dump((svm_classifier,X_test,y_test), f)


    cv.destroyAllWindows()
    print("\nDONE - exiting program")
    
if __name__=="__main__":
    main()