import os
import pickle
import cv2 as cv
import numpy as np
import sys
from sklearn import svm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

##You need to set the following paths to the location of the data on your machine
dir_path = os.path.dirname(os.path.realpath(__file__))
PATH_TO_DATA = os.path.join(dir_path,"Datastore","Supplied_Data")  #set each string as the path to the data
# each string is a directory
# This program expects the data to be .AVI files, held in directories named after the corrosponding label
# i.e directory name = 'Running'

FILE_EXTENTION = ".avi" 

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
##DEBUGGING FUNCTION - subset of data
def getSubsetOfData(filePathArray, samplesPerFeature):
    subsetData=[]
    for feature in filePathArray:
        featureList=[]
        for video in feature[:samplesPerFeature]:
            featureList.append(video)
        subsetData.append(featureList)    
        
    return subsetData    
##DEBUGGING FUNCTION - Close on error

def fileError():
    print("\033[91mBad file path, no data found\n")
    print(PATH_TO_DATA,"\nContains no .avi, has directory with no .avi or has wrong file structure\033[0m")
    print("\033[91mupdate PATH_TO_DATA with correct path\033[0m")
    sys.exit()


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
    # Create a VideoCapture object and read from input file
    cap = cv.VideoCapture(video_path)
    success, frame = cap.read() # Get 1st frame 
    prev_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # Convert to grayscale   
    h, w = prev_frame.shape[:2]
    mhi = np.zeros((h, w), np.float32) # Init mhi motion history image

    while True:
        # Read the next frame
        success, frame = cap.read()

        if not success: # cannot read capture object, at the end of file or file error
            break

        
        curr_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) # Convert the frame to grayscale

        # Compute absolute diff
        frame_diff = cv.absdiff(curr_frame, prev_frame)

        # binary motion image
        _, motion_mask = cv.threshold(frame_diff, MIN_DELTA, MAX_DELTA, cv.THRESH_BINARY)

        # Update the motion history image
        timestamp = cv.getTickCount() / cv.getTickFrequency()
        cv.motempl.updateMotionHistory(motion_mask, mhi, timestamp, MHI_DURATION)

        prev_frame = curr_frame

    cap.release()

    # Normalize to [0,255] and convert type
    mhi_uint8 = cv.normalize(mhi, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    # Return the uint8 MHI
    return mhi_uint8
  
def get_hog_features(image): # returns a histogram of gradients for a given image
    winSize = (64,64) # HOG parameters
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

def train_svm(features, labels):
    # Split data into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

    # Train SVM classifier
    svm_classifier = svm.SVC()
    svm_classifier.fit(X_train, y_train)

    return svm_classifier, X_test, y_test

def test_svm(svm_classifier, X_test, y_test):
    print("Testing SVM classifier")
    predictions = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    print("Done\n")
    

def cross_validate_svm(features, labels, n_splits=10):
    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    # Reshape features array to 2D if necessary
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []

    for train_index, test_index in kfold.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        svm_classifier = svm.SVC()
        svm_classifier.fit(X_train, y_train)

        predictions = svm_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracy_scores.append(accuracy)

    print(f'Cross-validation accuracy scores: {accuracy_scores}')
    print(f'Average accuracy: {np.mean(accuracy_scores)}')
def main():
    labels=(getLabels())
    filePathArray = getFilePaths(PATH_TO_DATA, labels) 
    
    #filePathArray=getSubsetOfData(filePathArray,5)    ## smaller sample for debugging

    # Motion History Constant Variables
    Min_Delta = 50  
    Max_Delta  = 100
    MHI_DURATION= 1
    MHI_array=getMHIFromFilePathArray(filePathArray, Min_Delta,Max_Delta, MHI_DURATION)
           
    # Extract HoG features
    print("DONE\n\nExtracting HoG features")
    hog_features, numerical_labels = extract_hog_features(MHI_array)
    
    #K-fold cross validation
    cross_validate_svm(hog_features, numerical_labels)
    
    
    
    # Train SVM classifier
    print("DONE\n\nTraining SVM classifier")
    svm_classifier, X_test, y_test = train_svm(hog_features, numerical_labels)

    # Save the SVM classifier
    
    print("DONE\n\nSaving Classifier")
    
    
    cwd = os.getcwd()  
    path = os.path.join(cwd,"Hog_svm_classifier.pkl")
    print("\nPATH",path,"\n")

    with open(path, 'wb') as f:
        pickle.dump((svm_classifier,X_test,y_test),f)
   


    cv.destroyAllWindows()
    print("\nDONE - exiting program")
    
if __name__=="__main__":
    main()