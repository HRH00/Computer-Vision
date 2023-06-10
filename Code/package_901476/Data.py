import os 
import pickle
import cv2 as cv
import pickle

def getFilePaths(PATH_TO_DATA, labels,FILE_EXTENTION):#creates a 2D array of paths, the first index corrolates with each label integer
    print("Getting File Paths")
    all_data=[]  
    
    for label in labels:
        label_data=[]
        label_path = os.path.join(PATH_TO_DATA,label)
        for item in os.listdir(label_path):
            item_path = os.path.join(label_path, item)    
            if item_path.endswith(FILE_EXTENTION):
                if avi_not_corrupt(item_path):
                    label_data.append(item_path)
        print("Found",len(label_data),"Files ending in",FILE_EXTENTION,"for Label,",label)
        all_data.append(label_data)   


    for data in all_data:
        if not (data):
            fileError()                       
    if not all_data:
        print("Error: No files found in at least one label directory")
    return all_data
       
def avi_not_corrupt(file_path):
    if cv.VideoCapture(file_path).isOpened():
       return True  
    else:
        print("Error in", file_path)
        return False         

def getLabels(PATH_TO_DATA):# Creates a list of labels, sourced from each child directory names in the specified path 
    print("Getting Labels")
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
    print("Enumerating Labels")
    LabelIntegers = [i for i in range(len(labels))] # list aprehention

    if LabelIntegers== []:
        print("CANNOT ENumerate, no labels found\n\n")    
    else:
        print("Labels enumerated:",LabelIntegers,"\n")
        
    return LabelIntegers          

##DEBUGGING FUNCTION - subset of data
def getSubsetOfData(filePathArray, samplesPerFeature):
    print("Getting subset of data")
    subsetData=[]
    for feature in filePathArray:
        featureList=[]
        for video in feature[:samplesPerFeature]:
            featureList.append(video)
        subsetData.append(featureList)    
        
    return subsetData    
##DEBUGGING FUNCTION - Close on error

def fileError(PATH_TO_DATA):
    print("\033[91mBad file path, no data found\n")
    print(PATH_TO_DATA,"\nContains no .avi, has directory with no .avi or has wrong file structure\033[0m")
    print("\033[91mupdate PATH_TO_DATA with correct path\033[0m")
    os.sys.exit()

def showImageForXms(windowName,image,showForMS):##Display a single image for a specified number of milliseconds

    cv.imshow(windowName,image)
    if showForMS == 0:
        return
    cv.waitKey(showForMS)

def showImageFromVidForXms(windowName,path,showForMS):##Display a single image for a specified number of milliseconds
    video = cv.VideoCapture(path)
    success, image = video.read()
    if success:
        cv.imshow(windowName,image)
        if showForMS == 0:
            return
        cv.waitKey(showForMS)

def saveData(tupleOfData, path):
    print("Saving data to",path)
    try:
        with open(path, 'wb') as f:
            pickle.dump((tupleOfData),f)
    except Exception as e:
        print("Error saving data to",path)
        return

def openSavedData(path):
    print("Opening data from",path)
    try:
        with open(path, 'rb') as f: 
            dataTuple = pickle.load(f)
            return dataTuple
    except Exception:
        print("Error opening data from",path)
        return False 

