import os
import cv2 as cv
import numpy as np
import sys

##You need to set the following paths to the location of the data on your machine
dir_path = os.path.dirname(os.path.realpath(__file__))
PATH_TO_DATA = os.path.join(dir_path,"Datastore","Supplied")

PATH_TO_MHI_STORAGE = os.path.join("q") #iqqn ot in use
FILE_EXTENTION = ".avi" 


# Creates a list of labels, sourced from each child directory names in the specified path 
def getLabels():
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

#Creates a list of integers which corrolate with the labels list, 
def enumerateLabels(labels):
    
    LabelIntegers = [i for i in range(len(labels))] # list aprehention

    if LabelIntegers== []:
        print("CANNOT ENumerate, no labels found\n\n")    
    else:
        print("Labels enumerated:",LabelIntegers,"\n")
        
    return LabelIntegers        
#creates a 2D array of paths, the first index corrolates with each label integer    
def getFilePaths(labels, PATH_TO_DATA):
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
        

                        
            
    return all_data

def getLabelNameFromInteger(label_int,labels):
    return labels[label_int]

def getLabelIntFromName(label_name,labels):
    return labels.index(label_name)
#creates an array of MHIs from a 2D array of file paths
def getMHIFromFilePathArray(filePathArray, MIN_DELTA, MAX_DELTA, MHI_DURATION):   

    print("Getting MHI from file path array")
    MHI_array=[]
    for row in filePathArray:
        Label_MHI=[]
        for path in row:
            MHI=getMHIFromVideo(path, MIN_DELTA, MAX_DELTA, MHI_DURATION)
            Label_MHI.append(MHI)
            
            
        MHI_array.append(Label_MHI)
             
    print("MHI array created")
    return (MHI_array)
#generates the motion history from a video file path

def fileError():
    print("\033[91mBad file path, no data found\n")
    print(PATH_TO_DATA,"\nContains no .avi, has directory with no .avi or has wrong file structure\033[0m")
    print("\033[91mupdate PATH_TO_DATA with correct path\033[0m")
    sys.exit()
    
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

    # Return the motion history image
    return (mhi)

#performs sift analysis on a MHI, 
def performSIFTAnalysis(mhi):
    
    sift = cv.SIFT_create()
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(mhi, None)
    # Draw keypoints 
    mhi_with_keypoints = cv.drawKeypoints(mhi, keypoints, None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Display keypoints
    showImageUntilKeyPress("Keypoint",mhi_with_keypoints)
    return keypoints, descriptors

##Debugging function 
def getFramesGreyscale(video_path):

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
##Debugging function 
def getFrames(video_path):
    # Open the video file
    
    cap = cv.VideoCapture(video_path)
    frames=[]

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print(" opening video file",video_path)
        return   

    while True:
        success, frame = cap.read()
        if success:
            frames.append(frame)     
        else:
            break

    cap.release()

    print("Frames extracted from ",video_path)
    
    return frames
##Debugging function
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
##Debugging function 
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
##Debugging function 
def showVideo(videopath):
    print("showing Video",videopath)
    video = cv.VideoCapture(videopath)
    showShape=True
    while True:
        success, frame = video.read()
        if success:
            if showShape:
                print(frame.shape)
                showShape=False
                # Extract the specified channel based on the index
            Windowsname="Video"+str(videopath)
            # Display the channel
            cv.imshow(Windowsname,frame)

            # Check for 'q' key press to quit
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            video.release()
            cv.destroyAllWindows()
            break
##Display a single image until a key is pressed 
def showImageUntilKeyPress(windowName,image):
    while True:
        cv.imshow(windowName,image)

        # Check for 'q' key press to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
##Display a single image for a specified number of milliseconds
def showImageForXms(windowName,image,showForMS):
    cv.imshow(windowName,image)
    cv.waitKey(showForMS)

##Debugging function 
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
##Debugging function       
def getSubsetOfFilePathArray(filePathArray, samplesPerFeature):
    subsetData=[]
    for feature in filePathArray:
        featureList=[]
        for video in feature[:samplesPerFeature]:
            featureList.append(video)
        subsetData.append(featureList)    
        
    return subsetData  
   
   

   
   
def main():
    

    labels=(getLabels())
    Label_Integers=enumerateLabels(labels) 
    filePathArray = getFilePaths(labels, PATH_TO_DATA)    
    
    if not filePathArray:
        print("Error: No files found in at least one label directory")
    
    # #debuging functions
    #GreyVideoFrames = getFramesGreyscale(filePathArray[0][0])
    #showChannelfromVideoFrames(GreyVideoFrames, 0)
    #showVideo(filePathArray[0][0])
    #showChannelFromPath(filePathArray[0][0],0)
    #videoFrames = getFrames(filePathArray)
    #print(getFirstFrame(filePathArray[0][0]).shape)
    
    ## get a subset of data for testing
    filePathArray=getSubsetOfFilePathArray(filePathArray, 1)
    
    Min_Delta = 50  
    Max_Delta  = 1
    MHI_DURATION= .01
    MHI_array=getMHIFromFilePathArray(filePathArray,Min_Delta,Max_Delta,MHI_DURATION) 
    
    ##debguigingq

        
    cv.destroyAllWindows()
    
    

    x=0
    for label in MHI_array:
        print("Showing Images for:",labels[x])
        y=0
        for MHIImg in label:
            showImageForXms("Original",getFirstFrame(filePathArray[x][y]),10) 
            showImageForXms("MHI",MHIImg,10)
            mhi=cv.convertScaleAbs(MHIImg)
            performSIFTAnalysis(mhi)
            y+=1
        x+=1
        
    cv.destroyAllWindows()
    print("DONE - exiting program")
    
if __name__=="__main__":
    main()


