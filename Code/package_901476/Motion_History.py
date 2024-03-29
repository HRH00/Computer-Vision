## MOTION HISTORY IMAGE FUNCTIONS
import package_901476.Data as Data
import cv2 as cv
import numpy as np

def getMHIFromFilePathArray(filePathArray, MIN_DELTA, MAX_DELTA, MHI_DURATION):  
#generates the motion history from a video file path   
    has_data = Data.openSavedData("MHI_array.pkl")
    if has_data:
        return has_data
    else:
        print("Calculating Motion History Image from the file path array\nTHIS MAY TAKE A WHILE\n")
        MHI_array=[]
        for row in filePathArray:
            Label_MHI=[]
            for path in row:
                MHI=getMHIFromVideo(path, MIN_DELTA, MAX_DELTA, MHI_DURATION)
                Data.showImageForXms("MHI",MHI,1)
                Data.showImageFromVidForXms("Frame",path,1)
                Label_MHI.append(MHI)
            MHI_array.append(Label_MHI)
        print("\nDone - MHI array created\n")
        cv.destroyAllWindows()
        Data.saveData(MHI_array,"MHI_array.pkl")
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

    # Normalize to [0,255] , convert to uint8
    mhi_uint8 = cv.normalize(mhi, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    # Return the uint8 MHI
    return mhi_uint8