import os
import cv2 as cv
from package_901476 import Hog as hog
from package_901476 import Data as data
from package_901476 import Motion_History as mhi
from package_901476 import Neural_Net as nn

##You need to set the PATH_TO_DATA path to the location of the data on your machine
dir_path = os.path.dirname(os.path.realpath(__file__))
PATH_TO_DATA = os.path.join(dir_path,"Datastore","Supplied_Data")  #set each string as the path to the data
# each string is a directory

# This program expects the data to be .AVI files, held in directories named after the corrosponding label
# i.e directory name = 'Running'
FILE_EXTENTION = ".avi" 

def main():
    labels=(data.getLabels(PATH_TO_DATA))
    filePathArray = data.getFilePaths(PATH_TO_DATA, labels,FILE_EXTENTION) 
    
    #filePathArray=data.getSubsetOfData(filePathArray,5)    ## smaller sample for debugging

    # Motion History Constant Variables
    Min_Delta = 50  
    Max_Delta  = 100
    MHI_DURATION= 1
    MHI_array=mhi.getMHIFromFilePathArray(filePathArray, Min_Delta,Max_Delta, MHI_DURATION)
           
    # Extract HoG features
    hog_features, numerical_labels = hog.extract_hog_features(MHI_array)
    
    #K-fold cross validation for testing the neural network
    NN_cross_validate = nn.cross_validate_nn(hog_features, numerical_labels)
    
    # Train classifier without cross validation
    NN_classifier, X_test, y_test = nn.train_neural_network(hog_features, numerical_labels)

    # Save the classifiers
    
    print("DONE\n\nSaving Classifier")
    
    
    cwd = os.getcwd()  
    path = os.path.join(cwd,"Hog_nn_classifier.pkl")
    print("\nPATH",path,"\n")

    dataTuple = (NN_classifier,X_test,y_test,labels,NN_cross_validate)
   
    data.saveData(dataTuple,path)

    cv.destroyAllWindows()
    print("\nDONE - exiting program")
    
if __name__=="__main__":
    main()