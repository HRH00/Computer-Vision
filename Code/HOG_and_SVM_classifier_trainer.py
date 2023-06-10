import os

from package_901476 import Hog as hog
from package_901476 import Data as data
from package_901476 import Motion_History as mhi
from package_901476 import SVM as svm

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
    print("DONE\n\nExtracting HoG features")
    hog_features, numerical_labels = hog.extract_hog_features(MHI_array)
    
    #K-fold cross validation
    
    SVM_cross_validate = svm.cross_validate_svm(hog_features, numerical_labels)
    
    
    # Train SVM classifier
    print("DONE\n\nTraining SVM classifier")
    svm_classifier, X_test, y_test = svm.train_svm(hog_features, numerical_labels)

    # Save the SVM classifiers
    
    print("DONE\n\nSaving Classifier")
    
    
    cwd = os.getcwd()  
    path = os.path.join(cwd,"Hog_svm_classifier.pkl")
    print("\nPATH",path,"\n")

    dataTuple = (svm_classifier,X_test,y_test,labels,SVM_cross_validate)
   
    data.saveData(dataTuple,path)

    print("\nDONE - exiting program")
    
if __name__=="__main__":
    main()