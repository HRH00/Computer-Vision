import package_901476.SVM as SVM
import package_901476.Motion_History as mhi
import package_901476.Sift as Sift
import package_901476.Data as data 
import os

##You need to set the following paths to the location of the data on your machine
dir_path = os.path.dirname(os.path.realpath(__file__))
PATH_TO_DATA = os.path.join(dir_path,"Datastore","Supplied")


FILE_EXTENTION = ".avi" 


def main():
    print("SIFT and SVM classifier trainer")
    labels=(data.getLabels(PATH_TO_DATA))
    filePathArray = data.getFilePaths(PATH_TO_DATA, labels,FILE_EXTENTION) 
    
    filePathArray = data.getSubsetOfData(filePathArray, 10) # subset of data
    
    print("filespatharray",len(filePathArray[0]))
    # Motion History Constant Variables
    Min_Delta = 50  
    Max_Delta  = 100
    MHI_DURATION= 1
    MHI_array=mhi.getMHIFromFilePathArray(filePathArray, Min_Delta,Max_Delta, MHI_DURATION)

    (keypoints, descriptors, numerical_labels) = Sift.SIFTAnalysisOnMHI_Array(MHI_array)   
    features = Sift.compute_image_features(keypoints,descriptors, len(labels))
    print("DONE\n\nTraining SVM classifier")
    svm_classifier, X_test, y_test = SVM.train_svm(features, numerical_labels)
    print("DONE\n\Kcross fold validation")
    
    SVM_cross_validate = SVM.cross_validate_svm(features, numerical_labels) #K-fold cross validation
    # Save the SVM classifiers
    
    print("DONE\n\nSaving Classifier")  
    cwd = os.getcwd()  
    path = os.path.join(cwd,"SIFT_svm_classifier.pkl")
    print("\nPATH",path,"\n")

    dataTuple = (svm_classifier,X_test,y_test,labels,SVM_cross_validate)
    data.saveData(dataTuple,path)
    print("\nDONE - exiting program")
  
if __name__=="__main__":
    main()


