import package_901476.Neural_Net as nn
import package_901476.Motion_History as mhi
import package_901476.Sift as Sift
import package_901476.Data as data 
import os

##You need to set the following paths to the location of the data on your machine
dir_path = os.path.dirname(os.path.realpath(__file__))
PATH_TO_DATA = os.path.join(dir_path,"Datastore","Supplied")


FILE_EXTENTION = ".avi" 


def main():
    print("SIFT and Neural Network classifier trainer")
    labels=(data.getLabels(PATH_TO_DATA))
    filePathArray = data.getFilePaths(PATH_TO_DATA, labels,FILE_EXTENTION) 
    
    #filePathArray = data.getSubsetOfData(filePathArray, 5) # subset of data
    
    # Motion History Constant Variables
    Min_Delta = 50  
    Max_Delta  = 100
    MHI_DURATION= 1
    MHI_array=mhi.getMHIFromFilePathArray(filePathArray, Min_Delta,Max_Delta, MHI_DURATION)
    features, num_labels = Sift.doSift(MHI_array)

    print(num_labels[0])

    nn_classifier, X_test, y_test = nn.train_neural_network(features, num_labels)

    
    
    print("DONE\n\nTraining NN classifier")
# 
    print("DONE\n\Kcross fold validation")
    
    NN_cross_validate = nn.cross_validate_nn(features, num_labels) #K-fold cross validation
    # Save the NN classifiers
    
    print("DONE\n\nSaving Classifier")  
    cwd = os.getcwd()  
    path = os.path.join(cwd,"SIFT_NN_classifier.pkl")
    print("\nPATH",path,"\n")

    dataTuple = (nn_classifier,X_test,y_test,labels,NN_cross_validate)
    data.saveData(dataTuple,path)
    print("\nDONE - exiting program")
    

if __name__=="__main__":
   main()