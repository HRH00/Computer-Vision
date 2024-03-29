import os
import numpy as np
from package_901476 import test 
import package_901476.Data as data

def main():
    print("Testing HOG and Neural Network Classifier")
    try:    
        # generate the path to data
        cwd = os.getcwd()  
        path = os.path.join(cwd,"Hog_nn_classifier.pkl")
        print("\n",path,"\n")


        #load the data
        dataTuple = data.openSavedData(path)
        (NN_classifier,X_test,y_test, labels, NN_cross_validate) = dataTuple   
        
        
        
        print("Benchmarking classifier on single test sample for time ") 
        test.pred(NN_classifier,X_test) # predict values for Xtest 

        #####Cross validation analysis
        
        
        ## print the confusion matrix
        (acc_scores_cval, acc_scores_cval,agg_cm_CV) = NN_cross_validate
        print("Confusion Matrix Cross Validation:")
        
        print(agg_cm_CV)
        title="Aggregate Confusion Matrix - HOG+NN Classifier with Cross Validation"
        test.plot_confusion_matrix(agg_cm_CV,labels,title)
        print("\nTesting Neural Network classifier with Cross Validation")
        print(f'NN Cross Validation Average accuracy : {np.mean(acc_scores_cval)}')
        

    except: # ERROR
        print("ERROR - run HOG_and_NN_classifier_trainer.py in same directory as this file first")
        print("If this doesn\'t work, then something has gone very wrong")
    print("\nDONE - exiting program")
   
if __name__=="__main__":
    main()


