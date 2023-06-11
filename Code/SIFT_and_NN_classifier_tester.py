import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from package_901476 import test 
from package_901476 import Data as data

def main():
    print("Testing SIFT and nn Classifier")
    # Test nn classifier
    # Open the nn classifier and test data unseen by the classifier``

    
    
    try:
        cwd = os.getcwd()  
        path = os.path.join(cwd,"SIFT_nn_classifier.pkl")
        print("\n",path,"\n")

        dataTuple = data.openSavedData(path)
        (nn_classifier,X_test,y_test, labels,nn_cross_validate) = dataTuple   


        
        print("Testing Standard nn classifier")
        predictions = nn_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy Score:", accuracy)
        
        print("Classify one image") 
        test.pred(nn_classifier,[X_test[0]]) 

        cm = confusion_matrix(y_test, predictions)
        print("Confusion Matrix:")
        
        print(cm)
        # Precision, recall, f-score 
        print(classification_report(y_test, predictions))
        title="Confusion Matrix for SIFT and nn Classifier"
        
        test.plot_confusion_matrix(cm,labels,title)
        print("Done\n")
        
        #####Cross validation analysis
        
        (acc_scores_cval, acc_scores_cval,agg_cm_CV) = nn_cross_validate
        print("Confusion Matrix Cross Validation:")
        
        print(agg_cm_CV)
        title="Aggregate Consufion Matrix - SIFT+nn Classifier with Cross Validation"
        test.plot_confusion_matrix(agg_cm_CV,labels,title)
        print("\nTesting nn classifier with Cross Validation")
        print(f'nn Cross Validation Average accuracy : {np.mean(acc_scores_cval)}')
        

    except: # ERROR
        print("ERROR - run SIFT_and_NN_classifier_trainer.py in same directory as this file first")
        print("If this doesn\'t work, then something has gone very wrong")
    print("\nDONE - exiting program")

    
    
    
if __name__=="__main__":
    main()