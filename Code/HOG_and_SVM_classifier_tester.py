import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from package_901476 import test 
from package_901476 import Data as data

def main():
    print("Testing HOG and SVM Classifier")
    # Test SVM classifier
    # Open the SVM classifier and test data unseen by the classifier``

    
    
    try:
        cwd = os.getcwd()  #load data 
        path = os.path.join(cwd,"Hog_svm_classifier.pkl")
        print("\n",path,"\n")

        dataTuple = data.openSavedData(path)
        (svm_classifier,X_test,y_test, labels,SVM_cross_validate) = dataTuple   


        
        print("Testing Standard SVM classifier")
        predictions = svm_classifier.predict(X_test) # make predictions on test set
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy Score:", accuracy) # output accuracy score
        
        print("Classify one image") 
        test.pred(svm_classifier,[X_test[0]]) 

        cm = confusion_matrix(y_test, predictions)
        print("Confusion Matrix:")
        
        print(cm)
        # Precision, recall, f-score 
        print(classification_report(y_test, predictions))
        title="Confusion Matrix for HOG and SVM Classifier"
        
        test.plot_confusion_matrix(cm,labels,title)
        print("Done\n")
        
        #####Cross validation analysis
        
        (acc_scores_cval, acc_scores_cval,agg_cm_CV) = SVM_cross_validate
        print("Confusion Matrix Cross Validation:")
        
        print(agg_cm_CV)
        title="Aggregate Confusion Matrix - HOG+SVM Classifier with Cross Validation"
        test.plot_confusion_matrix(agg_cm_CV,labels,title)
        print("\nTesting SVM classifier with Cross Validation")
        print(f'SVM Cross Validation Average accuracy : {np.mean(acc_scores_cval)}')
        

    except: # ERROR
        print("ERROR - run HOG_and_SVM_classifier_trainer.py in same directory as this file first")
        print("If this doesn\'t work, then something has gone very wrong")
    print("\nDONE - exiting program")
    
if __name__=="__main__":
    main()


