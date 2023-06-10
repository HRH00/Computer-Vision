import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from package_901476 import test 
import package_901476.Data as data

def main():
    print("Testing HOG and Neural Network Classifier")
    
    cwd = os.getcwd()  
    path = os.path.join(cwd,"Hog_nn_classifier.pkl")
    print("\n",path,"\n")

    dataTuple = data.openSavedData(path)
    (NN_classifier,X_test,y_test, labels, NN_cross_validate) = dataTuple   
    
    print("Testing Neural Network classifier")
    predictions = NN_classifier.predict(X_test).astype(np.int32)
    
    
    print("dtype pred",predictions.dtype)
    print("dtype y test",y_test.dtype)
    accuracy = accuracy_score(y_test, predictions)   ###This has a problem 
    
#    prediction is float32
    
    print("Accuracy Score:", accuracy)
    
    print("Benchmarking classifier on single test sample for time ") 
    test.pred(NN_classifier,X_test) 

    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")

    print(cm)
    # Precision, recall, f-score 
    print(classification_report(y_test, predictions))
    title="Confusion Matrix for HOG and NN Classifier"
    
    test.plot_confusion_matrix(cm,labels,title)
    print("Done\n")
    
    #####Cross validation analysis
    
    (acc_scores_cval, acc_scores_cval,agg_cm_CV) = NN_cross_validate
    print("Confusion Matrix Cross Validation:")
    
    print(agg_cm_CV)
    title="Aggregate Consufion Matrix - HOG+NN Classifier with Cross Validation"
    test.plot_confusion_matrix(agg_cm_CV,labels,title)
    print("\nTesting Neural Network classifier with Cross Validation")
    print(f'NN Cross Validation Average accuracy : {np.mean(acc_scores_cval)}')
        

#    except: # ERROR
#        print("ERROR - run HOG_and_NN_classifier_trainer.py in same directory as this file first")
#        print("If this doesn\'t work, then something has gone very wrong")
#    print("\nDONE - exiting program")
    
if __name__=="__main__":
    main()


