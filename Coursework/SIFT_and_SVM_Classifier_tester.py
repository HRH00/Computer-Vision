import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

def plot_confusion_matrix(confusion_matrix, labels,title):
    plt.imshow(confusion_matrix, cmap='Purples')
    plt.colorbar()

    num_labels = len(labels)
    labelsX = [string[:7] for string in labels]#shorten labels to 7 characters
    print(labelsX)
    plt.xticks(np.arange(num_labels), labelsX)
    plt.yticks(np.arange(num_labels), labels)

    # Add labels to each cell
    for i in range(num_labels):
        for j in range(num_labels):
            plt.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='black')

    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(title)
    plt.show()

##time bench marks a function 50 times and prints the average time
def benchmark(func):
    def wrapper(*args, **kwargs):
        average = 0
        N=50
        for i in range(N):    
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            #print(f"Run {i} of {N} - Execution time: {execution_time:.2f} ms")
            average+=execution_time
        average=average/N
        print(f"Average execution time: {average:.2f} ms")
        return result
    return wrapper
@benchmark
def pred(classifier, image):
    classifier.predict(image)
           
def main():
    print("Testing SIFT and SVM Classifier")
    # Test SVM classifier
    # Open the SVM classifier and test data unseen by the classifier``

    
    
    try:
        cwd = os.getcwd()  
        path = os.path.join(cwd,"SIFT_svm_classifier.pkl")
        print("\n",path,"\n")

        with open(path, 'rb') as f:
            (svm_classifier,X_test,y_test, labels,SVM_cross_validate) = pickle.load(f)##open cached data
            
    

        
        print("Testing Standard SVM classifier")
        predictions = svm_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy Score:", accuracy)
        
        print("Classify one image") 
        pred(svm_classifier,[X_test[0]]) 

        cm = confusion_matrix(y_test, predictions)
        print("Confusion Matrix:")
        
        print(cm)
        # Precision, recall, f-score 
        print(classification_report(y_test, predictions))
        title="Consufion Matrix for SIFT and SVM Classifier"
        
        plot_confusion_matrix(cm,labels,title)
        print("Done\n")
        
        #####Cross validation analysis
        
        (acc_scores_cval, svm_classifier_cval, x_test_cval, y_test_cval) = SVM_cross_validate
        input("Press Enter to view cross validation results...")
        predictions_CV = svm_classifier_cval.predict(x_test_cval)
        accuracy_CV = accuracy_score(y_test_cval, predictions_CV)
        print("Cross validation Accuracy Score:", accuracy_CV)
        
        print("Classify one image") 
        pred(svm_classifier_cval,[X_test[0]]) 

        cm_CV = confusion_matrix(y_test_cval, predictions_CV)
        print("Confusion Matrix Cross Validation:")
        
        print(cm_CV)
        # Precision, recall, f-score 
        print(classification_report(y_test_cval, predictions_CV))
        title="Consufion Matrix for SIFT and SVM Classifier with Cross Validation"
        plot_confusion_matrix(cm_CV,labels,title)
        print("\nTesting SVM classifier with Cross Validation")
        print(f'SVM Cross Validation Average accuracy : {np.mean(acc_scores_cval)}')
        

    except: # ERROR
        print("Cannot load Classifier - run SIFT_and_SVM_classifier_trainer.py first")
    
    input("Press Enter to Exit...")
    print("\nDONE - exiting program")
    
if __name__=="__main__":
    main()


