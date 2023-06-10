import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

def plot_confusion_matrix(confusion_matrix, labels):
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
    plt.title('Confusion Matrix for HOG and SVM Classifier')
    plt.legend("TEST")
    plt.show()


def benchmark(func):
    def wrapper(*args, **kwargs):
        average = 0
        N=50
        for i in range(N):    
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            print(f"Run {i} of {N} - Execution time: {execution_time:.2f} ms")
            average+=execution_time
        average=average/N
        print(f"Average execution time: {average:.2f} ms")
        return result
    return wrapper
@benchmark
def pred(classifier, image):
    classifier.predict(image)
           
def main():
    print("Testing HOG and SVM Classifier")
    # Test SVM classifier
    # Open the SVM classifier and test data unseen by the classifier``

    
    
    try:
        cwd = os.getcwd()  
        path = os.path.join(cwd,"Hog_svm_classifier.pkl")
        print("\n",path,"\n")

        with open(path, 'rb') as f:
            (svm_classifier,X_test,y_test, labels) = pickle.load(f)##open cached data
            
    

        
        print("Testing SVM classifier")
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
        plot_confusion_matrix(cm,labels)
        print("Done\n")

    except: # ERROR
        print("Cannot load Classifier - run HOG_and_SVM_classifier_trainer.py first")
    
    input("Press Enter to Exit...")
    print("\nDONE - exiting program")
    
if __name__=="__main__":
    main()


