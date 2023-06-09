import os
import pickle
from sklearn.metrics import accuracy_score
 
def main():
    print("Testing SIFT and SVM Classifier")
    # Test SVM classifier
    # Open the SVM classifier and test data unseen by the classifier
    try:
            
        cwd = os.getcwd()  
        path = os.path.join(cwd,"SIFT_svm_classifier.pkl")
        print("\n",path,"\n")

        with open(path, 'rb') as f:
            (svm_classifier,X_test,y_test) = pickle.load(f)##open cached data

        
        print("Testing SVM classifier")
        predictions = svm_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy:", accuracy)
        print("Done\n")

    except: # ERROR
        print("Cannot load Classifier - run HOG_and_SVM_classifier_trainer.py first")
    
    input("Press Enter to Exit...")
    print("\nDONE - exiting program")
    
if __name__=="__main__":
    main()


