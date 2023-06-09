


import pickle
from sklearn.metrics import accuracy_score
 
def main():
    print("Testing SVM Classifier")
    # Test SVM classifier
    # Open the SVM classifier and test data unseen by the classifier
    try:
        with open('./Coursework/Hog_svm_classifier.pkl', 'rb') as f:
            (svm_classifier,X_test,y_test) = pickle.load(f)##open cached data
    

        
        print("Testing SVM classifier")
        predictions = svm_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy:", accuracy)
        print("Done\n")

    except: # ERROR
        print("Cannot load Classifier - run HOG_and_SVM_classifier_trainer.py first")
        
    print("\nDONE - exiting program")
    
if __name__=="__main__":
    main()


