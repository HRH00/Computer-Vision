import os
import pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Testing HOG and SVM Classifier")
    # Test SVM classifier
    # Open the SVM classifier and test data unseen by the classifier

    
    
    try:
        cwd = os.getcwd()  
        path = os.path.join(cwd,"Hog_svm_classifier.pkl")
        print("\n",path,"\n")

        with open(path, 'rb') as f:
            (svm_classifier,X_test,y_test) = pickle.load(f)##open cached data
            
    

        
        print("Testing SVM classifier")
        predictions = svm_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy Score:", accuracy)
        

        # Load data
        iris = load_iris()
        X = iris.data
        y = iris.target

        # Define the model
        model = RandomForestClassifier()

        # Define the cross-validation procedure
        cv = KFold(n_splits=10, random_state=1, shuffle=True)

        # Evaluate model
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

        # Report performance
        print('Accuracy: %.3f (%.3f)' % (scores.mean(), scores.std()))

        
        print("Done\n")

    except: # ERROR
        print("Cannot load Classifier - run HOG_and_SVM_classifier_trainer.py first")
    
    input("Press Enter to Exit...")
    print("\nDONE - exiting program")
    
if __name__=="__main__":
    main()


