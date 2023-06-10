from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np

def train_svm(features, labels):
    print("Training SVM classifier")
    # Split data into training and testing subsets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=0)

    # Train SVM classifier
    svm_classifier = svm.SVC()
    svm_classifier.fit(X_train, y_train)

    return svm_classifier, X_test, y_test

def test_svm(svm_classifier, X_test, y_test):
    print
    print("Testing SVM classifier")
    predictions = svm_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    print("Done\n")
    from sklearn.metrics import confusion_matrix

def cross_validate_svm(features, labels, n_splits=10):
    print("Performing cross-validation on SVM classifier")
    # Convert to np arrays
    features = np.array(features)
    labels = np.array(labels)

    # Reshape features array to 2D if necessary
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores_cval = []
    
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    
    # Initialize an empty confusion matrix with the correct shape
    confusion_matrix_agg = np.zeros((num_labels, num_labels))
    
    fold_index = 0
    for train_index, test_index in kfold.split(features, labels):
        X_train_cval, x_test_cval = features[train_index], features[test_index]
        y_train_cval, y_test_cval = labels[train_index], labels[test_index]

        svm_classifier_cval = svm.SVC()
        svm_classifier_cval.fit(X_train_cval, y_train_cval)

        predictions = svm_classifier_cval.predict(x_test_cval)
        accuracy = accuracy_score(y_test_cval, predictions)
        acc_scores_cval.append(accuracy)

        # Calculate the confusion matrix for the current fold
        fold_confusion_matrix = confusion_matrix(y_test_cval, predictions)

        # Aggregate the confusion matrices
        confusion_matrix_agg += fold_confusion_matrix    
        fold_index += 1
    
    print("Done\n")
    return acc_scores_cval, acc_scores_cval, confusion_matrix_agg
