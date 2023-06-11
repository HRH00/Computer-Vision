import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers import Dense

def train_neural_network(features_input, y_train):
    print("Training Neural Network classifier")
    # Convert features and labels to numpy arrays
    X_train, X_test, y_train, y_test = train_test_split(features_input, y_train, test_size=0.2, random_state=0)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Define the neural network architecture
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
    print("Done\n")
    return model, X_test, y_test




def cross_validate_nn(features, labels, n_splits=10):
    print("Performing cross-validation on neural network classifier")
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
        X_train_cval, X_test_cval = features[train_index], features[test_index]
        y_train_cval, y_test_cval = labels[train_index], labels[test_index]

        # Create a neural network model
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=X_train_cval.shape[1]))
        model.add(Dense(num_labels, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model on the training data
        model.fit(X_train_cval, y_train_cval, epochs=10, batch_size=32, verbose=0)

        # Predict the labels for the testing data
        predictions = np.argmax(model.predict(X_test_cval), axis=-1)

        # Calculate the accuracy for the current fold
        accuracy = accuracy_score(y_test_cval, predictions)
        acc_scores_cval.append(accuracy)

        # Calculate the confusion matrix for the current fold
        fold_confusion_matrix = confusion_matrix(y_test_cval, predictions)

        # Aggregate the confusion matrices
        confusion_matrix_agg += fold_confusion_matrix    
        fold_index += 1

    print("Done\n")
    return acc_scores_cval, acc_scores_cval, confusion_matrix_agg