from matplotlib import pyplot as plt
import time
import numpy as np

def plot_confusion_matrix(confusion_matrix, labels,title):
    print("Plotting confusion matrix")
    plt.imshow(confusion_matrix, cmap='Purples')
    plt.colorbar()

    num_labels = len(labels)
    labelsX = [string[:7] for string in labels]#shorten labels to 7 characters
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

##time bench marks a function 50 times
def benchmark(func):
    def wrapper(*args, **kwargs):
        average = 0
        N=100
        for i in range(N):    
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  
            average+=execution_time
        average=average/N
        print(f"Average execution time, out of 100, to Classify single MHI: {average:.2f} ms")
        return result
    return wrapper

@benchmark
def pred(classifier, image):
    classifier.predict(image)
           