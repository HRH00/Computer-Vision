import HOG_and_SVM_classifier_tester
import HOG_and_SVM_classifier_trainer
import SIFT_and_SVM_classifier_tester
import SIFT_and_SVM_classifier_trainer
import HOG_and_NN_classifier_tester
import HOG_and_NN_classifier_trainer


Program_Issues=[]
def main():
    print("Running all scripts")
    run_script("HOG_and_SVM_classifier_trainer", HOG_and_SVM_classifier_trainer.main)
    run_script("HOG_and_SVM_classifier_tester", HOG_and_SVM_classifier_tester.main)
    run_script("HOG_and_NN_classifier_trainer", HOG_and_NN_classifier_trainer.main)
    run_script("HOG_and_NN_classifier_tester", HOG_and_NN_classifier_tester.main)


    run_script("SIFT_and_SVM_classifier_trainer", SIFT_and_SVM_classifier_trainer.main)
    run_script("SIFT_and_SVM_classifier_tester", SIFT_and_SVM_classifier_tester.main)


    print("Done Running all scripts\n\n")

    if Program_Issues:
        for i in Program_Issues:
            print(f"\033[91m{i}: Failed To Run\033[0m")
                  


def run_script(script_name, main_function):
    try:
        print(f"\033[94mRUNNING {script_name}\033[0m")
        main_function()
        print("Done\n")
    except Exception as e:
        Program_Issues.append(script_name)
        error_message = f"\033[91mERROR Executing {script_name}: {str(e)}\033[0m"
        print(error_message)
    
if __name__=="__main__":
    main()