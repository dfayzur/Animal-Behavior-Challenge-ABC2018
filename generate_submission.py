import pandas as pd
import numpy as np

from os import listdir
from os.path import isfile, join

def majority_vote_ensemble():
    last_train_stack_path = "./output/split_train_stack_vote/"
    last_test_stack_path = "./output/split_test_stack_vote/"
    together_train_stack_path = "./output/together_train_stack_vote/"
    together_test_stack_path = "./output/together_test_stack_vote/"

    together_onlyfiles_train = [f for f in listdir(together_train_stack_path) if isfile(join(together_train_stack_path, f))]
    together_onlyfiles_test = [f for f in listdir(together_test_stack_path) if isfile(join(together_test_stack_path, f))]

    last_onlyfiles_train = [f for f in listdir(last_train_stack_path) if isfile(join(last_train_stack_path, f))]
    last_onlyfiles_test = [f for f in listdir(last_test_stack_path) if isfile(join(last_test_stack_path, f))]

    label_2_test = pd.DataFrame()

    col_ext = 'split'
    for i in range(len(last_onlyfiles_test)):
        test_tmp = pd.read_csv(last_test_stack_path + last_onlyfiles_test[i])
        test_tmp.columns = [col_ext + '_' + s for s in test_tmp.columns]
        label_2_test = pd.concat([label_2_test, test_tmp], axis=1)

    col_ext = 'together'
    for i in range(len(together_onlyfiles_train)):
        test_tmp = pd.read_csv(together_test_stack_path + together_onlyfiles_train[i])
        test_tmp.columns = [col_ext + '_' + s for s in test_tmp.columns]
        label_2_test = pd.concat([label_2_test, test_tmp], axis=1)


    test_df = label_2_test

    from scipy import stats

    a, b = stats.mode(test_df, axis=1)
    sub_test = a.reshape(-1)

    submit = pd.DataFrame()
    submit["label"] = np.int32(sub_test)
    submit.to_csv("./Submission/y_submission.txt", index=False,header=False)


if __name__ == "__main__":
    # doing majority vote ensemble, and create submission file
    majority_vote_ensemble()