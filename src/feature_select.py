import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


def feature_selector(data_df, label_df):
    X = data_df.values
    for lab in label_df.columns:
        Y = label_df[lab]

        # feature extraction
        test = SelectKBest(score_func=f_classif, k=4)
        fit = test.fit(X, Y)

        # summarize scores
        inds = np.argpartition(fit.scores_, -20)[-20:]
        
        # summarize selected features
        print(f"The best features for classifying {lab} are: ", end='')
        print(data_df.columns[inds])
        


if __name__ == "__main__":
    path_data = ""
    path_labels = ""
    data_df = pd.read_csv(path_data)
    label_df = pd.read_csv(path_labels)
    feature_selector(data_df, label_df)
