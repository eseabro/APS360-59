import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

from features import feature_extractor

local_path = "C:/Users/Emma/Documents/SchoolWork/Fourth Year/APS360/"

def load_data(data_folder):
    """
    This function loads the raw data into variables that can be used for further processing.

    Parameters
    ----------
    data_folder : string
        The folder to load. Likely "warmup/train/" or "phase_1_v2/train/"


    Returns
    -------
    data: pandas dataframe
        Dataframe containing file contents

    """
    # full_path = os.path.join(local_path, data_folder)
    # for file in os.listdir(full_path):
    #     file_data = pd.read_csv(os.path.join(full_path, file), header=0)

    file_data = pd.read_csv(data_folder, header=0)
    return file_data


def visualize_data(data_folder, file):
    """
    This function loads the raw data into variables that can be used for further processing.

    Parameters
    ----------
    data_folder : string
        The folder to load. Likely "warmup/train/" or "phase_1_v2/train/"
    file : string
        The filename to visualize. 


    Returns
    -------
    no return value but produces plots of data.

    """
    full_path = os.path.join(local_path, data_folder)
    file_data = pd.read_csv(os.path.join(full_path, file), header=0)
    file_data.plot(y=['X (m)', 'Y (m)', 'Z (m)'], title="Data Plot All")
    plt.show()


def normalize_data(df):
    """
    This function normalizes the data of all features to be a value
    between 0 and 1.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe containing file contents.

    Returns
    -------
    normalized_df : pandas dataframe
        Dataframe altered to contain normalized features.

    """
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


def remove_duplicates(df):
    """
    This function removes any duplicate rows from the input df.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe containing file contents.

    Returns
    -------
    duplicates_removed_df : pandas dataframe
        Dataframe altered to remove any duplicate rows.

    """
    duplicates_removed_df = df.drop_duplicates()
    return duplicates_removed_df


def handle_missing_values(df):
    """
    This function imputes missing values from the input df. It does so by using
    the mean value from 5 nearest neighbours found in the df.
    
    https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html


    Parameters
    ----------
    df : pandas dataframe
        Dataframe containing file contents.

    Returns
    -------
    imputed_df : pandas dataframe
        Dataframe altered to impute any missing values.

    """
    imputer = KNNImputer()
    imputed_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return imputed_df


if __name__ == "__main__":
    # data_f = "warmup/train/"
    # fname = "1.csv"
    # dataf = load_data(data_f)
    # DATA = feature_extractor(dataf, "test_out.csv")
    # visualize_data(data_f, fname)

    # defining paths according to project's root directory
    data_path = "warmup_train_1.csv"
    df = load_data(data_path)

    # Data cleaning
    duplicates_removed_df = remove_duplicates(df)
    normalized_df = normalize_data(duplicates_removed_df)
    imputed_df = handle_missing_values(normalized_df)

    # Feature extraction
    DATA = feature_extractor(imputed_df, "test_out2.csv")