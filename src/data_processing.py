import os
import pandas as pd
import matplotlib.pyplot as plt
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
        dataframe containing file contents

    """
    full_path = os.path.join(local_path, data_folder)
    for file in os.listdir(full_path):
        file_data = pd.read_csv(os.path.join(full_path, file), header=0)
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




if __name__ == "__main__":
    data_f = "warmup/train/"
    fname = "1.csv"
    dataf = load_data(data_f)
    DATA = feature_extractor(dataf, "test_out.csv")
    # visualize_data(data_f, fname)
    # normalize_data()