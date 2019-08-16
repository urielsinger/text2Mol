import os
import pandas as pd

from utils.general_utils import load_object

def get_molecules_df():
    molecules_df = pd.read_csv(r'..\data\molecules.csv')
    return molecules_df

def get_papers(num_files=None):
    all_df = pd.DataFrame()
    folder_path = r'..\dump\data_pkl_vectorized'
    file_paths = os.listdir(folder_path)
    for i, file_path in enumerate(file_paths):
        if num_files and num_files == i:
            return all_df
        df = pd.DataFrame(load_object(os.path.join(folder_path, file_path)))
        all_df = all_df.append(df)
    return all_df
