import pandas as pd
import os

def load_dataset():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "spotify_clean_modeling.csv")
    return pd.read_csv(DATA_PATH)
