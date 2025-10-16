"""
main.py
"""

import pandas as pd
from predictor import Predictor

FILENAME = "data/boston_buildings.csv"

LABEL = "use_class"
FEATURES = ["yr_built", "sqft", "historic_district", "landmark", "flood", 
           "stormwater", "ct_perc_children_under_5", "ct_perc_over_65"]
BOOLS = ["historic_district", "landmark", "flood", "stormwater"]

ALL = FEATURES.copy()
ALL.append(LABEL)

def encode_bool(series):
    return series.map({"f": 0, "t": 1})

def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())

def preprocess(filename):
    df = pd.read_csv(filename, low_memory=False)
    df = df[ALL].dropna()

    df[BOOLS] = df[BOOLS].apply(lambda col: encode_bool(col))
    return df

def main():
    df = preprocess(FILENAME)
    features_df = df[FEATURES].apply(lambda col: min_max_normalize(col))
    labels_encoded_df = pd.get_dummies(df[LABEL], columns=[LABEL])

    predictor = Predictor(features_df, labels_encoded_df)
    predictor.fit_model()
    predictor.plot_loss()
    
main()