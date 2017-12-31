"""Adapted from
dataaspirant.com/2017/06/26/random-forest-classifier-python-scikit-learn/"""
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from light_curve_ml.utils.context_util import joinRoot




def main():
    dataPath = joinRoot("data/rf/breast-cancer-wisconsin.csv")
    dataset = pd.read_csv(dataPath)
    print(dataset.describe())






if __name__ == "__main__":
    main()
