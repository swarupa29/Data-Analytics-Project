import pandas as pd
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score
# from lightgbm import lightgbmClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

categorical_cols = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']

class bureau:

    def __init__(self):
        return None

    def create(self):
        return self

    def fit(self, df):
        self.df = df
        self.labelEncoders = {}

        for col in categorical_cols:
            enc = LabelEncoder()
            self.df[col] = enc.fit_transform(self.df[col])
            self.labelEncoders[col] = enc

    def predict(self, x_test):

        for col in categorical_cols:
            x_test[col] = self.labelEncoders[col].transform(x_test[col])

        return x_test