# Put here the functions that might be required by more than 1 model/table
import pandas as pd
from numpy import nan 
import numpy as np

def replace_w_nan(df):
    df = df.replace('XNA', nan)
    df = df.replace('XAP', nan)
    return df

def categorical_mode(arr):
    return np.argmax(np.bincount(arr))
    
class pipeline:
    """
        Fit the data and test each model in the pipeline individulally, while creating
        the final submission file
    """

    def __init__(self, stages):
        self.names = []
        self.models = []
        
        for name, model in stages:
            self.names.append(name)
            self.models.append(model)
        
        self.n_stages = len(stages)

    def fit(self, X):
        i=1
        for model, name in zip(self.models, self.names):
            print("Training model {i}/{self.n_stages} : {name}...")
            model.fit(X)
            print("Finished training model {name}...")
            i+=1

    def predict(self, X_test):
        res = pd.DataFrame()
        for i in range(self.n_stages):
            res[self.names[i]] = self.models[i].predict(X_test)

        return res