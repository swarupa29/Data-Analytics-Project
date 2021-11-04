from os import pipe
import pandas as pd
from bureau import bureau
from utils import pipeline

res = 0
df = pd.read_csv('credit_risk/bureau.csv')
X=  df[:10000]
X_test = df[10000:11000]

def main():

    b = bureau()

    pl = pipeline([
        b.create()
    ])

    pl.fit(X)
    res = pl.predict(X_test)
    print(res)