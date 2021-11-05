import pandas as pd
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder


categorical_cols = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']
to_drop = []

class bureau:

    def __init__(self):
        
        self.categorical_cols = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']
        self.to_drop = []
        return

    def fit(self, joined) -> object:
        self.labelEncoders = {}

       # LABEL ENCODING
        for col in categorical_cols:
            enc = LabelEncoder()
            joined[col] = enc.fit_transform(joined[col])
            self.labelEncoders[col] = enc


        # DROP UNECESSARY FEATURES
        for col in to_drop:
            joined.drop(col, inplace=True)
        
        # Creating dataset to store results in
        df = pd.DataFrame()
        df['SK_ID_CURR'] = joined['SK_ID_CURR'].unique()
        df['TARGET'] = joined[['SK_ID_CURR', 'TARGET']].groupby('SK_ID_CURR').agg({'TARGET':'mean'})
        df.dropna(subset=['TARGET'], inplace=True)
        
        
        # FEATURE ENGINEERING

        # No. of past credits with credit bureau
        bureau_counts = joined[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
        bureau_counts.rename(columns={'SK_ID_BUREAU':'COUNT_BUREAU_CREDITS'}, inplace=True)
        df = df.join(bureau_counts, on='SK_ID_CURR', how='left')
        del bureau_counts
        gc.collect()


        # Aggregations on features
        ops = ['min', 'max', 'mean', 'var']
        aggs = {
            # Numerical features
            'DAYS_CREDIT': ops,
            'CREDIT_DAY_OVERDUE': ops,
            'DAYS_CREDIT_ENDDATE': ops[:3],
            'DAYS_ENDDATE_FACT': ops[:3],
            'AMT_CREDIT_MAX_OVERDUE': ops,
            'CNT_CREDIT_PROLONG': ops,
            'AMT_CREDIT_SUM': ops,
            'AMT_CREDIT_SUM_DEBT': ops,
            'AMT_CREDIT_SUM_LIMIT': ops[:3],
            'AMT_CREDIT_SUM_OVERDUE':ops,
            'DAYS_CREDIT_UPDATE': ops[:3],
            'AMT_ANNUITY': ops

        }
        bureau_agg = joined[['SK_ID_CURR']+list(aggs.keys())].groupby('SK_ID_CURR').agg(aggs)
        bureau_agg.columns = ['_'.join(colname).upper() for colname in bureau_agg.columns]
        df = df.join(bureau_agg, how='left', on='SK_ID_CURR')

        self.aggs = aggs
        self.ops = ops

        del bureau_agg, ops, aggs
        gc.collect()

        # taking most recent value for each categorical feature for each ID 
        cat_df = (
            joined [ ['SK_ID_CURR', 'DAYS_CREDIT'] + categorical_cols]
            .loc[joined [['SK_ID_CURR', 'DAYS_CREDIT'] + categorical_cols]
            .sort_values(['SK_ID_CURR','DAYS_CREDIT']).drop_duplicates('SK_ID_CURR',keep='last').index
        ])

        df = df.merge(cat_df, on='SK_ID_CURR', how='left')
        del cat_df
        gc.collect()

        # JOINING WITH BUREAU_BALANCE
        # TODO

        # dropping ids and days_credit(as it is same as days_credit_max)
        df.drop(['SK_ID_CURR', 'DAYS_CREDIT'], inplace=True, axis=1)
        
        train_cols = list(set(df.columns)-{'TARGET'})

        # lgb 
        lgb = LGBMClassifier(class_weight='balanced')
        lgb.fit(df[train_cols], df['TARGET'])
        self.lgb = lgb
        
        return self
    
    def predict(self, X_test) -> pd.DataFrame:
        
        # Empty dataframe to store model results
        L1 = pd.DataFrame()

       # LABEL ENCODING
        for col in categorical_cols:
            enc = self.labelEncoders[col]
            X_test[col] = enc.transform(X_test[col])


        # DROP UNECESSARY FEATURES
        for col in to_drop:
            X_test.drop(col, inplace=True)
        
        # Creating new empty dataset to store results in
        df = pd.DataFrame()
        df['SK_ID_CURR'] = X_test['SK_ID_CURR'].unique()        
        
        # FEATURE ENGINEERING

        # No. of past credits with credit bureau
        bureau_counts = X_test[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
        bureau_counts.rename(columns={'SK_ID_BUREAU':'COUNT_BUREAU_CREDITS'}, inplace=True)
        df = df.join(bureau_counts, on='SK_ID_CURR', how='left')
        del bureau_counts
        gc.collect()


        # Aggregations on features  
        bureau_agg = X_test[['SK_ID_CURR']+list(self.aggs.keys())].groupby('SK_ID_CURR').agg(self.aggs)
        bureau_agg.columns = ['_'.join(colname).upper() for colname in bureau_agg.columns]
        df = df.join(bureau_agg, how='left', on='SK_ID_CURR')

        del bureau_agg
        gc.collect()

        # taking most recent value for each categorical feature for each ID 
        cat_df = (
            X_test [ ['SK_ID_CURR', 'DAYS_CREDIT'] + categorical_cols]
            .loc[X_test [['SK_ID_CURR', 'DAYS_CREDIT'] + categorical_cols]
            .sort_values(['SK_ID_CURR','DAYS_CREDIT']).drop_duplicates('SK_ID_CURR',keep='last').index
        ])

        df = df.merge(cat_df, on='SK_ID_CURR', how='left')
        del cat_df
        gc.collect()

        # JOINING WITH BUREAU_BALANCE
        # TODO

        # dropping ids and days_credit(as it is same as days_credit_max)
        df.drop(['SK_ID_CURR', 'DAYS_CREDIT'], inplace=True, axis=1)
        
        # lgb 
        lgb = self.lgb
        pred = lgb.predict(df)
        L1['LGB_Classifier_Bureau'] = pred

        return L1