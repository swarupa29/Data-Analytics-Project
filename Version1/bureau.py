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
from utils import replace_w_nan, categorical_mode as mode

categorical_cols = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']
to_drop = []

class bureau:

    def __init__(self):
        
        self.categorical_cols = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']
        self.to_drop = []
        return

    def fit(self, joined) -> object:
        print("Training on Bureau and BureauBalance")
        self.labelEncoders = {}

       # LABEL ENCODING
        print("Encoding labels...")
        for col in categorical_cols:
            enc = LabelEncoder()
            joined[col] = enc.fit_transform(joined[col])
            self.labelEncoders[col] = enc

        # DROP UNECESSARY FEATURES
        print("Dropping features...")
        for col in to_drop:
            joined.drop(col, inplace=True)
        
        joined = replace_w_nan(joined)
        joined.fillna(method='ffill', inplace=True)

        # Creating dataset to store results in
        df = pd.DataFrame()
        df['SK_ID_CURR'] = joined['SK_ID_CURR'].unique()
        targets = joined[['SK_ID_CURR', 'TARGET']].groupby('SK_ID_CURR', as_index=False).agg({'TARGET':'mean'})
        df = targets.merge(df, on='SK_ID_CURR', how='left') 
        del targets
        
        # FEATURE ENGINEERING
        ops = ['min', 'max', 'mean', 'var']
        self.ops = ops

        # JOINING WITH BUREAU_BALANCE
        bureau_ids = joined[['SK_ID_BUREAU', 'SK_ID_CURR']]
        bureau_balance = pd.read_csv('credit_risk/bureau_balance.csv')
        bureau_balance = bureau_balance.merge(bureau_ids, on='SK_ID_BUREAU', how='inner')
        bureau_balance = replace_w_nan(bureau_balance)
        bureau_balance.fillna(method='ffill', inplace=True)

        print("Encoding labels in bureau balance...")
        enc = LabelEncoder()
        bureau_balance['STATUS'] = enc.fit_transform(bureau_balance['STATUS'])
        self.labelEncoders['STATUS'] = enc

        # Aggregates on bureau balance by bureau_id
        self.balance_aggs = {
            'MONTHS_BALANCE':self.ops[:2],
            'STATUS':self.ops[:2] + [mode]
        }
        print("Applying aggregates on bureau balance ...")
        bureau_balance_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(self.balance_aggs)
        bureau_balance_agg.columns = pd.Index([e[0] + '_' + e[1].upper()  for e in bureau_balance_agg.columns.tolist()])
        bureau_ids = bureau_balance_agg.merge(bureau_ids, on='SK_ID_BUREAU', how='right')
        bureau_ids.fillna(method='ffill', inplace=True)
        del bureau_balance_agg
        gc.collect()

        # Aggregating the aggregates by curr_id
        self.nested_aggs = {
            'MONTHS_BALANCE_MIN': self.ops[:2],
            'MONTHS_BALANCE_MAX': self.ops[:2],
            'STATUS_MIN': ['min', mode],
            'STATUS_MAX': ['max', mode],
            'STATUS_CATEGORICAL_MODE': self.ops[:2]+[mode]
        }
        print("Applying aggregates on aggregated balance data...")
        bureau_balance_agg = bureau_ids.groupby('SK_ID_CURR').agg(self.nested_aggs)
        bureau_balance_agg.columns = pd.Index([e[0] + '_' + e[1].upper()  for e in bureau_balance_agg.columns.tolist()])
        df = df.merge(bureau_balance_agg, on='SK_ID_CURR', how='left')
        del bureau_balance_agg, bureau_ids
        gc.collect()
    
        # No. of past credits with credit bureau
        print("Applying count..")
        bureau_counts = joined[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
        bureau_counts.rename(columns={'SK_ID_BUREAU':'COUNT_BUREAU_CREDITS'}, inplace=True)
        df = df.join(bureau_counts, on='SK_ID_CURR', how='left')
        del bureau_counts
        gc.collect()


        # Aggregations on features
        print("Applying aggregates on numerical data..")
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
        print("Applying aggregates on categorical data")
        cat_df = (
            joined [ ['SK_ID_CURR', 'DAYS_CREDIT'] + categorical_cols]
            .loc[joined [['SK_ID_CURR', 'DAYS_CREDIT'] + categorical_cols]
            .sort_values(['SK_ID_CURR','DAYS_CREDIT']).drop_duplicates('SK_ID_CURR',keep='last').index
        ])

        df = df.merge(cat_df, on='SK_ID_CURR', how='left')
        del cat_df
        gc.collect()
        
        # dropping ids and days_credit(as it is same as days_credit_max)
        df.drop(['SK_ID_CURR', 'DAYS_CREDIT'], inplace=True, axis=1)
        
        train_cols = list(set(df.columns)-{'TARGET'})

        # lgb 
        print("Training weighted LGBM Classifier...")
        lgb = LGBMClassifier(class_weight='balanced')
        lgb.fit(df[train_cols], df['TARGET'])
        self.lgb = lgb
        
        print("Finished processing bureau and bureau_balance...")
        return self
    
    def predict(self, X_test) -> pd.DataFrame:
        print("Fitting bureau and bureau balance")
        # Empty dataframe to store model results
        L1 = pd.DataFrame(index=X_test['SK_ID_CURR'].unique())

       # LABEL ENCODING
        for col in categorical_cols:
            enc = self.labelEncoders[col]
            X_test[col] = enc.transform(X_test[col])


        # DROP UNECESSARY FEATURES
        for col in to_drop:
            X_test.drop(col, inplace=True)
        
        X_test = replace_w_nan(X_test)
        X_test.fillna(method='ffill', inplace=True)

        # Creating new empty dataset to store results in
        df = pd.DataFrame()
        df['SK_ID_CURR'] = X_test['SK_ID_CURR'].unique()      
        
        # FEATURE ENGINEERING

        # JOINING WITH BUREAU_BALANCE
        bureau_ids = X_test[['SK_ID_BUREAU', 'SK_ID_CURR']]
        bureau_balance = pd.read_csv('credit_risk/bureau_balance.csv')
        bureau_balance = bureau_balance.merge(bureau_ids, on='SK_ID_BUREAU', how='inner')
        bureau_balance = replace_w_nan(bureau_balance)
        bureau_balance.fillna(method='ffill', inplace=True)

        print("Encoding labels in bureau balance...")
        bureau_balance['STATUS'] = self.labelEncoders['STATUS'].transform(bureau_balance['STATUS'])

        # Aggregates on bureau balance by bureau_id
    
        print("Applying aggregates on bureau balance ...")
        bureau_balance_agg = bureau_balance.groupby('SK_ID_BUREAU').agg(self.balance_aggs)
        bureau_balance_agg.columns = pd.Index([e[0] + '_' + e[1].upper()  for e in bureau_balance_agg.columns.tolist()])
        bureau_ids = bureau_balance_agg.merge(bureau_ids, on='SK_ID_BUREAU', how='right')
        bureau_ids.fillna(method='ffill', inplace=True)

        del bureau_balance_agg
        gc.collect()

        # Aggregating the aggregates by curr_id

        print("Applying aggregates on aggregated balance data...")
        bureau_balance_agg = bureau_ids.groupby('SK_ID_CURR').agg(self.nested_aggs)
        bureau_balance_agg.columns = pd.Index([e[0] + '_' + e[1].upper()  for e in bureau_balance_agg.columns.tolist()])
        df = df.merge(bureau_balance_agg, on='SK_ID_CURR', how='left')
        del bureau_balance_agg, bureau_ids
        gc.collect()


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

        # dropping ids and days_credit(as it is same as days_credit_max)
        df.drop(['SK_ID_CURR', 'DAYS_CREDIT'], inplace=True, axis=1)
        
        # lgb 
        lgb = self.lgb
        pred = lgb.predict_proba(df)[:, 1]
        L1['LGB_Classifier_Bureau'] = pred

        return L1