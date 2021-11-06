import pandas as pd
import numpy as np
import os
import gc
from lightgbm import LGBMClassifier


class installments:

    def __init__(self):
        return None

    def create(self):
        return self

    def fit(self, df):
        df['PAYMENT_PERC'] = df['AMT_PAYMENT'] / df['AMT_INSTALMENT']
        df['PAYMENT_DIFF'] = df['AMT_INSTALMENT'] - df['AMT_PAYMENT']
        
        # Days past due and days before due (no negative values)
        df['DPD'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
        df['DBD'] = df['DAYS_INSTALMENT'] - df['DAYS_ENTRY_PAYMENT']
        df['DPD'] = df['DPD'].apply(lambda x: x if x > 0 else 0)
        df['DBD'] = df['DBD'].apply(lambda x: x if x > 0 else 0)
        
        aggs = {
                'NUM_INSTALMENT_VERSION': ['nunique'],
                'DPD': ['max', 'mean', 'sum'],
                'DBD': ['max', 'mean', 'sum'],
                'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
                'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
                'AMT_INSTALMENT': ['max', 'mean', 'sum'],
                'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
                'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
                }
                
        self.aggs = aggs
        
        ins_agg = df.groupby('SK_ID_CURR').agg(aggs)
        ins_agg.columns = pd.Index(['INSTAL_' + i[0] + "_" + i[1].upper() for i in ins_agg.columns.tolist()])
        
        # Count installments accounts
        ins_agg['INSTAL_COUNT'] = df.groupby('SK_ID_CURR').size()
        gc.collect()
        
        train_cols = list(set(ins_agg.columns)-{'TARGET'})

        # lgb 
        targets = df[['SK_ID_CURR', 'TARGET']].groupby('SK_ID_CURR').agg({'TARGET':'mean'})
        ins_agg = targets.join(ins_agg, on='SK_ID_CURR', how='left')
        lgb = LGBMClassifier(class_weight='balanced')
        lgb.fit(ins_agg[train_cols], ins_agg['TARGET'])
        self.lgb = lgb
        
        return self

    def predict(self, X_test):
        L1 = pd.DataFrame(index=X_test['SK_ID_CURR'].unique())
        edf = pd.DataFrame()
        edf['SK_ID_CURR'] = X_test['SK_ID_CURR'].unique()

        X_test['PAYMENT_PERC'] = X_test['AMT_PAYMENT'] / X_test['AMT_INSTALMENT']
        X_test['PAYMENT_DIFF'] = X_test['AMT_INSTALMENT'] - X_test['AMT_PAYMENT']
        
        # Days past due and days before due (no negative values)
        X_test['DPD'] = X_test['DAYS_ENTRY_PAYMENT'] - X_test['DAYS_INSTALMENT']
        X_test['DBD'] = X_test['DAYS_INSTALMENT'] - X_test['DAYS_ENTRY_PAYMENT']
        X_test['DPD'] = X_test['DPD'].apply(lambda x: x if x > 0 else 0)
        X_test['DBD'] = X_test['DBD'].apply(lambda x: x if x > 0 else 0)

        edf = X_test.groupby('SK_ID_CURR').agg(self.aggs)
        edf.columns = pd.Index(['INSTAL_' + i[0] + "_" + i[1].upper() for i in edf.columns.tolist()])
        edf['INSTAL_COUNT'] = X_test.groupby('SK_ID_CURR').size()
        
        lgb = self.lgb
        pred = lgb.predict(edf)
        L1['Installments'] = pred

        return L1