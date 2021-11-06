import pandas as pd
import numpy as np
import os
import gc
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier


categorical_cols = ['NAME_CONTRACT_STATUS']

class credit_balance:
    def __init__(self):
        return None

    def fit(self, df):
        print("Fitting credit_card_ balance")
        
        self.labelEncoders = {}
        for col in categorical_cols:
            enc = LabelEncoder()
            df[col] = enc.fit_transform(df[col])
            self.labelEncoders[col] = enc
            
            
        df['RATIO_AMT_BALANCE_TO_AMT_CREDIT_LIMIT_ACTUAL'] = df['AMT_BALANCE'] / df['AMT_CREDIT_LIMIT_ACTUAL']
        df['SUM_ALL_AMT_DRAWINGS'] = df[['AMT_DRAWINGS_ATM_CURRENT','AMT_DRAWINGS_CURRENT','AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT']].sum(axis=1)
        df['RATIO_AMT_PAYMENT_TOTAL_CURRENT_TO_AMT_TOTAL_RECEIVABLE'] = df['AMT_PAYMENT_TOTAL_CURRENT'] / df['AMT_TOTAL_RECEIVABLE']
        df['RATIO_AMT_PAYMENT_CURRENT_TO_AMT_RECIVABLE'] = df['AMT_PAYMENT_CURRENT'] / df['AMT_RECIVABLE']
        df['SUM_ALL_CNT_DRAWINGS'] = df[['CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT']].sum(axis=1)
        df['RATIO_ALL_AMT_DRAWINGS_TO_ALL_CNT_DRAWINGS'] = df['SUM_ALL_AMT_DRAWINGS'] / df['SUM_ALL_CNT_DRAWINGS']
        df['DIFF_AMT_TOTAL_RECEIVABLE_AMT_PAYMENT_TOTAL_CURRENT'] = df['AMT_TOTAL_RECEIVABLE'] / df['AMT_PAYMENT_TOTAL_CURRENT']
        df['RATIO_AMT_PAYMENT_CURRENT_TO_AMT_PAYMENT_TOTAL_CURRENT'] = df['AMT_PAYMENT_CURRENT'] / df['AMT_PAYMENT_TOTAL_CURRENT']
        df['RATIO_AMT_RECEIVABLE_PRINCIPAL_TO_AMT_RECIVABLE'] = df['AMT_RECEIVABLE_PRINCIPAL'] / df['AMT_RECIVABLE']
        
        aggs = {
            'MONTHS_BALANCE': ['min', 'max', 'size'],
            'AMT_BALANCE': ['min', 'max', 'mean', 'sum'],
            'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean', 'sum', 'var'],
            'AMT_DRAWINGS_ATM_CURRENT': ['max'],
            'AMT_DRAWINGS_CURRENT': ['max'],
            'AMT_DRAWINGS_OTHER_CURRENT': ['max'],
            'AMT_DRAWINGS_POS_CURRENT': ['max'],
            'AMT_INST_MIN_REGULARITY': ['max'],
            'AMT_PAYMENT_CURRENT': ['max'],
            'AMT_PAYMENT_TOTAL_CURRENT': ['max'],
            'AMT_RECEIVABLE_PRINCIPAL': ['mean', 'sum'],
            'AMT_RECIVABLE': ['mean', 'sum'],
            'AMT_TOTAL_RECEIVABLE': ['mean'],
            'CNT_DRAWINGS_ATM_CURRENT': ['max'], 
            'CNT_DRAWINGS_CURRENT': ['max'],
            'CNT_DRAWINGS_OTHER_CURRENT': ['max'],
            'CNT_DRAWINGS_POS_CURRENT': ['max'],
            'CNT_INSTALMENT_MATURE_CUM': ['mean', 'sum'],
            'SK_DPD': ['max', 'sum'],
            'SK_DPD_DEF': ['max', 'sum'],
            'NAME_CONTRACT_STATUS': ['mean'],
            
            #New features
            'RATIO_AMT_BALANCE_TO_AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean'],
            'SUM_ALL_AMT_DRAWINGS': ['min', 'max', 'mean'],
            'RATIO_AMT_PAYMENT_TOTAL_CURRENT_TO_AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean'],
            'RATIO_AMT_PAYMENT_CURRENT_TO_AMT_RECIVABLE': ['min', 'max', 'mean'],
            'SUM_ALL_CNT_DRAWINGS': ['min', 'max', 'mean'],
            'RATIO_ALL_AMT_DRAWINGS_TO_ALL_CNT_DRAWINGS': ['min', 'max', 'mean'],
            'DIFF_AMT_TOTAL_RECEIVABLE_AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean'],
            'RATIO_AMT_PAYMENT_CURRENT_TO_AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean'],
            'RATIO_AMT_RECEIVABLE_PRINCIPAL_TO_AMT_RECIVABLE': ['min', 'max', 'mean'],
        }
        self.aggs = aggs
        cc_aggs = df.groupby('SK_ID_CURR').agg(aggs)
        cc_aggs.columns = pd.Index([i[0] + "_" + i[1].upper() + '_(CREDIT_CARD)' for i in cc_aggs.columns.tolist()])
        cc_aggs['CC_COUNT'] = df.groupby('SK_ID_CURR').size()
        
        train_cols = list(set(cc_aggs.columns)-{'TARGET'})

        # lgb 
        targets = df[['SK_ID_CURR', 'TARGET']].groupby('SK_ID_CURR').agg({'TARGET':'mean'})
        cc_aggs = targets.join(cc_aggs, on='SK_ID_CURR', how='left')
        lgb = LGBMClassifier(class_weight='balanced')
        lgb.fit(cc_aggs[train_cols], cc_aggs['TARGET'])
        self.lgb = lgb
        
        return self
    
    def predict(self, X_test):
        print("Predicting installment_payments")
        
        L1 = pd.DataFrame(index=X_test['SK_ID_CURR'].unique())
        
        for col in categorical_cols:
            enc = self.labelEncoders[col]
            X_test[col] = enc.transform(X_test[col])
        
        edf = pd.DataFrame()
        edf['SK_ID_CURR'] = X_test['SK_ID_CURR'].unique()
        
        X_test['RATIO_AMT_BALANCE_TO_AMT_CREDIT_LIMIT_ACTUAL'] = X_test['AMT_BALANCE'] / X_test['AMT_CREDIT_LIMIT_ACTUAL']
        X_test['SUM_ALL_AMT_DRAWINGS'] = X_test[['AMT_DRAWINGS_ATM_CURRENT','AMT_DRAWINGS_CURRENT','AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT']].sum(axis=1)
        X_test['RATIO_AMT_PAYMENT_TOTAL_CURRENT_TO_AMT_TOTAL_RECEIVABLE'] = X_test['AMT_PAYMENT_TOTAL_CURRENT'] / X_test['AMT_TOTAL_RECEIVABLE']
        X_test['RATIO_AMT_PAYMENT_CURRENT_TO_AMT_RECIVABLE'] = X_test['AMT_PAYMENT_CURRENT'] / X_test['AMT_RECIVABLE']
        X_test['SUM_ALL_CNT_DRAWINGS'] = X_test[['CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT']].sum(axis=1)
        X_test['RATIO_ALL_AMT_DRAWINGS_TO_ALL_CNT_DRAWINGS'] = X_test['SUM_ALL_AMT_DRAWINGS'] / X_test['SUM_ALL_CNT_DRAWINGS']
        X_test['DIFF_AMT_TOTAL_RECEIVABLE_AMT_PAYMENT_TOTAL_CURRENT'] = X_test['AMT_TOTAL_RECEIVABLE'] / X_test['AMT_PAYMENT_TOTAL_CURRENT']
        X_test['RATIO_AMT_PAYMENT_CURRENT_TO_AMT_PAYMENT_TOTAL_CURRENT'] = X_test['AMT_PAYMENT_CURRENT'] / X_test['AMT_PAYMENT_TOTAL_CURRENT']
        X_test['RATIO_AMT_RECEIVABLE_PRINCIPAL_TO_AMT_RECIVABLE'] = X_test['AMT_RECEIVABLE_PRINCIPAL'] / X_test['AMT_RECIVABLE']
        
        edf = X_test.groupby('SK_ID_CURR').agg(self.aggs)
        edf.columns = pd.Index([i[0] + "_" + i[1].upper() + '_(CREDIT_CARD)' for i in edf.columns.tolist()])
        edf['CC_COUNT'] = X_test.groupby('SK_ID_CURR').size()
        
        lgb = self.lgb
        pred = lgb.predict_proba(edf)[:,1]
        L1['Credit_card_balance'] = pred

        return L1