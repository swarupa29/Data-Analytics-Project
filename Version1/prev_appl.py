import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder
from utils import categorical_mode as mode
from lightgbm import LGBMClassifier

class prev_applications:

    def preprocess(self, df):
        df.replace("NXA", np.nan, inplace=True)
        df.replace("XAP", np.nan, inplace=True)
        df.loc[:,['DAYS_FIRST_DRAWING','DAYS_FIRST_DUE','DAYS_LAST_DUE_1ST_VERSION','DAYS_LAST_DUE','DAYS_TERMINATION']].replace(365243.0, np.nan, inplace=True)
    
    def generate_features(self, df):
        df['RATIO_AMT_APPLICATION_TO_AMT_CREDIT'] = (df['AMT_APPLICATION'] / df['AMT_CREDIT'])
        df['RATIO_AMT_CREDIT_TO_AMT_ANNUITY'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
        df['RATIO_AMT_APPLICATION_TO_AMT_ANNUITY'] = df['AMT_APPLICATION'] / df['AMT_ANNUITY']
        df['DIFF_AMT_CREDIT_AMT_GOODS_PRICE'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
        df['DIFF_AMT_APPLICATION_AMT_GOODS_PRICE'] = df['AMT_APPLICATION'] - df['AMT_GOODS_PRICE']
        df['DIFF_RATE_DOWN_PAYMENT_RATE_INTEREST_PRIMARY'] = df['RATE_DOWN_PAYMENT'] - df['RATE_INTEREST_PRIMARY']
        df['DIFF_DAYS_LAST_DUE_DAYS_FIRST_DUE'] = df['DAYS_LAST_DUE'] - df['DAYS_FIRST_DUE']

        # Collect all numerical feautures
        numerical_features = [
            'AMT_ANNUITY',
            'AMT_APPLICATION', 
            'AMT_CREDIT', 
            'AMT_DOWN_PAYMENT', 
            'AMT_GOODS_PRICE', 
            'HOUR_APPR_PROCESS_START', 
            'RATE_DOWN_PAYMENT', 
            'RATE_INTEREST_PRIMARY',   
            'RATE_INTEREST_PRIVILEGED', 
            'SELLERPLACE_AREA', 
            'DAYS_FIRST_DRAWING', 
            'DAYS_FIRST_DUE', 
            'DAYS_LAST_DUE_1ST_VERSION', 
            'DAYS_LAST_DUE',
            'DAYS_TERMINATION',
            'DAYS_DECISION',
            'RATIO_AMT_APPLICATION_TO_AMT_CREDIT',
            'RATIO_AMT_CREDIT_TO_AMT_ANNUITY',
            'RATIO_AMT_APPLICATION_TO_AMT_ANNUITY',
            'DIFF_AMT_CREDIT_AMT_GOODS_PRICE',
            'DIFF_AMT_APPLICATION_AMT_GOODS_PRICE',
            'DIFF_RATE_DOWN_PAYMENT_RATE_INTEREST_PRIMARY',
            'DIFF_DAYS_LAST_DUE_DAYS_FIRST_DUE',         
        ]

        # Collect all categorical features
        categorical_features = [
            'NAME_CONTRACT_TYPE', 
            'WEEKDAY_APPR_PROCESS_START', 
            'FLAG_LAST_APPL_PER_CONTRACT', 
            'NFLAG_LAST_APPL_IN_DAY',
            'NAME_CASH_LOAN_PURPOSE', 
            'NAME_CONTRACT_STATUS',  
            'NAME_PAYMENT_TYPE', 
            'CODE_REJECT_REASON',  
            'NAME_TYPE_SUITE', 
            'NAME_CLIENT_TYPE', 
            'NAME_GOODS_CATEGORY', 
            'NAME_PORTFOLIO', 
            'NAME_PRODUCT_TYPE', 
            'CHANNEL_TYPE', 
            'NAME_SELLER_INDUSTRY', 
            'NAME_YIELD_GROUP', 
            'PRODUCT_COMBINATION', 
            'NFLAG_INSURED_ON_APPROVAL',
        ]

        # Label Encode each categorical feature
        if hasattr(self, "encoders") == False:
            self.encoders = {}
        
        for feature in categorical_features:
            label_encoder = self.encoders.get(feature, LabelEncoder())
            df[feature] = label_encoder.fit_transform(df[feature])
            self.encoders[feature] = label_encoder
        
        # Define aggregation function for each numerical variable
        aggregations = {}
        for feature in numerical_features:
            aggregations[feature] = ['min', 'max', 'mean', 'var']

        # Replace nan values with value in previous column
        df.fillna(method = 'ffill', inplace = True)
        
        # Define aggregation function for each categorical variable
        for feature in categorical_features:
            aggregations[feature] = [mode]
        
        '''
        df = entire table - prev_applications
        aggreagates = group by id and calculate summary
        '''
        df = df.groupby('SK_ID_CURR').agg({**aggregations})
        df.columns = pd.Index([col_name+"_"+method.upper() for col_name, method in df.columns.tolist()])
        

    def fit(self, df):
        #do preprocessing
        #feature engineering
        #fit model
        #return self
        self.preprocess(df)
        self.generate_features(df)
        lgb = LGBMClassifier(class_weight='balanced')
        train_cols  = set(df.columns) - {'TARGET'} 
        lgb.fit(df[train_cols], df['TARGET'])
        self.lgb = lgb
        return self

    def predict(self, x_test):

        #same preprocessing as in fit
        #predict from model instead of fit
        #return predictions
        L1 = pd.DataFrame(index = x_test['SK_ID_CURR'].unique())
        print("Pre-processing data.....")
        self.preprocess(x_test)
        print("Building new features.....")
        self.generate_features(x_test)
        print("Preparing output.....")
        pred = self.lgb.predict_proba(x_test)[:, 1]
        L1['LGB_PREV_APPLICATIONS'] = pred
        return L1