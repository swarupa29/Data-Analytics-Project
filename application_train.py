import numpy as np # linear algebra
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb






class application_train:
    def __init__(self):
        return None

    def create(self):
        return self

    def fit(self, df):
        self.df=df
        self.labelEncoders = {}
        self.model=None

        #dropping all columns with mv>170000
        lst=self.df.isnull().sum()
        l=[]
        for i in range(len(lst)):
            if lst[i] >170000:
                l.append(i)
        self.df.drop(self.df.columns[l],axis=1,inplace=True)

        #fill categorical values with mode
        cat_cols=self.df.select_dtypes('object').nunique().index.tolist()
        for i in cat_cols:
            self.df[i].fillna(self.df[i].mode()[0], inplace=True)

        #for numeric data
        df_num = self.df.select_dtypes(include=np.number) #make a seperate dataset with them
        df_num = df_num.iloc[: , 1:]
        num_cols= self.df.select_dtypes(include=np.number).nunique().index.tolist()

        #divide the numerical columns into numeric categorical and continous variable columns. If the column has less than 10 distinct values, it is treated to be categorical data
        num_cols.pop(0)
        dis_cols=[]
        for i in num_cols:
            if(len(df_num[i].unique())<=10):
                dis_cols.append(i)

        #mv for numeric categories is mode
        for i in dis_cols:
            self.df[i].fillna(value=self.df[i].mode(),inplace=True)

        #imputing other missing values
        df_num = self.df.select_dtypes(include=np.number)
        imputer = IterativeImputer(imputation_order='ascending') #by default the estimator used is bayesian ridge regressor
        imputer.fit(df_num)
        Xtrans = imputer.transform(df_num)

        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        cat_vals = self.df.select_dtypes('object')
        df_new = pd.DataFrame(data = Xtrans,columns = num_cols)
        final_df=pd.concat([df_new,cat_vals], axis=1, join='inner')

        #after getting dataset, pre processing it and training model
        for i in cat_cols:
            enc=LabelEncoder()
            final_df[i]=enc.fit_transform(final_df[i])
            self.labelEncoders[i] = enc

        y=final_df['TARGET']
        x=final_df.drop(['TARGET','SK_ID_CURR'],axis=1)
        model = lgb.LGBMRegressor(learning_rate=0.09,max_depth=-5,random_state=42,max_iterations=120)
        model.fit(x,y)
        self.model=model
        return self

    
    def predict(self, X_test) -> pd.DataFrame:

        L1 = pd.DataFrame()

        cat_cols=X_test.select_dtypes('object').nunique().index.tolist()
        for i in cat_cols:
            X_test[i]=self.labelEncoders[i].transform(X_test[i])
        
        X_test.drop('SK_ID_CURR', inplace=True, axis=1)
        ypred=self.model.predict(X_test)
        L1['main']=ypred
        return L1









    
        
    


