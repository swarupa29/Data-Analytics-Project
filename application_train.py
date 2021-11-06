import numpy as np # linear algebra
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb






class application_train:
    def __init__(self):
        self.to_drop=[]
        self.cat_cols=[]
        self.cont_cols=[]
        self.dis_cols=[]
        return None

    def create(self):
        return self

    def fit(self, df):
        self.df=df
        self.labelEncoders = {}
        self.catModes={}
        self.numModes={}
        self.numMeans={}
        self.model=None

        #dropping all columns with mv>170000
        lst=self.df.isnull().sum()
        l=[]
        for i in range(len(lst)):
            if lst[i] >170000:
                l.append(df.columns[i])
        self.df.drop(l,axis=1,inplace=True)
        self.to_drop=l


        #fill categorical values with mode
        cat_cols=self.df.select_dtypes('object').nunique().index.tolist()
        for i in cat_cols:
            self.df[i].fillna(self.df[i].mode()[0], inplace=True)
            self.catModes[i]=self.df[i].mode()[0]
        self.cat_cols=cat_cols


        #for numeric data
        df_num = self.df.select_dtypes(include=np.number) #make a seperate dataset with them
        num_cols= self.df.select_dtypes(include=np.number).nunique().index.tolist()

        #divide the numerical columns into numeric categorical and continous variable columns. If the column has less than 10 distinct values, it is treated to be categorical data
        dis_cols=[]
        cont_cols=[]
        for i in num_cols:
            if(len(df_num[i].unique())<=10 and i!='TARGET'):
                dis_cols.append(i)
            elif(i!='TARGET'):
                cont_cols.append(i)
        self.dis_cols = dis_cols
        self.cont_cols = cont_cols


        #mv for numeric categories is mode
        for i in dis_cols:
            self.df[i].fillna(value=self.df[i].mode()[0],inplace=True)
            self.numModes[i]=self.df[i].mode()[0]


        #imputing other missing values
        #df_num = self.df.select_dtypes(include=np.number)
        #imputer = IterativeImputer(imputation_order='ascending') #by default the estimator used is bayesian ridge regressor
        #imputer.fit(df_num)
        #Xtrans = imputer.transform(df_num)

        for i in cont_cols:
            val=self.df[i].mean()
            self.df[i].fillna(value=val,inplace=True)
            self.numMeans[i]=val


        #num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        cat_vals = self.df.select_dtypes('object')
        num_vals=self.df.select_dtypes(include=np.number)
        #df_new = pd.DataFrame(data = Xtrans,columns = num_cols)
        final_df=pd.concat([num_vals,cat_vals], axis=1, join='inner')

        #after getting dataset, pre processing it and training model
        for i in self.cat_cols:
            enc=LabelEncoder()
            final_df[i]=enc.fit_transform(final_df[i])
            self.labelEncoders[i] = enc


        y=final_df['TARGET']
        x=final_df.drop(['TARGET','SK_ID_CURR'],axis=1)
        model = lgb.LGBMClassifier(class_weight='balanced')
        model.fit(x,y)
        self.model=model
        return self

    
    def predict(self, X_test) -> pd.DataFrame:

        L1 = pd.DataFrame(index=X_test['SK_ID_CURR'])
    
        X_test.drop(self.to_drop,axis=1,inplace=True)
        for i in self.cat_cols:
            X_test[i].fillna(self.catModes[i], inplace=True)
        for i in self.dis_cols:
            X_test[i].fillna(self.numModes[i], inplace=True)
        
        for i in self.cont_cols:
            X_test[i].fillna(self.numMeans[i], inplace=True)


        for i in self.cat_cols:
            X_test[i]=self.labelEncoders[i].transform(X_test[i])

        
        X_test.drop('SK_ID_CURR', inplace=True, axis=1)
        ypred=self.model.predict_proba(X_test)[:, 1]
        L1['main']=ypred
        return L1









    
        
    


