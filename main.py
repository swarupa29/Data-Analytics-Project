from os import pipe
import pandas as pd
from bureau import bureau
from utils import pipeline

res = 0
main_df = pd.read_csv('credit_risk/application_train.csv')
test_df = pd.read_csv('credit_risk/application_test.csv')
train_target = main_df[['TARGET', 'SK_ID_CURR']]
test_target = test_df['SK_ID_CURR']

bureau_df = pd.read_csv('credit_risk/bureau.csv')
def main() -> None:

    # store results of each table in L1
    L1 = pd.DataFrame()
    
    bureau_joined = bureau_df.merge(train_target, on='SK_ID_CURR', how='right')
    bureau_model = bureau()
    bureau_model.fit(bureau_joined)
    bureau_test = bureau_df.merge(test_target, on='SK_ID_CURR', how='right')
    res = bureau_model.predict(bureau_test)
    
    for col in res.columns:
        L1[col] = res[col]

    L1.to_csv('output/L1.csv', index=False)

if __name__ == '__main__':
    main()