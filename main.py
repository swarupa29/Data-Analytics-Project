from os import pipe
import pandas as pd
import gc

from installments_payments import installments
from bureau import bureau
from application_train import application_train


def main() -> None:
    main_df = pd.read_csv('credit_risk/application_train.csv')
    test_df = pd.read_csv('credit_risk/application_test.csv')
    train_target = main_df[['TARGET', 'SK_ID_CURR']]
    test_target = test_df['SK_ID_CURR'].copy()
    
    # store results of each table in L1
    L1 = pd.DataFrame(index=test_target)

    # Bureau +  Bureau Balance
    bureau_df = pd.read_csv('credit_risk/bureau.csv')
    bureau_joined = bureau_df.merge(train_target, on='SK_ID_CURR', how='right')
    bureau_model = bureau()
    bureau_model.fit(bureau_joined)
    bureau_test = bureau_df.merge(test_target, on='SK_ID_CURR', how='right')
    res = bureau_model.predict(bureau_test)
    L1 = L1.join(res, on='SK_ID_CURR', how='left')
    del bureau_df
    gc.collect()

    # Installments Payments
    install_df = pd.read_csv('credit_risk/installments_payments.csv')
    install_joined = install_df.merge(train_target, on='SK_ID_CURR', how='right')
    install_model = installments()
    install_model.fit(install_joined)
    install_test = install_df.merge(test_target, on='SK_ID_CURR', how='right')
    res = install_model.predict(install_test)
    L1 = L1.join(res, on='SK_ID_CURR', how='left')
    del install_df
    gc.collect()

    # Application_train
    main_train=application_train()
    main_train.fit(main_df)
    res=main_train.predict(test_df)
    L1 = L1.join(res, on='SK_ID_CURR', how='left')
    del install_df
    gc.collect()

    L1.to_csv('output/L1.csv', index=False)

if __name__ == '__main__':
    main()