from os import pipe
import pandas as pd
import gc
from lightgbm import LGBMClassifier

from installments_payments import installments
from bureau import bureau
from application_train import application_train


def main(main_df, test_df, output) -> None:
    train_target = main_df[['TARGET', 'SK_ID_CURR']]
    test_target = test_df['SK_ID_CURR'].copy()
    
    # store results of each table in L1

    L1 = pd.DataFrame(index=test_target)

    to_train = {
        'bureau': True,
        'application_train': True,
        'installments_payments': True,
        'credit_card_balance': True,
        'POS_CASH_balance': True,
        'previous_application': False
    }

    # Bureau +  Bureau Balance
    if to_train['bureau']:
        bureau_df = pd.read_csv('credit_risk/bureau.csv')
        bureau_joined = bureau_df.merge(train_target, on='SK_ID_CURR', how='right')
        bureau_model = bureau()
        bureau_model.fit(bureau_joined)
        bureau_test = bureau_df.merge(test_target, on='SK_ID_CURR', how='right')
        res = bureau_model.predict(bureau_test)
        try:
            L1.drop(res.columns, axis=1, inplace=True)
        except:
            pass
        L1 = L1.join(res, on='SK_ID_CURR', how='left')
        del bureau_df, res, bureau_model
        gc.collect()
        L1.to_csv(f'output/{output}.csv')

    # Installments Payments
    if to_train['installments_payments']:
        install_df = pd.read_csv('credit_risk/installments_payments.csv')
        install_joined = install_df.merge(train_target, on='SK_ID_CURR', how='right')
        install_model = installments()
        install_model.fit(install_joined)
        install_test = install_df.merge(test_target, on='SK_ID_CURR', how='right')
        res = install_model.predict(install_test)
        try:
            L1.drop(res.columns, axis=1, inplace=True)
        except:
            pass
        L1 = L1.join(res, on='SK_ID_CURR', how='left')
        del install_df, install_model, res
        gc.collect()
        L1.to_csv(f'output/{output}.csv')

    # Application_train
    if to_train['application_train']:
        main_train=application_train()
        main_train.fit(main_df)
        res=main_train.predict(test_df)
        print(res)
        try:
            L1.drop(res.columns, axis=1, inplace=True)
        except:
            pass
        L1 = L1.join(res, on='SK_ID_CURR', how='left')
        del main_train, res
        L1.to_csv(f'output/{output}.csv')


    # 
if __name__ == '__main__':
    main_df = pd.read_csv('credit_risk/application_train.csv')
    test_df = pd.read_csv('credit_risk/application_test.csv')
    train_csv='L2_train'
    test_csv='L2_test'
    pred_csv='L2_pred'

    # create training file for L2 model using L1 models
    print("Preparing train data for L2")
    main(main_df, main_df.drop('TARGET', axis=1), train_csv)
    print("Preparing test data for L2")
    main(main_df, test_df, test_csv)
    # train model on file

    L2_train = pd.read_csv(f'output/{train_csv}.csv', index_col=0).join(main_df['TARGET'], on='SK_ID_CURR', how='inner')
    L2_test = pd.read_csv(f'output/{test_csv}.csv', index_col=0)

    print("Fitting L2 on train data")
    lgb = LGBMClassifier(class_weight='balanced')
    lgb.fit(L2_train.drop('TARGET', axis=1), L2_train['TARGET'])

    print("Running predictions from L2 on test data")
    pred = lgb.predict(L2_test)
    L2_pred = pd.DataFrame(index=L2_test.index)
    L2_pred['TARGET'] = pred

    print("Saving predictions")
    L2_pred.to_csv(f'output/{pred_csv}.csv')

    exit(0)