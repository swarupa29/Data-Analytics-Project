# Data-Analytics-Project
Code for the Data Analytics project on Home Credit Default Risk Prediction

## About the Dataset
The dataset consists of 9 csv files

## Glossary
* Credit: Small loan of fixed amount to be paid back later
* Annuity: A payment made at fixed intervals
* Cash Loan: A quick loan requiring minimal documentation
* POS: Point of Sales (Time and place of transaction)

## `application_train.csv`/`application_test.csv`: 
Consists of the target column and other application data of the current applicant. Indexed by `SK_ID_CUR`
which is the unique ID given to each loan application (One row -> One loan).
This ID can also be used to index the column by same name in the following tables
- bureau.csv
- previous_application.csv,
- POS_CASH_balance.csv, 
- credit_card_balance.csv, 
- installments_payments.csv
  
## `bureau.csv`:
Past data from Credit Bureau (An organisation in the US responsible for collecting information to help creditors and lendors. Data is given to it voluntarily and thus the data is not complete). One row represents one credit. Can be indexed by `SK_ID_BUREAU`

## `bureau_balance.csv`:

Monthly balance of previous credits in the Credit Bureau
One row for each month of credit data in each row in bureau.csv. `SK_ID_BUREAU` references the corresponding row in bureau.csv

## `POS_CASH_balance.csv`
Monthly balance of past POS and cash loans

## `Credit_card_balance.csv`
Monthly balance of past credit cards of applicant

## `previous_application.csv`
Details of previous loan applications at Home Credit
One row one application

## `installments_payments.csv`
Repayment history of previous disbursed loans 
1 row for each payment made of payment missed

## `HomeCredit_columns_description.csv`
Description of all the columns used in the dataset






