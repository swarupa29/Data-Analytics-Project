# Data-Analytics-Project
Code for the Data Analytics project on Home Credit Default Risk Prediction

## About the Dataset
The dataset consists of 9 csv files

### Glossary
* Credit: Small loan of fixed amount to be paid back later
* Annuity: A payment made at fixed intervals
* Cash Loan: A quick loan requiring minimal documentation
* POS: Point of Sales (Time and place of transaction)

### `application_train.csv`/`application_test.csv`: 
Consists of the target column and other application data of the current applicant. Indexed by `SK_ID_CUR`
which is the unique ID given to each loan application (One row -> One loan).
This ID can also be used to index the column by same name in the following tables
- bureau.csv
- previous_application.csv,
- POS_CASH_balance.csv, 
- credit_card_balance.csv, 
- installments_payments.csv
  
### `bureau.csv`:
Past data from Credit Bureau (An organisation in the US responsible for collecting information to help creditors and lendors. Data is given to it voluntarily and thus the data is not complete).\ One row represents one credit. Can be indexed by `SK_ID_BUREAU`

### `bureau_balance.csv`:
Monthly balance of previous credits in the Credit Bureau.\
One row for each month of credit data in each row in bureau.csv. `SK_ID_BUREAU` references the corresponding row in bureau.csv

### `POS_CASH_balance.csv`:
Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.\
One row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans.

### `Credit_card_balance.csv`:
Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.\
One row for each month of history of every previous credit in Home Credit.

### `previous_application.csv`:
Details of previous loan applications at Home Credit.\
One row one application related to loans.

### `installments_payments.csv`:
Repayment history of previous disbursed loans.\
One row for each payment made of payment missed.

### `HomeCredit_columns_description.csv`:
Description of all the columns used in the dataset.

![image](https://user-images.githubusercontent.com/54891659/142798878-33484585-0f6f-4939-a564-464c05c5bb51.png)

## Summary of Kernel
Using Stacking Ensemble Model

### EDA
Columns having in excess of 70% data missing was dropped. \
Categorical values were imputed with mode. Quantitative variables were imputed using ridge regression. Checked for skew in the data, and balanced it. \
Visualization : Checked for distribution, relation of data via density plots, normal plots and box plots to check for outliers. Catplot of features for each target class.\
PCA and Dimensionality Reduction : Standardised all variables, with total variance at 90%. Shape reduced from 89 to 47.

### Level 0 Model
Each table of the dataset is trained parallely. LightGBM was used as the performance was highly dependent upon feature engineering of each table.\
ROCAUC Score - Train data : 0.90, Test data : 0.76\
Output of L0 model is input to L1 model.

### Level 1 Model
Combine predictions of L0 models and give final prediction.\
Each table had varying influence on the target variable, hence a simple ANN is used.\
ROCAUC Score - Train data : 0.91, Test data : 0.92

### Final
ROCAUC Score - 0.75 (Kaggle)
